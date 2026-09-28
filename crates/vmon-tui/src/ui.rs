// SPDX-License-Identifier: Apache-2.0

use std::collections::{HashMap, HashSet, VecDeque};

use ratatui::Frame;
use ratatui::layout::{Constraint, Direction, Layout, Rect};
use ratatui::style::{Color, Modifier, Style};
use ratatui::symbols::Marker;
use ratatui::text::{Line, Span};
use ratatui::widgets::{
    Axis, Block, BorderType, Borders, Cell, Chart, Dataset, GraphType, Paragraph, Row, Table,
    TableState, Tabs, Wrap,
};

use vmon_core::cluster::ClusterState;
use vmon_core::gpu::{GpuMetrics, GpuScrape};
use vmon_core::node::{LatencyPct, NodeMetrics};

use crate::theme;

// ── Constants ──

const MAX_HISTORY: usize = 30;
const SPARK_MAX: usize = 20;
const MAX_GRAPH_HISTORY: usize = 120;
const COL_WIDTH: usize = 34;
const SPARK_CHARS: [char; 8] = ['▁', '▂', '▃', '▄', '▅', '▆', '▇', '█'];
// Table layout tiers: Narrow / Normal / Wide
const NARROW_MAX: u16 = 99;
const WIDE_MIN: u16 = 140;

// ── Tabs ──

#[derive(Clone, Copy, PartialEq, Debug)]
pub enum Tab {
    Overview,
    KvXfer,
    /// Mooncake Store dashboard. Only offered (tab bar + Tab-key cycling)
    /// when `--mooncake` targets exist — see [`UiState::mooncake_tab`].
    Mooncake,
    Hardware,
    Info,
}

const TAB_TITLES: [&str; Tab::COUNT] = ["Overview", "KV / Xfer", "Mooncake", "Hardware", "Info"];

impl Tab {
    const COUNT: usize = 5;

    fn index(self) -> usize {
        match self {
            Tab::Overview => 0,
            Tab::KvXfer => 1,
            Tab::Mooncake => 2,
            Tab::Hardware => 3,
            Tab::Info => 4,
        }
    }

    fn from_index(i: usize) -> Self {
        match i % Self::COUNT {
            0 => Tab::Overview,
            1 => Tab::KvXfer,
            2 => Tab::Mooncake,
            3 => Tab::Hardware,
            _ => Tab::Info,
        }
    }
}

/// One entry per rendered detail section (grid columns and full-width blocks
/// alike). Update `ALL` when adding a variant — `section_tab`'s exhaustive
/// match is the compile-time reminder.
#[derive(Clone, Copy, PartialEq, Debug)]
enum DetailSection {
    Latency,
    RequestStats,
    TokenStats,
    HttpStats,
    SpecDecode,
    DynamoConfig,
    CacheStats,
    KvBlockResidency,
    NixlTransfers,
    RemoteKvFetch,
    KvEvents,
    Mooncake,
    PerfMfu,
    GpuHardware,
    RdmaIb,
}

impl DetailSection {
    #[cfg(test)]
    const ALL: [DetailSection; 15] = [
        DetailSection::Latency,
        DetailSection::RequestStats,
        DetailSection::TokenStats,
        DetailSection::HttpStats,
        DetailSection::SpecDecode,
        DetailSection::DynamoConfig,
        DetailSection::CacheStats,
        DetailSection::KvBlockResidency,
        DetailSection::NixlTransfers,
        DetailSection::RemoteKvFetch,
        DetailSection::KvEvents,
        DetailSection::Mooncake,
        DetailSection::PerfMfu,
        DetailSection::GpuHardware,
        DetailSection::RdmaIb,
    ];
}

/// Single source of truth for the section → tab mapping.
fn section_tab(s: DetailSection) -> Tab {
    use DetailSection::*;
    match s {
        Latency | RequestStats | TokenStats | HttpStats | SpecDecode | DynamoConfig => {
            Tab::Overview
        }
        CacheStats | KvBlockResidency | NixlTransfers | RemoteKvFetch | KvEvents => Tab::KvXfer,
        Mooncake => Tab::Mooncake,
        PerfMfu | GpuHardware | RdmaIb => Tab::Hardware,
    }
}

/// server_info keys and their display titles on the Info tab, in render order.
const INFO_SECTIONS: [(&str, &str); 3] = [
    ("vllm_config", "vLLM Config"),
    ("vllm_env", "vLLM Env"),
    ("system_env", "System Env"),
];

// ── Sparkline history snapshot ──

#[derive(Clone, Default)]
struct Snapshot {
    ttft: f64,
    itl: f64,
    e2e: f64,
    queue: f64,
    prefill: f64,
    decode: f64,
    inference: f64,
    tpot: f64,
    kv_cache: f64,  // KV cache usage ratio (0.0-1.0)
    cache_hit: f64, // Prefix cache hit rate (0.0-1.0)
    ext_hit: f64,   // External cache hit rate (0.0-1.0) — meaningful with NIXL
    // KV cache events (aggregate across DP ranks)
    kv_stored: f64,
    kv_evicted: f64,
    kv_active: f64,
    kv_tokens: f64,
    kv_churn: f64,
    // Async remote-KV fetch stage gauges
    kv_fetch_wait: f64,
    kv_fetch_recv: f64,
    kv_fetch_done: f64,
}

impl Snapshot {
    fn from_metrics(n: &NodeMetrics) -> Self {
        let (kv_stored, kv_evicted, kv_active, kv_tokens, kv_churn) =
            if let Some(ranks) = &n.kv_events {
                let s: u64 = ranks.iter().map(|k| k.blocks_stored).sum();
                let e: u64 = ranks.iter().map(|k| k.blocks_removed).sum();
                let a: i64 = ranks.iter().map(|k| k.active_blocks).sum();
                let t: u64 = ranks.iter().map(|k| k.tokens_stored).sum();
                (s as f64, e as f64, a as f64, t as f64, (s + e) as f64)
            } else {
                (0.0, 0.0, 0.0, 0.0, 0.0)
            };
        Self {
            ttft: n.win_ttft.p99,
            itl: n.win_itl.p99,
            e2e: n.win_e2e.p99,
            queue: n.win_queue.p99,
            prefill: n.win_prefill.p99,
            decode: n.win_decode.p99,
            inference: n.win_inference.p99,
            tpot: n.win_tpot.p99,
            kv_cache: n.kv_cache_usage,
            cache_hit: n.prefix_cache_hit_rate,
            ext_hit: n.external_cache_hit_rate,
            kv_stored,
            kv_evicted,
            kv_active,
            kv_tokens,
            kv_churn,
            kv_fetch_wait: n.kv_fetch_waiting_to_start,
            kv_fetch_recv: n.kv_fetch_in_progress,
            kv_fetch_done: n.kv_fetch_completed_waiting,
        }
    }
}

// ── Graph history sample ──

#[derive(Clone)]
struct GraphSample {
    prompt_tps: f64,
    generation_tps: f64,
    req_per_sec: f64,
    ts: std::time::Instant,
}

impl GraphSample {
    fn from_metrics(n: &NodeMetrics) -> Self {
        Self {
            prompt_tps: n.prompt_tps,
            generation_tps: n.generation_tps,
            req_per_sec: n.requests_per_sec,
            ts: std::time::Instant::now(),
        }
    }
}

// ── UI State ──

/// A pending user action that requires confirmation.
#[derive(Clone)]
pub enum PendingAction {
    Sleep(String),
    WakeUp(String),
    ResetCache(String),
    ResetCacheAll(Vec<String>),
}

impl PendingAction {
    pub fn prompt(&self) -> String {
        match self {
            PendingAction::Sleep(addr) => format!("Sleep {addr}? (y/N)"),
            PendingAction::WakeUp(addr) => format!("Wake up {addr}? (y/N)"),
            PendingAction::ResetCache(addr) => format!("Reset prefix cache on {addr}? (y/N)"),
            PendingAction::ResetCacheAll(addrs) => {
                format!("Reset prefix cache on all {} nodes? (y/N)", addrs.len())
            }
        }
    }
}

/// Replay playback state shared between the replay task and the TUI.
pub struct ReplayState {
    pub total_samples: usize,
    pub total_duration_secs: f64,
    pub current_sample: std::sync::atomic::AtomicUsize,
    pub current_elapsed: std::sync::atomic::AtomicU64, // f64 bits
    pub paused: std::sync::atomic::AtomicBool,
    pub speed: std::sync::atomic::AtomicU32, // x10: 10=1x, 20=2x ... up to 10000=1000x
}

impl ReplayState {
    pub fn new(total_samples: usize, total_duration_secs: f64) -> Self {
        Self {
            total_samples,
            total_duration_secs,
            current_sample: std::sync::atomic::AtomicUsize::new(0),
            current_elapsed: std::sync::atomic::AtomicU64::new(0),
            paused: std::sync::atomic::AtomicBool::new(false),
            speed: std::sync::atomic::AtomicU32::new(10),
        }
    }

    pub fn get_elapsed(&self) -> f64 {
        f64::from_bits(self.current_elapsed.load(std::sync::atomic::Ordering::Relaxed))
    }

    pub fn set_elapsed(&self, v: f64) {
        self.current_elapsed.store(v.to_bits(), std::sync::atomic::Ordering::Relaxed);
    }

    pub fn get_sample(&self) -> usize {
        self.current_sample.load(std::sync::atomic::Ordering::Relaxed)
    }

    pub fn is_paused(&self) -> bool {
        self.paused.load(std::sync::atomic::Ordering::Relaxed)
    }

    pub fn toggle_pause(&self) {
        self.paused.fetch_xor(true, std::sync::atomic::Ordering::Relaxed);
    }

    pub fn get_speed(&self) -> f64 {
        self.speed.load(std::sync::atomic::Ordering::Relaxed) as f64 / 10.0
    }

    pub fn speed_up(&self) {
        let cur = self.speed.load(std::sync::atomic::Ordering::Relaxed);
        // Stored value is ×10 (get_speed divides by 10), so these map to the
        // displayed 1/2/5/10/20/50/100/200/500/1000x ladder.
        let new = match cur {
            10 => 20,
            20 => 50,
            50 => 100,
            100 => 200,
            200 => 500,
            500 => 1000,
            1000 => 2000,
            2000 => 5000,
            5000 => 10000,
            _ => cur,
        };
        self.speed.store(new, std::sync::atomic::Ordering::Relaxed);
    }

    pub fn speed_down(&self) {
        let cur = self.speed.load(std::sync::atomic::Ordering::Relaxed);
        let new = match cur {
            10000 => 5000,
            5000 => 2000,
            2000 => 1000,
            1000 => 500,
            500 => 200,
            200 => 100,
            100 => 50,
            50 => 20,
            20 => 10,
            _ => cur,
        };
        self.speed.store(new, std::sync::atomic::Ordering::Relaxed);
    }
}

/// Active in-TUI collection session. Records `ClusterState` snapshots into a
/// `TimeSeriesCollector`, and on stop renders the same JSON shape as
/// `vmon collect` so the file is replay-compatible.
pub struct CollectSession {
    /// Output file path (rendered once, on stop).
    pub path: std::path::PathBuf,
    pub collector: vmon_report::collector::TimeSeriesCollector,
    /// True when paused. Paused samples are skipped and the collector's start
    /// instant is shifted forward on resume so `elapsed_secs` excludes idle gaps.
    pub paused: bool,
    /// Wall-clock instant pause began (Some only while paused).
    pub pause_started_at: Option<tokio::time::Instant>,
}

impl CollectSession {
    pub fn sample_count(&self) -> usize {
        self.collector.samples.len()
    }

    /// Active recording duration (excludes paused time, since `start` is
    /// shifted forward on resume).
    pub fn active_elapsed(&self) -> std::time::Duration {
        let now = tokio::time::Instant::now();
        // While paused, use the pause-start instant as "now" so the displayed
        // timer freezes during pause.
        let effective_now = self.pause_started_at.unwrap_or(now);
        effective_now.saturating_duration_since(self.collector.start)
    }

    pub fn pause(&mut self) {
        if !self.paused {
            self.paused = true;
            self.pause_started_at = Some(tokio::time::Instant::now());
        }
    }

    pub fn resume(&mut self) {
        if self.paused {
            if let Some(t0) = self.pause_started_at.take() {
                let paused_for = tokio::time::Instant::now().saturating_duration_since(t0);
                self.collector.start += paused_for;
            }
            self.paused = false;
        }
    }
}

pub struct UiState {
    pub table_state: TableState,
    pub selected: usize,
    pub show_detail: bool,
    pub active_tab: Tab,
    pub detail_scroll: u16,
    pub search_active: bool,
    pub search_query: String,
    /// Engine sub-tab: 0 = All (aggregate), 1+ = per-engine (DP0, DP1, ...).
    pub engine_tab: usize,
    /// Maps visual table row → (node_index, engine_tab). Built each frame by draw_table.
    row_map: Vec<(usize, usize)>,
    /// Addresses of nodes with expanded DP engine sub-rows. Everything else
    /// renders collapsed, so nodes joining later (follow mode picking up a
    /// new job) start collapsed too. Keyed by addr, not index — indices
    /// shift whenever the node list changes and would flip fold state on
    /// unrelated rows.
    expanded: HashSet<String>,
    history: HashMap<String, VecDeque<Snapshot>>,
    graph_history: HashMap<String, VecDeque<GraphSample>>,
    pub show_graph: bool,
    /// Pending action awaiting confirmation.
    pub pending_action: Option<PendingAction>,
    /// Transient status message (e.g., action result).
    pub status_message: Option<(String, tokio::time::Instant)>,
    /// Replay state (None = live mode).
    pub replay: Option<std::sync::Arc<ReplayState>>,
    /// Active in-TUI collection session (None = not recording).
    pub collect: Option<CollectSession>,
    /// Diff mode: index of the second node to compare against selected node.
    /// None = diff mode inactive.
    pub diff_target: Option<usize>,
    /// True while user is picking the diff target node.
    pub diff_picking: bool,
    /// Sticky table column flags — once a column becomes visible, it stays.
    sticky_flags: TableFlags,
    /// Scrape interval (live mode only). Used to detect stale rows. None
    /// disables staleness checks (replay mode).
    pub scrape_interval: Option<std::time::Duration>,
    /// Whether the Mooncake tab is offered. Refreshed from the cluster each
    /// frame: true when `--mooncake` targets exist (reachable or not).
    pub mooncake_tab: bool,
    /// Node table area captured each draw, for mouse hit-testing.
    table_area: Rect,
    /// Tab bar area captured each draw; zero-sized when the bar isn't drawn
    /// (graph mode, detail hidden).
    tabs_area: Rect,
    /// Last table left-click (visual row, time) for double-click detection.
    last_table_click: Option<(usize, std::time::Instant)>,
}

impl Default for UiState {
    fn default() -> Self {
        Self::new()
    }
}

impl UiState {
    pub fn new() -> Self {
        let mut table_state = TableState::default();
        table_state.select(Some(0));
        Self {
            table_state,
            selected: 0,
            show_detail: true,
            active_tab: Tab::Overview,
            detail_scroll: 0,
            search_active: false,
            search_query: String::new(),
            engine_tab: 0,
            row_map: Vec::new(),
            expanded: HashSet::new(),
            history: HashMap::new(),
            graph_history: HashMap::new(),
            show_graph: false,
            pending_action: None,
            status_message: None,
            replay: None,
            collect: None,
            diff_target: None,
            diff_picking: false,
            sticky_flags: TableFlags {
                has_gpu: false,
                has_mfu: false,
                has_load: false,
                has_sleep: false,
                has_ib: false,
                has_pcie: false,
                has_tc: false,
                has_nixl: false,
                has_multi_job: false,
                show_power: false,
                show_nvl: false,
                show_model: false,
            },
            scrape_interval: None,
            mooncake_tab: false,
            table_area: Rect::default(),
            tabs_area: Rect::default(),
            last_table_click: None,
        }
    }

    /// Total selectable rows in the table.
    pub fn row_count(&self) -> usize {
        self.row_map.len()
    }

    pub fn record_cluster(&mut self, cluster: &ClusterState) {
        for node in &cluster.nodes {
            if !node.is_healthy {
                continue;
            }
            let hist = self.history.entry(node.addr.clone()).or_default();
            hist.push_back(Snapshot::from_metrics(node));
            if hist.len() > MAX_HISTORY {
                hist.pop_front();
            }
            let ghist = self.graph_history.entry(node.addr.clone()).or_default();
            ghist.push_back(GraphSample::from_metrics(node));
            if ghist.len() > MAX_GRAPH_HISTORY {
                ghist.pop_front();
            }
        }
    }

    pub fn select_next(&mut self) {
        let count = self.row_map.len();
        if count == 0 {
            return;
        }
        let row = self.table_state.selected().unwrap_or(0);
        let new_row = (row + 1).min(count - 1);
        self.apply_row_selection(new_row);
    }

    pub fn select_prev(&mut self) {
        let row = self.table_state.selected().unwrap_or(0);
        let new_row = row.saturating_sub(1);
        self.apply_row_selection(new_row);
    }

    fn apply_row_selection(&mut self, row: usize) {
        self.table_state.select(Some(row));
        if let Some(&(node_idx, eng)) = self.row_map.get(row) {
            let node_changed = self.selected != node_idx;
            self.selected = node_idx;
            self.engine_tab = eng;
            if node_changed {
                self.detail_scroll = 0;
            }
        }
    }

    pub fn toggle_detail(&mut self) {
        self.show_detail = !self.show_detail;
    }

    pub fn toggle_graph(&mut self) {
        self.show_graph = !self.show_graph;
    }

    pub fn next_tab(&mut self) {
        self.step_tab(1);
    }

    pub fn prev_tab(&mut self) {
        self.step_tab(Tab::COUNT - 1);
    }

    /// Advance the active tab by `step` (modulo), skipping the Mooncake tab
    /// while it isn't offered. `step` is 1 (forward) or COUNT-1 (backward),
    /// so applying it twice hops over exactly the one hidden tab.
    fn step_tab(&mut self, step: usize) {
        let mut t = Tab::from_index(self.active_tab.index() + step);
        if t == Tab::Mooncake && !self.mooncake_tab {
            t = Tab::from_index(t.index() + step);
        }
        self.active_tab = t;
        self.detail_scroll = 0;
    }

    /// Route a mouse event using the layout areas captured by the last draw:
    /// left-click selects the table row / switches to the clicked tab, a
    /// double-click on a table row toggles its DP engine fold (same as
    /// Space); the wheel moves the selection over the table and scrolls the
    /// detail pane elsewhere. Confirm dialogs stay keyboard-only.
    pub fn handle_mouse(&mut self, ev: crossterm::event::MouseEvent, nodes: &[NodeMetrics]) {
        use crossterm::event::{MouseButton, MouseEventKind};
        /// Two clicks on the same row within this window count as a double-click.
        const DOUBLE_CLICK_WINDOW: std::time::Duration = std::time::Duration::from_millis(400);
        if self.pending_action.is_some() {
            return;
        }
        let pos = ratatui::layout::Position::new(ev.column, ev.row);
        let over_table = self.table_area.contains(pos);
        match ev.kind {
            MouseEventKind::Down(MouseButton::Left) => {
                if self.tabs_area.contains(pos) {
                    if let Some(t) = self.tab_at(ev.column - self.tabs_area.x) {
                        if t != self.active_tab {
                            self.active_tab = t;
                            self.detail_scroll = 0;
                        }
                    }
                } else if over_table {
                    // Table layout: row 0 border, row 1 header, data rows
                    // after — shifted by the widget's internal scroll offset.
                    let y = (ev.row - self.table_area.y) as usize;
                    if let Some(rel) = y.checked_sub(2) {
                        let row = rel + self.table_state.offset();
                        if row < self.row_map.len() {
                            self.apply_row_selection(row);
                            // take() resets the anchor, so a triple-click is
                            // one double plus a fresh single.
                            let is_double = self.last_table_click.take().is_some_and(|(r, t)| {
                                r == row && t.elapsed() < DOUBLE_CLICK_WINDOW
                            });
                            if is_double {
                                self.toggle_engines(nodes);
                            } else {
                                self.last_table_click = Some((row, std::time::Instant::now()));
                            }
                        }
                    }
                }
            }
            MouseEventKind::ScrollUp if over_table => self.select_prev(),
            MouseEventKind::ScrollDown if over_table => self.select_next(),
            MouseEventKind::ScrollUp => self.scroll_detail_up(),
            MouseEventKind::ScrollDown => self.scroll_detail_down(),
            _ => {}
        }
    }

    /// Which tab a click at `x` (column relative to the tab bar's left edge)
    /// lands on. Replicates ratatui's `Tabs` layout for the visible tabs:
    /// one padding column, the pre-padded title, one padding column, then a
    /// one-column divider between tabs. Divider columns hit nothing.
    fn tab_at(&self, x: u16) -> Option<Tab> {
        let mut cursor: u16 = 0;
        for i in 0..Tab::COUNT {
            let t = Tab::from_index(i);
            if t == Tab::Mooncake && !self.mooncake_tab {
                continue;
            }
            // " {title} " from draw_tabs plus the widget's outer padding.
            let w = TAB_TITLES[t.index()].chars().count() as u16 + 4;
            if x < cursor + w {
                return (x >= cursor).then_some(t);
            }
            cursor += w + 1; // divider column
        }
        None
    }

    pub fn scroll_detail_down(&mut self) {
        self.detail_scroll = self.detail_scroll.saturating_add(3);
    }

    pub fn scroll_detail_up(&mut self) {
        self.detail_scroll = self.detail_scroll.saturating_sub(3);
    }

    pub fn enter_search(&mut self) {
        self.search_active = true;
        self.search_query.clear();
        self.detail_scroll = 0;
    }

    pub fn exit_search(&mut self) {
        if self.search_active {
            self.search_active = false;
        } else if !self.search_query.is_empty() {
            self.search_query.clear();
            self.detail_scroll = 0;
        }
    }

    pub fn search_push(&mut self, c: char) {
        self.search_query.push(c);
        self.detail_scroll = 0;
    }

    pub fn search_pop(&mut self) {
        self.search_query.pop();
        self.detail_scroll = 0;
    }

    /// Toggle DP engine sub-row visibility.
    /// On cluster row: toggle all nodes. On node/sub-row: toggle that node.
    pub fn toggle_engines(&mut self, nodes: &[NodeMetrics]) {
        if self.selected == usize::MAX {
            // Cluster row: toggle all
            let any_expanded = nodes.iter().any(|n| self.expanded.contains(&n.addr));
            if any_expanded {
                self.expanded.clear();
            } else {
                self.expanded.extend(nodes.iter().map(|n| n.addr.clone()));
            }
        } else if let Some(node) = nodes.get(self.selected) {
            if !self.expanded.remove(&node.addr) {
                self.expanded.insert(node.addr.clone());
            }
        }
    }
}

// ── Gauge bar rendering ──

/// KV cache usage percentage cell.
fn kv_cell(ratio: f64) -> Cell<'static> {
    let pct = ratio * 100.0;
    Cell::from(Span::styled(
        format!("{:.1}%", pct),
        Style::default().fg(theme::load_color(pct)),
    ))
}

/// KV cache cell with sparkline trend (Wide tier).
fn kv_cell_spark(ratio: f64, spark: &str) -> Cell<'static> {
    let pct = ratio * 100.0;
    let color = theme::load_color(pct);
    Cell::from(Line::from(vec![
        Span::styled(format!("{:>5.1}%", pct), Style::default().fg(color)),
        Span::styled(spark.to_string(), Style::default().fg(color)),
    ]))
}

/// Build a horizontal bar for KV cache usage. Format: " {bar8}" where the bar
/// uses sub-cell precision (eighths). Trend is read from frame-to-frame motion.
fn kv_sparkline(history: &VecDeque<Snapshot>) -> String {
    let current = history.back().map(|s| s.kv_cache).unwrap_or(0.0);
    bar_only(current, history)
}

/// Prefix cache hit rate percentage cell.
fn hit_cell(ratio: f64) -> Cell<'static> {
    let pct = ratio * 100.0;
    Cell::from(Span::styled(
        format!("{:.1}%", pct),
        Style::default().fg(theme::TEXT),
    ))
}

/// Cache hit cell with sparkline trend (Wide tier).
fn hit_cell_spark(ratio: f64, spark: &str) -> Cell<'static> {
    let pct = ratio * 100.0;
    Cell::from(Line::from(vec![
        Span::styled(format!("{:>5.1}%", pct), Style::default().fg(theme::TEXT)),
        Span::styled(spark.to_string(), Style::default().fg(theme::ACCENT)),
    ]))
}

/// Build a horizontal bar for prefix cache hit rate.
fn hit_sparkline(history: &VecDeque<Snapshot>) -> String {
    let current = history.back().map(|s| s.cache_hit).unwrap_or(0.0);
    bar_only(current, history)
}

/// External cache hit rate percentage cell. Same layout as `hit_cell`; only
/// rendered when NIXL/PD-disaggregation makes the external cache meaningful.
fn ext_cell(ratio: f64) -> Cell<'static> {
    let pct = ratio * 100.0;
    Cell::from(Span::styled(
        format!("{:.1}%", pct),
        Style::default().fg(theme::TEXT),
    ))
}

/// External cache hit rate cell with sparkline trend (Wide tier). Uses
/// `theme::PURPLE` for the bar so it visually separates from the adjacent
/// prefix-cache bar which uses `theme::ACCENT`.
fn ext_cell_spark(ratio: f64, spark: &str) -> Cell<'static> {
    let pct = ratio * 100.0;
    Cell::from(Line::from(vec![
        Span::styled(format!("{:>5.1}%", pct), Style::default().fg(theme::TEXT)),
        Span::styled(spark.to_string(), Style::default().fg(theme::PURPLE)),
    ]))
}

fn ext_sparkline(history: &VecDeque<Snapshot>) -> String {
    let current = history.back().map(|s| s.ext_hit).unwrap_or(0.0);
    bar_only(current, history)
}

/// Render `" {bar8}"` (1 leading space + 8-cell bar with eighths precision).
/// Returns empty string if history is empty.
fn bar_only(current: f64, history: &VecDeque<Snapshot>) -> String {
    if history.is_empty() {
        return String::new();
    }
    format!(" {}", bar_str(current * 100.0, 8))
}

const BAR_CHARS: [char; 9] = [' ', '▏', '▎', '▍', '▌', '▋', '▊', '▉', '█'];

/// Render a horizontal bar of given cell width (each cell = 1/width of 100%),
/// using sub-cell precision in 1/8 increments. Output is exactly `width` chars.
fn bar_str(pct: f64, width: usize) -> String {
    let pct = pct.clamp(0.0, 100.0);
    let total_eighths = (pct / 100.0 * width as f64 * 8.0).round() as usize;
    let full_cells = (total_eighths / 8).min(width);
    let partial = total_eighths % 8;
    let mut s = String::with_capacity(width * 3);
    for _ in 0..full_cells {
        s.push('█');
    }
    if full_cells < width {
        s.push(BAR_CHARS[partial]);
        for _ in (full_cells + 1)..width {
            s.push(' ');
        }
    }
    s
}

// ── Sparkline helpers ──

/// Sparkline over the last `min(max_points, SPARK_MAX)` values. Rows that
/// must fit a fixed column pass the remaining display width as `max_points`
/// so the chart shrinks (down to nothing) instead of wrapping the line.
fn sparkline_capped(values: &[f64], max_points: usize) -> String {
    let cap = max_points.min(SPARK_MAX);
    if values.is_empty() || cap == 0 {
        return String::new();
    }
    let start = values.len().saturating_sub(cap);
    let vals = &values[start..];
    let min = vals.iter().copied().fold(f64::INFINITY, f64::min);
    let max = vals.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    let range = max - min;
    vals.iter()
        .map(|&v| {
            if range < 1e-9 {
                SPARK_CHARS[3]
            } else {
                let idx = ((v - min) / range * 7.0).round() as usize;
                SPARK_CHARS[idx.min(7)]
            }
        })
        .collect()
}

fn extract(history: &VecDeque<Snapshot>, f: fn(&Snapshot) -> f64) -> Vec<f64> {
    history.iter().map(f).collect()
}

/// Minimum column width for a `latency_entry` to be worth two-column layout:
/// label (12) + right-aligned value (23) + a few sparkline points.
const LATENCY_ENTRY_MIN_W: usize = 43;

fn latency_entry(label: &str, pct: &LatencyPct, hist: &[f64], max_width: usize) -> Line<'static> {
    use unicode_width::UnicodeWidthStr;
    let val = format!(
        "{} {}/{}/{}",
        format_ms(pct.mean),
        format_ms(pct.p50),
        format_ms(pct.p90),
        format_ms(pct.p99),
    );
    let label_str = format!("  {:<10}", label);
    let val_str = format!("{:>22} ", val);
    // Sparkline only gets what's left of the column so the row never wraps
    // (second-scale latencies push the value span past its 22-col minimum).
    let used =
        UnicodeWidthStr::width(label_str.as_str()) + UnicodeWidthStr::width(val_str.as_str());
    let spark = sparkline_capped(hist, max_width.saturating_sub(used));
    let color = theme::latency_color(pct.p99);
    Line::from(vec![
        Span::styled(label_str, Style::default().fg(theme::ACCENT)),
        Span::styled(
            val_str,
            Style::default().fg(theme::BRIGHT).add_modifier(Modifier::BOLD),
        ),
        Span::styled(spark, Style::default().fg(color)),
    ])
}

fn stat_row_spark(
    label: &str,
    value: &str,
    hist: &[f64],
    color: Color,
    max_width: usize,
) -> Line<'static> {
    use unicode_width::UnicodeWidthStr;
    let label_str = format!("  {:<12}", label);
    let val_str = format!("{:>10} ", value);
    let used =
        UnicodeWidthStr::width(label_str.as_str()) + UnicodeWidthStr::width(val_str.as_str());
    let spark = sparkline_capped(hist, max_width.saturating_sub(used));
    Line::from(vec![
        Span::styled(label_str, Style::default().fg(theme::ACCENT)),
        Span::styled(val_str, Style::default().fg(theme::BRIGHT)),
        Span::styled(spark, Style::default().fg(color)),
    ])
}

fn stat_row(label: &str, value: &str) -> Line<'static> {
    Line::from(vec![
        Span::styled(
            format!("  {:<12}", label),
            Style::default().fg(theme::ACCENT),
        ),
        Span::styled(value.to_string(), Style::default().fg(theme::TEXT)),
    ])
}

/// `stat_row` with a custom value color (e.g. for thresholded alerts).
fn stat_row_color(label: &str, value: &str, color: ratatui::style::Color) -> Line<'static> {
    Line::from(vec![
        Span::styled(
            format!("  {:<12}", label),
            Style::default().fg(theme::ACCENT),
        ),
        Span::styled(value.to_string(), Style::default().fg(color)),
    ])
}

// ── N-column merge ──

/// Wrap spans into multiple lines, each fitting within `max_width` display
/// columns. Continuation lines are left-aligned under the value (indented by
/// first span width).
///
/// Uses display column width (CJK / emoji = 2, ASCII = 1) so a model name
/// containing wide characters doesn't overflow the column. Wraps on character
/// boundaries — wide chars are atomic: we don't split them and we don't place
/// one in a 1-column gap.
fn wrap_spans(spans: Vec<Span<'static>>, max_width: usize) -> Vec<Vec<Span<'static>>> {
    use unicode_width::{UnicodeWidthChar, UnicodeWidthStr};
    if max_width == 0 {
        return vec![spans];
    }
    let total: usize = spans.iter().map(|s| UnicodeWidthStr::width(s.content.as_ref())).sum();
    if total <= max_width {
        return vec![spans];
    }
    // Continuation indent = first span display width (the label), capped at half column.
    let indent = spans
        .first()
        .map(|s| UnicodeWidthStr::width(s.content.as_ref()))
        .unwrap_or(0)
        .min(max_width / 2);
    let cont_capacity = max_width.saturating_sub(indent).max(1);

    let mut lines: Vec<Vec<Span<'static>>> = Vec::new();
    let mut cur_line: Vec<Span<'static>> = Vec::new();
    let mut cur_w: usize = 0;
    let mut cur_limit = max_width; // first line: full width

    for span in spans {
        let style = span.style;
        let mut chars: std::collections::VecDeque<char> = span.content.chars().collect();
        while !chars.is_empty() {
            let mut avail = cur_limit.saturating_sub(cur_w);
            if avail == 0 {
                lines.push(std::mem::take(&mut cur_line));
                cur_line = vec![Span::raw(" ".repeat(indent))];
                cur_w = 0;
                cur_limit = cont_capacity;
                continue;
            }
            // Accumulate chars whose summed display width fits `avail`.
            let mut chunk = String::new();
            let mut chunk_w = 0usize;
            while let Some(&c) = chars.front() {
                let cw = UnicodeWidthChar::width(c).unwrap_or(0);
                if cw > avail {
                    // Wide char doesn't fit; wrap to next line if we already
                    // emitted something, else accept overflow on empty line
                    // to make forward progress.
                    if chunk_w == 0 {
                        chunk.push(c);
                        chunk_w += cw;
                        chars.pop_front();
                    }
                    break;
                }
                chunk.push(c);
                chunk_w += cw;
                avail -= cw;
                chars.pop_front();
            }
            cur_w += chunk_w;
            cur_line.push(Span::styled(chunk, style));
        }
    }
    if !cur_line.is_empty() {
        lines.push(cur_line);
    }
    if lines.is_empty() {
        lines.push(Vec::new());
    }
    lines
}

fn merge_n_columns(columns: Vec<Vec<Line<'static>>>, col_width: usize) -> Vec<Line<'static>> {
    let max_len = columns.iter().map(|c| c.len()).max().unwrap_or(0);
    let empty = Line::from("");
    let mut result = Vec::new();
    for i in 0..max_len {
        // Wrap each non-last column's line to fit within col_width
        let wrapped: Vec<Vec<Vec<Span<'static>>>> = columns
            .iter()
            .enumerate()
            .map(|(ci, col)| {
                let l = col.get(i).cloned().unwrap_or_else(|| empty.clone());
                if ci == columns.len() - 1 {
                    vec![l.spans]
                } else {
                    wrap_spans(l.spans, col_width)
                }
            })
            .collect();
        let max_rows = wrapped.iter().map(|w| w.len()).max().unwrap_or(1);
        for row in 0..max_rows {
            let mut spans: Vec<Span<'static>> = Vec::new();
            for (ci, col_wrapped) in wrapped.iter().enumerate() {
                let is_last = ci == columns.len() - 1;
                if let Some(line_spans) = col_wrapped.get(row) {
                    let w: usize = line_spans
                        .iter()
                        .map(|s| unicode_width::UnicodeWidthStr::width(s.content.as_ref()))
                        .sum();
                    spans.extend(line_spans.iter().cloned());
                    if !is_last && w < col_width {
                        spans.push(Span::raw(" ".repeat(col_width - w)));
                    }
                } else if !is_last {
                    spans.push(Span::raw(" ".repeat(col_width)));
                }
            }
            result.push(Line::from(spans));
        }
    }
    result
}

fn section_header(title: &str) -> Line<'static> {
    Line::from(vec![Span::styled(
        format!("  {title}"),
        Style::default().fg(theme::PURPLE).add_modifier(Modifier::BOLD),
    )])
}

/// Render the per-host RDMA/InfiniBand detail block:
///   header line + summary (active/total, total link, total TX/RX) + per-device rows.
fn rdma_detail_lines(ib: &vmon_core::ib::IbScrape, content_width: usize) -> Vec<Line<'static>> {
    let active = ib.active_count();
    let total = ib.devices.len();
    let total_tx = ib.total_tx_gbps();
    let total_rx = ib.total_rx_gbps();
    let peak = ib.total_link_gbps();
    let mut lines = vec![section_header(&format!(
        "RDMA / InfiniBand  ({active}/{total} active)"
    ))];
    if peak > 0.0 {
        lines.push(stat_row(
            "Total",
            &format!(
                "TX {} RX {}  / peak {}",
                format_ib_bw(total_tx),
                format_ib_bw(total_rx),
                format_ib_bw(peak),
            ),
        ));
    } else {
        lines.push(stat_row(
            "Total",
            &format!(
                "TX {} RX {}",
                format_ib_bw(total_tx),
                format_ib_bw(total_rx)
            ),
        ));
    }
    // Per-device table-style rows. Sort active first, then by device name.
    let mut devs: Vec<&vmon_core::ib::IbDevice> = ib.devices.iter().collect();
    devs.sort_by(|a, b| {
        b.is_active()
            .cmp(&a.is_active())
            .then_with(|| a.device.cmp(&b.device))
            .then_with(|| a.port.cmp(&b.port))
    });

    // Build a compact per-device "card" (no state word — use color on the
    // device name; active/down counts already in the section header):
    //   "  mlx5_5:1  200G  TX 12.4G/8.7G  (62/43%)"
    // Pack multiple cards per line when content_width allows.
    const CARD_WIDTH: usize = 42; // generous so the longest possible card fits
    const GAP: usize = 2;
    let cols = ((content_width + GAP) / (CARD_WIDTH + GAP)).max(1);

    let mut row_spans: Vec<Span<'static>> = Vec::new();
    let mut col_idx = 0usize;
    for d in devs {
        let name_color = if d.is_active() {
            theme::HEALTHY
        } else {
            theme::MUTED
        };
        let link_str = match d.link_gbps() {
            Some(g) => format_ib_bw(g),
            None => "-".to_string(),
        };
        let util_str = match (d.tx_utilization(), d.rx_utilization()) {
            (Some(tx), Some(rx)) => format!("({:.0}/{:.0}%)", tx * 100.0, rx * 100.0),
            _ => String::new(),
        };
        // Build the card text body separately so we can pad to CARD_WIDTH.
        let label = format!("{}:{}", d.device, d.port);
        let body = format!(
            "{label:<10}  {link:>4}  TX {tx}/{rx}  {util}",
            link = link_str,
            tx = format_ib_bw(d.tx_gbps),
            rx = format_ib_bw(d.rx_gbps),
            util = util_str,
        );
        let body_w = unicode_width::UnicodeWidthStr::width(body.as_str());
        let pad = CARD_WIDTH.saturating_sub(body_w);

        if col_idx == 0 {
            row_spans.push(Span::raw("  "));
        } else {
            row_spans.push(Span::raw(" ".repeat(GAP)));
        }
        // Re-compose with styled device name (the rest stays default).
        // body starts with "{label:<10}  ..." — split off the label segment
        // so we can color just the name.
        let label_padded = format!("{label:<10}");
        let rest = body
            .strip_prefix(&label_padded)
            .map(|s| s.to_string())
            .unwrap_or_else(|| body.clone());
        row_spans.push(Span::styled(label_padded, Style::default().fg(name_color)));
        row_spans.push(Span::styled(rest, Style::default().fg(theme::TEXT)));
        if pad > 0 {
            row_spans.push(Span::raw(" ".repeat(pad)));
        }

        col_idx += 1;
        if col_idx >= cols {
            lines.push(Line::from(std::mem::take(&mut row_spans)));
            col_idx = 0;
        }
    }
    if !row_spans.is_empty() {
        lines.push(Line::from(row_spans));
    }
    lines
}

/// Render the Mooncake Store details section. Cluster-wide block: status,
/// memory bar, key/client counts, request rate breakdown, evictions, segment
/// fill range, and (when non-quiescent) HA oplog state. `job_id` labels the
/// owning SLURM job when several stores are tracked (one per job).
fn mooncake_detail_lines(
    m: &vmon_core::mooncake::MooncakeMetrics,
    job_id: Option<&str>,
) -> Vec<Line<'static>> {
    let mut lines = Vec::new();
    let mut header = "Mooncake Store".to_string();
    if let Some(id) = job_id {
        header.push_str(&format!(" · job {id}"));
    }
    if !m.addr.is_empty() {
        header.push_str(&format!("  ({})", m.addr));
    }
    lines.push(section_header(&header));

    // Status line
    let status = match &m.health {
        Some(h) => {
            let ready = if h.service_ready {
                "ready"
            } else {
                "not-ready"
            };
            let role = if h.role.is_empty() {
                "?"
            } else {
                h.role.as_str()
            };
            let state = if h.ha_state.is_empty() {
                "-"
            } else {
                h.ha_state.as_str()
            };
            format!("{role} · {state} · {ready}")
        }
        None => "unknown (health probe failed)".to_string(),
    };
    let status_color = match &m.health {
        Some(h) if h.service_ready && h.ha_state.eq_ignore_ascii_case("serving") => theme::HEALTHY,
        Some(_) => theme::WARNING,
        None => theme::MUTED,
    };
    lines.push(stat_row_color("Status", &status, status_color));

    // Memory bar
    let mem_str = if m.mem_total_bytes == 0 {
        format!("{} / -", format_bytes(m.mem_allocated_bytes))
    } else {
        let bar = ascii_bar(m.mem_util, 10);
        format!(
            "{} / {}  {bar} {:.1}%",
            format_bytes(m.mem_allocated_bytes),
            format_bytes(m.mem_total_bytes),
            m.mem_util * 100.0,
        )
    };
    lines.push(stat_row("Memory", &mem_str));

    // Keys + clients
    lines.push(stat_row(
        "Keys",
        &format!(
            "{}  (soft-pinned {})",
            format_count(m.key_count as i64),
            format_count(m.soft_pin_key_count as i64),
        ),
    ));
    lines.push(stat_row("Clients", &format_count(m.active_clients as i64)));

    // Request rates
    let fail_color = if m.failure_rate > 0.0 {
        theme::DANGER
    } else {
        theme::TEXT
    };
    lines.push(stat_row_color(
        "Requests",
        &format!(
            "{:.1}/s  (fail {:.1}/s)",
            m.total_request_rate, m.failure_rate
        ),
        if m.failure_rate > 0.0 {
            fail_color
        } else {
            theme::TEXT
        },
    ));
    let mut by_op = format!(
        "get {:.1}/s  put {:.1}/s  exist {:.1}/s",
        m.get_rate, m.put_rate, m.exist_rate,
    );
    if m.remove_rate > 0.0 {
        by_op.push_str(&format!("  remove {:.1}/s", m.remove_rate));
    }
    lines.push(stat_row("  by op", &by_op));
    if m.ping_rate > 0.0 {
        lines.push(stat_row("Pings", &format!("{:.1}/s", m.ping_rate)));
    }

    // Evictions
    lines.push(stat_row(
        "Evictions",
        &format!(
            "{:.1}/s  ({}/s)",
            m.eviction_rate,
            format_bytes_per_sec(m.eviction_bytes_rate),
        ),
    ));

    // Segments
    if m.segment_count > 0 {
        let fill_str = if m.segment_fill_max > 0.0 {
            format!(
                "  fill {:.1}%–{:.1}%",
                m.segment_fill_min * 100.0,
                m.segment_fill_max * 100.0,
            )
        } else {
            String::new()
        };
        lines.push(stat_row(
            "Segments",
            &format!("{} mounted{}", m.segment_count, fill_str),
        ));
    }

    // HA — only show when something non-zero would surprise the operator.
    let role_is_leader =
        m.health.as_ref().map(|h| h.role.eq_ignore_ascii_case("leader")).unwrap_or(true);
    if m.ha_standby_state != 0
        || m.ha_oplog_standby_lag != 0
        || m.ha_oplog_pending_entries != 0
        || !role_is_leader
    {
        lines.push(stat_row(
            "HA",
            &format!(
                "standby_state={} lag={} pending={}",
                m.ha_standby_state, m.ha_oplog_standby_lag, m.ha_oplog_pending_entries,
            ),
        ));
    }

    lines
}

// ── Main draw ──

pub fn draw(f: &mut Frame, cluster: &ClusterState, ui: &mut UiState) {
    // Offer the Mooncake tab only when --mooncake targets exist; if the
    // active tab just disappeared (targets gone with their job), fall back
    // to the KV/Xfer tab it split off from.
    ui.mooncake_tab = !cluster.mooncakes.is_empty() || !cluster.mooncake_addrs.is_empty();
    if ui.active_tab == Tab::Mooncake && !ui.mooncake_tab {
        ui.active_tab = Tab::KvXfer;
    }
    let search_height = if ui.search_active { 1 } else { 0 };
    // Table height: border(2) + header(1) + cluster_row(0|1) + node_rows + visible engine sub-rows
    let cluster_extra = if cluster.nodes.len() > 1 { 1 } else { 0 };
    let engine_extra: usize = cluster
        .nodes
        .iter()
        .filter(|n| ui.expanded.contains(&n.addr))
        .map(sub_row_count)
        .sum();
    let table_rows = cluster.nodes.len() + cluster_extra + engine_extra + 3; // +3 = 2 border + 1 header
    let table_h = (table_rows as u16).max(4); // minimum 4 to avoid collapse

    // Graph mode: table on top, graphs below (replaces detail panel)
    if ui.show_graph {
        let chunks = Layout::default()
            .direction(Direction::Vertical)
            .constraints([
                Constraint::Length(1),
                Constraint::Length(table_h),
                Constraint::Fill(1),
                Constraint::Length(1),
            ])
            .split(f.area());
        ui.table_area = chunks[1];
        ui.tabs_area = Rect::default(); // no tab bar in graph mode
        draw_header(f, chunks[0], cluster, ui);
        draw_table(f, chunks[1], cluster, ui);
        draw_graphs(f, chunks[2], cluster, ui);
        draw_footer(f, chunks[3], ui);
        return;
    }
    let chunks = Layout::default()
        .direction(Direction::Vertical)
        .constraints(if ui.show_detail {
            vec![
                Constraint::Length(1),       // header
                Constraint::Length(table_h), // table (fit content)
                Constraint::Length(1),       // tabs
                Constraint::Length(search_height),
                Constraint::Fill(1),   // detail
                Constraint::Length(1), // footer
            ]
        } else {
            vec![
                Constraint::Length(1),
                Constraint::Fill(1),
                Constraint::Length(0),
                Constraint::Length(0),
                Constraint::Length(0),
                Constraint::Length(1),
            ]
        })
        .split(f.area());

    ui.table_area = chunks[1];
    ui.tabs_area = if ui.show_detail {
        chunks[2]
    } else {
        Rect::default()
    };
    draw_header(f, chunks[0], cluster, ui);
    draw_table(f, chunks[1], cluster, ui);
    if ui.show_detail {
        draw_tabs(f, chunks[2], ui);
        if ui.search_active {
            draw_search_bar(f, chunks[3], ui);
        }
        draw_detail(f, chunks[4], cluster, ui);
    }
    draw_footer(f, chunks[5], ui);
}

// ── Header ──

/// Longest job name shown in the header before ellipsis truncation. Keeps
/// config-encoding sweep names from pushing the uptime/status off-screen.
const MAX_JOB_NAME_CHARS: usize = 32;

fn draw_header(f: &mut Frame, area: Rect, cluster: &ClusterState, ui: &UiState) {
    // Show version only when a specific node is selected (not the cluster summary row)
    let version_str = if ui.selected != usize::MAX {
        cluster
            .node_infos
            .values()
            .find_map(|i| {
                if i.vllm_version.is_empty() {
                    None
                } else {
                    Some(format!("vLLM {}", i.vllm_version))
                }
            })
            .unwrap_or_default()
    } else {
        String::new()
    };

    let sep = Span::styled(" \u{258F} ", Style::default().fg(theme::MUTED)); // ▏

    let all_loading = cluster.nodes.iter().all(|n| n.is_loading);
    let status_span = if all_loading {
        Span::styled(" (connecting…)", Style::default().fg(theme::MUTED))
    } else {
        Span::styled(
            format!(" ({} up)", cluster.healthy_count),
            Style::default().fg(theme::HEALTHY),
        )
    };
    let mut spans = vec![
        Span::styled(" vmon ", theme::accent()),
        sep.clone(),
        Span::styled(
            format!("{} nodes", cluster.nodes.len()),
            Style::default().fg(theme::BRIGHT),
        ),
        status_span,
    ];

    // Cluster-wide sinfo totals: how big the whole SLURM cluster is and how
    // many nodes are free right now, as opposed to the scrape targets above
    // which only cover the tracked jobs.
    if let Some(sc) = &cluster.slurm_cluster {
        spans.push(sep.clone());
        spans.push(Span::styled(
            format!("slurm {}", sc.total_nodes),
            Style::default().fg(theme::BRIGHT),
        ));
        let idle_color = if sc.idle_nodes > 0 {
            theme::HEALTHY
        } else {
            theme::MUTED
        };
        spans.push(Span::styled(
            format!(" ({} idle)", sc.idle_nodes),
            Style::default().fg(idle_color),
        ));
    }

    match cluster.slurm_jobs.len() {
        0 => {}
        1 => {
            let job = &cluster.slurm_jobs[0];
            spans.push(sep.clone());
            let id_style = if job.is_ended() {
                Style::default().fg(theme::MUTED).add_modifier(Modifier::CROSSED_OUT)
            } else {
                Style::default().fg(theme::job_color(&job.job_id)).add_modifier(Modifier::BOLD)
            };
            spans.push(Span::styled(format!("job {}", job.job_id), id_style));
            if !job.job_name.is_empty() {
                spans.push(Span::styled(
                    format!(
                        " ({})",
                        truncate_ellipsis(&job.job_name, MAX_JOB_NAME_CHARS)
                    ),
                    Style::default().fg(theme::MUTED),
                ));
            }
            let uptime_style = if job.is_ended() {
                Style::default().fg(theme::MUTED)
            } else {
                Style::default().fg(theme::TEXT)
            };
            spans.push(Span::styled(
                format!(" · {}", job.format_uptime()),
                uptime_style,
            ));
            if job.is_ended() {
                spans.push(Span::styled(
                    " (ended)",
                    Style::default().fg(theme::MUTED).add_modifier(Modifier::ITALIC),
                ));
            }
        }
        n => {
            // Multiple jobs — always list each with its own uptime; the
            // header line clips on narrow terminals rather than collapsing
            // to an aggregate.
            spans.push(sep.clone());
            spans.push(Span::styled(
                format!("{n} jobs: "),
                Style::default().fg(theme::BRIGHT),
            ));
            for (i, job) in cluster.slurm_jobs.iter().enumerate() {
                if i > 0 {
                    spans.push(Span::styled(", ", Style::default().fg(theme::MUTED)));
                }
                let id_style = if job.is_ended() {
                    Style::default().fg(theme::MUTED).add_modifier(Modifier::CROSSED_OUT)
                } else {
                    Style::default().fg(theme::job_color(&job.job_id)).add_modifier(Modifier::BOLD)
                };
                spans.push(Span::styled(job.job_id.clone(), id_style));
                let uptime_style = if job.is_ended() {
                    Style::default().fg(theme::MUTED)
                } else {
                    Style::default().fg(theme::TEXT)
                };
                spans.push(Span::styled(
                    format!("/{}", job.format_uptime()),
                    uptime_style,
                ));
                if job.is_ended() {
                    spans.push(Span::styled(
                        " (ended)",
                        Style::default().fg(theme::MUTED).add_modifier(Modifier::ITALIC),
                    ));
                }
            }
        }
    }

    if !version_str.is_empty() {
        spans.push(sep.clone());
        spans.push(Span::styled(version_str, Style::default().fg(theme::TEXT)));
    }

    // TPGS: total generation tok/s / total GPU count across healthy nodes.
    // This value updates every scrape and changes width, so it is rendered
    // right-aligned at the very end (see below) — that keeps its jitter from
    // shifting everything to its left (e.g. the replay progress bar).
    let total_gpu_count: usize = cluster
        .nodes
        .iter()
        .filter(|n| n.is_healthy)
        .filter_map(|n| n.gpu_scrape.as_ref())
        .map(|gs| gs.gpus.len())
        .sum();
    let tpgs_span = (total_gpu_count > 0).then(|| {
        let tpgs =
            (cluster.total_generation_tps + cluster.total_prompt_tps) / total_gpu_count as f64;
        Span::styled(
            format!("{:.1} tok/s/gpu", tpgs),
            Style::default().fg(theme::BRIGHT),
        )
    });

    // Replay progress indicator
    if let Some(replay) = &ui.replay {
        let sample = replay.get_sample();
        let elapsed = replay.get_elapsed();
        let total = replay.total_duration_secs;
        let speed = replay.get_speed();
        let paused = replay.is_paused();

        spans.push(sep.clone());
        spans.push(Span::styled(
            "▶ REPLAY ",
            Style::default().fg(theme::WARNING).add_modifier(Modifier::BOLD),
        ));

        let state = if paused { "⏸" } else { "▶" };
        spans.push(Span::styled(
            format!(
                "{state} {:.0}s/{:.0}s ({}/{}) {speed:.0}x",
                elapsed, total, sample, replay.total_samples
            ),
            Style::default().fg(theme::TEXT),
        ));

        // Progress bar
        let pct = if total > 0.0 { elapsed / total } else { 0.0 };
        let bar_w: usize = 15;
        let filled = (pct * bar_w as f64).round() as usize;
        let empty = bar_w.saturating_sub(filled);
        spans.push(Span::styled(" ", Style::default()));
        spans.push(Span::styled(
            "█".repeat(filled),
            Style::default().fg(theme::ACCENT),
        ));
        spans.push(Span::styled(
            "░".repeat(empty),
            Style::default().fg(theme::BAR_EMPTY),
        ));
        spans.push(Span::styled(
            format!(" {:.0}%", pct * 100.0),
            Style::default().fg(theme::TEXT),
        ));
    }

    // Right-align the throughput at the far edge: pad the gap between the
    // left content and the value so its changing width only grows/shrinks the
    // blank padding, never displacing anything else.
    if let Some(tpgs_span) = tpgs_span {
        let used: usize = spans.iter().map(|s| s.width()).sum();
        let avail = area.width as usize;
        let pad = avail.saturating_sub(used + tpgs_span.width());
        spans.push(Span::raw(" ".repeat(pad.max(1))));
        spans.push(tpgs_span);
    }

    let header = Line::from(spans);
    f.render_widget(
        Paragraph::new(header).style(Style::default().bg(theme::HEADER_BG)),
        area,
    );
}

// ── Footer ──

/// `● REC mm:ss (N)` while recording, `❚❚ PAUSED mm:ss (N)` while paused.
fn rec_badge(session: &CollectSession) -> Span<'static> {
    let elapsed = session.active_elapsed();
    let secs = elapsed.as_secs();
    let mm = secs / 60;
    let ss = secs % 60;
    let n = session.sample_count();
    let (text, color) = if session.paused {
        (
            format!(" \u{2759}\u{2759} PAUSED {mm:02}:{ss:02} ({n}) "),
            theme::WARNING,
        )
    } else {
        (
            format!(" \u{25CF} REC {mm:02}:{ss:02} ({n}) "),
            theme::DANGER,
        )
    };
    Span::styled(
        text,
        Style::default().fg(color).add_modifier(Modifier::BOLD),
    )
}

fn draw_footer(f: &mut Frame, area: Rect, ui: &UiState) {
    // Confirmation prompt takes priority
    if let Some(action) = &ui.pending_action {
        let footer = Line::from(vec![Span::styled(
            format!(" {} ", action.prompt()),
            Style::default().fg(theme::WARNING).add_modifier(Modifier::BOLD),
        )]);
        f.render_widget(
            Paragraph::new(footer).style(Style::default().bg(Color::DarkGray)),
            area,
        );
        return;
    }

    // Transient status message (auto-expires after 3s)
    if let Some((msg, when)) = &ui.status_message {
        if when.elapsed() < std::time::Duration::from_secs(3) {
            let color = if msg.starts_with("Error") {
                theme::DANGER
            } else {
                theme::HEALTHY
            };
            let footer = Line::from(Span::styled(format!(" {msg}"), Style::default().fg(color)));
            f.render_widget(
                Paragraph::new(footer).style(Style::default().bg(theme::HEADER_BG)),
                area,
            );
            return;
        }
    }

    let sep = Span::styled(" \u{2502} ", Style::default().fg(theme::MUTED)); // │

    let key = |k: &str| -> Span<'static> {
        Span::styled(
            k.to_string(),
            Style::default().fg(theme::ACCENT).add_modifier(Modifier::BOLD),
        )
    };
    let hint = |h: &str| -> Span<'static> {
        Span::styled(h.to_string(), Style::default().fg(theme::MUTED))
    };

    let mut spans: Vec<Span<'static>> = Vec::new();
    let push_pair = |spans: &mut Vec<Span<'static>>, k: &str, h: &str| {
        if !spans.is_empty() {
            spans.push(sep.clone());
        }
        spans.push(key(k));
        spans.push(hint(h));
    };

    if ui.replay.is_some() {
        // Replay mode footer
        push_pair(&mut spans, " j/k", " select");
        push_pair(&mut spans, "Tab", " switch");
        push_pair(&mut spans, "Enter", " detail");
        if ui.show_detail {
            push_pair(&mut spans, "[/]", " scroll");
        }
        push_pair(&mut spans, "Space", " fold");
        push_pair(&mut spans, "p", " pause");
        push_pair(&mut spans, ">", " faster");
        push_pair(&mut spans, "<", " slower");
        push_pair(&mut spans, "n/N", " step");
        push_pair(&mut spans, "/", " search");
        push_pair(&mut spans, "g", " graph");
        push_pair(&mut spans, "q", " quit");
    } else {
        // Live mode footer — recording badge takes the leading slot when active.
        if let Some(session) = &ui.collect {
            spans.push(rec_badge(session));
            spans.push(sep.clone());
        }
        push_pair(&mut spans, " j/k", " select");
        push_pair(&mut spans, "Tab", " switch");
        push_pair(&mut spans, "Enter", " detail");
        if ui.show_detail {
            push_pair(&mut spans, "[/]", " scroll");
        }
        push_pair(&mut spans, "Space", " fold");
        if ui.active_tab == Tab::Info {
            push_pair(&mut spans, "d", " diff");
        }
        push_pair(&mut spans, "s", " sleep");
        push_pair(&mut spans, "R", " reset cache");
        if ui.collect.is_some() {
            push_pair(&mut spans, "c", " pause");
            push_pair(&mut spans, "C", " save");
        } else {
            push_pair(&mut spans, "c", " rec");
        }
        push_pair(&mut spans, "/", " search");
        push_pair(&mut spans, "g", " graph");
        push_pair(&mut spans, "q", " quit");
    }
    let footer = Line::from(spans);
    f.render_widget(
        Paragraph::new(footer).style(Style::default().bg(theme::HEADER_BG)),
        area,
    );
}

// ── Graph view ──

fn draw_graphs(f: &mut Frame, area: Rect, cluster: &ClusterState, ui: &UiState) {
    // 2x2 grid: toks/s charts on the left, req/s charts on the right.
    let cols = Layout::default()
        .direction(Direction::Horizontal)
        .constraints([Constraint::Percentage(50), Constraint::Percentage(50)])
        .split(area);
    let halves = Layout::default()
        .direction(Direction::Vertical)
        .constraints([Constraint::Percentage(50), Constraint::Percentage(50)])
        .split(cols[0]);

    // Capture a single `now` so all charts share the same time reference.
    let now = std::time::Instant::now();

    // Dynamo PD deployment (both prefill and decode worker roles present):
    // split the charts by role so each chart shows only the workers that
    // actually serve that phase — prefill workers' prompt TPS on top, decode
    // workers' generation TPS below. Frontends/routers are dropped from both.
    let healthy = || cluster.nodes.iter().filter(|n| n.is_healthy);
    let is_pd = healthy().any(|n| role_is_prefill(n.dynamo_role.as_deref()))
        && healthy().any(|n| role_is_decode(n.dynamo_role.as_deref()));

    // Single pass per node: build prefill/decode toks/s and req/s series
    // simultaneously. This avoids traversing graph_history multiple times
    // and clocks timestamps once.
    let mut prefill_series: Vec<(String, Vec<(f64, f64)>)> = Vec::new();
    let mut decode_series: Vec<(String, Vec<(f64, f64)>)> = Vec::new();
    let mut prefill_req_series: Vec<(String, Vec<(f64, f64)>)> = Vec::new();
    let mut decode_req_series: Vec<(String, Vec<(f64, f64)>)> = Vec::new();
    for node in healthy() {
        let role = node.dynamo_role.as_deref();
        // In PD mode a role-less node is a plain vLLM instance serving both
        // phases; a labeled non-worker (frontend/router) is skipped entirely.
        let (want_prefill, want_decode) = if is_pd && role.is_some() {
            (role_is_prefill(role), role_is_decode(role))
        } else {
            (true, true)
        };
        if !want_prefill && !want_decode {
            continue;
        }
        let label = label_addr(&node.addr);
        let mut prefill = Vec::new();
        let mut decode = Vec::new();
        let mut req = Vec::new();
        if let Some(h) = ui.graph_history.get(&node.addr) {
            prefill.reserve(h.len());
            decode.reserve(h.len());
            req.reserve(h.len());
            for s in h.iter() {
                let x = -(now.duration_since(s.ts).as_secs_f64());
                prefill.push((x, s.prompt_tps));
                decode.push((x, s.generation_tps));
                req.push((x, s.req_per_sec));
            }
        }
        if want_prefill {
            prefill_series.push((label.clone(), prefill));
            prefill_req_series.push((label.clone(), req.clone()));
        }
        if want_decode {
            decode_series.push((label.clone(), decode));
            decode_req_series.push((label, req));
        }
    }

    // Compute x_min across both series (in PD mode they hold different nodes).
    let x_min = prefill_series
        .iter()
        .chain(decode_series.iter())
        .flat_map(|(_, pts)| pts.iter().map(|(x, _)| *x))
        .fold(-1.0_f64, f64::min);

    let (prefill_title, decode_title, prefill_req_title, decode_req_title) = if is_pd {
        (
            "Prefill toks/s [P]",
            "Decode toks/s [D]",
            "Prefill req/s [P]",
            "Decode req/s [D]",
        )
    } else {
        ("Prefill toks/s", "Decode toks/s", "Req/s", "Req/s")
    };

    // Top charts: no x-axis labels (saved for bottom charts to avoid duplication).
    draw_tps_chart(f, halves[0], &prefill_series, x_min, prefill_title, false);
    draw_tps_chart(f, halves[1], &decode_series, x_min, decode_title, true);

    if is_pd {
        let req_halves = Layout::default()
            .direction(Direction::Vertical)
            .constraints([Constraint::Percentage(50), Constraint::Percentage(50)])
            .split(cols[1]);
        draw_tps_chart(
            f,
            req_halves[0],
            &prefill_req_series,
            x_min,
            prefill_req_title,
            false,
        );
        draw_tps_chart(
            f,
            req_halves[1],
            &decode_req_series,
            x_min,
            decode_req_title,
            true,
        );
    } else {
        // Non-PD: every node serves both phases, so prefill/decode req/s are
        // identical — render a single full-height chart instead of duplicates.
        draw_tps_chart(f, cols[1], &prefill_req_series, x_min, "Req/s", true);
    }
}

/// Average the per-node series into one cluster-mean line, aligning samples
/// from the newest end (nodes that joined later have shorter histories).
/// Samples across nodes come from the same scrape tick (their x offsets are
/// microseconds apart), so slot alignment by distance-from-end is safe. The
/// mean — rather than the sum — keeps the y-scale of the per-node lines, so
/// they stay readable behind the accent line.
fn mean_series(series: &[(String, Vec<(f64, f64)>)]) -> Vec<(f64, f64)> {
    let max_len = series.iter().map(|(_, pts)| pts.len()).max().unwrap_or(0);
    let mut mean = Vec::with_capacity(max_len);
    for back in (1..=max_len).rev() {
        let mut x = None;
        let mut sum = 0.0;
        let mut n = 0u32;
        for (_, pts) in series {
            if pts.len() >= back {
                let (px, py) = pts[pts.len() - back];
                x.get_or_insert(px);
                sum += py;
                n += 1;
            }
        }
        if let Some(x) = x {
            mean.push((x, sum / f64::from(n)));
        }
    }
    mean
}

fn draw_tps_chart(
    f: &mut Frame,
    area: Rect,
    series: &[(String, Vec<(f64, f64)>)],
    x_min: f64,
    title: &'static str,
    show_x_labels: bool,
) {
    // Cluster-mean line over uniform muted per-node lines. A braille cell
    // can only hold one foreground color (ratatui canvas), so many colored
    // node lines bunched at similar values merged into multicolor speckle;
    // muted background + one accent line painted last keeps overlapping
    // cells readable.
    let mean = mean_series(series);

    // Dynamic y-axis with floor of 1.0 to avoid zero-bounds panic. The mean
    // never exceeds the per-node max, so folding over `series` bounds both.
    let y_max = series
        .iter()
        .flat_map(|(_, pts)| pts.iter().map(|(_, y)| *y))
        .fold(1.0_f64, f64::max)
        * 1.2;

    // X-axis: 5 evenly-distributed time labels (only on bottom chart)
    let x_range = -x_min; // x_min is negative
    let x_labels: Vec<Span> = (0..5)
        .map(|i| {
            let t = x_min + i as f64 * x_range / 4.0;
            let s = if i == 4 {
                "now".to_string()
            } else {
                fmt_secs(t)
            };
            Span::styled(s, theme::muted())
        })
        .collect();

    // Y-axis: 3 labels — 0, midpoint, max. Use one decimal for small ranges
    // (e.g. req/s charts) where whole numbers would collapse to duplicates.
    let fmt_y = |v: f64| {
        if y_max < 10.0 {
            format!("{:.1}", v)
        } else {
            format!("{:.0}", v)
        }
    };
    let y_labels = vec![
        Span::styled("0", theme::muted()),
        Span::styled(fmt_y(y_max / 2.0), theme::muted()),
        Span::styled(fmt_y(y_max), theme::muted()),
    ];

    // Dataset<'a> borrows &'a [(f64,f64)]; series and mean outlive datasets.
    // Per-node lines first in a single muted tone; the mean is painted last
    // so cells it shares with node lines resolve to the accent color. With a
    // single node the mean equals the node's own line — draw just that one.
    // All datasets are unnamed: no legend box — the title carries the value.
    let mut datasets: Vec<Dataset> = Vec::with_capacity(series.len() + 1);
    if series.len() > 1 {
        datasets.extend(series.iter().map(|(_, pts)| {
            Dataset::default()
                .data(pts)
                .marker(Marker::Braille)
                .graph_type(GraphType::Line)
                .style(Style::default().fg(theme::MUTED))
        }));
    }
    datasets.push(
        Dataset::default()
            .data(&mean)
            .marker(Marker::Braille)
            .graph_type(GraphType::Line)
            .style(Style::default().fg(theme::ACCENT)),
    );

    // The mean line's latest value lives in the chart title (accent-styled
    // to match the line). A legend box would be hidden by ratatui whenever
    // it can't fit in 1/4 of the plot area — exactly the small-window case
    // where the value matters most — and the title row is always drawn.
    let title_line = Line::from(vec![
        Span::styled(title, theme::bold()),
        Span::styled(
            mean.last().map(|&(_, v)| format!(" avg {}", fmt_y(v))).unwrap_or_default(),
            Style::default().fg(theme::ACCENT),
        ),
    ]);

    let x_axis = {
        let ax = Axis::default().bounds([x_min, 0.0]).style(Style::default().fg(theme::MUTED));
        if show_x_labels {
            ax.labels(x_labels)
        } else {
            ax
        }
    };

    let chart = Chart::new(datasets)
        .block(
            Block::default()
                .title(title_line)
                .borders(Borders::ALL)
                .border_type(BorderType::Rounded)
                .border_style(Style::default().fg(theme::MUTED)),
        )
        .x_axis(x_axis)
        .y_axis(
            Axis::default()
                .bounds([0.0, y_max])
                .labels(y_labels)
                .style(Style::default().fg(theme::MUTED)),
        );

    f.render_widget(chart, area);
}

/// Format a negative seconds value as a human-readable time label, e.g. "-90s" or "-3m20s".
fn fmt_secs(t: f64) -> String {
    let secs = t.abs().round() as u64;
    if secs < 60 {
        format!("-{}s", secs)
    } else {
        let m = secs / 60;
        let s = secs % 60;
        if s == 0 {
            format!("-{}m", m)
        } else {
            format!("-{}m{}s", m, s)
        }
    }
}

fn label_addr(addr: &str) -> String {
    let s = addr
        .strip_prefix("http://")
        .or_else(|| addr.strip_prefix("https://"))
        .unwrap_or(addr);
    let mut chars = s.chars();
    let head: String = chars.by_ref().take(17).collect();
    if chars.next().is_some() {
        format!("{}...", head)
    } else {
        head
    }
}

// ── Table with gauge bars (3-tier adaptive) ──

#[derive(Clone, Copy)]
enum TableTier {
    Narrow,
    Normal,
    Wide,
}

impl TableTier {
    fn from_width(w: u16) -> Self {
        if w <= NARROW_MAX {
            Self::Narrow
        } else if w >= WIDE_MIN {
            Self::Wide
        } else {
            Self::Normal
        }
    }
}

#[derive(Clone, Copy)]
struct TableFlags {
    has_gpu: bool,
    has_mfu: bool,
    has_load: bool,
    has_sleep: bool,
    has_ib: bool,
    has_pcie: bool,
    has_tc: bool,
    has_nixl: bool,
    /// True when targets span ≥2 SLURM jobs and a "Job" column should be shown.
    has_multi_job: bool,
    /// Width-derived display toggles, recomputed every frame in `draw_table`
    /// (never sticky). On narrow terminals columns are shed until the table
    /// fits: PCIe↓/PCIe↑ first (via clearing `has_pcie` on the frame-local
    /// copy), then Power, then NVL, then Model.
    show_power: bool,
    show_nvl: bool,
    show_model: bool,
}

fn table_header(tier: TableTier, f: TableFlags) -> Row<'static> {
    let s = Style::default().fg(theme::MUTED).add_modifier(Modifier::BOLD);
    let mut cells = vec![Cell::from("Node")];
    if f.has_multi_job {
        cells.push(Cell::from("Job"));
    }
    if f.show_model {
        cells.push(Cell::from("Model"));
    }
    cells.push(Cell::from("KV%"));
    cells.push(Cell::from("Cache%"));
    if f.has_nixl {
        cells.push(Cell::from("Ext%"));
    }
    cells.extend([
        Cell::from("Run"),
        Cell::from("Wait"),
        Cell::from("Decode"),
        Cell::from("Prefill"),
    ]);
    cells.push(Cell::from("Req/s"));
    if f.has_load {
        cells.push(Cell::from("Load"));
    }
    if f.has_sleep {
        cells.push(Cell::from("Sleep"));
    }
    if f.has_gpu {
        cells.push(Cell::from("GPU%"));
        cells.push(Cell::from("MBW%"));
        if f.has_tc && matches!(tier, TableTier::Wide) {
            cells.push(Cell::from("TC%"));
        }
        cells.push(Cell::from("VRAM%"));
        if f.show_power {
            cells.push(Cell::from("Power"));
        }
    }
    // Interconnect bandwidth group: NVL (intra-host GPU↔GPU) →
    // PCIe (host↔GPU) → RDMA (host↔host). Ordered by physical hop distance.
    if f.show_nvl {
        cells.push(Cell::from("NVL"));
    }
    if f.has_pcie {
        cells.push(Cell::from("PCIe↓"));
        cells.push(Cell::from("PCIe↑"));
    }
    if f.has_ib {
        cells.push(Cell::from("RDMA↓"));
        cells.push(Cell::from("RDMA↑"));
    }
    if f.has_mfu {
        cells.push(Cell::from("FLOPS"));
        cells.push(Cell::from("Mem"));
    }
    if f.has_nixl && matches!(tier, TableTier::Wide) {
        cells.push(Cell::from("KV xfer"));
    }
    if matches!(tier, TableTier::Wide) {
        cells.push(Cell::from("Preemptions"));
    } else {
        cells.push(Cell::from("Prmpt"));
    }
    Row::new(cells).style(s).bottom_margin(0)
}

fn table_widths(tier: TableTier, f: TableFlags, node_col: u16) -> Vec<Constraint> {
    let spark_width = if matches!(tier, TableTier::Wide) {
        16
    } else {
        6
    };
    // Node column width is content-sized by `node_col_width` in draw_table so
    // hostnames, stale tags, and role badges are never truncated.
    let mut w = vec![Constraint::Length(node_col)]; // Node
    if f.has_multi_job {
        // 4 trailing chars of job_id + 1 padding column = 5 chars.
        w.push(Constraint::Length(5)); // Job
    }
    if f.show_model {
        w.push(Constraint::Length(12)); // Model
    }
    w.push(Constraint::Length(spark_width)); // KV%   (wider in Wide tier for sparkline)
    w.push(Constraint::Length(spark_width)); // Hit%  (wider in Wide tier for sparkline)
    if f.has_nixl {
        w.push(Constraint::Length(spark_width)); // Ext% (external cache hit, PD-disagg)
    }
    w.extend([
        Constraint::Length(5), // Run
        Constraint::Length(5), // Wait
        Constraint::Length(8), // Decode
        Constraint::Length(8), // Prefill
    ]);
    w.push(Constraint::Length(6)); // Req/s
    if f.has_load {
        w.push(Constraint::Length(6)); // Load
    }
    if f.has_sleep {
        w.push(Constraint::Length(6)); // Sleep
    }
    if f.has_gpu {
        w.push(Constraint::Length(5)); // GPU%
        w.push(Constraint::Length(5)); // MBW%
        if f.has_tc && matches!(tier, TableTier::Wide) {
            w.push(Constraint::Length(5)); // TC%
        }
        w.push(Constraint::Length(5)); // VRAM%
        if f.show_power {
            w.push(Constraint::Length(6)); // Power
        }
    }
    // Interconnect bandwidth group.
    if f.show_nvl {
        w.push(Constraint::Length(6)); // NVL
    }
    if f.has_pcie {
        w.push(Constraint::Length(5)); // PCIe↓
        w.push(Constraint::Length(5)); // PCIe↑
    }
    if f.has_ib {
        w.push(Constraint::Length(6)); // RDMA↓
        w.push(Constraint::Length(6)); // RDMA↑
    }
    if f.has_mfu {
        w.push(Constraint::Length(6)); // FLOPS
        w.push(Constraint::Length(6)); // Mem (read+write per GPU)
    }
    if f.has_nixl && matches!(tier, TableTier::Wide) {
        w.push(Constraint::Length(10)); // NIXL MB/s
    }
    // Preemptions: full header in Wide, abbreviated "Prmpt" elsewhere.
    w.push(Constraint::Length(if matches!(tier, TableTier::Wide) {
        12
    } else {
        6
    }));
    w
}

/// Total terminal width the table needs to render every column: column
/// widths + 1-char spacing between columns + 2 block border columns.
fn required_table_width(tier: TableTier, f: TableFlags, node_col: u16) -> u16 {
    let widths = table_widths(tier, f, node_col);
    let cols: u16 = widths
        .iter()
        .map(|c| match c {
            Constraint::Length(n) => *n,
            _ => 0,
        })
        .sum();
    cols + widths.len().saturating_sub(1) as u16 + 2
}

/// Content-sized Node column width: the widest first cell across all rows
/// (status glyph + fold arrow + shortened addr + stale tag + role badge),
/// so the fixed-width columns to the right never squeeze the node name.
fn node_col_width(
    cluster: &ClusterState,
    scrape_interval: Option<std::time::Duration>,
    prefix: &str,
) -> u16 {
    let mut w: usize = 14; // floor: matches the pre-sizing default width
    if cluster.nodes.len() > 1 {
        let cluster_label = format!("CLUSTER ({})", cluster.healthy_count);
        w = w.max(2 + cluster_label.chars().count());
    }
    for n in &cluster.nodes {
        let fold = if sub_row_count(n) > 0 { 2 } else { 0 };
        // 2 = status glyph + space, mirroring node_row's addr line.
        let mut lw = 2 + fold + short_node_label(&n.addr, prefix).chars().count();
        if let Some(iv) = scrape_interval {
            if n.is_healthy && n.last_updated.elapsed() > iv.saturating_mul(3) {
                let age = n.last_updated.elapsed().as_secs();
                lw += format!(" (stale {}s)", age).chars().count();
            }
        }
        if let Some(badge) = role_badge(n.dynamo_role.as_deref()) {
            lw += 1 + badge.content.chars().count();
        }
        w = w.max(lw);
    }
    w as u16
}

/// Chars of the Node cell left for the node name: cell width minus the
/// status glyph + space, fold arrow, and role badge. The badge must survive
/// a squeezed column, so the name is what gets truncated.
fn node_name_budget(node_col: u16, fold: &str, badge: Option<&Span>) -> usize {
    let badge_w = badge.map_or(0, |b| 1 + b.content.chars().count());
    (node_col as usize).saturating_sub(2 + fold.chars().count() + badge_w)
}

#[allow(clippy::too_many_arguments)]
fn node_row(
    n: &NodeMetrics,
    tier: TableTier,
    info_model: &str,
    f: TableFlags,
    fold: &str,
    kv_spark: &str,
    hit_spark: &str,
    ext_hit_spark: &str,
    job_id: &str,
    job_ended: bool,
    prefix: &str,
    stale: bool,
    node_col: u16,
) -> Row<'static> {
    if !n.is_healthy {
        // Transient ("…") covers loading + booting-under-loaded-GPU; otherwise "-".
        let in_transition = n.is_loading
            || n.gpu_scrape
                .as_ref()
                .map(|g| {
                    g.gpus.iter().any(|gpu| {
                        gpu.mem_total_bytes > 0
                            && (gpu.mem_used_bytes as f64 / gpu.mem_total_bytes as f64) > 0.2
                    })
                })
                .unwrap_or(false);
        let filler = if in_transition { "…" } else { "-" };
        let badge = role_badge(n.dynamo_role.as_deref());
        let name = truncate_ellipsis(
            &short_node_label(&n.addr, prefix),
            node_name_budget(node_col, fold, badge.as_ref()),
        );
        let mut addr_line = vec![
            status_indicator(n),
            Span::raw(" "),
            Span::styled(fold.to_string(), Style::default().fg(theme::MUTED)),
            Span::styled(name, Style::default().fg(theme::MUTED)),
        ];
        if let Some(role) = badge {
            addr_line.push(Span::raw(" "));
            addr_line.push(role);
        }
        let mut cells = vec![Cell::from(Line::from(addr_line))];
        if f.has_multi_job {
            cells.push(job_cell(job_id, job_ended));
        }
        // [Model] + KV% + Hit% (+Ext% if nixl) + Run + Wait + Gen
        let scalar_count = 5 + f.show_model as usize + f.has_nixl as usize;
        for _ in 0..scalar_count {
            cells.push(Cell::from(Span::styled(
                filler,
                Style::default().fg(theme::MUTED),
            )));
        }
        // Prefill tok/s
        cells.push(Cell::from(Span::styled(
            filler,
            Style::default().fg(theme::MUTED),
        )));
        // Req/s
        cells.push(Cell::from(Span::styled(
            filler,
            Style::default().fg(theme::MUTED),
        )));
        if f.has_load {
            cells.push(Cell::from(Span::styled(
                filler,
                Style::default().fg(theme::MUTED),
            )));
        }
        if f.has_sleep {
            cells.push(Cell::from(Span::styled(
                filler,
                Style::default().fg(theme::MUTED),
            )));
        }
        let render_tc = f.has_tc && matches!(tier, TableTier::Wide);
        if f.has_gpu {
            if n.gpu_scrape.is_some() {
                gpu_table_cells(&mut cells, n.gpu_scrape.as_ref(), render_tc, f.show_power);
            } else {
                // GPU columns: GPU%, MBW%, VRAM% (+TC%/+Power when shown).
                // NVL is rendered separately below.
                let dash_n = 3 + render_tc as usize + f.show_power as usize;
                for _ in 0..dash_n {
                    cells.push(Cell::from(Span::styled(
                        filler,
                        Style::default().fg(theme::MUTED),
                    )));
                }
            }
        }
        if f.show_nvl {
            // NVL is per-host DCGM, available even when vLLM is loading.
            nvl_table_cell(&mut cells, n.gpu_scrape.as_ref());
        }
        if f.has_pcie {
            // PCIe is per-host (DCGM), independent of vLLM health.
            pcie_table_cells(&mut cells, n.gpu_scrape.as_ref());
        }
        if f.has_ib {
            // RDMA is per-host (node_exporter), independent of vLLM health.
            // Render whatever the scraper attached, even when vLLM is down.
            ib_table_cells(&mut cells, n.ib_scrape.as_ref());
        }
        if f.has_mfu {
            cells.push(Cell::from(Span::styled(
                filler,
                Style::default().fg(theme::MUTED),
            )));
        }
        if f.has_nixl && matches!(tier, TableTier::Wide) {
            cells.push(Cell::from(Span::styled(
                filler,
                Style::default().fg(theme::MUTED),
            )));
        }
        // Preemptions
        cells.push(Cell::from(Span::styled(
            filler,
            Style::default().fg(theme::MUTED),
        )));
        return Row::new(cells).style(Style::default().fg(theme::MUTED));
    }

    let model_src = if info_model.is_empty() {
        &n.model_name
    } else {
        info_model
    };
    let model = short_model(model_src);
    let label_color = if stale { theme::MUTED } else { theme::BRIGHT };
    let badge = role_badge(n.dynamo_role.as_deref());
    let name = truncate_ellipsis(
        &short_node_label(&n.addr, prefix),
        node_name_budget(node_col, fold, badge.as_ref()),
    );
    let mut addr_spans = vec![
        status_indicator(n),
        Span::raw(" "),
        Span::styled(fold.to_string(), Style::default().fg(theme::MUTED)),
        Span::styled(name, Style::default().fg(label_color)),
    ];
    // Badge before the stale tag: when the cell clips, the transient stale
    // tag is what falls off the edge, never the P/D role badge.
    if let Some(role) = badge {
        addr_spans.push(Span::raw(" "));
        addr_spans.push(role);
    }
    if stale {
        let age = n.last_updated.elapsed().as_secs();
        addr_spans.push(Span::styled(
            format!(" (stale {}s)", age),
            Style::default().fg(theme::WARNING),
        ));
    }
    let mut cells = vec![Cell::from(Line::from(addr_spans))];
    if f.has_multi_job {
        cells.push(job_cell(job_id, job_ended));
    }
    if f.show_model {
        cells.push(Cell::from(Span::styled(
            model,
            Style::default().fg(theme::TEXT),
        )));
    }
    if kv_spark.is_empty() {
        cells.push(kv_cell(n.kv_cache_usage));
    } else {
        cells.push(kv_cell_spark(n.kv_cache_usage, kv_spark));
    }
    if hit_spark.is_empty() {
        cells.push(hit_cell(n.prefix_cache_hit_rate));
    } else {
        cells.push(hit_cell_spark(n.prefix_cache_hit_rate, hit_spark));
    }
    if f.has_nixl {
        if ext_hit_spark.is_empty() {
            cells.push(ext_cell(n.external_cache_hit_rate));
        } else {
            cells.push(ext_cell_spark(n.external_cache_hit_rate, ext_hit_spark));
        }
    }
    cells.push(Cell::from(Span::styled(
        format!("{:.0}", n.requests_running),
        Style::default().fg(theme::BRIGHT),
    )));
    cells.push(Cell::from(Span::styled(
        format!("{:.0}", n.requests_waiting),
        Style::default().fg(theme::wait_color(n.requests_waiting)),
    )));
    cells.push(Cell::from(Span::styled(
        format!("{:.0}", n.generation_tps),
        Style::default().fg(theme::HEALTHY),
    )));
    cells.push(Cell::from(Span::styled(
        format!("{:.0}", n.prompt_tps),
        Style::default().fg(theme::HEALTHY),
    )));
    cells.push(Cell::from(Span::styled(
        format!("{:.1}", n.requests_per_sec),
        Style::default().fg(theme::TEXT),
    )));
    if f.has_load {
        match n.server_load {
            Some(load) => cells.push(Cell::from(Span::styled(
                format!("{:.1}", load),
                Style::default().fg(theme::TEXT),
            ))),
            None => cells.push(Cell::from(Span::styled(
                "-",
                Style::default().fg(theme::MUTED),
            ))),
        }
    }
    if f.has_sleep {
        match n.is_sleeping {
            Some(true) => cells.push(Cell::from(Span::styled(
                "zzz",
                Style::default().fg(theme::WARNING),
            ))),
            Some(false) => cells.push(Cell::from(Span::styled(
                "-",
                Style::default().fg(theme::MUTED),
            ))),
            None => cells.push(Cell::from(Span::styled(
                "-",
                Style::default().fg(theme::MUTED),
            ))),
        }
    }
    if f.has_gpu {
        let render_tc = f.has_tc && matches!(tier, TableTier::Wide);
        gpu_table_cells(&mut cells, n.gpu_scrape.as_ref(), render_tc, f.show_power);
    }
    if f.show_nvl {
        nvl_table_cell(&mut cells, n.gpu_scrape.as_ref());
    }
    if f.has_pcie {
        pcie_table_cells(&mut cells, n.gpu_scrape.as_ref());
    }
    if f.has_ib {
        ib_table_cells(&mut cells, n.ib_scrape.as_ref());
    }
    if f.has_mfu {
        flops_table_cell(&mut cells, n.estimated_flops_per_gpu_per_sec);
        mem_table_cell(
            &mut cells,
            n.estimated_read_bytes_per_gpu_per_sec + n.estimated_write_bytes_per_gpu_per_sec,
        );
    }
    if f.has_nixl && matches!(tier, TableTier::Wide) {
        cells.push(nixl_throughput_cell(
            n.has_nixl,
            n.nixl_throughput_mb_per_sec,
            n.external_kv_transfer_tokens_per_sec,
        ));
    }
    cells.push(Cell::from(Span::styled(
        format_count(n.preemptions_total as i64),
        Style::default().fg(theme::TEXT),
    )));
    let row = Row::new(cells);
    if stale {
        row.style(Style::default().fg(theme::MUTED))
    } else {
        row
    }
}

/// Append GPU%, MBW%, [TC% if render_tc], VRAM%, [Power if render_power]
/// cells to a row. NVL is rendered separately by `nvl_table_cell` so the
/// interconnect columns (NVL / PCIe / RDMA) stay grouped.
fn gpu_table_cells(
    cells: &mut Vec<Cell<'static>>,
    gpu: Option<&GpuScrape>,
    render_tc: bool,
    render_power: bool,
) {
    let dash = || Cell::from(Span::styled("-", Style::default().fg(theme::MUTED)));
    match gpu {
        Some(g) => {
            // GPU% — prefers DCGM PROF GR_ENGINE_ACTIVE (cycle-accurate),
            // falls back to DCGM_FI_DEV_GPU_UTIL when Profiling disabled.
            let util = g.avg_gpu_pct();
            cells.push(Cell::from(Span::styled(
                format!("{:.0}%", util),
                Style::default().fg(theme::load_color(util)),
            )));
            // MBW% — prefers DCGM PROF DRAM_ACTIVE.
            let mbw = g.avg_mbw_pct();
            cells.push(Cell::from(Span::styled(
                format!("{:.0}%", mbw),
                Style::default().fg(theme::load_color(mbw)),
            )));
            // TC% — DCGM PROF PIPE_TENSOR_ACTIVE; the most LLM-meaningful
            // signal (matmul pipeline activity).
            if render_tc {
                match g.avg_tc_pct() {
                    Some(tc) => cells.push(Cell::from(Span::styled(
                        format!("{:.0}%", tc),
                        Style::default().fg(theme::load_color(tc)),
                    ))),
                    None => cells.push(dash()),
                }
            }
            // VRAM%
            let (vram_used, vram_total) = g.mem_used_total();
            let vram_pct = if vram_total > 0 {
                vram_used as f64 / vram_total as f64 * 100.0
            } else {
                0.0
            };
            cells.push(Cell::from(Span::styled(
                format!("{:.0}%", vram_pct),
                Style::default().fg(theme::load_color(vram_pct)),
            )));
            // Power
            if render_power {
                cells.push(Cell::from(Span::styled(
                    format!("{:.0}W", g.total_power()),
                    Style::default().fg(theme::TEXT),
                )));
            }
        }
        None => {
            let n = 3 + render_tc as usize + render_power as usize;
            for _ in 0..n {
                cells.push(dash());
            }
        }
    }
}

/// Append the single NVL cell. Per-host NVLink TX and RX are conserved
/// (every intra-node byte counts once on each side), so summed across the
/// host's GPUs they're equal; show TX as the representative value.
fn nvl_table_cell(cells: &mut Vec<Cell<'static>>, gpu: Option<&GpuScrape>) {
    let dash = || Cell::from(Span::styled("-", Style::default().fg(theme::MUTED)));
    match gpu {
        Some(g) => {
            let nvl = g.total_nvlink_tx_kbps() as f64 / (1024.0 * 1024.0);
            cells.push(Cell::from(Span::styled(
                format_nvlink_bw(nvl),
                Style::default().fg(theme::TEXT),
            )));
        }
        None => cells.push(dash()),
    }
}

/// Append PCIe↓ (RX) and PCIe↑ (TX) cells in GB/s to a row. Sourced from
/// DCGM `DCGM_FI_PROF_PCIE_*_BYTES` (preferred) or `DCGM_FI_DEV_PCIE_*_THROUGHPUT`.
fn pcie_table_cells(cells: &mut Vec<Cell<'static>>, gpu: Option<&GpuScrape>) {
    let dash = || Cell::from(Span::styled("-", Style::default().fg(theme::MUTED)));
    match gpu {
        Some(g) => {
            let rx = g.total_pcie_rx_gbps();
            let tx = g.total_pcie_tx_gbps();
            cells.push(Cell::from(Span::styled(
                format_pcie_bw(rx),
                Style::default().fg(theme::TEXT),
            )));
            cells.push(Cell::from(Span::styled(
                format_pcie_bw(tx),
                Style::default().fg(theme::TEXT),
            )));
        }
        None => {
            cells.push(dash());
            cells.push(dash());
        }
    }
}

/// Append RDMA↓ (RX) and RDMA↑ (TX) cells in Gbps to a row. Backed by
/// node_exporter's `node_infiniband_*` metrics, which cover both InfiniBand
/// and RoCE links.
fn ib_table_cells(cells: &mut Vec<Cell<'static>>, ib: Option<&vmon_core::ib::IbScrape>) {
    let dash = || Cell::from(Span::styled("-", Style::default().fg(theme::MUTED)));
    match ib {
        Some(s) => {
            let rx = s.total_rx_gbps();
            let tx = s.total_tx_gbps();
            // Heat color by utilization vs aggregate one-direction link peak,
            // when the rate is reported. Otherwise neutral text color.
            let peak = s.total_link_gbps();
            let rx_color = if peak > 0.0 {
                theme::load_color(rx / peak * 100.0)
            } else {
                theme::TEXT
            };
            let tx_color = if peak > 0.0 {
                theme::load_color(tx / peak * 100.0)
            } else {
                theme::TEXT
            };
            cells.push(Cell::from(Span::styled(
                format_ib_bw(rx),
                Style::default().fg(rx_color),
            )));
            cells.push(Cell::from(Span::styled(
                format_ib_bw(tx),
                Style::default().fg(tx_color),
            )));
        }
        None => {
            cells.push(dash());
            cells.push(dash());
        }
    }
}

/// Format IB bandwidth for a 6-char column. Input is Gbps (the unit used
/// internally by node_exporter-derived counters); display is GB/s with
/// K/M/G suffixes, matching the NVLink column for visual consistency.
fn format_ib_bw(gbps: f64) -> String {
    if !gbps.is_finite() {
        return "-".to_string();
    }
    let gbytes = gbps / 8.0;
    if gbytes < 0.000_001 {
        "-".to_string()
    } else if gbytes < 1.0 {
        let mbytes = gbytes * 1024.0;
        if mbytes < 1.0 {
            format!("{:.0}K", mbytes * 1024.0)
        } else if mbytes < 100.0 {
            format!("{:.1}M", mbytes)
        } else {
            format!("{:.0}M", mbytes)
        }
    } else if gbytes < 10.0 {
        format!("{:.1}G", gbytes)
    } else {
        format!("{:.0}G", gbytes)
    }
}

/// Append FLOPS cell to a row (per-GPU TFLOP/s).
fn flops_table_cell(cells: &mut Vec<Cell<'static>>, flops_per_gpu: f64) {
    if flops_per_gpu > 0.0 {
        let tflops = flops_per_gpu / 1e12;
        cells.push(Cell::from(Span::styled(
            if tflops >= 100.0 {
                format!("{:.0}T", tflops)
            } else if tflops >= 1.0 {
                format!("{:.1}T", tflops)
            } else {
                format!("{:.0}G", tflops * 1000.0)
            },
            Style::default().fg(theme::TEXT),
        )));
    } else {
        cells.push(Cell::from(Span::styled(
            "-",
            Style::default().fg(theme::MUTED),
        )));
    }
}

/// Append memory bandwidth cell to a row (per-GPU read+write, bytes/s).
/// Mirrors `flops_table_cell` styling: single-letter unit, no "/s" — the
/// `Mem` column header implies the rate.
fn mem_table_cell(cells: &mut Vec<Cell<'static>>, bytes_per_sec: f64) {
    if bytes_per_sec > 0.0 {
        cells.push(Cell::from(Span::styled(
            if bytes_per_sec >= 1e12 {
                format!("{:.1}T", bytes_per_sec / 1e12)
            } else if bytes_per_sec >= 1e10 {
                format!("{:.0}G", bytes_per_sec / 1e9)
            } else if bytes_per_sec >= 1e9 {
                format!("{:.1}G", bytes_per_sec / 1e9)
            } else if bytes_per_sec >= 1e6 {
                format!("{:.0}M", bytes_per_sec / 1e6)
            } else {
                format!("{:.0}", bytes_per_sec)
            },
            Style::default().fg(theme::TEXT),
        )));
    } else {
        cells.push(Cell::from(Span::styled(
            "-",
            Style::default().fg(theme::MUTED),
        )));
    }
}

fn cluster_row(
    cluster: &ClusterState,
    tier: TableTier,
    f: TableFlags,
    kv_spark: &str,
    hit_spark: &str,
    ext_hit_spark: &str,
) -> Row<'static> {
    let healthy: Vec<&NodeMetrics> = cluster.nodes.iter().filter(|n| n.is_healthy).collect();
    let cnt = healthy.len() as f64;
    let total_req_s: f64 = healthy.iter().map(|n| n.requests_per_sec).sum();
    let avg_hit = if healthy.is_empty() {
        0.0
    } else {
        healthy.iter().map(|n| n.prefix_cache_hit_rate).sum::<f64>() / cnt
    };
    let avg_ext_hit = if healthy.is_empty() {
        0.0
    } else {
        healthy.iter().map(|n| n.external_cache_hit_rate).sum::<f64>() / cnt
    };

    let mut cells = vec![Cell::from(Line::from(vec![
        Span::raw("  "), // status-glyph slot, keeps addresses aligned across rows
        Span::styled(
            format!("CLUSTER ({})", cluster.healthy_count),
            theme::accent(),
        ),
    ]))];
    if f.has_multi_job {
        cells.push(Cell::from(""));
    }
    if f.show_model {
        cells.push(Cell::from("")); // Model
    }
    if kv_spark.is_empty() {
        cells.push(kv_cell(cluster.avg_kv_cache_usage));
    } else {
        cells.push(kv_cell_spark(cluster.avg_kv_cache_usage, kv_spark));
    }
    if hit_spark.is_empty() {
        cells.push(hit_cell(avg_hit));
    } else {
        cells.push(hit_cell_spark(avg_hit, hit_spark));
    }
    if f.has_nixl {
        if ext_hit_spark.is_empty() {
            cells.push(ext_cell(avg_ext_hit));
        } else {
            cells.push(ext_cell_spark(avg_ext_hit, ext_hit_spark));
        }
    }
    cells.push(Cell::from(Span::styled(
        format!("{:.0}", cluster.total_requests_running),
        Style::default().fg(theme::BRIGHT),
    )));
    cells.push(Cell::from(Span::styled(
        format!("{:.0}", cluster.total_requests_waiting),
        Style::default().fg(theme::wait_color(cluster.total_requests_waiting)),
    )));
    cells.push(Cell::from(Span::styled(
        format!("{:.0}", cluster.total_generation_tps),
        Style::default().fg(theme::HEALTHY),
    )));
    cells.push(Cell::from(Span::styled(
        format!("{:.0}", cluster.total_prompt_tps),
        Style::default().fg(theme::HEALTHY),
    )));
    cells.push(Cell::from(Span::styled(
        format!("{:.1}", total_req_s),
        Style::default().fg(theme::TEXT),
    )));
    if f.has_load {
        let load_vals: Vec<f64> = cluster
            .nodes
            .iter()
            .filter(|n| n.is_healthy)
            .filter_map(|n| n.server_load)
            .collect();
        if load_vals.is_empty() {
            cells.push(Cell::from(""));
        } else {
            let avg_load = load_vals.iter().sum::<f64>() / load_vals.len() as f64;
            cells.push(Cell::from(Span::styled(
                format!("{:.1}", avg_load),
                Style::default().fg(theme::TEXT),
            )));
        }
    }
    if f.has_sleep {
        let sleeping_count = cluster.nodes.iter().filter(|n| n.is_sleeping == Some(true)).count();
        if sleeping_count > 0 {
            cells.push(Cell::from(Span::styled(
                format!("{}", sleeping_count),
                Style::default().fg(theme::WARNING),
            )));
        } else {
            cells.push(Cell::from(""));
        }
    }
    // Interconnect group must match node_row order: GPU stats → NVL → PCIe →
    // RDMA. Omitting NVL here shifts PCIe/RDMA one column left of the header.
    let gpu_agg = if f.has_gpu {
        aggregate_gpu_scrapes(&cluster.nodes)
    } else {
        None
    };
    if f.has_gpu {
        let render_tc = f.has_tc && matches!(tier, TableTier::Wide);
        gpu_table_cells(&mut cells, gpu_agg.as_ref(), render_tc, f.show_power);
    }
    if f.show_nvl {
        nvl_table_cell(&mut cells, gpu_agg.as_ref());
    }
    if f.has_pcie {
        pcie_table_cells(&mut cells, gpu_agg.as_ref());
    }
    if f.has_ib {
        let agg = aggregate_ib_scrapes(&cluster.nodes);
        ib_table_cells(&mut cells, agg.as_ref());
    }
    if f.has_mfu {
        // Cluster FLOPS + Mem: average per-GPU across healthy nodes that have
        // FLOPs data. Use the same node set for both so the columns line up.
        let healthy_with_flops: Vec<&NodeMetrics> = cluster
            .nodes
            .iter()
            .filter(|n| n.is_healthy && n.estimated_flops_per_gpu_per_sec > 0.0)
            .collect();
        let (avg_flops, avg_mem) = if healthy_with_flops.is_empty() {
            (0.0, 0.0)
        } else {
            let n = healthy_with_flops.len() as f64;
            let f_sum: f64 =
                healthy_with_flops.iter().map(|m| m.estimated_flops_per_gpu_per_sec).sum();
            let m_sum: f64 = healthy_with_flops
                .iter()
                .map(|m| {
                    m.estimated_read_bytes_per_gpu_per_sec + m.estimated_write_bytes_per_gpu_per_sec
                })
                .sum();
            (f_sum / n, m_sum / n)
        };
        flops_table_cell(&mut cells, avg_flops);
        mem_table_cell(&mut cells, avg_mem);
    }
    cells.push(Cell::from("")); // Preempt — no aggregate
    Row::new(cells).style(Style::default().fg(theme::BRIGHT)).bottom_margin(0)
}

/// Aggregate IB scrapes across unique hosts for the cluster row.
/// IB scrapes are per-host and the same scrape is attached to every vLLM
/// process on that host, so dedupe by hostname before summing. Includes
/// hosts whose vLLM is offline — RDMA traffic is independent of vLLM health.
fn aggregate_ib_scrapes(nodes: &[NodeMetrics]) -> Option<vmon_core::ib::IbScrape> {
    use std::collections::HashSet;
    let mut seen: HashSet<&str> = HashSet::new();
    let mut devices = Vec::new();
    for n in nodes.iter() {
        let host = n.addr.split(':').next().unwrap_or(&n.addr);
        if !seen.insert(host) {
            continue;
        }
        if let Some(s) = &n.ib_scrape {
            devices.extend(s.devices.iter().cloned());
        }
    }
    if devices.is_empty() {
        None
    } else {
        Some(vmon_core::ib::IbScrape { devices })
    }
}

/// Aggregate GPU scrapes across all nodes for cluster row. Includes nodes
/// whose vLLM is offline — DCGM/agent data is independent of vLLM health,
/// and dropping offline-vLLM peers would lose their TP-partitioned GPU slice.
fn aggregate_gpu_scrapes(nodes: &[NodeMetrics]) -> Option<GpuScrape> {
    let scrapes: Vec<&GpuScrape> = nodes.iter().filter_map(|n| n.gpu_scrape.as_ref()).collect();
    if scrapes.is_empty() {
        return None;
    }
    let mut all_gpus = Vec::new();
    for s in &scrapes {
        all_gpus.extend(s.gpus.iter().cloned());
    }
    // Aggregate host CPU: average across nodes that report it
    let cpu_vals: Vec<f64> = scrapes.iter().filter_map(|s| s.cpu_percent).collect();
    let cpu_percent = if cpu_vals.is_empty() {
        None
    } else {
        Some(cpu_vals.iter().sum::<f64>() / cpu_vals.len() as f64)
    };
    // Aggregate host MEM: sum across unique hosts
    let mem_used: u64 = scrapes.iter().filter_map(|s| s.mem_used_bytes).sum();
    let mem_total: u64 = scrapes.iter().filter_map(|s| s.mem_total_bytes).sum();
    let mem_used_bytes = if mem_total > 0 { Some(mem_used) } else { None };
    let mem_total_bytes = if mem_total > 0 { Some(mem_total) } else { None };
    Some(GpuScrape {
        gpus: all_gpus,
        cpu_percent,
        mem_used_bytes,
        mem_total_bytes,
    })
}

/// Compute how many sub-rows a node should have (GPUs or engines, whichever is larger).
fn sub_row_count(n: &NodeMetrics) -> usize {
    let gpu_count = n.gpu_scrape.as_ref().map_or(0, |g| g.gpus.len());
    let engine_count = n.engine_metrics.as_ref().map_or(0, |e| e.len());
    let count = gpu_count.max(engine_count);
    // Only show sub-rows if there are at least 2
    if count > 1 { count } else { 0 }
}

fn engine_sub_row(
    engine: Option<&NodeMetrics>,
    gpu: Option<&GpuMetrics>,
    gpu_idx: usize,
    tier: TableTier,
    f: TableFlags,
) -> Row<'static> {
    // 4-space indent: 2 chars for the status-glyph slot + 2 for the tree
    // indent. Aligns the "GPU{n}" label below the parent row's hostname.
    let label = format!("    GPU{gpu_idx}");
    let mut label_line = vec![Span::styled(label, Style::default().fg(theme::MUTED))];
    if let Some(role) = engine.and_then(|n| role_badge(n.dynamo_role.as_deref())) {
        label_line.push(Span::raw(" "));
        label_line.push(role);
    }
    let mut cells = vec![Cell::from(Line::from(label_line))];
    if f.has_multi_job {
        cells.push(Cell::from("")); // Job — same as parent
    }
    if f.show_model {
        cells.push(Cell::from("")); // Model — same as parent
    }
    // vLLM engine metrics (if available for this rank)
    if let Some(n) = engine {
        // Wide tier: show the same horizontal bar as the parent host row,
        // computed from this rank's current value (no per-rank history kept).
        if matches!(tier, TableTier::Wide) {
            let kv_bar = format!(" {}", bar_str(n.kv_cache_usage * 100.0, 8));
            let hit_bar = format!(" {}", bar_str(n.prefix_cache_hit_rate * 100.0, 8));
            cells.push(kv_cell_spark(n.kv_cache_usage, &kv_bar));
            cells.push(hit_cell_spark(n.prefix_cache_hit_rate, &hit_bar));
            if f.has_nixl {
                let ext_bar = format!(" {}", bar_str(n.external_cache_hit_rate * 100.0, 8));
                cells.push(ext_cell_spark(n.external_cache_hit_rate, &ext_bar));
            }
        } else {
            cells.push(kv_cell(n.kv_cache_usage));
            cells.push(hit_cell(n.prefix_cache_hit_rate));
            if f.has_nixl {
                cells.push(ext_cell(n.external_cache_hit_rate));
            }
        }
        cells.push(Cell::from(Span::styled(
            format!("{:.0}", n.requests_running),
            Style::default().fg(theme::TEXT),
        )));
        cells.push(Cell::from(Span::styled(
            format!("{:.0}", n.requests_waiting),
            Style::default().fg(theme::wait_color(n.requests_waiting)),
        )));
        cells.push(Cell::from(Span::styled(
            format!("{:.0}", n.generation_tps),
            Style::default().fg(theme::HEALTHY),
        )));
        cells.push(Cell::from(Span::styled(
            format!("{:.0}", n.prompt_tps),
            Style::default().fg(theme::HEALTHY),
        )));
        cells.push(Cell::from(Span::styled(
            format!("{:.1}", n.requests_per_sec),
            Style::default().fg(theme::TEXT),
        )));
    } else {
        // No engine data — show dashes for vLLM columns
        // KV, Hit (+Ext if nixl), Run, Wait, Gen, Prefill
        let dash_n = if f.has_nixl { 7 } else { 6 };
        for _ in 0..dash_n {
            cells.push(Cell::from(Span::styled(
                "-",
                Style::default().fg(theme::MUTED),
            )));
        }
        // Req/s
        cells.push(Cell::from(Span::styled(
            "-",
            Style::default().fg(theme::MUTED),
        )));
    }
    if f.has_load {
        cells.push(Cell::from(Span::styled(
            "-",
            Style::default().fg(theme::MUTED),
        )));
    }
    if f.has_sleep {
        cells.push(Cell::from(Span::styled(
            "-",
            Style::default().fg(theme::MUTED),
        )));
    }
    if f.has_gpu {
        let render_tc = f.has_tc && matches!(tier, TableTier::Wide);
        single_gpu_cells(&mut cells, gpu, render_tc, f.show_power);
    }
    if f.show_nvl {
        single_nvl_cell(&mut cells, gpu);
    }
    if f.has_pcie {
        single_pcie_cells(&mut cells, gpu);
    }
    if f.has_ib {
        // RDMA is per-host, not per-GPU — sub-rows show dashes.
        let dash = || Cell::from(Span::styled("-", Style::default().fg(theme::MUTED)));
        cells.push(dash());
        cells.push(dash());
    }
    if f.has_mfu {
        let flops = engine.map_or(0.0, |n| n.estimated_flops_per_gpu_per_sec);
        flops_table_cell(&mut cells, flops);
    }
    if f.has_nixl && matches!(tier, TableTier::Wide) {
        match engine {
            Some(n) => cells.push(nixl_throughput_cell(
                n.has_nixl,
                n.nixl_throughput_mb_per_sec,
                n.external_kv_transfer_tokens_per_sec,
            )),
            None => cells.push(Cell::from(Span::styled(
                "-",
                Style::default().fg(theme::MUTED),
            ))),
        }
    }
    if let Some(n) = engine {
        cells.push(Cell::from(Span::styled(
            format_count(n.preemptions_total as i64),
            Style::default().fg(theme::TEXT),
        )));
    } else {
        cells.push(Cell::from(""));
    }
    Row::new(cells).style(Style::default().fg(theme::TEXT))
}

/// Render per-GPU cells for a sub-row: GPU%, MBW%, [TC% if render_tc], VRAM%,
/// [Power if render_power]. NVL is rendered separately by `single_nvl_cell`
/// to keep the interconnect columns (NVL / PCIe / RDMA) grouped.
fn single_gpu_cells(
    cells: &mut Vec<Cell<'static>>,
    gpu: Option<&GpuMetrics>,
    render_tc: bool,
    render_power: bool,
) {
    let dash = || Cell::from(Span::styled("-", Style::default().fg(theme::MUTED)));
    match gpu {
        Some(g) => {
            // GPU% / MBW% prefer DCGM PROF, fallback to DEV — see gpu_table_cells.
            let util = g.gpu_pct();
            cells.push(Cell::from(Span::styled(
                format!("{:.0}%", util),
                Style::default().fg(theme::load_color(util)),
            )));
            let mbw = g.mbw_pct();
            cells.push(Cell::from(Span::styled(
                format!("{:.0}%", mbw),
                Style::default().fg(theme::load_color(mbw)),
            )));
            if render_tc {
                match g.tc_pct() {
                    Some(tc) => cells.push(Cell::from(Span::styled(
                        format!("{:.0}%", tc),
                        Style::default().fg(theme::load_color(tc)),
                    ))),
                    None => cells.push(dash()),
                }
            }
            let vram_pct = if g.mem_total_bytes > 0 {
                g.mem_used_bytes as f64 / g.mem_total_bytes as f64 * 100.0
            } else {
                0.0
            };
            cells.push(Cell::from(Span::styled(
                format!("{:.0}%", vram_pct),
                Style::default().fg(theme::load_color(vram_pct)),
            )));
            if render_power {
                cells.push(Cell::from(Span::styled(
                    format!("{:.0}W", g.power_watts),
                    Style::default().fg(theme::TEXT),
                )));
            }
        }
        None => {
            let n = 3 + render_tc as usize + render_power as usize;
            for _ in 0..n {
                cells.push(dash());
            }
        }
    }
}

/// Render the single per-GPU NVL cell for a sub-row. TX only (RX is symmetric
/// for collective ops).
fn single_nvl_cell(cells: &mut Vec<Cell<'static>>, gpu: Option<&GpuMetrics>) {
    let dash = || Cell::from(Span::styled("-", Style::default().fg(theme::MUTED)));
    match gpu {
        Some(g) => {
            let nvl = g.nvlink_tx_kbps as f64 / (1024.0 * 1024.0);
            cells.push(Cell::from(Span::styled(
                format_nvlink_bw(nvl),
                Style::default().fg(theme::TEXT),
            )));
        }
        None => cells.push(dash()),
    }
}

/// Render per-GPU PCIe cells for a sub-row: PCIe↓, PCIe↑.
fn single_pcie_cells(cells: &mut Vec<Cell<'static>>, gpu: Option<&GpuMetrics>) {
    let dash = || Cell::from(Span::styled("-", Style::default().fg(theme::MUTED)));
    match gpu {
        Some(g) => {
            cells.push(Cell::from(Span::styled(
                format_pcie_bw(g.pcie_rx_gbps()),
                Style::default().fg(theme::TEXT),
            )));
            cells.push(Cell::from(Span::styled(
                format_pcie_bw(g.pcie_tx_gbps()),
                Style::default().fg(theme::TEXT),
            )));
        }
        None => {
            cells.push(dash());
            cells.push(dash());
        }
    }
}

fn draw_table(f: &mut Frame, area: Rect, cluster: &ClusterState, ui: &mut UiState) {
    let tier = TableTier::from_width(area.width);
    // Compute current flags and merge with sticky (once on, stays on).
    let current = TableFlags {
        has_gpu: cluster.nodes.iter().any(|n| n.gpu_scrape.is_some()),
        has_mfu: cluster.nodes.iter().any(|n| n.estimated_flops_per_gpu_per_sec > 0.0),
        has_load: cluster.nodes.iter().any(|n| n.server_load.unwrap_or(0.0) > 0.0),
        has_sleep: cluster.nodes.iter().any(|n| n.is_sleeping.is_some()),
        has_ib: cluster
            .nodes
            .iter()
            .any(|n| n.ib_scrape.as_ref().is_some_and(|s| !s.devices.is_empty())),
        has_pcie: cluster.nodes.iter().any(|n| {
            n.gpu_scrape.as_ref().is_some_and(|g| {
                g.gpus.iter().any(|gpu| {
                    gpu.pcie_prof_tx_bytes_per_sec > 0
                        || gpu.pcie_prof_rx_bytes_per_sec > 0
                        || gpu.pcie_tx_kbps > 0
                        || gpu.pcie_rx_kbps > 0
                })
            })
        }),
        has_tc: cluster.nodes.iter().any(|n| {
            n.gpu_scrape
                .as_ref()
                .is_some_and(|g| g.gpus.iter().any(|gpu| gpu.tensor_active.is_some()))
        }),
        has_nixl: cluster.nodes.iter().any(|n| n.has_nixl),
        has_multi_job: cluster.slurm_jobs.len() >= 2,
        show_power: false,
        show_nvl: false,
        show_model: false,
    };
    ui.sticky_flags.has_gpu |= current.has_gpu;
    ui.sticky_flags.has_mfu |= current.has_mfu;
    ui.sticky_flags.has_load |= current.has_load;
    ui.sticky_flags.has_sleep |= current.has_sleep;
    ui.sticky_flags.has_ib |= current.has_ib;
    ui.sticky_flags.has_pcie |= current.has_pcie;
    ui.sticky_flags.has_tc |= current.has_tc;
    ui.sticky_flags.has_nixl |= current.has_nixl;
    ui.sticky_flags.has_multi_job |= current.has_multi_job;
    let node_prefix =
        common_node_prefix(&cluster.nodes.iter().map(|n| n.addr.clone()).collect::<Vec<_>>());
    let mut node_col = node_col_width(cluster, ui.scrape_interval, &node_prefix);

    let mut flags = ui.sticky_flags;
    flags.show_power = flags.has_gpu;
    flags.show_nvl = flags.has_gpu;
    flags.show_model = true;
    // Narrow terminals: shed low-priority columns until the table fits.
    // Order: PCIe↓/PCIe↑ first, then Power, then NVL, then Model. The sticky
    // flags are untouched, so the columns come back when the terminal widens.
    if required_table_width(tier, flags, node_col) > area.width && flags.has_pcie {
        flags.has_pcie = false;
    }
    if required_table_width(tier, flags, node_col) > area.width && flags.show_power {
        flags.show_power = false;
    }
    if required_table_width(tier, flags, node_col) > area.width && flags.show_nvl {
        flags.show_nvl = false;
    }
    if required_table_width(tier, flags, node_col) > area.width && flags.show_model {
        flags.show_model = false;
    }
    // Still overflowing with every shed-able column hidden: take the rest out
    // of the Node column, down to a floor that keeps the status glyph, fold
    // arrow, and at least 7 chars of the node name visible. node_row truncates
    // the name itself so the P/D role badge survives the cut.
    let overflow = required_table_width(tier, flags, node_col).saturating_sub(area.width);
    if overflow > 0 {
        let fold_w = if cluster.nodes.iter().any(|n| sub_row_count(n) > 0) {
            2
        } else {
            0
        };
        node_col = node_col.saturating_sub(overflow).max(2 + fold_w + 7);
    }

    let mut rows: Vec<Row> = Vec::new();
    let mut row_map: Vec<(usize, usize)> = Vec::new(); // (node_idx, engine_tab)

    let is_wide = matches!(tier, TableTier::Wide);

    // Cluster row bar: average current value across all healthy nodes' histories.
    let (cluster_kv_spark, cluster_hit_spark, cluster_ext_hit_spark) =
        if is_wide && cluster.nodes.len() > 1 {
            let hists: Vec<&VecDeque<Snapshot>> = cluster
                .nodes
                .iter()
                .filter(|n| n.is_healthy)
                .filter_map(|n| ui.history.get(&n.addr))
                .collect();
            if hists.is_empty() {
                (String::new(), String::new(), String::new())
            } else {
                let build = |get: fn(&Snapshot) -> f64| -> String {
                    let (sum, cnt) = hists.iter().fold((0.0, 0usize), |(s, c), h| match h.back() {
                        Some(snap) => (s + get(snap), c + 1),
                        None => (s, c),
                    });
                    if cnt == 0 {
                        return String::new();
                    }
                    let cur = sum / cnt as f64;
                    format!(" {}", bar_str(cur * 100.0, 8))
                };
                (
                    build(|s| s.kv_cache),
                    build(|s| s.cache_hit),
                    build(|s| s.ext_hit),
                )
            }
        } else {
            (String::new(), String::new(), String::new())
        };

    if cluster.nodes.len() > 1 {
        rows.push(cluster_row(
            cluster,
            tier,
            flags,
            &cluster_kv_spark,
            &cluster_hit_spark,
            &cluster_ext_hit_spark,
        ));
        row_map.push((usize::MAX, 0)); // sentinel: cluster aggregate row
    }
    for (ni, n) in cluster.nodes.iter().enumerate() {
        let info = cluster.node_infos.get(&n.addr);
        let info_model = info.and_then(|i| i.model_name.as_deref()).unwrap_or("");
        let n_sub = sub_row_count(n);
        let is_collapsed = !ui.expanded.contains(&n.addr);
        let fold_indicator = if n_sub > 0 {
            if is_collapsed { "▸ " } else { "▾ " }
        } else {
            ""
        };
        let (kv_spark, hit_spark_str, ext_hit_spark_str) = if is_wide {
            let h = ui.history.get(&n.addr);
            (
                h.map(kv_sparkline).unwrap_or_default(),
                h.map(hit_sparkline).unwrap_or_default(),
                h.map(ext_sparkline).unwrap_or_default(),
            )
        } else {
            (String::new(), String::new(), String::new())
        };
        let (row_job_id, row_job_ended) = if flags.has_multi_job {
            let host = n.addr.split(':').next().unwrap_or(&n.addr);
            cluster
                .slurm_job_for_host(host)
                .map(|j| (j.job_id.clone(), j.is_ended()))
                .unwrap_or_default()
        } else {
            (String::new(), false)
        };
        let stale = ui
            .scrape_interval
            .map(|iv| n.is_healthy && n.last_updated.elapsed() > iv.saturating_mul(3))
            .unwrap_or(false);
        rows.push(node_row(
            n,
            tier,
            info_model,
            flags,
            fold_indicator,
            &kv_spark,
            &hit_spark_str,
            &ext_hit_spark_str,
            &row_job_id,
            row_job_ended,
            &node_prefix,
            stale,
            node_col,
        ));
        row_map.push((ni, 0)); // aggregate row

        // GPU / engine sub-rows (skip if collapsed)
        if !is_collapsed && n_sub > 0 {
            let gpu_count = n.gpu_scrape.as_ref().map_or(0, |g| g.gpus.len());
            let engine_count = n.engine_metrics.as_ref().map_or(0, |e| e.len());
            let tp_size = if engine_count > 0 && gpu_count >= engine_count {
                gpu_count / engine_count
            } else {
                1
            };
            for gi in 0..n_sub {
                let gpu_metric = n.gpu_scrape.as_ref().and_then(|g| g.gpus.get(gi));
                let engine_idx = gi / tp_size;
                let engine = n.engine_metrics.as_ref().and_then(|e| e.get(engine_idx));
                rows.push(engine_sub_row(engine, gpu_metric, gi, tier, flags));
                row_map.push((ni, engine_idx + 1));
            }
        }
    }

    // Sync row_map and restore selection
    ui.row_map = row_map;
    // Ensure table_state selection is valid
    let max_row = ui.row_map.len().saturating_sub(1);
    let current = ui.table_state.selected().unwrap_or(0).min(max_row);
    ui.table_state.select(Some(current));
    // Sync selected/engine_tab from current row
    if let Some(&(ni, eng)) = ui.row_map.get(current) {
        ui.selected = ni;
        ui.engine_tab = eng;
    }

    let mut widths = table_widths(tier, flags, node_col);
    if required_table_width(tier, flags, node_col) > area.width {
        // Overflow persists even at the Node-column floor. With equal-priority
        // Length constraints ratatui squeezes every column across the board;
        // Min outranks Length in the layout solver, so pin the Node column and
        // let the right-hand metric columns absorb the squeeze instead.
        widths[0] = Constraint::Min(node_col);
    }
    let table = Table::new(rows, widths)
        .header(table_header(tier, flags))
        .block(
            Block::default()
                .borders(Borders::ALL)
                .border_type(BorderType::Rounded)
                .border_style(Style::default().fg(theme::MUTED)),
        )
        .row_highlight_style(Style::default().bg(theme::SELECTED_BG));

    f.render_stateful_widget(table, area, &mut ui.table_state);
}

// ── Tabs ──

fn draw_tabs(f: &mut Frame, area: Rect, ui: &UiState) {
    // Hide the Mooncake tab from the bar while it isn't offered; the select
    // index then counts visible tabs only.
    let visible: Vec<Tab> = (0..Tab::COUNT)
        .map(Tab::from_index)
        .filter(|t| *t != Tab::Mooncake || ui.mooncake_tab)
        .collect();
    let titles: Vec<Line> = visible
        .iter()
        .map(|t| Line::from(format!(" {} ", TAB_TITLES[t.index()])))
        .collect();
    let selected = visible.iter().position(|t| *t == ui.active_tab).unwrap_or(0);
    let tabs = Tabs::new(titles)
        .select(selected)
        .style(Style::default().fg(theme::MUTED))
        .highlight_style(
            Style::default()
                .fg(theme::ACCENT)
                .add_modifier(Modifier::BOLD | Modifier::UNDERLINED),
        )
        .divider(Span::styled("\u{2502}", Style::default().fg(theme::MUTED)));
    f.render_widget(tabs, area);
}

fn draw_search_bar(f: &mut Frame, area: Rect, ui: &UiState) {
    let line = Line::from(vec![
        Span::styled(
            " / ",
            Style::default().fg(theme::WARNING).add_modifier(Modifier::BOLD),
        ),
        Span::styled(&*ui.search_query, Style::default().fg(theme::BRIGHT)),
        Span::styled("_", Style::default().add_modifier(Modifier::SLOW_BLINK)),
    ]);
    f.render_widget(
        Paragraph::new(line).style(Style::default().bg(theme::HEADER_BG)),
        area,
    );
}

// ── Search / filter ──

fn is_section_header(line: &Line) -> bool {
    let text: String = line.spans.iter().map(|s| s.content.as_ref()).collect();
    let trimmed = text.trim();
    (trimmed.starts_with('[') && trimmed.ends_with(']'))
        || (trimmed.starts_with("──") && trimmed.ends_with("──"))
}

fn filter_lines<'a>(lines: Vec<Line<'a>>, query: &str) -> Vec<Line<'a>> {
    if query.is_empty() {
        return lines;
    }
    let q = query.to_lowercase();
    let mut result = Vec::new();
    let mut current_header: Option<Line<'a>> = None;
    let mut header_emitted = false;

    for line in lines {
        if is_section_header(&line) {
            current_header = Some(line);
            header_emitted = false;
            continue;
        }
        let text: String = line.spans.iter().map(|s| s.content.as_ref()).collect();
        if text.to_lowercase().contains(&q) {
            if !header_emitted {
                if let Some(h) = current_header.take() {
                    result.push(h);
                }
                header_emitted = true;
            }
            result.push(line);
        }
    }
    result
}

fn highlight_lines<'a>(lines: Vec<Line<'a>>, query: &str) -> Vec<Line<'a>> {
    if query.is_empty() {
        return lines;
    }
    let q_lower = query.to_lowercase();
    lines
        .into_iter()
        .map(|line| {
            let full_text: String = line.spans.iter().map(|s| s.content.as_ref()).collect();
            let text_lower = full_text.to_lowercase();
            if !text_lower.contains(&q_lower) {
                return line;
            }
            let mut spans = Vec::new();
            let mut pos = 0;
            let bytes = full_text.as_str();
            while pos < bytes.len() {
                if let Some(idx) = text_lower[pos..].find(&q_lower) {
                    let abs = pos + idx;
                    if abs > pos {
                        spans.push(Span::raw(String::from(&bytes[pos..abs])));
                    }
                    spans.push(Span::styled(
                        String::from(&bytes[abs..abs + query.len()]),
                        Style::default().fg(Color::Black).bg(theme::WARNING),
                    ));
                    pos = abs + query.len();
                } else {
                    spans.push(Span::raw(String::from(&bytes[pos..])));
                    break;
                }
            }
            Line::from(spans)
        })
        .collect()
}

fn apply_search<'a>(lines: Vec<Line<'a>>, query: &str) -> Vec<Line<'a>> {
    if query.is_empty() {
        return lines;
    }
    let filtered = filter_lines(lines, query);
    highlight_lines(filtered, query)
}

// ── Detail pane dispatch ──

fn draw_detail(f: &mut Frame, area: Rect, cluster: &ClusterState, ui: &UiState) {
    match ui.active_tab {
        Tab::Overview | Tab::KvXfer | Tab::Mooncake | Tab::Hardware => {
            draw_metrics_detail(f, area, cluster, ui, ui.active_tab)
        }
        Tab::Info => draw_server_info_tab(f, area, cluster, ui),
    }
}

// ── Metrics detail with gauge bars + sparklines ──

/// Build a cluster-level aggregate NodeMetrics by summing all healthy nodes.
fn cluster_aggregate(cluster: &ClusterState) -> NodeMetrics {
    let healthy: Vec<&NodeMetrics> = cluster.nodes.iter().filter(|n| n.is_healthy).collect();
    let mut agg = NodeMetrics::offline("CLUSTER".to_string());
    if healthy.is_empty() {
        return agg;
    }
    agg.is_healthy = true;
    agg.is_loading = false;
    let cnt = healthy.len() as f64;

    // Additive rates
    agg.requests_running = healthy.iter().map(|n| n.requests_running).sum();
    agg.requests_waiting = healthy.iter().map(|n| n.requests_waiting).sum();
    agg.kv_cache_usage = healthy.iter().map(|n| n.kv_cache_usage).sum::<f64>() / cnt;
    agg.prompt_tokens_total = healthy.iter().map(|n| n.prompt_tokens_total).sum();
    agg.generation_tokens_total = healthy.iter().map(|n| n.generation_tokens_total).sum();
    agg.prompt_tps = healthy.iter().map(|n| n.prompt_tps).sum();
    agg.generation_tps = healthy.iter().map(|n| n.generation_tps).sum();
    agg.requests_per_sec = healthy.iter().map(|n| n.requests_per_sec).sum();
    agg.preemptions_per_sec = healthy.iter().map(|n| n.preemptions_per_sec).sum();
    agg.preemptions_total = healthy.iter().map(|n| n.preemptions_total).sum();
    agg.http_qps = healthy.iter().map(|n| n.http_qps).sum();
    agg.kv_fetch_waiting_to_start = healthy.iter().map(|n| n.kv_fetch_waiting_to_start).sum();
    agg.kv_fetch_in_progress = healthy.iter().map(|n| n.kv_fetch_in_progress).sum();
    agg.kv_fetch_completed_waiting = healthy.iter().map(|n| n.kv_fetch_completed_waiting).sum();
    agg.has_kv_fetch = healthy.iter().any(|n| n.has_kv_fetch);

    // Latency (average across nodes)
    macro_rules! avg_pct {
        ($field:ident) => {
            LatencyPct {
                p50: healthy.iter().map(|n| n.$field.p50).sum::<f64>() / cnt,
                p90: healthy.iter().map(|n| n.$field.p90).sum::<f64>() / cnt,
                p99: healthy.iter().map(|n| n.$field.p99).sum::<f64>() / cnt,
                mean: healthy.iter().map(|n| n.$field.mean).sum::<f64>() / cnt,
            }
        };
    }
    agg.cum_ttft = avg_pct!(cum_ttft);
    agg.cum_itl = avg_pct!(cum_itl);
    agg.cum_e2e = avg_pct!(cum_e2e);
    agg.cum_queue = avg_pct!(cum_queue);
    agg.cum_prefill = avg_pct!(cum_prefill);
    agg.cum_decode = avg_pct!(cum_decode);
    agg.cum_inference = avg_pct!(cum_inference);
    agg.cum_tpot = avg_pct!(cum_tpot);
    agg.win_ttft = avg_pct!(win_ttft);
    agg.win_itl = avg_pct!(win_itl);
    agg.win_e2e = avg_pct!(win_e2e);
    agg.win_queue = avg_pct!(win_queue);
    agg.win_prefill = avg_pct!(win_prefill);
    agg.win_decode = avg_pct!(win_decode);
    agg.win_inference = avg_pct!(win_inference);
    agg.win_tpot = avg_pct!(win_tpot);

    // Request stats
    agg.request_success_total = healthy.iter().map(|n| n.request_success_total).sum();
    let mut reasons: HashMap<String, u64> = HashMap::new();
    for n in &healthy {
        for (reason, count) in &n.success_by_reason {
            *reasons.entry(reason.clone()).or_default() += count;
        }
    }
    let mut reasons_vec: Vec<_> = reasons.into_iter().collect();
    reasons_vec.sort_by(|a, b| a.0.cmp(&b.0));
    agg.success_by_reason = reasons_vec;

    // Token stats (weighted average by request count — or simple average)
    agg.avg_prompt_tokens = healthy.iter().map(|n| n.avg_prompt_tokens).sum::<f64>() / cnt;
    agg.avg_generation_tokens = healthy.iter().map(|n| n.avg_generation_tokens).sum::<f64>() / cnt;
    agg.avg_prefill_kv_computed =
        healthy.iter().map(|n| n.avg_prefill_kv_computed).sum::<f64>() / cnt;
    agg.iteration_tokens_mean = healthy.iter().map(|n| n.iteration_tokens_mean).sum::<f64>() / cnt;
    agg.win_prompt_tokens = avg_pct!(win_prompt_tokens);
    agg.win_generation_tokens = avg_pct!(win_generation_tokens);

    // Cache stats (additive totals)
    agg.prompt_tokens_cached_total = healthy.iter().map(|n| n.prompt_tokens_cached_total).sum();
    agg.prompt_tokens_recomputed_total =
        healthy.iter().map(|n| n.prompt_tokens_recomputed_total).sum();
    agg.prefix_cache_hit_rate = healthy.iter().map(|n| n.prefix_cache_hit_rate).sum::<f64>() / cnt;
    agg.external_cache_hit_rate =
        healthy.iter().map(|n| n.external_cache_hit_rate).sum::<f64>() / cnt;
    agg.mm_cache_hit_rate = healthy.iter().map(|n| n.mm_cache_hit_rate).sum::<f64>() / cnt;

    // Speculative decoding (sum totals, average rates)
    agg.spec_decode_draft_tokens_total =
        healthy.iter().map(|n| n.spec_decode_draft_tokens_total).sum();
    agg.spec_decode_accepted_tokens_total =
        healthy.iter().map(|n| n.spec_decode_accepted_tokens_total).sum();
    agg.spec_decode_acceptance_rate =
        healthy.iter().map(|n| n.spec_decode_acceptance_rate).sum::<f64>() / cnt;
    agg.spec_decode_drafts_per_sec = healthy.iter().map(|n| n.spec_decode_drafts_per_sec).sum();

    // Performance / MFU (average across nodes)
    agg.estimated_flops_per_gpu_per_sec =
        healthy.iter().map(|n| n.estimated_flops_per_gpu_per_sec).sum::<f64>() / cnt;
    agg.estimated_read_bytes_per_gpu_per_sec =
        healthy.iter().map(|n| n.estimated_read_bytes_per_gpu_per_sec).sum::<f64>() / cnt;
    agg.estimated_write_bytes_per_gpu_per_sec =
        healthy.iter().map(|n| n.estimated_write_bytes_per_gpu_per_sec).sum::<f64>() / cnt;
    let mfu_vals: Vec<f64> = healthy.iter().filter_map(|n| n.mfu_percent).collect();
    agg.mfu_percent = if mfu_vals.is_empty() {
        None
    } else {
        Some(mfu_vals.iter().sum::<f64>() / mfu_vals.len() as f64)
    };

    // Prompt tokens by source (merge)
    let mut sources: HashMap<String, u64> = HashMap::new();
    for n in &healthy {
        for (src, count) in &n.prompt_tokens_by_source {
            *sources.entry(src.clone()).or_default() += count;
        }
    }
    let mut sources_vec: Vec<_> = sources.into_iter().collect();
    sources_vec.sort_by(|a, b| a.0.cmp(&b.0));
    agg.prompt_tokens_by_source = sources_vec;

    // HTTP status (merge)
    for n in &healthy {
        for (status, count) in &n.http_requests_by_status {
            *agg.http_requests_by_status.entry(status.clone()).or_default() += count;
        }
    }
    let total_http: u64 = agg.http_requests_by_status.values().sum();
    let error_http: u64 = agg
        .http_requests_by_status
        .iter()
        .filter(|(k, _)| k.starts_with('4') || k.starts_with('5'))
        .map(|(_, v)| *v)
        .sum();
    agg.http_error_rate = if total_http > 0 {
        error_http as f64 / total_http as f64
    } else {
        0.0
    };

    // KV Block Residency (average latencies)
    agg.kv_block_lifetime = avg_pct!(kv_block_lifetime);
    agg.kv_block_idle_before_evict = avg_pct!(kv_block_idle_before_evict);
    agg.kv_block_reuse_gap = avg_pct!(kv_block_reuse_gap);
    agg.has_kv_block_metrics = healthy.iter().any(|n| n.has_kv_block_metrics);

    // NIXL (sum totals, average latencies)
    agg.nixl_failed_transfers_total = healthy.iter().map(|n| n.nixl_failed_transfers_total).sum();
    agg.nixl_failed_notifications_total =
        healthy.iter().map(|n| n.nixl_failed_notifications_total).sum();
    agg.nixl_kv_expired_reqs_total = healthy.iter().map(|n| n.nixl_kv_expired_reqs_total).sum();
    agg.nixl_xfer_time = avg_pct!(nixl_xfer_time);
    agg.nixl_post_time = avg_pct!(nixl_post_time);
    agg.nixl_avg_bytes_transferred =
        healthy.iter().map(|n| n.nixl_avg_bytes_transferred).sum::<f64>() / cnt;
    agg.nixl_avg_descriptors = healthy.iter().map(|n| n.nixl_avg_descriptors).sum::<f64>() / cnt;
    agg.nixl_transfers_per_sec = healthy.iter().map(|n| n.nixl_transfers_per_sec).sum();
    agg.nixl_throughput_mb_per_sec =
        healthy.iter().map(|n| n.nixl_throughput_mb_per_sec).sum::<f64>() / cnt;
    agg.has_nixl = healthy.iter().any(|n| n.has_nixl);

    // GPU hardware (aggregate all GPUs across nodes)
    agg.gpu_scrape = aggregate_gpu_scrapes(&cluster.nodes);
    // IB (sum across unique hosts)
    agg.ib_scrape = aggregate_ib_scrapes(&cluster.nodes);

    agg
}

fn draw_metrics_detail(
    f: &mut Frame,
    area: Rect,
    cluster: &ClusterState,
    ui: &UiState,
    group: Tab,
) {
    let is_cluster_row = ui.selected == usize::MAX;
    let cluster_agg;
    let node = if is_cluster_row {
        cluster_agg = cluster_aggregate(cluster);
        Some(&cluster_agg)
    } else {
        cluster.nodes.get(ui.selected)
    };
    let usable = area.width.saturating_sub(2) as usize;

    let lines = match node {
        Some(n) if n.is_healthy => {
            // Hardware sections are per host, never per DP engine — keep the
            // host-level node around before the per-engine shadow below.
            let host = n;
            // Resolve which metrics to display: aggregate or per-engine
            let display_node: &NodeMetrics = n
                .engine_metrics
                .as_ref()
                .filter(|_| ui.engine_tab > 0)
                .and_then(|engines| engines.get(ui.engine_tab - 1))
                .unwrap_or(n);
            let n = display_node;

            let empty_hist = VecDeque::new();
            let h = ui.history.get(&n.addr).unwrap_or(&empty_hist);

            let mut lines = Vec::new();

            if group == Tab::Hardware && ui.engine_tab > 0 {
                lines.push(Line::from(Span::styled(
                    "  (host-level view — hardware is per host, not per DP engine)",
                    Style::default().fg(theme::MUTED),
                )));
            }

            if section_tab(DetailSection::Latency) == group {
                // ── Latency sparklines (2-column, adaptive width) ──
                // Fall back to a single full-width column when half the panel
                // can't fit a whole entry — otherwise merge_n_columns would wrap
                // each row and push the sparkline onto its own line.
                let two_col = usable / 2 >= LATENCY_ENTRY_MIN_W;
                let latency_col_w = if two_col { usable / 2 } else { usable };
                let col_latency_l = vec![
                    section_header("Latency (mean p50/p90/p99)"),
                    latency_entry("TTFT", &n.cum_ttft, &extract(h, |s| s.ttft), latency_col_w),
                    latency_entry("ITL", &n.cum_itl, &extract(h, |s| s.itl), latency_col_w),
                    latency_entry(
                        "Prefill",
                        &n.cum_prefill,
                        &extract(h, |s| s.prefill),
                        latency_col_w,
                    ),
                    latency_entry(
                        "Inference",
                        &n.cum_inference,
                        &extract(h, |s| s.inference),
                        latency_col_w,
                    ),
                ];
                let col_latency_r = vec![
                    Line::from(""),
                    latency_entry("E2E", &n.cum_e2e, &extract(h, |s| s.e2e), latency_col_w),
                    latency_entry(
                        "Queue",
                        &n.cum_queue,
                        &extract(h, |s| s.queue),
                        latency_col_w,
                    ),
                    latency_entry(
                        "Decode",
                        &n.cum_decode,
                        &extract(h, |s| s.decode),
                        latency_col_w,
                    ),
                    latency_entry("TPOT", &n.cum_tpot, &extract(h, |s| s.tpot), latency_col_w),
                ];

                if two_col {
                    lines.extend(merge_n_columns(
                        vec![col_latency_l, col_latency_r],
                        latency_col_w,
                    ));
                } else {
                    lines.extend(col_latency_l);
                    lines.extend(col_latency_r.into_iter().skip(1)); // drop spacer
                }
                lines.push(Line::from(""));
            }

            // ── Stats column groups ──
            // Grid geometry is needed up front: sections with sparkline rows
            // size their charts to the column they will land in.
            let grid_cols = (usable / COL_WIDTH).max(1);
            let grid_col_w = usable / grid_cols.max(1);
            // Sort by status label so the row keeps a stable order across
            // renders (HashMap iteration order is not deterministic).
            let mut status_vec: Vec<_> = n.http_requests_by_status.iter().collect();
            status_vec.sort_by(|a, b| a.0.cmp(b.0));
            let http_status: String = status_vec
                .into_iter()
                .map(|(s, c)| format!("{s}={c}"))
                .collect::<Vec<_>>()
                .join(" ");

            let mut col_request = vec![
                section_header("Request Stats"),
                stat_row("Total", &format_count(n.request_success_total as i64)),
            ];
            for (reason, count) in &n.success_by_reason {
                col_request.push(stat_row(
                    &format!("  {reason}"),
                    &format_count(*count as i64),
                ));
            }
            col_request.push(stat_row(
                "Preemptions",
                &format_count(n.preemptions_total as i64),
            ));
            let col_tokens = vec![
                section_header("Token Stats"),
                stat_row("Prompt Total", &format_count(n.prompt_tokens_total as i64)),
                stat_row(
                    "  per-req",
                    &format!(
                        "avg {:.0}  p50 {:.0}  p99 {:.0}",
                        n.avg_prompt_tokens, n.win_prompt_tokens.p50, n.win_prompt_tokens.p99
                    ),
                ),
                stat_row("Gen Total", &format_count(n.generation_tokens_total as i64)),
                stat_row(
                    "  per-req",
                    &format!(
                        "avg {:.0}  p50 {:.0}  p99 {:.0}",
                        n.avg_generation_tokens,
                        n.win_generation_tokens.p50,
                        n.win_generation_tokens.p99
                    ),
                ),
                stat_row(
                    "PF Computed",
                    &format!("{:.0} tok", n.avg_prefill_kv_computed),
                ),
                stat_row("Iter Tokens", &format!("{:.0}", n.iteration_tokens_mean)),
            ];
            let mut col_cache = vec![
                section_header("Cache Stats"),
                stat_row(
                    "Prefix Hit",
                    &format!("{:.1}%", n.prefix_cache_hit_rate * 100.0),
                ),
                stat_row(
                    "External Hit",
                    &format!("{:.1}%", n.external_cache_hit_rate * 100.0),
                ),
                stat_row("MM Hit", &format!("{:.1}%", n.mm_cache_hit_rate * 100.0)),
                stat_row(
                    "Cached Tok",
                    &format_count(n.prompt_tokens_cached_total as i64),
                ),
                stat_row(
                    "Recomputed",
                    &format_count(n.prompt_tokens_recomputed_total as i64),
                ),
            ];
            for (i, (src, cnt)) in n.prompt_tokens_by_source.iter().enumerate() {
                let label = if i == 0 { "Sources" } else { "" };
                col_cache.push(stat_row(
                    label,
                    &format!("{src}={}", format_count(*cnt as i64)),
                ));
            }
            let col_http = vec![
                section_header("HTTP Stats"),
                stat_row("QPS", &format!("{:.1}", n.http_qps)),
                stat_row_color(
                    "Error Rate",
                    &format!("{:.1}%", n.http_error_rate * 100.0),
                    theme::error_rate_color(n.http_error_rate),
                ),
                stat_row("Status", &http_status),
            ];

            // ── Speculative Decoding (conditional) ──
            let col_spec = if n.spec_decode_draft_tokens_total > 0 {
                let mut lines = vec![
                    section_header("Spec Decode"),
                    stat_row(
                        "Accept Rate",
                        &format!("{:.1}%", n.spec_decode_acceptance_rate * 100.0),
                    ),
                    stat_row("Drafts/s", &format!("{:.1}", n.spec_decode_drafts_per_sec)),
                    stat_row(
                        "Draft Tok",
                        &format_count(n.spec_decode_draft_tokens_total as i64),
                    ),
                    stat_row(
                        "Accepted",
                        &format_count(n.spec_decode_accepted_tokens_total as i64),
                    ),
                ];
                if !n.spec_decode_acceptance_per_pos.is_empty() {
                    let pos_str: String = n
                        .spec_decode_acceptance_per_pos
                        .iter()
                        .map(|(pos, rate)| format!("{}:{:.0}%", pos + 1, rate * 100.0))
                        .collect::<Vec<_>>()
                        .join(" ");
                    lines.push(stat_row("Per-Pos", &pos_str));
                }
                lines
            } else {
                Vec::new()
            };

            // ── Performance / MFU (conditional, host-scoped) ──
            let col_perf = if host.estimated_flops_per_gpu_per_sec > 0.0 {
                let flops_str = format_flops(host.estimated_flops_per_gpu_per_sec);
                let read_bps = host.estimated_read_bytes_per_gpu_per_sec;
                let write_bps = host.estimated_write_bytes_per_gpu_per_sec;
                let read_str = format_bytes_per_sec(read_bps);
                let write_str = format_bytes_per_sec(write_bps);
                let total_str = format_bytes_per_sec(read_bps + write_bps);
                let mfu_str = match host.mfu_percent {
                    Some(pct) => format!("{:.1}%", pct),
                    None => "-".to_string(),
                };
                vec![
                    section_header("Performance (per GPU)"),
                    stat_row("MFU", &mfu_str),
                    stat_row("FLOPs", &flops_str),
                    stat_row("Mem Total", &total_str),
                    stat_row("Mem Read", &read_str),
                    stat_row("Mem Write", &write_str),
                ]
            } else {
                Vec::new()
            };

            // ── KV Cache Residency (conditional) ──
            let col_kv_block = if n.has_kv_block_metrics {
                vec![
                    section_header("KV Block Residency"),
                    stat_row(
                        "Lifetime",
                        &format!(
                            "p50 {} p99 {}",
                            format_ms(n.kv_block_lifetime.p50),
                            format_ms(n.kv_block_lifetime.p99)
                        ),
                    ),
                    stat_row(
                        "Idle→Evict",
                        &format!(
                            "p50 {} p99 {}",
                            format_ms(n.kv_block_idle_before_evict.p50),
                            format_ms(n.kv_block_idle_before_evict.p99)
                        ),
                    ),
                    stat_row(
                        "Reuse Gap",
                        &format!(
                            "p50 {} p99 {}",
                            format_ms(n.kv_block_reuse_gap.p50),
                            format_ms(n.kv_block_reuse_gap.p99)
                        ),
                    ),
                ]
            } else {
                Vec::new()
            };

            // ── NIXL KV Connector (conditional) ──
            let col_nixl = if n.has_nixl {
                let bytes_display = if n.nixl_avg_bytes_transferred >= 1e9 {
                    format!("{:.1} GB", n.nixl_avg_bytes_transferred / 1e9)
                } else if n.nixl_avg_bytes_transferred >= 1e6 {
                    format!("{:.1} MB", n.nixl_avg_bytes_transferred / 1e6)
                } else if n.nixl_avg_bytes_transferred >= 1e3 {
                    format!("{:.1} KB", n.nixl_avg_bytes_transferred / 1e3)
                } else {
                    format!("{:.0} B", n.nixl_avg_bytes_transferred)
                };
                let throughput_display = if n.nixl_throughput_mb_per_sec >= 1024.0 {
                    format!("{:.1} GB/s", n.nixl_throughput_mb_per_sec / 1024.0)
                } else {
                    format!("{:.1} MB/s", n.nixl_throughput_mb_per_sec)
                };
                vec![
                    section_header("NIXL Transfers"),
                    stat_row("Xfer/s", &format!("{:.1}", n.nixl_transfers_per_sec)),
                    stat_row("Throughput", &throughput_display),
                    stat_row(
                        "Xfer Time",
                        &format!(
                            "p50 {} p99 {}",
                            format_ms(n.nixl_xfer_time.p50),
                            format_ms(n.nixl_xfer_time.p99)
                        ),
                    ),
                    stat_row(
                        "Post Time",
                        &format!(
                            "p50 {} p99 {}",
                            format_ms(n.nixl_post_time.p50),
                            format_ms(n.nixl_post_time.p99)
                        ),
                    ),
                    stat_row("Avg Bytes", &bytes_display),
                    stat_row("Avg Descs", &format!("{:.0}", n.nixl_avg_descriptors)),
                    stat_row(
                        "Failed Xfer",
                        &format_count(n.nixl_failed_transfers_total as i64),
                    ),
                    stat_row(
                        "Failed Notif",
                        &format_count(n.nixl_failed_notifications_total as i64),
                    ),
                    stat_row(
                        "KV Expired",
                        &format_count(n.nixl_kv_expired_reqs_total as i64),
                    ),
                ]
            } else {
                Vec::new()
            };

            // ── Async Remote-KV Fetch Stages (conditional) ──
            // `vllm:num_requests_kv_fetch_by_stage`: how many requests sit in
            // each phase of an asynchronous remote-KV fetch (Dynamo PD /
            // KV-offload). Stages: waiting_to_start → in_progress →
            // completed_waiting (received, waiting to run).
            let col_kv_fetch = if n.has_kv_fetch {
                let h_wait = extract(h, |s| s.kv_fetch_wait);
                let h_recv = extract(h, |s| s.kv_fetch_recv);
                let h_done = extract(h, |s| s.kv_fetch_done);
                vec![
                    section_header("Remote KV Fetch"),
                    stat_row_spark(
                        "Wait Start",
                        &format!("{:.0}", n.kv_fetch_waiting_to_start),
                        &h_wait,
                        theme::WARNING,
                        grid_col_w,
                    ),
                    stat_row_spark(
                        "In Progress",
                        &format!("{:.0}", n.kv_fetch_in_progress),
                        &h_recv,
                        theme::ACCENT,
                        grid_col_w,
                    ),
                    stat_row_spark(
                        "Completed",
                        &format!("{:.0}", n.kv_fetch_completed_waiting),
                        &h_done,
                        theme::HEALTHY,
                        grid_col_w,
                    ),
                ]
            } else {
                Vec::new()
            };

            // ── Dynamo Frontend Config ──
            let col_dynamo = if n.has_dynamo_config {
                let mut rows = vec![section_header("Dynamo Config")];
                if n.dynamo_context_length > 0 {
                    rows.push(stat_row("Ctx Length", &n.dynamo_context_length.to_string()));
                }
                if n.dynamo_max_num_seqs > 0 {
                    rows.push(stat_row("Max Seqs", &n.dynamo_max_num_seqs.to_string()));
                }
                if n.dynamo_max_num_batched_tokens > 0 {
                    rows.push(stat_row(
                        "Max Batched",
                        &n.dynamo_max_num_batched_tokens.to_string(),
                    ));
                }
                if n.dynamo_total_kv_blocks > 0 {
                    let kv_capacity = n.dynamo_total_kv_blocks * n.dynamo_kv_block_size;
                    rows.push(stat_row(
                        "KV Blocks",
                        &format!("{} ({}tok)", n.dynamo_total_kv_blocks, kv_capacity),
                    ));
                }
                if n.dynamo_kv_block_size > 0 {
                    rows.push(stat_row(
                        "Block Size",
                        &format!("{} tok", n.dynamo_kv_block_size),
                    ));
                }
                if n.dynamo_disconnected_clients > 0.0 {
                    rows.push(stat_row(
                        "Disconnected",
                        &format!("{:.0}", n.dynamo_disconnected_clients),
                    ));
                }
                if n.dynamo_uptime_secs > 0.0 {
                    rows.push(stat_row(
                        "Uptime",
                        &format_uptime_secs(n.dynamo_uptime_secs),
                    ));
                }
                if n.dynamo_model_load_secs > 0.0 {
                    rows.push(stat_row(
                        "Model Load",
                        &format!("{:.1}s", n.dynamo_model_load_secs),
                    ));
                }
                if n.dynamo_component_requests_total > 0
                    || n.dynamo_component_requests_per_sec > 0.0
                {
                    rows.push(stat_row(
                        "Worker Reqs",
                        &format!(
                            "{} ({:.1}/s)",
                            n.dynamo_component_requests_total, n.dynamo_component_requests_per_sec
                        ),
                    ));
                }
                if let Some(inflight) = n.dynamo_component_inflight {
                    if inflight > 0.0 {
                        rows.push(stat_row("Inflight", &format!("{inflight:.0}")));
                    }
                }
                if n.dynamo_tokenize_latency.p99 > 0.0 {
                    rows.push(stat_row(
                        "Tokenize",
                        &format!(
                            "p50 {} p99 {}",
                            format_ms(n.dynamo_tokenize_latency.p50),
                            format_ms(n.dynamo_tokenize_latency.p99)
                        ),
                    ));
                }
                if n.dynamo_detokenize_latency.p99 > 0.0 {
                    rows.push(stat_row(
                        "Detokenize",
                        &format!(
                            "p50 {} p99 {}",
                            format_ms(n.dynamo_detokenize_latency.p50),
                            format_ms(n.dynamo_detokenize_latency.p99)
                        ),
                    ));
                }
                rows
            } else {
                Vec::new()
            };

            // ── KV Events (only if ZMQ subscriber is active) ──
            let col_kv = if let Some(ranks) = &n.kv_events {
                let h_stored = extract(h, |s| s.kv_stored);
                let h_evicted = extract(h, |s| s.kv_evicted);
                let h_active = extract(h, |s| s.kv_active);
                let h_tokens = extract(h, |s| s.kv_tokens);
                let h_churn = extract(h, |s| s.kv_churn);

                let mut lines_kv = Vec::new();
                let single = ranks.len() == 1;
                if single {
                    // Single DP rank: flat display with sparklines
                    let kv = &ranks[0];
                    lines_kv.push(section_header("KV Cache Events (ZMQ)"));
                    lines_kv.push(stat_row_spark(
                        "Blk Stored",
                        &format!("{} blk", kv.blocks_stored),
                        &h_stored,
                        theme::HEALTHY,
                        usable,
                    ));
                    lines_kv.push(stat_row_spark(
                        "Blk Evicted",
                        &format!("{} blk", kv.blocks_removed),
                        &h_evicted,
                        theme::WARNING,
                        usable,
                    ));
                    lines_kv.push(stat_row_spark(
                        "Active Blks",
                        &format_count(kv.active_blocks),
                        &h_active,
                        theme::ACCENT,
                        usable,
                    ));
                    lines_kv.push(stat_row_spark(
                        "Tok Cached",
                        &format!("{} tok", kv.tokens_stored),
                        &h_tokens,
                        theme::HEALTHY,
                        usable,
                    ));
                    lines_kv.push(stat_row_spark(
                        "Churn",
                        &format!("{} evt", kv.blocks_stored + kv.blocks_removed),
                        &h_churn,
                        theme::PURPLE,
                        usable,
                    ));
                    lines_kv.push(stat_row("Seq Gaps", &kv.seq_gaps.to_string()));
                } else {
                    // Multiple DP ranks: per-rank columns with aggregate sparklines
                    let rank_cols: Vec<Vec<Line<'static>>> = ranks
                        .iter()
                        .enumerate()
                        .map(|(i, kv)| {
                            let has_data = kv.total_events > 0 || kv.active_blocks != 0;
                            // Use the recorded DP rank, with an index fallback
                            // for replays of old recordings.
                            let rank = kv.dp_rank.unwrap_or(i as u16);
                            let mut col = vec![section_header(&format!("KV Events DP{rank}"))];
                            if has_data {
                                col.extend([
                                    stat_row("Blk Stored", &format!("{} blk", kv.blocks_stored)),
                                    stat_row("Blk Evicted", &format!("{} blk", kv.blocks_removed)),
                                    stat_row("Active Blks", &format_count(kv.active_blocks)),
                                    stat_row("Tok Cached", &format!("{} tok", kv.tokens_stored)),
                                    stat_row(
                                        "Churn",
                                        &format!("{} evt", kv.blocks_stored + kv.blocks_removed),
                                    ),
                                    stat_row("Seq Gaps", &kv.seq_gaps.to_string()),
                                ]);
                            } else {
                                col.push(Line::from(Span::styled(
                                    "  (no events received)",
                                    Style::default().fg(theme::MUTED),
                                )));
                            }
                            col
                        })
                        .collect();
                    lines_kv.extend(merge_n_columns(rank_cols, COL_WIDTH));
                    // Aggregate sparklines below per-rank columns
                    lines_kv.push(Line::from(""));
                    lines_kv.push(section_header("KV Aggregate Trends"));
                    lines_kv.push(stat_row_spark(
                        "Blk Stored",
                        "",
                        &h_stored,
                        theme::HEALTHY,
                        usable,
                    ));
                    lines_kv.push(stat_row_spark(
                        "Blk Evicted",
                        "",
                        &h_evicted,
                        theme::WARNING,
                        usable,
                    ));
                    lines_kv.push(stat_row_spark(
                        "Active Blks",
                        "",
                        &h_active,
                        theme::ACCENT,
                        usable,
                    ));
                    lines_kv.push(stat_row_spark("Churn", "", &h_churn, theme::PURPLE, usable));
                }
                lines_kv
            } else {
                Vec::new()
            };

            // ── Adaptive grid: flow this tab's stat sections into N columns ──
            let mut sections: Vec<Vec<Line<'static>>> = Vec::new();
            for (sec, col) in [
                (DetailSection::RequestStats, col_request),
                (DetailSection::TokenStats, col_tokens),
                (DetailSection::CacheStats, col_cache),
                (DetailSection::HttpStats, col_http),
                (DetailSection::SpecDecode, col_spec),
                (DetailSection::PerfMfu, col_perf),
                (DetailSection::KvBlockResidency, col_kv_block),
                (DetailSection::NixlTransfers, col_nixl),
                (DetailSection::RemoteKvFetch, col_kv_fetch),
                (DetailSection::DynamoConfig, col_dynamo),
            ] {
                if section_tab(sec) == group && !col.is_empty() {
                    sections.push(col);
                }
            }

            let mut sec_iter = sections.into_iter().peekable();
            while sec_iter.peek().is_some() {
                let row: Vec<Vec<Line<'static>>> = sec_iter.by_ref().take(grid_cols).collect();
                lines.extend(merge_n_columns(row, grid_col_w));
            }

            // Host-scoped like the hardware sections: per-engine metrics never
            // carry KV-transfer data (kv_events is None for DP-rank rows), and
            // the cluster aggregate has no kv_events either, so check members.
            let has_kv_transfer = host.has_kv_block_metrics
                || host.has_nixl
                || host.has_kv_fetch
                || host.kv_events.is_some()
                || (is_cluster_row && cluster.nodes.iter().any(|m| m.kv_events.is_some()));
            if group == Tab::KvXfer && !has_kv_transfer {
                lines.push(Line::from(Span::styled(
                    "  (no KV-transfer metrics detected on this node)",
                    Style::default().fg(theme::MUTED),
                )));
            }

            // ── Mooncake Store (cluster-wide, own tab) — one block per
            // tracked store (multi-job auto mode tracks one per job).
            if section_tab(DetailSection::Mooncake) == group {
                for mc in &cluster.mooncakes {
                    lines.push(Line::from(""));
                    let host = mc.addr.split(':').next().unwrap_or(&mc.addr);
                    let job_id = cluster.slurm_job_for_host(host).map(|j| j.job_id.as_str());
                    lines.extend(mooncake_detail_lines(mc, job_id));
                }
                for addr in cluster
                    .mooncake_addrs
                    .iter()
                    .filter(|a| !cluster.mooncakes.iter().any(|m| &m.addr == *a))
                {
                    lines.push(Line::from(""));
                    lines.push(section_header(&format!(
                        "Mooncake Store · unreachable ({addr})"
                    )));
                }
            }

            // ── GPU Hardware (full-width, host-scoped) ──
            if section_tab(DetailSection::GpuHardware) == group {
                if let Some(gs) = &host.gpu_scrape {
                    let gpu_count = gs.gpus.len();
                    let tpgs = if gpu_count > 0 {
                        (host.generation_tps + host.prompt_tps) / gpu_count as f64
                    } else {
                        0.0
                    };
                    let (vram_used, vram_total) = gs.mem_used_total();
                    let used_gib = vram_used as f64 / (1024.0 * 1024.0 * 1024.0);
                    let total_gib = vram_total as f64 / (1024.0 * 1024.0 * 1024.0);
                    let gpu_model = gs
                        .gpus
                        .iter()
                        .find_map(|g| {
                            if g.name.is_empty() {
                                None
                            } else {
                                Some(g.name.as_str())
                            }
                        })
                        .unwrap_or("-");
                    let power_limit = gs.total_power_limit();
                    let power_str = if power_limit > 0.0 {
                        format!("{:.0}/{:.0} W", gs.total_power(), power_limit)
                    } else {
                        format!("{:.0} W", gs.total_power())
                    };
                    let mut col_gpu = vec![
                        section_header("GPU Hardware"),
                        stat_row("Model", &format!("{} × {}", gpu_count, gpu_model)),
                        stat_row("TPGS", &format!("{:.1} tok/s/gpu", tpgs)),
                        stat_row("Avg Util", &format!("{:.0}%", gs.avg_utilization())),
                        stat_row("Avg MBW", &format!("{:.0}%", gs.avg_mem_utilization())),
                        stat_row(
                            "NVLink",
                            &format!(
                                "TX {}  RX {}",
                                format_nvlink_bw_long(
                                    gs.total_nvlink_tx_kbps() as f64 / (1024.0 * 1024.0)
                                ),
                                format_nvlink_bw_long(
                                    gs.total_nvlink_rx_kbps() as f64 / (1024.0 * 1024.0)
                                ),
                            ),
                        ),
                        stat_row("VRAM", &format!("{:.1}/{:.1} GiB", used_gib, total_gib)),
                        stat_row("Power", &power_str),
                        stat_row("Avg Temp", &format!("{:.0}°C", gs.avg_temperature())),
                    ];
                    if let Some(cpu) = gs.cpu_percent {
                        col_gpu.push(stat_row("Host CPU", &format!("{:.0}%", cpu)));
                    }
                    if let (Some(mu), Some(mt)) = (gs.mem_used_bytes, gs.mem_total_bytes) {
                        let mu_g = mu as f64 / (1024.0 * 1024.0 * 1024.0);
                        let mt_g = mt as f64 / (1024.0 * 1024.0 * 1024.0);
                        col_gpu.push(stat_row(
                            "Host MEM",
                            &format!("{:.0}/{:.0} GiB", mu_g, mt_g),
                        ));
                    }
                    lines.push(Line::from(""));
                    lines.extend(col_gpu);
                }
            }

            // ── RDMA / InfiniBand (full-width, per-device, host-scoped) ──
            if section_tab(DetailSection::RdmaIb) == group {
                if let Some(ib) = &host.ib_scrape {
                    if !ib.devices.is_empty() {
                        lines.push(Line::from(""));
                        lines.extend(rdma_detail_lines(ib, usable));
                    }
                }
            }

            let has_hw = host.gpu_scrape.is_some()
                || host.ib_scrape.as_ref().is_some_and(|ib| !ib.devices.is_empty())
                || host.estimated_flops_per_gpu_per_sec > 0.0;
            if group == Tab::Hardware && !has_hw {
                lines.push(Line::from(Span::styled(
                    "  (no GPU agent / DCGM / InfiniBand data for this host)",
                    Style::default().fg(theme::MUTED),
                )));
            }

            if section_tab(DetailSection::KvEvents) == group && !col_kv.is_empty() {
                lines.push(Line::from(""));
                lines.extend(col_kv);
            }

            lines
        }
        Some(n) => {
            let gpu_mem_loaded = n
                .gpu_scrape
                .as_ref()
                .map(|g| {
                    g.gpus.iter().any(|gpu| {
                        gpu.mem_total_bytes > 0
                            && (gpu.mem_used_bytes as f64 / gpu.mem_total_bytes as f64) > 0.2
                    })
                })
                .unwrap_or(false);
            let (header_text, header_color) = if gpu_mem_loaded {
                (format!(" {} — starting…", n.addr), theme::WARNING)
            } else if n.is_loading {
                (format!(" {} — connecting…", n.addr), theme::TEXT)
            } else {
                (format!(" {} (offline)", n.addr), theme::DANGER)
            };
            let mut lines = vec![Line::from(Span::styled(
                header_text,
                Style::default().fg(header_color),
            ))];
            // Stale hardware honors the same section → tab routing as the
            // healthy path: GPU info lives on the Hardware tab only.
            if group == Tab::Hardware {
                if let Some(gs) = &n.gpu_scrape {
                    let gpu_count = gs.gpus.len();
                    let gpu_model = gs
                        .gpus
                        .iter()
                        .find_map(|g| {
                            if g.name.is_empty() {
                                None
                            } else {
                                Some(g.name.as_str())
                            }
                        })
                        .unwrap_or("-");
                    let (vram_used, vram_total) = gs.mem_used_total();
                    let used_gib = vram_used as f64 / (1024.0 * 1024.0 * 1024.0);
                    let total_gib = vram_total as f64 / (1024.0 * 1024.0 * 1024.0);
                    let power_limit = gs.total_power_limit();
                    let power_str = if power_limit > 0.0 {
                        format!("{:.0}/{:.0} W", gs.total_power(), power_limit)
                    } else {
                        format!("{:.0} W", gs.total_power())
                    };
                    lines.push(Line::from(""));
                    lines.push(section_header(if gpu_mem_loaded {
                        "GPU Hardware"
                    } else {
                        "GPU Hardware (stale)"
                    }));
                    lines.push(stat_row("Model", &format!("{} × {}", gpu_count, gpu_model)));
                    lines.push(stat_row(
                        "Avg Util",
                        &format!("{:.0}%", gs.avg_utilization()),
                    ));
                    lines.push(stat_row(
                        "Avg MBW",
                        &format!("{:.0}%", gs.avg_mem_utilization()),
                    ));
                    lines.push(stat_row(
                        "NVLink",
                        &format!(
                            "TX {}  RX {}",
                            format_nvlink_bw_long(
                                gs.total_nvlink_tx_kbps() as f64 / (1024.0 * 1024.0)
                            ),
                            format_nvlink_bw_long(
                                gs.total_nvlink_rx_kbps() as f64 / (1024.0 * 1024.0)
                            ),
                        ),
                    ));
                    lines.push(stat_row(
                        "VRAM",
                        &format!("{:.1}/{:.1} GiB", used_gib, total_gib),
                    ));
                    lines.push(stat_row("Power", &power_str));
                    lines.push(stat_row(
                        "Avg Temp",
                        &format!("{:.0}°C", gs.avg_temperature()),
                    ));
                    if let Some(cpu) = gs.cpu_percent {
                        lines.push(stat_row("Host CPU", &format!("{:.0}%", cpu)));
                    }
                    if let (Some(mu), Some(mt)) = (gs.mem_used_bytes, gs.mem_total_bytes) {
                        let mu_g = mu as f64 / (1024.0 * 1024.0 * 1024.0);
                        let mt_g = mt as f64 / (1024.0 * 1024.0 * 1024.0);
                        lines.push(stat_row(
                            "Host MEM",
                            &format!("{:.0}/{:.0} GiB", mu_g, mt_g),
                        ));
                    }
                } else {
                    lines.push(Line::from(Span::styled(
                        "  No data available",
                        Style::default().fg(theme::MUTED),
                    )));
                }
            }
            lines
        }
        None => vec![Line::from(Span::styled(
            " No node selected",
            Style::default().fg(theme::MUTED),
        ))],
    };

    let mut lines = apply_search(lines, &ui.search_query);
    let match_count = lines.len();
    if !ui.search_query.is_empty() && lines.is_empty() {
        lines.push(Line::from(Span::styled(
            "  no matches on this tab — other tabs may match",
            Style::default().fg(theme::MUTED),
        )));
    }
    let match_title = if ui.search_query.is_empty() {
        String::new()
    } else {
        format!(" {} matches ", match_count)
    };
    let mut block = Block::default()
        .borders(Borders::ALL)
        .border_type(BorderType::Rounded)
        .border_style(Style::default().fg(theme::MUTED));
    if !match_title.is_empty() {
        block = block.title(Span::styled(match_title, theme::accent()));
    }
    let paragraph = Paragraph::new(lines).block(block).scroll((ui.detail_scroll, 0));
    f.render_widget(paragraph, area);
}

// ── Server info (Info tab) ──

fn draw_server_info_tab(f: &mut Frame, area: Rect, cluster: &ClusterState, ui: &UiState) {
    // Diff picking prompt
    if ui.diff_picking {
        let mut lines = vec![Line::from(Span::styled(
            "  Select a node with j/k, then press d to compare",
            Style::default().fg(theme::WARNING).add_modifier(Modifier::BOLD),
        ))];
        lines.push(Line::from(Span::styled(
            "  Press d again to cancel",
            Style::default().fg(theme::MUTED),
        )));
        lines.push(Line::from(""));
        for (i, n) in cluster.nodes.iter().enumerate() {
            let marker = if i == ui.selected { "▶ " } else { "  " };
            let color = if i == ui.selected {
                theme::ACCENT
            } else {
                theme::TEXT
            };
            lines.push(Line::from(Span::styled(
                format!("{marker}{}", n.addr),
                Style::default().fg(color),
            )));
        }
        let block = Block::default()
            .borders(Borders::ALL)
            .border_type(BorderType::Rounded)
            .border_style(Style::default().fg(theme::WARNING))
            .title(Span::styled(
                " Diff: pick target node ",
                Style::default().fg(theme::WARNING).add_modifier(Modifier::BOLD),
            ));
        let paragraph = Paragraph::new(lines).block(block);
        f.render_widget(paragraph, area);
        return;
    }

    // Diff mode: compare selected node with diff_target
    if let Some(target_idx) = ui.diff_target {
        draw_diff_view(f, area, cluster, ui, target_idx);
        return;
    }

    // Normal view: all three server_info sections in one scrollable view.
    let node = cluster.nodes.get(ui.selected);
    let addr = node.map(|n| &n.addr);
    let info = addr.and_then(|a| cluster.node_infos.get(a));
    let server_info = info.and_then(|i| i.server_info.as_ref());

    let lines: Vec<Line> = if ui.selected == usize::MAX {
        vec![Line::from(Span::styled(
            "  Select a node — the cluster row has no server_info",
            Style::default().fg(theme::MUTED),
        ))]
    } else {
        match server_info {
            Some(si) => {
                let mut lines = Vec::new();
                for (key, title) in INFO_SECTIONS {
                    lines.push(Line::from(Span::styled(
                        format!(" [{title}]"),
                        Style::default().fg(theme::PURPLE).add_modifier(Modifier::BOLD),
                    )));
                    match si.get(key) {
                        Some(data) => lines.extend(format_json_lines(data, key)),
                        None => lines.push(Line::from(Span::styled(
                            "  (not available)",
                            Style::default().fg(theme::MUTED),
                        ))),
                    }
                    lines.push(Line::from(""));
                }
                lines
            }
            None => vec![Line::from(Span::styled(
                "  No server_info available (endpoint may not be supported)",
                Style::default().fg(theme::MUTED),
            ))],
        }
    };

    let mut lines = apply_search(lines, &ui.search_query);
    let match_count = lines.len();
    if !ui.search_query.is_empty() && lines.is_empty() {
        lines.push(Line::from(Span::styled(
            "  no matches on this tab — other tabs may match",
            Style::default().fg(theme::MUTED),
        )));
    }
    let match_title = if ui.search_query.is_empty() {
        String::new()
    } else {
        format!(" {} matches ", match_count)
    };
    let mut block = Block::default()
        .borders(Borders::ALL)
        .border_type(BorderType::Rounded)
        .border_style(Style::default().fg(theme::MUTED));
    if !match_title.is_empty() {
        block = block.title(Span::styled(match_title, theme::accent()));
    }
    let paragraph = Paragraph::new(lines)
        .block(block)
        .wrap(Wrap { trim: false })
        .scroll((ui.detail_scroll, 0));
    f.render_widget(paragraph, area);
}

/// Render a diff view comparing two nodes' server_info (all sections).
fn draw_diff_view(
    f: &mut Frame,
    area: Rect,
    cluster: &ClusterState,
    ui: &UiState,
    target_idx: usize,
) {
    let get_info = |idx: usize| {
        cluster
            .nodes
            .get(idx)
            .and_then(|n| cluster.node_infos.get(&n.addr))
            .and_then(|i| i.server_info.as_ref())
    };

    let left_addr = cluster.nodes.get(ui.selected).map(|n| n.addr.as_str()).unwrap_or("?");
    let right_addr = cluster.nodes.get(target_idx).map(|n| n.addr.as_str()).unwrap_or("?");

    let left = get_info(ui.selected);
    let right = get_info(target_idx);

    let mut lines: Vec<Line> = Vec::new();

    match (left, right) {
        (Some(l), Some(r)) => {
            for (key, title) in INFO_SECTIONS {
                let mut sec_lines: Vec<Line<'static>> = Vec::new();
                match (l.get(key), r.get(key)) {
                    (None, None) => continue,
                    (Some(a), Some(b)) => diff_json_objects(&mut sec_lines, a, b, key),
                    (Some(_), None) => sec_lines.push(Line::from(Span::styled(
                        format!("  only present on {left_addr}"),
                        Style::default().fg(theme::MUTED),
                    ))),
                    (None, Some(_)) => sec_lines.push(Line::from(Span::styled(
                        format!("  only present on {right_addr}"),
                        Style::default().fg(theme::MUTED),
                    ))),
                }
                if !sec_lines.is_empty() {
                    lines.push(Line::from(Span::styled(
                        format!(" [{title}]"),
                        Style::default().fg(theme::PURPLE).add_modifier(Modifier::BOLD),
                    )));
                    lines.extend(sec_lines);
                    lines.push(Line::from(""));
                }
            }
        }
        (None, None) => {
            lines.push(Line::from(Span::styled(
                "  No server_info on either node",
                Style::default().fg(theme::MUTED),
            )));
        }
        (None, _) => {
            lines.push(Line::from(Span::styled(
                format!("  No server_info on {left_addr}"),
                Style::default().fg(theme::MUTED),
            )));
        }
        (_, None) => {
            lines.push(Line::from(Span::styled(
                format!("  No server_info on {right_addr}"),
                Style::default().fg(theme::MUTED),
            )));
        }
    }

    if lines.is_empty() {
        lines.push(Line::from(Span::styled(
            "  No differences found",
            Style::default().fg(theme::HEALTHY),
        )));
    }

    let lines = apply_search(lines, &ui.search_query);

    let title = format!(" Diff: {} vs {} (d to exit) ", left_addr, right_addr);
    let block = Block::default()
        .borders(Borders::ALL)
        .border_type(BorderType::Rounded)
        .border_style(Style::default().fg(theme::PURPLE))
        .title(Span::styled(
            title,
            Style::default().fg(theme::PURPLE).add_modifier(Modifier::BOLD),
        ));
    let paragraph = Paragraph::new(lines)
        .block(block)
        .wrap(Wrap { trim: false })
        .scroll((ui.detail_scroll, 0));
    f.render_widget(paragraph, area);
}

/// Compare two JSON objects and produce diff lines.
fn diff_json_objects(
    lines: &mut Vec<Line<'static>>,
    left: &serde_json::Value,
    right: &serde_json::Value,
    section: &str,
) {
    if section == "vllm_config" {
        diff_vllm_config(lines, left, right);
    } else {
        diff_flat(lines, left, right);
    }
}

fn diff_vllm_config(
    lines: &mut Vec<Line<'static>>,
    left: &serde_json::Value,
    right: &serde_json::Value,
) {
    let empty_map = serde_json::Map::new();
    let l_obj = left.as_object().unwrap_or(&empty_map);
    let r_obj = right.as_object().unwrap_or(&empty_map);

    // Collect all section names
    let mut sections: Vec<&String> = l_obj.keys().chain(r_obj.keys()).collect();
    sections.sort();
    sections.dedup();

    for sec in sections {
        let l_sec = l_obj.get(sec.as_str());
        let r_sec = r_obj.get(sec.as_str());

        let l_inner = l_sec.and_then(|v| v.as_object()).unwrap_or(&empty_map);
        let r_inner = r_sec.and_then(|v| v.as_object()).unwrap_or(&empty_map);

        let mut sec_lines: Vec<Line<'static>> = Vec::new();
        let mut keys: Vec<&String> = l_inner.keys().chain(r_inner.keys()).collect();
        keys.sort();
        keys.dedup();

        for key in keys {
            let lv = l_inner.get(key.as_str());
            let rv = r_inner.get(key.as_str());

            match (lv, rv) {
                (Some(a), Some(b)) if a == b => continue, // Same: skip
                (Some(a), Some(b)) => {
                    // Different values
                    sec_lines.push(Line::from(vec![
                        Span::styled(format!("  - {key}: "), Style::default().fg(theme::DANGER)),
                        Span::styled(format_value_compact(a), Style::default().fg(theme::DANGER)),
                    ]));
                    sec_lines.push(Line::from(vec![
                        Span::styled(format!("  + {key}: "), Style::default().fg(theme::HEALTHY)),
                        Span::styled(format_value_compact(b), Style::default().fg(theme::HEALTHY)),
                    ]));
                }
                (Some(a), None) => {
                    sec_lines.push(Line::from(Span::styled(
                        format!("  - {key}: {}", format_value_compact(a)),
                        Style::default().fg(theme::DANGER),
                    )));
                }
                (None, Some(b)) => {
                    sec_lines.push(Line::from(Span::styled(
                        format!("  + {key}: {}", format_value_compact(b)),
                        Style::default().fg(theme::HEALTHY),
                    )));
                }
                (None, None) => unreachable!(),
            }
        }

        if !sec_lines.is_empty() {
            lines.push(Line::from(Span::styled(
                format!("  [{sec}]"),
                Style::default().fg(theme::PURPLE).add_modifier(Modifier::BOLD),
            )));
            lines.extend(sec_lines);
            lines.push(Line::from(""));
        }
    }
}

fn diff_flat(lines: &mut Vec<Line<'static>>, left: &serde_json::Value, right: &serde_json::Value) {
    let empty_map = serde_json::Map::new();
    let l_obj = left.as_object().unwrap_or(&empty_map);
    let r_obj = right.as_object().unwrap_or(&empty_map);

    let mut keys: Vec<&String> = l_obj.keys().chain(r_obj.keys()).collect();
    keys.sort();
    keys.dedup();

    for key in keys {
        let lv = l_obj.get(key.as_str());
        let rv = r_obj.get(key.as_str());

        match (lv, rv) {
            (Some(a), Some(b)) if a == b => continue,
            (Some(a), Some(b)) => {
                lines.push(Line::from(vec![
                    Span::styled(format!("  - {key}: "), Style::default().fg(theme::DANGER)),
                    Span::styled(format_value_compact(a), Style::default().fg(theme::DANGER)),
                ]));
                lines.push(Line::from(vec![
                    Span::styled(format!("  + {key}: "), Style::default().fg(theme::HEALTHY)),
                    Span::styled(format_value_compact(b), Style::default().fg(theme::HEALTHY)),
                ]));
            }
            (Some(a), None) => {
                lines.push(Line::from(Span::styled(
                    format!("  - {key}: {}", format_value_compact(a)),
                    Style::default().fg(theme::DANGER),
                )));
            }
            (None, Some(b)) => {
                lines.push(Line::from(Span::styled(
                    format!("  + {key}: {}", format_value_compact(b)),
                    Style::default().fg(theme::HEALTHY),
                )));
            }
            (None, None) => unreachable!(),
        }
    }
}

/// Compact value formatting for diff display.
fn format_value_compact(v: &serde_json::Value) -> String {
    match v {
        serde_json::Value::String(s) => {
            if s.len() > 80 {
                format!("\"{}...\"", &s[..80])
            } else {
                format!("\"{s}\"")
            }
        }
        serde_json::Value::Null => "null".to_string(),
        serde_json::Value::Bool(b) => b.to_string(),
        serde_json::Value::Number(n) => n.to_string(),
        other => {
            let s = serde_json::to_string(other).unwrap_or_default();
            if s.len() > 80 {
                format!("{}...", &s[..80])
            } else {
                s
            }
        }
    }
}

fn format_json_lines(value: &serde_json::Value, section: &str) -> Vec<Line<'static>> {
    match section {
        "vllm_config" => format_vllm_config(value),
        _ => format_flat_object(value),
    }
}

fn format_vllm_config(value: &serde_json::Value) -> Vec<Line<'static>> {
    let mut lines = Vec::new();
    if let Some(obj) = value.as_object() {
        for (section_name, section_val) in obj {
            lines.push(Line::from(Span::styled(
                format!("  [{section_name}]"),
                Style::default().fg(theme::PURPLE).add_modifier(Modifier::BOLD),
            )));
            if let Some(inner) = section_val.as_object() {
                for (k, v) in inner {
                    format_config_kv(&mut lines, k, v, "    ");
                }
            } else {
                let val_str = format_value(section_val);
                lines.push(Line::from(vec![
                    Span::styled("    ", Style::default()),
                    Span::styled(val_str, Style::default().fg(theme::TEXT)),
                ]));
            }
            lines.push(Line::from(""));
        }
    }
    lines
}

/// Format a key-value pair, pretty-printing complex/embedded JSON values.
fn format_config_kv(
    lines: &mut Vec<Line<'static>>,
    key: &str,
    val: &serde_json::Value,
    indent: &str,
) {
    // Try to resolve the actual JSON value: if it's a string containing JSON, parse it.
    let parsed_embedded: Option<serde_json::Value> = match val {
        serde_json::Value::String(s) if s.starts_with('{') || s.starts_with('[') => {
            serde_json::from_str(s).ok()
        }
        _ => None,
    };
    let resolved = parsed_embedded.as_ref().unwrap_or(val);

    match resolved {
        serde_json::Value::Object(obj) if !obj.is_empty() => {
            lines.push(Line::from(Span::styled(
                format!("{indent}{key}:"),
                Style::default().fg(theme::ACCENT),
            )));
            append_pretty_json_lines(lines, resolved, &format!("{indent}  "));
        }
        serde_json::Value::Array(arr) if !arr.is_empty() => {
            lines.push(Line::from(Span::styled(
                format!("{indent}{key}:"),
                Style::default().fg(theme::ACCENT),
            )));
            append_pretty_json_lines(lines, resolved, &format!("{indent}  "));
        }
        _ => {
            let val_str = format_value(val);
            let style = if val_str == "true" || val_str == "True" {
                Style::default().fg(theme::HEALTHY)
            } else if val_str == "false" || val_str == "False" || val_str == "null" {
                Style::default().fg(theme::MUTED)
            } else {
                Style::default().fg(theme::TEXT)
            };
            lines.push(Line::from(vec![
                Span::styled(
                    format!("{indent}{key}: "),
                    Style::default().fg(theme::ACCENT),
                ),
                Span::styled(val_str, style),
            ]));
        }
    }
}

/// Append pretty-printed JSON lines with syntax coloring.
fn append_pretty_json_lines(
    lines: &mut Vec<Line<'static>>,
    value: &serde_json::Value,
    indent: &str,
) {
    let pretty = serde_json::to_string_pretty(value).unwrap_or_default();
    for line in pretty.lines() {
        let trimmed = line.trim();
        if trimmed.is_empty() {
            continue;
        }
        // Determine color based on content
        let style = if trimmed.starts_with('"') && trimmed.contains(':') {
            // JSON key line: "key": value
            Style::default().fg(theme::ACCENT)
        } else if trimmed == "{"
            || trimmed == "}"
            || trimmed == "["
            || trimmed == "]"
            || trimmed == "{}"
            || trimmed == "[]"
            || trimmed == "},"
            || trimmed == "],"
        {
            Style::default().fg(theme::MUTED)
        } else if trimmed == "true" || trimmed == "true," {
            Style::default().fg(theme::HEALTHY)
        } else if trimmed == "false"
            || trimmed == "false,"
            || trimmed == "null"
            || trimmed == "null,"
        {
            Style::default().fg(theme::MUTED)
        } else {
            Style::default().fg(theme::TEXT)
        };
        lines.push(Line::from(Span::styled(format!("{indent}{line}"), style)));
    }
}

const MULTILINE_MAX_LINES: usize = 6;

/// Keys to hide in flat object display (too verbose / unreadable in TUI).
const HIDDEN_KEYS: &[&str] = &["gpu_topo", "cpu_info"];

fn format_flat_object(value: &serde_json::Value) -> Vec<Line<'static>> {
    let mut lines = Vec::new();
    if let Some(obj) = value.as_object() {
        for (k, v) in obj {
            if HIDDEN_KEYS.contains(&k.as_str()) {
                continue;
            }
            let val_str = format_value(v);
            let val_lines: Vec<&str> = val_str.lines().collect();

            if val_lines.len() <= 1 {
                let display = if val_str.len() > 120 {
                    format!("{}...", &val_str[..120])
                } else {
                    val_str
                };
                let style = if display == "true" || display == "True" {
                    Style::default().fg(theme::HEALTHY)
                } else if display == "false" || display == "False" {
                    Style::default().fg(theme::MUTED)
                } else {
                    Style::default().fg(theme::TEXT)
                };
                lines.push(Line::from(vec![
                    Span::styled(format!("  {k}: "), Style::default().fg(theme::ACCENT)),
                    Span::styled(display, style),
                ]));
            } else {
                let show = val_lines.len().min(MULTILINE_MAX_LINES);
                let truncated = val_lines.len() > MULTILINE_MAX_LINES;
                lines.push(Line::from(Span::styled(
                    format!("  {k}: ({} lines)", val_lines.len()),
                    Style::default().fg(theme::ACCENT),
                )));
                for vl in &val_lines[..show] {
                    let trimmed = vl.trim();
                    if trimmed.is_empty() {
                        continue;
                    }
                    let display = if trimmed.len() > 100 {
                        format!("    {}...", &trimmed[..100])
                    } else {
                        format!("    {trimmed}")
                    };
                    lines.push(Line::from(Span::styled(
                        display,
                        Style::default().fg(theme::TEXT),
                    )));
                }
                if truncated {
                    lines.push(Line::from(Span::styled(
                        format!("    ... ({} more lines)", val_lines.len() - show),
                        Style::default().fg(theme::MUTED),
                    )));
                }
            }
        }
    }
    lines
}

fn format_value(v: &serde_json::Value) -> String {
    match v {
        serde_json::Value::String(s) => s.clone(),
        serde_json::Value::Null => "null".to_string(),
        serde_json::Value::Bool(b) => b.to_string(),
        serde_json::Value::Number(n) => n.to_string(),
        serde_json::Value::Array(arr) if arr.is_empty() => "[]".to_string(),
        serde_json::Value::Object(obj) if obj.is_empty() => "{}".to_string(),
        other => {
            let s = serde_json::to_string(other).unwrap_or_default();
            if s.len() > 120 {
                format!("{}...", &s[..120])
            } else {
                s
            }
        }
    }
}

// ── Utilities ──

/// Longest common host prefix across `addrs`, cut at the last `-`/`_` boundary
/// so a token (e.g. `node09`) is never split. Returns "" when there are <2 nodes,
/// hosts share no separator-aligned prefix, or any host fails to start with it.
fn common_node_prefix(addrs: &[String]) -> String {
    if addrs.len() < 2 {
        return String::new();
    }
    let hosts: Vec<&str> = addrs.iter().map(|a| a.split(':').next().unwrap_or(a)).collect();
    let first = hosts[0];
    let mut end = first.len();
    for h in &hosts[1..] {
        let common = first
            .as_bytes()
            .iter()
            .zip(h.as_bytes().iter())
            .take_while(|(a, b)| a == b)
            .count();
        end = end.min(common);
    }
    if end == 0 {
        return String::new();
    }
    let cut = first[..end].rfind(['-', '_']).map(|i| i + 1).unwrap_or(0);
    first[..cut].to_string()
}

/// Strip the auto-detected common prefix from `addr`, keeping the port.
/// Falls back to the full address when the prefix doesn't apply.
fn short_node_label(addr: &str, prefix: &str) -> String {
    if !prefix.is_empty() && addr.starts_with(prefix) {
        addr[prefix.len()..].to_string()
    } else {
        addr.to_string()
    }
}

/// Single-character status glyph for a node row, color carries the state.
fn status_indicator(n: &NodeMetrics) -> Span<'static> {
    let (glyph, color) = if n.is_healthy {
        ("\u{25CF}", theme::HEALTHY) // ● up
    } else if n.is_loading {
        ("\u{25CB}", theme::MUTED) // ○ first connection
    } else {
        // "starting": GPU memory has been allocated but /metrics isn't responding.
        let gpu_mem_loaded = n
            .gpu_scrape
            .as_ref()
            .map(|g| {
                g.gpus.iter().any(|gpu| {
                    gpu.mem_total_bytes > 0
                        && (gpu.mem_used_bytes as f64 / gpu.mem_total_bytes as f64) > 0.2
                })
            })
            .unwrap_or(false);
        if gpu_mem_loaded {
            ("\u{25D0}", theme::WARNING) // ◐ starting
        } else {
            ("\u{25CF}", theme::DANGER) // ● down
        }
    };
    Span::styled(glyph, Style::default().fg(color))
}

/// Trailing 4 chars of a SLURM job_id, used as a compact JOB-column tag
/// when targets span multiple jobs.
fn job_tag(job_id: &str) -> String {
    if job_id.chars().count() > 4 {
        // Avoid slicing inside a multi-byte char boundary (job_ids are ASCII
        // in practice, but stay defensive).
        let start = job_id.len().saturating_sub(4);
        job_id[start..].to_string()
    } else {
        job_id.to_string()
    }
}

fn job_cell(job_id: &str, ended: bool) -> Cell<'static> {
    if job_id.is_empty() {
        return Cell::from(Span::styled("-", Style::default().fg(theme::MUTED)));
    }
    let tag = job_tag(job_id);
    let style = if ended {
        Style::default().fg(theme::MUTED).add_modifier(Modifier::CROSSED_OUT)
    } else {
        Style::default().fg(theme::job_color(job_id))
    };
    Cell::from(Span::styled(tag, style))
}

/// Render a NIXL throughput cell (MB/s, auto-scaling to GB/s above 1024 MB/s).
/// Render the KV-transfer column. Prefers NIXL bandwidth (MB/s) when vLLM
/// publishes it; falls back to `external_kv_transfer` token rate when the
/// NIXL histogram is empty (e.g. some Dynamo builds expose the histogram
/// declaration but never observe it, even with active disagg traffic).
fn nixl_throughput_cell(has_nixl: bool, mb_per_sec: f64, kv_tok_per_sec: f64) -> Cell<'static> {
    if !has_nixl {
        return Cell::from(Span::styled("-", Style::default().fg(theme::MUTED)));
    }
    if mb_per_sec > 0.0 {
        let (text, color) = if mb_per_sec >= 1024.0 {
            (format!("{:.1}G/s", mb_per_sec / 1024.0), theme::HEALTHY)
        } else if mb_per_sec >= 100.0 {
            (format!("{:.0}M/s", mb_per_sec), theme::HEALTHY)
        } else {
            (format!("{:.1}M/s", mb_per_sec), theme::TEXT)
        };
        return Cell::from(Span::styled(text, Style::default().fg(color)));
    }
    if kv_tok_per_sec > 0.0 {
        let (text, color) = if kv_tok_per_sec >= 1.0e6 {
            (format!("{:.1}Mt/s", kv_tok_per_sec / 1.0e6), theme::HEALTHY)
        } else if kv_tok_per_sec >= 1.0e3 {
            (format!("{:.1}Kt/s", kv_tok_per_sec / 1.0e3), theme::HEALTHY)
        } else {
            (format!("{:.0}t/s", kv_tok_per_sec), theme::TEXT)
        };
        return Cell::from(Span::styled(text, Style::default().fg(color)));
    }
    Cell::from(Span::styled("0", Style::default().fg(theme::MUTED)))
}

/// True when a Dynamo role string includes the prefill worker role,
/// e.g. "P", "P+D", and raw "prefill" all match.
fn role_is_prefill(role: Option<&str>) -> bool {
    role.is_some_and(|r| r.split('+').any(|p| p == "P" || p == "prefill"))
}

/// True when a Dynamo role string includes the decode worker role.
fn role_is_decode(role: Option<&str>) -> bool {
    role.is_some_and(|r| r.split('+').any(|p| p == "D" || p == "decode"))
}

/// Render a Dynamo P/D role badge styled span. Returns None when role is absent.
fn role_badge(role: Option<&str>) -> Option<Span<'static>> {
    let role = role?;
    // Normalize raw labels to short badges; pre-aggregated values (P, D, P+D) pass through.
    // "backend" is normally resolved upstream (ClusterState::resolve_backend_roles)
    // into "D" (PD decode worker) or None (aggregated worker). If a raw value
    // still leaks here it's ambiguous, so err on no badge.
    let badge = match role {
        "prefill" => "P",
        "decode" => "D",
        "backend" => return None,
        other => other,
    };
    let color = if badge.contains("+") {
        theme::WARNING
    } else if badge == "P" {
        theme::ACCENT
    } else if badge == "D" {
        theme::HEALTHY
    } else {
        theme::TEXT
    };
    Some(Span::styled(
        format!("[{badge}]"),
        Style::default().fg(color),
    ))
}

fn short_model(name: &str) -> String {
    use unicode_width::UnicodeWidthChar;
    let short = name.rsplit('/').next().unwrap_or(name);
    // Truncate to 12 display cells. If clipped, leave room for an ellipsis (1 cell)
    // so the visible result is still ≤ 12 cells and the next column can't drift.
    let mut budget = 12usize;
    let mut buf = String::new();
    let mut clipped = false;
    let chars: Vec<char> = short.chars().collect();
    for (i, c) in chars.iter().enumerate() {
        let w = c.width().unwrap_or(0);
        let remaining_chars = chars.len() - i - 1;
        // Reserve 1 cell for ellipsis if more chars follow.
        let needed_now = if remaining_chars > 0 { w + 1 } else { w };
        if needed_now > budget {
            clipped = true;
            break;
        }
        buf.push(*c);
        budget -= w;
    }
    if clipped {
        buf.push('…');
    }
    buf
}

/// Format NVLink bandwidth (input in GB/s) with auto-scaling: KB/s, MB/s, GB/s.
fn format_nvlink_bw(gbps: f64) -> String {
    if gbps < 0.001 {
        "-".to_string()
    } else if gbps < 1.0 {
        let mbps = gbps * 1024.0;
        if mbps < 1.0 {
            format!("{:.0}K", mbps * 1024.0)
        } else if mbps < 100.0 {
            format!("{:.1}M", mbps)
        } else {
            format!("{:.0}M", mbps)
        }
    } else if gbps < 10.0 {
        format!("{:.1}G", gbps)
    } else {
        format!("{:.0}G", gbps)
    }
}

/// Format PCIe bandwidth (input in GB/s) — same scaling as NVLink but with a
/// lower dash threshold (1 KB/s) since PCIe at idle still moves real traffic
/// worth surfacing.
fn format_pcie_bw(gbps: f64) -> String {
    if gbps < 0.000_001 || !gbps.is_finite() {
        "-".to_string()
    } else if gbps < 1.0 {
        let mbps = gbps * 1024.0;
        if mbps < 1.0 {
            format!("{:.0}K", mbps * 1024.0)
        } else if mbps < 100.0 {
            format!("{:.1}M", mbps)
        } else {
            format!("{:.0}M", mbps)
        }
    } else if gbps < 10.0 {
        format!("{:.1}G", gbps)
    } else {
        format!("{:.0}G", gbps)
    }
}

/// Format NVLink bandwidth (input in GB/s) with unit suffix for detail panel.
fn format_nvlink_bw_long(gbps: f64) -> String {
    if gbps < 0.001 {
        "-".to_string()
    } else if gbps < 1.0 {
        let mbps = gbps * 1024.0;
        if mbps < 1.0 {
            format!("{:.0} KB/s", mbps * 1024.0)
        } else {
            format!("{:.1} MB/s", mbps)
        }
    } else {
        format!("{:.1} GB/s", gbps)
    }
}

fn format_count(n: i64) -> String {
    let abs = n.unsigned_abs();
    if abs >= 1_000_000_000_000 {
        format!("{:.1}T", n as f64 / 1_000_000_000_000.0)
    } else if abs >= 1_000_000_000 {
        format!("{:.1}G", n as f64 / 1_000_000_000.0)
    } else if abs >= 1_000_000 {
        format!("{:.1}M", n as f64 / 1_000_000.0)
    } else if abs >= 1_000 {
        format!("{:.1}K", n as f64 / 1_000.0)
    } else {
        n.to_string()
    }
}

fn format_ms(ms: f64) -> String {
    if ms == 0.0 {
        "-".to_string()
    } else if ms < 1.0 {
        format!("{:.1}ms", ms)
    } else if ms < 1000.0 {
        format!("{:.0}ms", ms)
    } else {
        format!("{:.1}s", ms / 1000.0)
    }
}

/// Format a duration in seconds as "Xd Yh", "Xh Ym", "Xm Ys", or "Xs".
fn format_uptime_secs(secs: f64) -> String {
    if secs <= 0.0 {
        return "-".to_string();
    }
    let total = secs as u64;
    let days = total / 86400;
    let hours = (total % 86400) / 3600;
    let mins = (total % 3600) / 60;
    let s = total % 60;
    if days > 0 {
        format!("{days}d {hours}h")
    } else if hours > 0 {
        format!("{hours}h {mins}m")
    } else if mins > 0 {
        format!("{mins}m {s}s")
    } else {
        format!("{s}s")
    }
}

fn format_flops(flops: f64) -> String {
    if flops >= 1e15 {
        format!("{:.1} PFLOP/s", flops / 1e15)
    } else if flops >= 1e12 {
        format!("{:.1} TFLOP/s", flops / 1e12)
    } else if flops >= 1e9 {
        format!("{:.1} GFLOP/s", flops / 1e9)
    } else {
        format!("{:.0} FLOP/s", flops)
    }
}

fn format_bytes_per_sec(bps: f64) -> String {
    if bps >= 1e12 {
        format!("{:.1} TB/s", bps / 1e12)
    } else if bps >= 1e9 {
        format!("{:.1} GB/s", bps / 1e9)
    } else if bps >= 1e6 {
        format!("{:.1} MB/s", bps / 1e6)
    } else {
        format!("{:.0} B/s", bps)
    }
}

fn format_bytes(bytes: u64) -> String {
    let v = bytes as f64;
    if v >= 1e12 {
        format!("{:.2} TB", v / 1e12)
    } else if v >= 1e9 {
        format!("{:.2} GB", v / 1e9)
    } else if v >= 1e6 {
        format!("{:.1} MB", v / 1e6)
    } else if v >= 1e3 {
        format!("{:.0} KB", v / 1e3)
    } else {
        format!("{bytes} B")
    }
}

fn ascii_bar(frac: f64, width: usize) -> String {
    let clamped = frac.clamp(0.0, 1.0);
    let filled = (clamped * width as f64).round() as usize;
    let filled = filled.min(width);
    let empty = width - filled;
    format!("{}{}", "█".repeat(filled), "░".repeat(empty))
}

/// Truncate to at most `max` characters, appending `…` when cut. Char-based
/// (not byte-based) so multi-byte names never split mid-codepoint. Used for
/// SLURM job names in the header — sweep launchers encode the whole config
/// into the name, easily hundreds of characters.
fn truncate_ellipsis(s: &str, max: usize) -> String {
    if s.chars().count() <= max {
        return s.to_string();
    }
    let cut: String = s.chars().take(max.saturating_sub(1)).collect();
    format!("{cut}…")
}

#[cfg(test)]
mod tab_tests {
    use super::*;

    #[test]
    fn tab_index_round_trips_and_wraps() {
        for i in 0..Tab::COUNT {
            assert_eq!(Tab::from_index(i).index(), i);
        }
        assert_eq!(Tab::from_index(Tab::COUNT), Tab::Overview);
        let mut ui = UiState::new();
        ui.mooncake_tab = true;
        assert_eq!(ui.active_tab, Tab::Overview);
        ui.prev_tab();
        assert_eq!(ui.active_tab, Tab::Info); // backwards from first lands on last
        ui.next_tab();
        assert_eq!(ui.active_tab, Tab::Overview);
    }

    #[test]
    fn tab_click_hit_testing_matches_tabs_layout() {
        // Rendered bar: " Overview │ KV / Xfer │ Hardware │ Info" — each tab
        // spans title+4 columns (inner+outer padding), dividers 1 column.
        let mut ui = UiState::new();
        assert_eq!(ui.tab_at(0), Some(Tab::Overview));
        assert_eq!(ui.tab_at(11), Some(Tab::Overview)); // "Overview" = 8+4
        assert_eq!(ui.tab_at(12), None); // divider
        assert_eq!(ui.tab_at(13), Some(Tab::KvXfer));
        // Mooncake hidden: the slot after KV / Xfer (9+4 wide) is Hardware.
        assert_eq!(ui.tab_at(13 + 13 + 1), Some(Tab::Hardware));
        // Mooncake offered: same column now lands in the Mooncake span.
        ui.mooncake_tab = true;
        assert_eq!(ui.tab_at(13 + 13 + 1), Some(Tab::Mooncake));
        // Past the last tab: no hit.
        assert_eq!(ui.tab_at(200), None);
    }

    #[test]
    fn tab_cycling_skips_mooncake_unless_offered() {
        let mut ui = UiState::new();
        // Hidden (no --mooncake targets): KvXfer steps straight to Hardware
        // in both directions.
        ui.active_tab = Tab::KvXfer;
        ui.next_tab();
        assert_eq!(ui.active_tab, Tab::Hardware);
        ui.prev_tab();
        assert_eq!(ui.active_tab, Tab::KvXfer);
        // Offered: the Mooncake tab is part of the cycle.
        ui.mooncake_tab = true;
        ui.next_tab();
        assert_eq!(ui.active_tab, Tab::Mooncake);
        ui.next_tab();
        assert_eq!(ui.active_tab, Tab::Hardware);
        ui.prev_tab();
        assert_eq!(ui.active_tab, Tab::Mooncake);
    }

    #[test]
    fn every_section_routes_to_a_metric_tab() {
        for s in DetailSection::ALL {
            assert_ne!(section_tab(s), Tab::Info, "{s:?} must not route to Info");
        }
    }

    #[test]
    fn moved_sections_land_on_their_new_tabs() {
        use DetailSection::*;
        assert_eq!(section_tab(Latency), Tab::Overview);
        assert_eq!(section_tab(DynamoConfig), Tab::Overview);
        assert_eq!(section_tab(CacheStats), Tab::KvXfer);
        assert_eq!(section_tab(RemoteKvFetch), Tab::KvXfer);
        assert_eq!(section_tab(Mooncake), Tab::Mooncake);
        assert_eq!(section_tab(KvEvents), Tab::KvXfer);
        assert_eq!(section_tab(PerfMfu), Tab::Hardware);
        assert_eq!(section_tab(GpuHardware), Tab::Hardware);
        assert_eq!(section_tab(RdmaIb), Tab::Hardware);
    }

    #[test]
    fn tab_switch_resets_detail_scroll() {
        let mut ui = UiState::new();
        ui.detail_scroll = 42;
        ui.next_tab();
        assert_eq!(ui.detail_scroll, 0);
        ui.detail_scroll = 7;
        ui.prev_tab();
        assert_eq!(ui.detail_scroll, 0);
    }
}

#[cfg(test)]
mod fold_state_tests {
    use super::*;

    fn node(addr: &str) -> NodeMetrics {
        NodeMetrics::offline(addr.to_string())
    }

    /// Fold state must be keyed by addr, not index: when a new job's nodes
    /// join in follow mode the list reorders, and index-keyed state expanded
    /// old jobs' rows at random.
    #[test]
    fn expansion_follows_node_addr_when_list_reorders() {
        let mut ui = UiState::new();
        let before = vec![node("node09"), node("node10")];
        ui.selected = 1;
        ui.toggle_engines(&before); // expand node10
        // A new job's node joins ahead of the old ones: node10 stays expanded,
        // node09 stays collapsed, and the newcomer defaults to collapsed.
        let after = [node("node08"), node("node09"), node("node10")];
        let expanded: Vec<&str> = after
            .iter()
            .filter(|n| ui.expanded.contains(&n.addr))
            .map(|n| n.addr.as_str())
            .collect();
        assert_eq!(expanded, vec!["node10"]);
    }

    #[test]
    fn cluster_row_toggles_all_nodes() {
        let mut ui = UiState::new();
        let nodes = vec![node("a"), node("b")];
        ui.selected = usize::MAX;
        ui.toggle_engines(&nodes); // none expanded → expand all
        assert_eq!(ui.expanded.len(), 2);
        ui.toggle_engines(&nodes); // any expanded → collapse all
        assert!(ui.expanded.is_empty());
    }
}

#[cfg(test)]
mod role_split_tests {
    use super::*;

    #[test]
    fn worker_roles_match_short_long_and_combined_forms() {
        for r in ["P", "prefill", "P+D", "prefill+decode", "frontend+P"] {
            assert!(role_is_prefill(Some(r)), "{r} should be prefill");
        }
        for r in ["D", "decode", "P+D", "prefill+decode"] {
            assert!(role_is_decode(Some(r)), "{r} should be decode");
        }
    }

    #[test]
    fn non_worker_roles_match_neither() {
        for r in ["frontend", "router", "backend"] {
            assert!(!role_is_prefill(Some(r)));
            assert!(!role_is_decode(Some(r)));
        }
        assert!(!role_is_prefill(None));
        assert!(!role_is_decode(None));
    }
}

#[cfg(test)]
mod graph_series_tests {
    use super::*;

    fn series(v: &[&[(f64, f64)]]) -> Vec<(String, Vec<(f64, f64)>)> {
        v.iter().enumerate().map(|(i, pts)| (format!("n{i}"), pts.to_vec())).collect()
    }

    #[test]
    fn mean_of_equal_length_series_averages_each_slot() {
        let s = series(&[&[(-2.0, 10.0), (-1.0, 20.0)], &[(-2.0, 30.0), (-1.0, 40.0)]]);
        assert_eq!(mean_series(&s), vec![(-2.0, 20.0), (-1.0, 30.0)]);
    }

    #[test]
    fn mean_aligns_shorter_series_at_newest_end() {
        // Second node joined one tick later: its single sample only
        // contributes to the newest slot.
        let s = series(&[(&[(-2.0, 10.0), (-1.0, 20.0)]) as &[_], &[(-1.0, 40.0)]]);
        assert_eq!(mean_series(&s), vec![(-2.0, 10.0), (-1.0, 30.0)]);
    }

    #[test]
    fn mean_of_empty_input_is_empty() {
        assert!(mean_series(&[]).is_empty());
        assert!(mean_series(&series(&[&[]])).is_empty());
    }
}

#[cfg(test)]
mod node_label_tests {
    use super::*;

    fn s(v: &[&str]) -> Vec<String> {
        v.iter().map(|x| x.to_string()).collect()
    }

    #[test]
    fn common_prefix_cuts_at_dash_boundary() {
        let p = common_node_prefix(&s(&["example-node09:8080", "example-node10:8080"]));
        assert_eq!(p, "example-");
    }

    #[test]
    fn common_prefix_empty_for_single_host() {
        let p = common_node_prefix(&s(&["example-node09:8080"]));
        assert_eq!(p, "");
    }

    #[test]
    fn common_prefix_empty_when_no_separator_in_common_part() {
        // Hosts share "host" but there's no `-`/`_` inside that prefix, so we
        // refuse to commit (could chop a real token).
        let p = common_node_prefix(&s(&["host01", "host02"]));
        assert_eq!(p, "");
    }

    #[test]
    fn common_prefix_empty_for_disjoint_hosts() {
        let p = common_node_prefix(&s(&["alpha-1", "beta-1"]));
        assert_eq!(p, "");
    }

    #[test]
    fn short_label_strips_prefix_and_keeps_port() {
        assert_eq!(
            short_node_label("example-node09:8080", "example-"),
            "node09:8080"
        );
    }

    #[test]
    fn short_label_falls_back_when_prefix_missing() {
        assert_eq!(
            short_node_label("other-host:80", "example-"),
            "other-host:80"
        );
        assert_eq!(
            short_node_label("example-node09:8080", ""),
            "example-node09:8080"
        );
    }

    #[test]
    fn truncate_ellipsis_cuts_long_names_and_keeps_short_ones() {
        assert_eq!(truncate_ellipsis("short", 32), "short");
        // Exactly at the limit: untouched.
        let exact: String = "x".repeat(32);
        assert_eq!(truncate_ellipsis(&exact, 32), exact);
        // Over the limit: 31 chars + ellipsis = 32 total.
        let long: String = "y".repeat(40);
        let out = truncate_ellipsis(&long, 32);
        assert_eq!(out.chars().count(), 32);
        assert!(out.ends_with('…'));
        // Multi-byte safe.
        assert_eq!(truncate_ellipsis("测试一二三", 3), "测试…");
    }
}

#[cfg(test)]
mod format_count_tests {
    use super::format_count;

    #[test]
    fn raw_under_a_thousand() {
        assert_eq!(format_count(0), "0");
        assert_eq!(format_count(42), "42");
        assert_eq!(format_count(999), "999");
    }

    #[test]
    fn scales_to_k_m_g_t() {
        assert_eq!(format_count(1_500), "1.5K");
        assert_eq!(format_count(2_400_000), "2.4M");
        assert_eq!(format_count(3_500_000_000), "3.5G");
        assert_eq!(format_count(1_200_000_000_000), "1.2T");
    }

    #[test]
    fn negative_values_scale_too() {
        assert_eq!(format_count(-1_500), "-1.5K");
        assert_eq!(format_count(-2_000_000_000), "-2.0G");
    }
}
