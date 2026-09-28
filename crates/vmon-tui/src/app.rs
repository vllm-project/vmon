// SPDX-License-Identifier: Apache-2.0

use std::io;
use std::sync::Arc;
use std::time::Duration;

use crossterm::event::{self, DisableMouseCapture, EnableMouseCapture, Event};
use crossterm::execute;
use crossterm::terminal::{
    EnterAlternateScreen, LeaveAlternateScreen, disable_raw_mode, enable_raw_mode,
};
use ratatui::Terminal;
use ratatui::backend::CrosstermBackend;
use tokio::sync::watch;

/// RAII guard that restores the terminal (disable raw mode, leave alt-screen,
/// show cursor) when dropped. Ensures cleanup runs whether the event loop
/// exits normally, returns an error via `?`, or unwinds from a panic.
struct TerminalGuard;

impl TerminalGuard {
    fn new() -> io::Result<Self> {
        enable_raw_mode()?;
        let mut stdout = io::stdout();
        // Mouse capture enables click-to-select / click-tabs / wheel-scroll.
        // Terminal-native text selection still works with Shift held.
        if let Err(e) = execute!(stdout, EnterAlternateScreen, EnableMouseCapture) {
            // Roll back raw mode so the terminal isn't left half-initialized
            // when we return Err (Drop never runs because we don't return Self).
            let _ = disable_raw_mode();
            return Err(e);
        }
        Ok(Self)
    }
}

impl Drop for TerminalGuard {
    fn drop(&mut self) {
        let _ = disable_raw_mode();
        let _ = execute!(io::stdout(), DisableMouseCapture, LeaveAlternateScreen);
        let _ = execute!(io::stdout(), crossterm::cursor::Show);
    }
}

/// Install a panic hook that restores the terminal before delegating to the
/// previous hook (so the backtrace still reaches stderr legibly instead of
/// being eaten by raw mode + alt-screen). Idempotent across multiple
/// `run`/`run_replay` invocations.
fn install_panic_hook() {
    use std::sync::Once;
    static INSTALL: Once = Once::new();
    INSTALL.call_once(|| {
        let prev = std::panic::take_hook();
        std::panic::set_hook(Box::new(move |info| {
            let _ = disable_raw_mode();
            let _ = execute!(io::stdout(), DisableMouseCapture, LeaveAlternateScreen);
            let _ = execute!(io::stdout(), crossterm::cursor::Show);
            prev(info);
        }));
    });
}

use vmon_core::actions::NodeActions;
use vmon_core::cluster::ClusterState;

use crate::input::{self, Action};
use crate::ui::{self, CollectSession, PendingAction, ReplayState, UiState};

pub async fn run(rx: watch::Receiver<ClusterState>, scrape_interval: Duration) -> io::Result<()> {
    run_inner(rx, None, Some(scrape_interval)).await
}

/// Run TUI in replay mode with playback state.
pub async fn run_replay(
    rx: watch::Receiver<ClusterState>,
    replay: Arc<ReplayState>,
) -> io::Result<()> {
    run_inner(rx, Some(replay), None).await
}

async fn run_inner(
    mut rx: watch::Receiver<ClusterState>,
    replay: Option<Arc<ReplayState>>,
    scrape_interval: Option<Duration>,
) -> io::Result<()> {
    install_panic_hook();
    // Guard restores the terminal on drop, regardless of how this function
    // exits (normal return, error via `?`, or panic unwind).
    let _guard = TerminalGuard::new()?;

    let backend = CrosstermBackend::new(io::stdout());
    let mut terminal = Terminal::new(backend)?;

    let mut ui_state = UiState::new();
    ui_state.replay = replay.clone();
    ui_state.scrape_interval = scrape_interval;
    let actions = NodeActions::new();
    let replay_mode = replay.is_some();

    // Dedicated input thread → channel. A single long-lived reader forwards
    // crossterm events over an mpsc channel that the select loop drains.
    // `recv()` is cancel-safe, so a keypress is never lost when the other
    // select branch (cluster update) wins the race — which is exactly what
    // happened with a per-iteration `spawn_blocking(event::poll)`: at high
    // replay speed the poll arm was dropped while still running, and the
    // orphaned task would `read()` and silently discard the keypress.
    let (key_tx, mut key_rx) = tokio::sync::mpsc::unbounded_channel::<Event>();
    std::thread::spawn(move || {
        loop {
            match event::poll(Duration::from_millis(200)) {
                Ok(true) => match event::read() {
                    Ok(ev) => {
                        if key_tx.send(ev).is_err() {
                            break; // receiver dropped — loop has exited
                        }
                    }
                    Err(_) => break,
                },
                Ok(false) => {
                    // No event this interval; bail out if the receiver is gone.
                    if key_tx.is_closed() {
                        break;
                    }
                }
                Err(_) => break,
            }
        }
    });

    // Cap the redraw rate (~30 fps). At high replay speed the watch channel
    // emits an update every millisecond or so; drawing the full TUI on each one
    // floods the terminal faster than it can render, so a keypress appears
    // laggy even though it was handled immediately. We still record every
    // sample (below) for history/collect fidelity — only the draw is throttled.
    let frame_budget = Duration::from_millis(33);
    // Force an immediate first draw.
    let mut last_draw = tokio::time::Instant::now()
        .checked_sub(frame_budget)
        .unwrap_or_else(tokio::time::Instant::now);

    loop {
        let now = tokio::time::Instant::now();
        if now.duration_since(last_draw) >= frame_budget {
            // Clone so the watch read-lock is released before the (relatively
            // slow) render, otherwise we'd block the replay sender each frame.
            let cluster = rx.borrow().clone();
            terminal.draw(|f| {
                ui::draw(f, &cluster, &mut ui_state);
            })?;
            last_draw = now;
        }
        // Time until the next allowed frame; the select sleeps at most this long
        // so a pending change/keypress still paints promptly (within one frame).
        let until_next_frame = frame_budget.saturating_sub(now.duration_since(last_draw));

        // Wait for keyboard input, a cluster state change, or the frame timer
        tokio::select! {
            _ = rx.changed() => {
                let cluster = rx.borrow();
                ui_state.record_cluster(&cluster);
                // Append to active collect session (skipped while paused).
                if let Some(session) = ui_state.collect.as_mut() {
                    if !session.paused {
                        session.collector.record(&cluster);
                    }
                }
            }
            _ = tokio::time::sleep(until_next_frame) => {
                // Frame timer elapsed — loop back and redraw with latest state.
            }
            event = key_rx.recv() => {
                // Mouse first, by reference (MouseEvent is Copy) so the
                // by-value key match below still owns `event`.
                if let Some(Event::Mouse(m)) = &event {
                    let cluster = rx.borrow().clone();
                    ui_state.handle_mouse(*m, &cluster.nodes);
                }
                if let Some(Event::Key(key)) = event {
                    let cluster = rx.borrow().clone();
                    let confirm_pending = ui_state.pending_action.is_some();
                    match input::handle_key(key, ui_state.search_active, confirm_pending, replay_mode) {
                        Action::Quit => break,
                        Action::Up => ui_state.select_prev(),
                        Action::Down => ui_state.select_next(),
                        Action::ToggleDetail => {
                            if !ui_state.show_graph {
                                ui_state.toggle_detail();
                            }
                        }
                        Action::ToggleGraph => ui_state.toggle_graph(),
                        Action::ToggleEngines => {
                            ui_state.toggle_engines(&cluster.nodes);
                        }
                        Action::TabNext => ui_state.next_tab(),
                        Action::TabPrev => ui_state.prev_tab(),
                        Action::ScrollDetailDown => ui_state.scroll_detail_down(),
                        Action::ScrollDetailUp => ui_state.scroll_detail_up(),
                        Action::EnterSearch => ui_state.enter_search(),
                        Action::ExitSearch => {
                            if ui_state.diff_picking {
                                ui_state.diff_picking = false;
                            } else if ui_state.diff_target.is_some() {
                                ui_state.diff_target = None;
                            } else {
                                ui_state.exit_search();
                            }
                        }
                        Action::SearchInput(c) => ui_state.search_push(c),
                        Action::SearchBackspace => ui_state.search_pop(),
                        Action::ToggleSleep => {
                            if let Some(addr) = selected_node_addr(&cluster, &ui_state) {
                                let node = cluster.nodes.iter().find(|n| n.addr == addr);
                                let is_sleeping = node.and_then(|n| n.is_sleeping).unwrap_or(false);
                                ui_state.pending_action = Some(if is_sleeping {
                                    PendingAction::WakeUp(addr)
                                } else {
                                    PendingAction::Sleep(addr)
                                });
                            }
                        }
                        Action::ResetPrefixCache => {
                            if ui_state.selected == usize::MAX {
                                let addrs: Vec<String> = cluster.nodes.iter()
                                    .filter(|n| n.is_healthy)
                                    .map(|n| n.addr.clone())
                                    .collect();
                                if !addrs.is_empty() {
                                    ui_state.pending_action = Some(PendingAction::ResetCacheAll(addrs));
                                }
                            } else if let Some(addr) = selected_node_addr(&cluster, &ui_state) {
                                ui_state.pending_action = Some(PendingAction::ResetCache(addr));
                            }
                        }
                        Action::Confirm => {
                            if let Some(pending) = ui_state.pending_action.take() {
                                let msg = execute_action(&actions, &pending).await;
                                ui_state.status_message = Some((msg, tokio::time::Instant::now()));
                            }
                        }
                        Action::Cancel => {
                            ui_state.pending_action = None;
                        }
                        Action::DiffMode => {
                            let is_info_tab =
                                matches!(ui_state.active_tab, crate::ui::Tab::Info);
                            if ui_state.diff_picking {
                                // Confirm selection: use currently selected node as diff target
                                let target = ui_state.selected;
                                ui_state.diff_picking = false;
                                ui_state.diff_target = Some(target);
                            } else if ui_state.diff_target.is_some() {
                                // Exit diff mode
                                ui_state.diff_target = None;
                            } else if is_info_tab && cluster.nodes.len() >= 2 {
                                // Enter diff picking mode
                                ui_state.diff_picking = true;
                            }
                        }
                        Action::ReplayPause => {
                            if let Some(ref r) = replay { r.toggle_pause(); }
                        }
                        Action::ReplayFaster => {
                            if let Some(ref r) = replay { r.speed_up(); }
                        }
                        Action::ReplaySlower => {
                            if let Some(ref r) = replay { r.speed_down(); }
                        }
                        Action::ReplayStepForward => {
                            if let Some(ref r) = replay {
                                if !r.is_paused() { r.toggle_pause(); }
                                // Signal step via a special sample increment
                                r.current_sample.fetch_add(0, std::sync::atomic::Ordering::Relaxed);
                                // Step is handled by the replay task checking a step flag
                            }
                        }
                        Action::ReplayStepBackward => {
                            // Step backward not supported in v1
                        }
                        Action::CollectToggle => {
                            let msg = match ui_state.collect.as_mut() {
                                None => {
                                    let cluster = rx.borrow().clone();
                                    let addrs: Vec<String> =
                                        cluster.nodes.iter().map(|n| n.addr.clone()).collect();
                                    let path = build_collect_path(&cluster);
                                    let collector =
                                        vmon_report::collector::TimeSeriesCollector::new(addrs);
                                    let mut session = CollectSession {
                                        path: path.clone(),
                                        collector,
                                        paused: false,
                                        pause_started_at: None,
                                    };
                                    // Record the current cluster immediately so the file is non-empty
                                    // even if the user stops before the next scrape tick.
                                    session.collector.record(&cluster);
                                    ui_state.collect = Some(session);
                                    format!("Recording → {}", path.display())
                                }
                                Some(session) => {
                                    if session.paused {
                                        session.resume();
                                        "Recording resumed".to_string()
                                    } else {
                                        session.pause();
                                        "Recording paused".to_string()
                                    }
                                }
                            };
                            ui_state.status_message = Some((msg, tokio::time::Instant::now()));
                        }
                        Action::CollectStop => {
                            if let Some(session) = ui_state.collect.take() {
                                let msg = finish_collect_session(session);
                                ui_state.status_message = Some((msg, tokio::time::Instant::now()));
                            }
                        }
                        Action::None => {}
                    }
                }
            }
        }
    }

    // Terminal restore happens via TerminalGuard's Drop impl.
    Ok(())
}

fn selected_node_addr(cluster: &ClusterState, ui: &UiState) -> Option<String> {
    if ui.selected == usize::MAX {
        return None; // cluster row
    }
    cluster.nodes.get(ui.selected).map(|n| n.addr.clone())
}

/// Build the output path for a new collect session. Lands in the current
/// working directory; filename includes the SLURM job ID (when exactly one
/// matches) and a UTC timestamp.
fn build_collect_path(cluster: &ClusterState) -> std::path::PathBuf {
    let ts = format_timestamp_utc();
    let stem = match cluster.slurm_jobs.as_slice() {
        [job] => format!("vmon-{}-{ts}", job.job_id),
        _ => format!("vmon-{ts}"),
    };
    std::path::PathBuf::from(format!("{stem}.json"))
}

/// `YYYYMMDD-HHMMSSZ` from the current SystemTime, computed in UTC without an
/// external date crate. See Howard Hinnant's "civil_from_days" algorithm.
fn format_timestamp_utc() -> String {
    let secs = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0);
    let days = (secs / 86_400) as i64;
    let sod = (secs % 86_400) as u32;
    let h = sod / 3600;
    let m = (sod % 3600) / 60;
    let s = sod % 60;
    let (y, mo, d) = civil_from_days(days);
    format!("{y:04}{mo:02}{d:02}-{h:02}{m:02}{s:02}Z")
}

/// Convert days-since-Unix-epoch to (year, month, day) in the proleptic
/// Gregorian calendar. Algorithm from H. Hinnant; valid for all i64 days.
fn civil_from_days(z: i64) -> (i32, u32, u32) {
    let z = z + 719_468;
    let era = if z >= 0 { z } else { z - 146_096 } / 146_097;
    let doe = (z - era * 146_097) as u64;
    let yoe = (doe - doe / 1460 + doe / 36_524 - doe / 146_096) / 365;
    let y = yoe as i64 + era * 400;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
    let mp = (5 * doy + 2) / 153;
    let d = doy - (153 * mp + 2) / 5 + 1;
    let m = if mp < 10 { mp + 3 } else { mp - 9 };
    let y = if m <= 2 { y + 1 } else { y };
    (y as i32, m as u32, d as u32)
}

/// Render the session to JSON and write it to the chosen path. Returns the
/// status message to surface in the footer.
fn finish_collect_session(session: CollectSession) -> String {
    let CollectSession {
        path, collector, ..
    } = session;
    let n = collector.samples.len();
    let content = vmon_report::json::render(&collector, None);
    match std::fs::write(&path, content) {
        Ok(()) => format!("Saved {n} samples → {}", path.display()),
        Err(e) => format!("Error writing {}: {e}", path.display()),
    }
}

async fn execute_action(actions: &NodeActions, pending: &PendingAction) -> String {
    match pending {
        PendingAction::Sleep(addr) => match actions.sleep(addr).await {
            Ok(()) => format!("Sent sleep to {addr}"),
            Err(e) => format!("Error: {e}"),
        },
        PendingAction::WakeUp(addr) => match actions.wake_up(addr).await {
            Ok(()) => format!("Sent wake_up to {addr}"),
            Err(e) => format!("Error: {e}"),
        },
        PendingAction::ResetCache(addr) => match actions.reset_prefix_cache(addr).await {
            Ok(()) => format!("Prefix cache reset on {addr}"),
            Err(e) => format!("Error: {e}"),
        },
        PendingAction::ResetCacheAll(addrs) => {
            let futs: Vec<_> = addrs.iter().map(|addr| actions.reset_prefix_cache(addr)).collect();
            let results = futures::future::join_all(futs).await;
            let ok = results.iter().filter(|r| r.is_ok()).count();
            let fail = results.len() - ok;
            if fail == 0 {
                format!("Prefix cache reset on all {ok} nodes")
            } else {
                format!("Prefix cache reset: {ok} ok, {fail} failed")
            }
        }
    }
}
