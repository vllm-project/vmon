// SPDX-License-Identifier: Apache-2.0

use std::collections::{HashMap, HashSet};
use std::sync::{Arc, Mutex};
use std::time::Duration;

use futures::future::join_all;
use tokio::sync::watch;
use tokio::time::Instant;

use crate::cluster::ClusterState;
use crate::gpu::{GpuScrape, extract_gpu_metrics};
use crate::ib::{IbRateTracker, IbScrape, extract_ib_metrics};
use crate::kv_events::KVEventSubscriber;
use crate::metrics::{VllmScrape, detect_engines, extract_engine_metrics, extract_vllm_metrics};
use crate::mooncake::{MooncakeHealth, MooncakeScrape, MooncakeState, parse_mooncake_metrics};
use crate::node::{NodeInfo, NodeMetrics, NodeState};
use crate::parser::parse_prometheus_text;
use crate::slurm::{SlurmClusterStats, SlurmJobInfo};

/// Shared handle to the SLURM job list embedded in each emitted
/// `ClusterState`. The CLI updates this concurrently from a refresh task
/// (e.g. to mark ended jobs) while the scraper reads it once per tick.
pub type SharedSlurmJobs = Arc<Mutex<Vec<SlurmJobInfo>>>;

/// Shared handle to the cluster-wide SLURM node counts (`sinfo` totals)
/// embedded in each emitted `ClusterState`. The CLI updates this from a
/// periodic sinfo poll task; the scraper reads it once per tick.
pub type SharedSlurmCluster = Arc<Mutex<Option<SlurmClusterStats>>>;

/// Shared handle to the scrape address list. Used by the CLI follow task to
/// add nodes for newly-discovered SLURM jobs and remove nodes for jobs that
/// have been gone past the grace period; the scraper snapshots the contents
/// once per tick and reconciles its internal `NodeState` map against it.
pub type SharedNodes = Arc<Mutex<Vec<String>>>;

#[derive(Debug, thiserror::Error)]
pub enum ScrapeError {
    #[error("HTTP error: {0}")]
    Http(#[from] reqwest::Error),
    #[error("parse error: {0}")]
    Parse(#[from] crate::parser::ParseError),
    #[error(transparent)]
    Response(#[from] crate::http::ResponseError),
}

#[derive(Debug, Clone, serde::Deserialize)]
struct VersionResp {
    version: String,
}

/// How often to retry fetching server_info after a failure (in scrape ticks).
const INFO_RETRY_TICKS: u64 = 15;

pub struct Scraper {
    client: reqwest::Client,
    interval: Duration,
    nodes: SharedNodes,
    zmq_port: Option<u16>,
    zmq_topic: String,
    zmq_dp_ranks: u16,
    gpu_port: Option<u16>,
    ib_port: Option<u16>,
    mooncake_addrs: Vec<String>,
    mooncake_auto_port: Option<u16>,
    dynamo: bool,
    slurm_jobs: SharedSlurmJobs,
    slurm_cluster: SharedSlurmCluster,
}

impl Scraper {
    /// Snapshot of the current scrape address list.
    pub fn node_addrs(&self) -> Vec<String> {
        self.nodes.lock().expect("nodes mutex poisoned").clone()
    }

    /// Shared handle to the scrape address list. Clone and pass to the CLI
    /// follow task so it can mutate the same list the scraper reads.
    pub fn nodes_handle(&self) -> SharedNodes {
        self.nodes.clone()
    }

    pub fn new(nodes: Vec<String>, interval: Duration) -> Self {
        let client = reqwest::Client::builder()
            .timeout(Duration::from_secs(15))
            .build()
            .expect("failed to build HTTP client");

        Self {
            client,
            interval,
            nodes: Arc::new(Mutex::new(nodes)),
            zmq_port: None,
            zmq_topic: String::new(),
            zmq_dp_ranks: 1,
            gpu_port: None,
            ib_port: None,
            mooncake_addrs: Vec::new(),
            mooncake_auto_port: None,
            dynamo: false,
            slurm_jobs: Arc::new(Mutex::new(Vec::new())),
            slurm_cluster: Arc::new(Mutex::new(None)),
        }
    }

    /// Replace the internal address list handle with an externally-owned one.
    /// The follow task in the CLI uses this to feed new SLURM-discovered nodes
    /// into the running scraper without restart.
    pub fn with_nodes_handle(mut self, handle: SharedNodes) -> Self {
        self.nodes = handle;
        self
    }

    /// Enable NVIDIA Dynamo mode: skip vLLM-specific endpoints (/version, /server_info, etc.)
    pub fn with_dynamo(mut self) -> Self {
        self.dynamo = true;
        self
    }

    /// Attach SLURM job metadata as a static snapshot; will be embedded in
    /// every emitted `ClusterState`. Use [`Scraper::with_slurm_jobs_handle`]
    /// when the list needs to be updated after construction (e.g. to freeze
    /// uptime when a job leaves `squeue`).
    pub fn with_slurm_jobs(mut self, jobs: Vec<SlurmJobInfo>) -> Self {
        self.slurm_jobs = Arc::new(Mutex::new(jobs));
        self
    }

    /// Attach a shared, mutable SLURM job list. Each tick the scraper
    /// snapshots the current contents into the emitted `ClusterState`,
    /// so any external writer (e.g. a `squeue` refresh task) can update
    /// the same handle and have the change reflected on the next tick.
    pub fn with_slurm_jobs_handle(mut self, handle: SharedSlurmJobs) -> Self {
        self.slurm_jobs = handle;
        self
    }

    /// Attach a shared handle to cluster-wide SLURM node counts. Each tick
    /// the scraper snapshots the current value into the emitted
    /// `ClusterState`; the CLI's sinfo poll task is the writer.
    pub fn with_slurm_cluster_handle(mut self, handle: SharedSlurmCluster) -> Self {
        self.slurm_cluster = handle;
        self
    }

    /// Enable ZMQ KV event subscription for all nodes.
    /// `dp_ranks` controls how many DP rank ports to subscribe (port, port+1, ...).
    pub fn with_zmq(mut self, port: u16, topic: String, dp_ranks: u16) -> Self {
        self.zmq_port = Some(port);
        self.zmq_topic = topic;
        self.zmq_dp_ranks = dp_ranks.max(1);
        self
    }

    /// Enable GPU metrics scraping from vmon agent on each node.
    pub fn with_gpu(mut self, port: u16) -> Self {
        self.gpu_port = Some(port);
        self
    }

    /// Enable InfiniBand metrics scraping from node_exporter (default port 9100).
    pub fn with_ib(mut self, port: u16) -> Self {
        self.ib_port = Some(port);
        self
    }

    /// Enable Mooncake Store scraping of an explicit leader endpoint
    /// (`host:port` exposing `/metrics` and `/health`). May be called more
    /// than once to track several stores.
    pub fn with_mooncake(mut self, addr: String) -> Self {
        self.mooncake_addrs.push(addr);
        self
    }

    /// Enable Mooncake Store auto-targeting: each tick, derive one leader
    /// address per tracked SLURM job (the job's first node on `port`).
    /// Jobs appearing later in
    /// follow mode are picked up automatically; with no SLURM context, falls
    /// back to the first scrape target's host.
    pub fn with_mooncake_auto(mut self, port: u16) -> Self {
        self.mooncake_auto_port = Some(port);
        self
    }

    /// Scrape GPU metrics from a vmon agent running on `host`.
    async fn scrape_gpu(&self, host: &str) -> Option<GpuScrape> {
        let port = self.gpu_port?;
        let url = format!("http://{host}:{port}/metrics");
        let resp = match self.client.get(&url).timeout(Duration::from_secs(5)).send().await {
            Ok(r) => r,
            Err(e) => {
                tracing::warn!(host, port, %e, "GPU scrape HTTP failed");
                return None;
            }
        };
        let body = match crate::http::text(resp).await {
            Ok(b) => b,
            Err(e) => {
                tracing::warn!(host, port, %e, "GPU scrape body read failed");
                return None;
            }
        };
        let families = match parse_prometheus_text(&body) {
            Ok(f) => f,
            Err(e) => {
                tracing::warn!(host, port, %e, "GPU scrape parse failed");
                return None;
            }
        };
        let scrape = extract_gpu_metrics(&families);
        if scrape.gpus.is_empty()
            && scrape.cpu_percent.is_none()
            && scrape.mem_total_bytes.is_none()
        {
            tracing::warn!(
                host,
                port,
                families = families.len(),
                "GPU scrape: no usable metrics found"
            );
            None
        } else {
            for g in &scrape.gpus {
                tracing::debug!(
                    host, gpu = g.index, name = %g.name,
                    util = g.utilization, mem_util = g.mem_utilization,
                    mem_used = g.mem_used_bytes, mem_total = g.mem_total_bytes,
                    power = g.power_watts, temp = g.temperature,
                    "GPU scrape result"
                );
            }
            Some(scrape)
        }
    }

    /// Scrape InfiniBand metrics from `node_exporter` on `host`. Returns the
    /// raw cumulative-byte counters; rates are computed later by `IbRateTracker`.
    async fn scrape_ib(&self, host: &str) -> Option<IbScrape> {
        let port = self.ib_port?;
        let url = format!("http://{host}:{port}/metrics");
        let resp = match self.client.get(&url).timeout(Duration::from_secs(5)).send().await {
            Ok(r) => r,
            Err(e) => {
                tracing::debug!(host, port, %e, "IB scrape HTTP failed");
                return None;
            }
        };
        let body = match crate::http::text(resp).await {
            Ok(b) => b,
            Err(e) => {
                tracing::debug!(host, port, %e, "IB scrape body read failed");
                return None;
            }
        };
        let families = match parse_prometheus_text(&body) {
            Ok(f) => f,
            Err(e) => {
                tracing::debug!(host, port, %e, "IB scrape parse failed");
                return None;
            }
        };
        let scrape = extract_ib_metrics(&families);
        if scrape.devices.is_empty() {
            tracing::debug!(host, port, "IB scrape: no infiniband devices found");
            None
        } else {
            Some(scrape)
        }
    }

    /// Scrape Mooncake Store `/metrics` (Prometheus text). Returns `None` on
    /// any HTTP/parse failure; the caller decides whether to surface
    /// "unreachable" in the UI.
    async fn scrape_mooncake(&self, addr: &str) -> Option<MooncakeScrape> {
        let url = format!("http://{addr}/metrics");
        let resp = match self.client.get(&url).timeout(Duration::from_secs(5)).send().await {
            Ok(r) => r,
            Err(e) => {
                tracing::warn!(addr, %e, "Mooncake scrape HTTP failed");
                return None;
            }
        };
        let body = match crate::http::text(resp).await {
            Ok(b) => b,
            Err(e) => {
                tracing::warn!(addr, %e, "Mooncake scrape body read failed");
                return None;
            }
        };
        let families = match parse_prometheus_text(&body) {
            Ok(f) => f,
            Err(e) => {
                tracing::warn!(addr, %e, "Mooncake scrape parse failed");
                return None;
            }
        };
        Some(parse_mooncake_metrics(&families))
    }

    /// Scrape Mooncake Store `/health` (JSON). Best-effort: `None` on any
    /// failure — health is supplementary to the `/metrics` payload.
    async fn scrape_mooncake_health(&self, addr: &str) -> Option<MooncakeHealth> {
        let url = format!("http://{addr}/health");
        let resp = self.client.get(&url).timeout(Duration::from_secs(3)).send().await.ok()?;
        crate::http::json::<MooncakeHealth>(resp).await.ok()
    }

    /// Fetch static info from a node (version + server_info).
    pub async fn probe_node(&self, addr: &str) -> Option<NodeInfo> {
        let mut info = NodeInfo::default();
        let mut ok = false;

        if self.dynamo {
            // Dynamo frontend doesn't expose /version or /server_info.
            // Attempt /v1/models only.
            if let Ok(resp) = self
                .client
                .get(format!("http://{addr}/v1/models"))
                .timeout(Duration::from_secs(5))
                .send()
                .await
            {
                if let Ok(body) = crate::http::json::<serde_json::Value>(resp).await {
                    if let Some(id) = body["data"][0]["id"].as_str() {
                        info.model_name = Some(id.to_string());
                        ok = true;
                    }
                }
            }
            return if ok { Some(info) } else { None };
        }

        // /version
        if let Ok(resp) = self
            .client
            .get(format!("http://{addr}/version"))
            .timeout(Duration::from_secs(5))
            .send()
            .await
        {
            if let Ok(v) = crate::http::json::<VersionResp>(resp).await {
                info.vllm_version = v.version;
                ok = true;
            }
        }

        // /v1/models — model name
        if let Ok(resp) = self
            .client
            .get(format!("http://{addr}/v1/models"))
            .timeout(Duration::from_secs(5))
            .send()
            .await
        {
            if let Ok(body) = crate::http::json::<serde_json::Value>(resp).await {
                if let Some(id) = body["data"][0]["id"].as_str() {
                    info.model_name = Some(id.to_string());
                    ok = true;
                }
            }
        }

        // /server_info — best effort
        if let Ok(resp) = self
            .client
            .get(format!("http://{addr}/server_info?config_format=json"))
            .timeout(Duration::from_secs(5))
            .send()
            .await
        {
            if let Ok(body) = crate::http::json::<serde_json::Value>(resp).await {
                if let Some(gpu) = body["system_env"]["nvidia_gpu_models"].as_str() {
                    info.gpu_info = Some(compact_gpu_info(gpu));
                }
                let mut body = body;
                crate::privacy::redact(&mut body);
                info.server_info = Some(body);
                ok = true;
            }
        }

        if ok { Some(info) } else { None }
    }

    /// Scrape /metrics and parse into VllmScrape.
    ///
    pub async fn scrape_metrics(&self, addr: &str) -> Result<VllmScrape, ScrapeError> {
        let response = self
            .client
            .get(format!("http://{addr}/metrics"))
            .timeout(Duration::from_secs(10))
            .send()
            .await?;
        let body = crate::http::text(response).await?;
        let families = parse_prometheus_text(&body)?;
        let now = Instant::now();
        let mut scrape = extract_vllm_metrics(&families, now);
        let engines = detect_engines(&families);
        if !engines.is_empty() {
            scrape.engine_scrapes =
                engines.iter().map(|eng| extract_engine_metrics(&families, eng, now)).collect();
        }
        Ok(scrape)
    }

    /// Run the scrape loop, sending ClusterState updates through the watch channel.
    /// Returns the watch receiver.
    pub async fn run_loop(self) -> watch::Receiver<ClusterState> {
        // Seed the initial cluster state from whatever nodes are currently in
        // the shared list. Empty list (e.g. follow mode before the first
        // squeue poll lands) is fine — the TUI will just render an empty
        // body until the first reconcile adds rows.
        let nodes_snapshot: Vec<String> = self.nodes.lock().expect("nodes mutex poisoned").clone();
        // In Dynamo mode, ranks fold into one row per host after the first
        // scrape — seed the initial state at host granularity so the TUI
        // doesn't briefly render N×ranks loading rows that then collapse.
        let initial_nodes: Vec<NodeMetrics> = if self.dynamo {
            let mut seen: HashSet<String> = HashSet::new();
            let mut nodes = Vec::new();
            for addr in &nodes_snapshot {
                let host = addr.split(':').next().unwrap_or(addr).to_string();
                if seen.insert(host.clone()) {
                    nodes.push(NodeMetrics::loading(host));
                }
            }
            nodes
        } else {
            nodes_snapshot.iter().map(|a| NodeMetrics::loading(a.clone())).collect()
        };
        let mut initial = ClusterState::aggregate(initial_nodes, HashMap::new());
        initial.slurm_jobs = self.slurm_jobs.lock().expect("slurm_jobs mutex poisoned").clone();
        initial.slurm_cluster = *self.slurm_cluster.lock().expect("slurm_cluster mutex poisoned");
        let (tx, rx) = watch::channel(initial);

        tokio::spawn(async move {
            let mut states: HashMap<String, NodeState> =
                nodes_snapshot.iter().map(|a| (a.clone(), NodeState::new(a.clone()))).collect();

            // Spawn ZMQ subscribers (one per DP rank per node) if zmq_port is set
            let mut kv_subs: HashMap<String, Vec<KVEventSubscriber>> =
                if let Some(port) = self.zmq_port {
                    tracing::info!(
                        port,
                        dp_ranks = self.zmq_dp_ranks,
                        "Enabling ZMQ KV event subscribers"
                    );
                    nodes_snapshot
                        .iter()
                        .map(|addr| {
                            let subs =
                                spawn_kv_subs_for(addr, port, self.zmq_dp_ranks, &self.zmq_topic);
                            (addr.clone(), subs)
                        })
                        .collect()
                } else {
                    HashMap::new()
                };

            // Per-target Mooncake rate state, keyed by address. Reconciled
            // each tick: auto-derived targets come and go with SLURM jobs.
            let mut mooncake_states: HashMap<String, MooncakeState> = HashMap::new();

            // Addrs that have completed at least one successful /metrics
            // scrape. `regroup_dp_engines` uses this to tell headless DP
            // secondaries (which never serve metrics) apart from standalone
            // nodes that died mid-run.
            let mut ever_scraped_ok: HashSet<String> = HashSet::new();

            let mut tick_count: u64 = 0;
            let mut interval = tokio::time::interval(self.interval);
            // If a scrape tick overruns (e.g. one slow node hits the 15s
            // reqwest timeout), `Delay` keeps the *next* tick at least one
            // interval away instead of firing back-to-back to "catch up".
            // Catch-up bursts compound congestion and skew rate denominators.
            interval.set_missed_tick_behavior(tokio::time::MissedTickBehavior::Delay);
            let mut ib_tracker = IbRateTracker::new();

            loop {
                interval.tick().await;
                tick_count += 1;

                // Snapshot the current address list and reconcile state. New
                // addrs (added by the CLI follow task) get a fresh NodeState
                // and ZMQ subscribers; addrs that have been removed are
                // dropped so their TUI rows disappear next frame.
                let addrs: Vec<String> = self.nodes.lock().expect("nodes mutex poisoned").clone();
                let jobs_snapshot: Vec<SlurmJobInfo> =
                    self.slurm_jobs.lock().expect("slurm_jobs mutex poisoned").clone();
                reconcile_node_states(&mut states, &addrs);
                if let Some(port) = self.zmq_port {
                    reconcile_kv_subs(
                        &mut kv_subs,
                        &addrs,
                        port,
                        self.zmq_dp_ranks,
                        &self.zmq_topic,
                    );
                }

                // Per-tick set of unique hosts (for GPU + IB scraping).
                // Sort for deterministic ordering — keeps debug logs / traces
                // reproducible across ticks.
                let mut unique_hosts: Vec<String> = addrs
                    .iter()
                    .map(|a| a.split(':').next().unwrap_or(a).to_string())
                    .collect::<HashSet<_>>()
                    .into_iter()
                    .collect();
                unique_hosts.sort();

                // ── Scrape /metrics + GPU + IB in parallel ──
                let scrape_futs = join_all(addrs.iter().map(|addr| {
                    let addr = addr.clone();
                    let scraper = &self;
                    async move {
                        let metrics = scraper.scrape_metrics(&addr).await;
                        (addr, metrics)
                    }
                }));

                let gpu_futs = async {
                    if self.gpu_port.is_none() {
                        return HashMap::<String, Option<GpuScrape>>::new();
                    }
                    join_all(unique_hosts.iter().map(|host| {
                        let host = host.clone();
                        let scraper = &self;
                        async move { (host.clone(), scraper.scrape_gpu(&host).await) }
                    }))
                    .await
                    .into_iter()
                    .collect()
                };

                let ib_futs = async {
                    if self.ib_port.is_none() {
                        return HashMap::<String, Option<IbScrape>>::new();
                    }
                    join_all(unique_hosts.iter().map(|host| {
                        let host = host.clone();
                        let scraper = &self;
                        async move { (host.clone(), scraper.scrape_ib(&host).await) }
                    }))
                    .await
                    .into_iter()
                    .collect()
                };

                let mc_targets = mooncake_targets(
                    &self.mooncake_addrs,
                    self.mooncake_auto_port,
                    &jobs_snapshot,
                    &addrs,
                );
                let mooncake_fut = join_all(mc_targets.iter().map(|addr| {
                    let addr = addr.clone();
                    let scraper = &self;
                    async move {
                        let (scrape, health) = tokio::join!(
                            scraper.scrape_mooncake(&addr),
                            scraper.scrape_mooncake_health(&addr),
                        );
                        (addr, scrape, health)
                    }
                }));

                let (scrape_results, gpu_results, mut ib_results, mooncake_results) =
                    tokio::join!(scrape_futs, gpu_futs, ib_futs, mooncake_fut);

                // Convert raw cumulative IB byte counters into per-second Gbps.
                // IB traffic is per-host; all peers on a host share the same
                // numbers (no per-node slicing).
                let ib_now = Instant::now();
                for (host, maybe_scrape) in ib_results.iter_mut() {
                    if let Some(s) = maybe_scrape {
                        ib_tracker.apply(host, s, ib_now);
                    }
                }

                // Build per-host node ordering (sorted by address) for GPU index assignment.
                // When N vLLM nodes share a host (one process per GPU), the i-th node by
                // address order is assumed to own GPU[i]. This avoids showing every node all
                // GPUs on the host when each process only uses one.
                let mut host_node_order: HashMap<String, Vec<String>> = HashMap::new();
                for addr in &addrs {
                    let host = addr.split(':').next().unwrap_or(addr).to_string();
                    host_node_order.entry(host).or_default().push(addr.clone());
                }
                for peers in host_node_order.values_mut() {
                    peers.sort(); // lower port → lower GPU index
                }

                // ── Compute node metrics ──
                let mut node_metrics: Vec<NodeMetrics> = Vec::with_capacity(addrs.len());
                for (addr, result) in &scrape_results {
                    let state = states.get_mut(addr).unwrap();
                    let host = addr.split(':').next().unwrap_or(addr);
                    let kv: Option<Vec<_>> = kv_subs.get(addr).map(|subs| {
                        subs.iter()
                            .enumerate()
                            .map(|(i, sub)| {
                                let mut m = sub.drain();
                                m.dp_rank = Some(i as u16);
                                m
                            })
                            .collect()
                    });
                    // In Dynamo fold mode, host-level metrics (GPU/IB) are
                    // attached after the per-host fold below — leave per-rank
                    // metrics with None to avoid duplication.
                    let gpu = if self.dynamo {
                        None
                    } else {
                        let full = gpu_results.get(host).and_then(|g| g.clone());
                        // When N nodes share a host, divide the GPU list evenly by port order:
                        //   tp_size = gpu_count / node_count
                        //   node[i] gets gpus[i*tp_size .. (i+1)*tp_size]
                        // Examples:
                        //   4 GPUs, 4 nodes (TP=1 each): each node gets 1 GPU
                        //   4 GPUs, 2 nodes (TP=2 each): node0→GPU[0,1], node1→GPU[2,3]
                        // Single-node hosts keep the full scrape (e.g. TP=N single process).
                        if let Some(mut gs) = full {
                            let peers =
                                host_node_order.get(host).map(|v| v.as_slice()).unwrap_or(&[]);
                            if peers.len() > 1 && !gs.gpus.is_empty() {
                                if let Some(idx) = peers.iter().position(|a| a == addr) {
                                    let tp_size = (gs.gpus.len() / peers.len()).max(1);
                                    let start = idx * tp_size;
                                    let end = ((idx + 1) * tp_size).min(gs.gpus.len());
                                    gs.gpus = if start < gs.gpus.len() {
                                        gs.gpus[start..end].to_vec()
                                    } else {
                                        vec![]
                                    };
                                }
                            }
                            Some(gs)
                        } else {
                            None
                        }
                    };
                    let ib = if self.dynamo {
                        None
                    } else {
                        ib_results.get(host).and_then(|i| i.clone())
                    };
                    match result {
                        Ok(scrape) => {
                            // Re-probe server_info on offline→online transition
                            if !state.is_healthy {
                                state.info_fetched = false;
                            }
                            state.is_healthy = true;
                            ever_scraped_ok.insert(addr.clone());
                            let m = state.update(scrape.clone(), kv, gpu, ib);
                            node_metrics.push(m);
                        }
                        Err(e) => {
                            tracing::debug!(addr, %e, "scrape failed");
                            node_metrics.push(state.on_scrape_error(gpu, ib));
                        }
                    }
                }

                // ── Multi-node DP: re-attribute centralized engines ──
                // A vLLM multi-node DP head publishes every rank's metrics
                // while its headless secondaries expose nothing; slice the
                // head's engines across the listed peer nodes so each row
                // shows its own share next to its own GPUs.
                if !self.dynamo {
                    crate::node::regroup_dp_engines(&mut node_metrics, &ever_scraped_ok);
                }

                // ── Dynamo: fold per-rank metrics into one row per host ──
                if self.dynamo && !node_metrics.is_empty() {
                    let mut by_host: HashMap<String, Vec<NodeMetrics>> = HashMap::new();
                    let mut host_order: Vec<String> = Vec::new();
                    for m in node_metrics.drain(..) {
                        let host = m.addr.split(':').next().unwrap_or(m.addr.as_str()).to_string();
                        if !by_host.contains_key(&host) {
                            host_order.push(host.clone());
                        }
                        by_host.entry(host).or_default().push(m);
                    }
                    // Sort each host's ranks by addr (port order) so engine_metrics[i] = rank i
                    for ranks in by_host.values_mut() {
                        ranks.sort_by(|a, b| a.addr.cmp(&b.addr));
                    }
                    for host in host_order {
                        let ranks = by_host.remove(&host).unwrap();
                        let mut folded = NodeMetrics::aggregate_dynamo_ranks(host.clone(), ranks);
                        folded.gpu_scrape = gpu_results.get(&host).and_then(|g| g.clone());
                        folded.ib_scrape = ib_results.get(&host).and_then(|i| i.clone());
                        node_metrics.push(folded);
                    }
                }

                // ── Lazy-fetch server_info for nodes that are online but not yet probed ──
                // Also retry periodically for nodes where probe previously failed.
                let probe_addrs: Vec<String> = states
                    .iter()
                    .filter(|(_, s)| {
                        s.is_healthy
                            && (!s.info_fetched
                                || (tick_count.is_multiple_of(INFO_RETRY_TICKS)
                                    && s.info.server_info.is_none()))
                    })
                    .map(|(a, _)| a.clone())
                    .collect();

                if !probe_addrs.is_empty() {
                    let probe_futs: Vec<_> = probe_addrs
                        .iter()
                        .map(|addr| {
                            let addr = addr.clone();
                            let scraper = &self;
                            async move { (addr.clone(), scraper.probe_node(&addr).await) }
                        })
                        .collect();
                    let probe_results = join_all(probe_futs).await;
                    for (addr, maybe_info) in probe_results {
                        if let Some(state) = states.get_mut(&addr) {
                            if let Some(info) = maybe_info {
                                state.info = info;
                                state.info_fetched = true;
                            } else {
                                // Mark as attempted so we don't retry every tick
                                state.info_fetched = true;
                            }
                        }
                    }
                }

                let node_infos: HashMap<String, NodeInfo> =
                    states.iter().map(|(addr, s)| (addr.clone(), s.info.clone())).collect();
                let mut cluster = ClusterState::aggregate(node_metrics, node_infos);
                cluster.slurm_jobs = jobs_snapshot;
                cluster.slurm_cluster =
                    *self.slurm_cluster.lock().expect("slurm_cluster mutex poisoned");
                // "backend" → "D" (PD decode) or no badge (aggregated worker),
                // decided per SLURM job now that the job list is attached.
                cluster.resolve_backend_roles();
                // Drop rate state for targets that disappeared (job ended and
                // left the tracked list) so the map doesn't grow unbounded.
                mooncake_states.retain(|addr, _| mc_targets.contains(addr));
                cluster.mooncake_addrs = mc_targets;
                for (addr, scrape, health) in mooncake_results {
                    let Some(scrape) = scrape else { continue };
                    let state = mooncake_states
                        .entry(addr.clone())
                        .or_insert_with(|| MooncakeState::new(addr));
                    cluster.mooncakes.push(state.update(scrape, health, Instant::now()));
                }
                if tx.send(cluster).is_err() {
                    break; // All receivers dropped
                }
            }
        });

        rx
    }
}

/// Spawn one ZMQ KV-event subscriber per DP rank for `addr`. Used both at
/// scraper startup and when reconciling against an updated address list in
/// follow mode.
fn spawn_kv_subs_for(
    addr: &str,
    base_port: u16,
    dp_ranks: u16,
    topic: &str,
) -> Vec<KVEventSubscriber> {
    let host = addr.split(':').next().unwrap_or(addr);
    (0..dp_ranks.max(1))
        .filter_map(|rank| {
            // base_port + rank can exceed u16::MAX with large DP counts —
            // skip rather than panic (debug) or wrap (release).
            let Some(port) = base_port.checked_add(rank) else {
                tracing::warn!(
                    base_port,
                    rank,
                    "ZMQ port overflow (base_port + rank > 65535); skipping subscriber"
                );
                return None;
            };
            let ep = format!("tcp://{host}:{port}");
            tracing::info!(%ep, rank, "Spawning ZMQ subscriber");
            Some(KVEventSubscriber::spawn(ep, topic.to_string()))
        })
        .collect()
}

/// Reconcile a `NodeState` map against the desired address list. Adds missing
/// entries with a fresh `NodeState`; drops entries that are no longer in
/// `desired`. Pure function: no IO, safe to unit-test.
fn reconcile_node_states(states: &mut HashMap<String, NodeState>, desired: &[String]) {
    let want: HashSet<&str> = desired.iter().map(String::as_str).collect();
    states.retain(|addr, _| want.contains(addr.as_str()));
    for addr in desired {
        states.entry(addr.clone()).or_insert_with(|| NodeState::new(addr.clone()));
    }
}

/// Reconcile the per-address ZMQ subscriber map. Drops entries for addresses
/// no longer present (dropping a subscriber signals its thread to exit and
/// close its socket); spawns subscribers for new addresses.
fn reconcile_kv_subs(
    kv_subs: &mut HashMap<String, Vec<KVEventSubscriber>>,
    desired: &[String],
    base_port: u16,
    dp_ranks: u16,
    topic: &str,
) {
    let want: HashSet<&str> = desired.iter().map(String::as_str).collect();
    kv_subs.retain(|addr, _| want.contains(addr.as_str()));
    for addr in desired {
        if !kv_subs.contains_key(addr) {
            kv_subs.insert(
                addr.clone(),
                spawn_kv_subs_for(addr, base_port, dp_ranks, topic),
            );
        }
    }
}

/// Compute this tick's Mooncake scrape targets: explicit addresses first,
/// then (in auto mode) one target per live SLURM job — the job's first node
/// on `auto_port`. With auto mode but no SLURM context at all (plain host
/// list on a non-SLURM cluster),
/// falls back to the first scrape target's host so a bare `--mooncake` still
/// monitors something. De-duplicated, order stable. Pure function.
fn mooncake_targets(
    explicit: &[String],
    auto_port: Option<u16>,
    jobs: &[SlurmJobInfo],
    node_addrs: &[String],
) -> Vec<String> {
    let mut targets: Vec<String> = explicit.to_vec();
    let Some(port) = auto_port else {
        return targets;
    };
    let mut auto: Vec<String> = jobs
        .iter()
        .filter(|j| !j.is_ended())
        .filter_map(|j| j.nodes.first())
        .map(|h| format!("{h}:{port}"))
        .collect();
    if jobs.is_empty() {
        if let Some(first) = node_addrs.first() {
            let host = first.split(':').next().unwrap_or(first);
            auto.push(format!("{host}:{port}"));
        }
    }
    for a in auto {
        if !targets.contains(&a) {
            targets.push(a);
        }
    }
    targets
}

/// Condense multi-line GPU list like "GPU 0: Example GPU\nGPU 1: Example GPU\n..."
/// into "4x Example GPU" (or list distinct models if mixed).
fn compact_gpu_info(raw: &str) -> String {
    use std::collections::HashMap as Map;
    let mut counts: Map<&str, usize> = Map::new();
    for line in raw.lines() {
        let model = line.find(':').map(|i| line[i + 1..].trim()).unwrap_or(line.trim());
        if !model.is_empty() {
            *counts.entry(model).or_default() += 1;
        }
    }
    if counts.is_empty() {
        return raw.lines().next().unwrap_or("").to_string();
    }
    counts
        .iter()
        .map(|(model, &cnt)| {
            if cnt > 1 {
                format!("{cnt}x {model}")
            } else {
                model.to_string()
            }
        })
        .collect::<Vec<_>>()
        .join(", ")
}

#[cfg(test)]
mod tests {
    use super::*;

    fn job(id: &str, nodes: &[&str], ended: bool) -> SlurmJobInfo {
        SlurmJobInfo {
            job_id: id.into(),
            job_name: format!("job-{id}"),
            start_time: std::time::SystemTime::now(),
            end_time: ended.then(std::time::SystemTime::now),
            nodelist_compact: nodes.join(","),
            nodes: nodes.iter().map(|s| s.to_string()).collect(),
        }
    }

    #[test]
    fn mooncake_targets_one_per_live_job_skipping_ended() {
        let jobs = vec![
            job("100", &["node01", "node02"], false),
            job("200", &["node09", "node10"], false),
            job("300", &["node20"], true),
        ];
        let t = mooncake_targets(&[], Some(8702), &jobs, &["node01:7500".into()]);
        assert_eq!(
            t,
            vec!["node01:8702".to_string(), "node09:8702".to_string()]
        );
    }

    #[test]
    fn mooncake_targets_explicit_first_and_deduped_against_auto() {
        let jobs = vec![
            job("100", &["node01"], false),
            job("200", &["node09"], false),
        ];
        let explicit = vec!["node01:8702".to_string()];
        let t = mooncake_targets(&explicit, Some(8702), &jobs, &[]);
        assert_eq!(
            t,
            vec!["node01:8702".to_string(), "node09:8702".to_string()]
        );
    }

    #[test]
    fn mooncake_targets_falls_back_to_first_node_without_slurm_context() {
        let t = mooncake_targets(&[], Some(8702), &[], &["example-01:8000".into()]);
        assert_eq!(t, vec!["example-01:8702".to_string()]);
        // But not when jobs are known and merely all ended — their masters
        // are gone with them.
        let t = mooncake_targets(&[], Some(8702), &[job("1", &["node01"], true)], &[]);
        assert!(t.is_empty());
    }

    #[test]
    fn mooncake_targets_explicit_only_without_auto_port() {
        let jobs = vec![job("100", &["node01"], false)];
        let explicit = vec!["h:9003".to_string()];
        let t = mooncake_targets(&explicit, None, &jobs, &[]);
        assert_eq!(t, vec!["h:9003".to_string()]);
    }

    #[test]
    fn reconcile_node_states_inserts_new_and_keeps_existing() {
        let mut states: HashMap<String, NodeState> = HashMap::new();
        states.insert("a:8000".into(), NodeState::new("a:8000".into()));

        reconcile_node_states(&mut states, &["a:8000".to_string(), "b:8000".to_string()]);

        assert_eq!(states.len(), 2);
        assert!(states.contains_key("a:8000"));
        assert!(states.contains_key("b:8000"));
    }

    #[test]
    fn reconcile_node_states_removes_addrs_no_longer_desired() {
        let mut states: HashMap<String, NodeState> = HashMap::new();
        states.insert("a:8000".into(), NodeState::new("a:8000".into()));
        states.insert("b:8000".into(), NodeState::new("b:8000".into()));

        reconcile_node_states(&mut states, &["a:8000".to_string()]);

        assert_eq!(states.len(), 1);
        assert!(states.contains_key("a:8000"));
        assert!(!states.contains_key("b:8000"));
    }

    #[test]
    fn reconcile_node_states_empty_desired_clears_all() {
        let mut states: HashMap<String, NodeState> = HashMap::new();
        states.insert("a:8000".into(), NodeState::new("a:8000".into()));
        states.insert("b:8000".into(), NodeState::new("b:8000".into()));

        reconcile_node_states(&mut states, &[]);

        assert!(states.is_empty());
    }

    #[test]
    fn reconcile_node_states_idempotent_on_unchanged_set() {
        let mut states: HashMap<String, NodeState> = HashMap::new();
        states.insert("a:8000".into(), NodeState::new("a:8000".into()));
        states.insert("b:8000".into(), NodeState::new("b:8000".into()));
        let before_keys: HashSet<String> = states.keys().cloned().collect();

        reconcile_node_states(&mut states, &["a:8000".to_string(), "b:8000".to_string()]);

        let after_keys: HashSet<String> = states.keys().cloned().collect();
        assert_eq!(before_keys, after_keys);
    }
}
