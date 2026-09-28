// SPDX-License-Identifier: Apache-2.0

use std::time::{Duration, SystemTime, UNIX_EPOCH};

use tokio::sync::watch;
use tokio::time::Instant;

use vmon_core::cluster::ClusterState;
// The sample structs (NodeSample, GpuSample, IbSample, TimeSample) and their
// `From<&NodeMetrics>` / sanitize / serde impls live in `vmon-core::sample` so
// the on-disk JSON shape has a single source of truth shared with the replay
// reader. We re-export them here to keep `vmon_report::collector::NodeSample`
// imports working.
pub use vmon_core::sample::{GpuSample, IbSample, MooncakeSample, NodeSample, TimeSample};

/// All available metric field names (for `--list-metrics`).
///
/// Entries without a dot select top-level `NodeSample` scalar fields.
/// Entries prefixed `gpus.` select per-GPU fields on each `GpuSample`
/// inside `NodeSample.gpus[]` (identity fields `index` and `uuid` are always
/// included when any `gpus.*` filter is active).
/// Entries prefixed `ibs.` select per-device fields on each `IbSample`
/// inside `NodeSample.ibs[]` (identity fields `device` and `port` are
/// always included when any `ibs.*` filter is active).
pub const METRIC_NAMES: &[&str] = &[
    "kv_cache",
    "running",
    "waiting",
    "generation_tps",
    "prompt_tps",
    "ttft_p50_ms",
    "ttft_p99_ms",
    "itl_p50_ms",
    "itl_p99_ms",
    "e2e_p50_ms",
    "e2e_p99_ms",
    "queue_p50_ms",
    "queue_p99_ms",
    "prefill_p50_ms",
    "prefill_p99_ms",
    "decode_p50_ms",
    "decode_p99_ms",
    "inference_p50_ms",
    "inference_p99_ms",
    "tpot_p50_ms",
    "tpot_p99_ms",
    "prefix_cache_hit_rate",
    "external_cache_hit_rate",
    "mm_cache_hit_rate",
    "spec_decode_acceptance_rate",
    "spec_decode_drafts_per_sec",
    "estimated_flops_per_gpu_per_sec",
    "estimated_read_bytes_per_gpu_per_sec",
    "estimated_write_bytes_per_gpu_per_sec",
    "mfu_percent",
    "avg_prompt_tokens",
    "avg_generation_tokens",
    "iteration_tokens_mean",
    "preemptions_per_sec",
    "requests_per_sec",
    "http_qps",
    "http_error_rate",
    "server_load",
    "gpu_utilization",
    "gpu_mem_utilization",
    "gpu_power_watts",
    "gpu_temperature",
    "gpu_vram_used_bytes",
    "gpu_vram_total_bytes",
    "gpu_nvlink_tx_kbps",
    "gpu_nvlink_rx_kbps",
    // Per-GPU series (one value per GPU per tick, inside NodeSample.gpus[]).
    "gpus.index",
    "gpus.uuid",
    "gpus.name",
    "gpus.utilization",
    "gpus.mem_utilization",
    "gpus.power_watts",
    "gpus.temperature",
    "gpus.mem_used_bytes",
    "gpus.mem_total_bytes",
    "gpus.clock_mhz",
    "gpus.mem_clock_mhz",
    "gpus.pcie_tx_kbps",
    "gpus.pcie_rx_kbps",
    "gpus.nvlink_tx_kbps",
    "gpus.nvlink_rx_kbps",
    "gpus.power_limit_watts",
    "gpus.throttle_reasons",
    "gpus.ecc_sbe_volatile",
    "gpus.ecc_dbe_volatile",
    "gpus.dram_active",
    "gpus.gr_engine_active",
    "gpus.tensor_active",
    "gpus.enc_utilization",
    "gpus.dec_utilization",
    "gpus.mem_temperature",
    "gpus.total_energy_mj",
    "gpus.pcie_replay_count",
    "gpus.xid_errors",
    "gpus.remapped_rows_correctable",
    "gpus.remapped_rows_uncorrectable",
    "gpus.row_remap_failure",
    "gpus.vgpu_license_status",
    "gpus.pcie_prof_tx_bytes_per_sec",
    "gpus.pcie_prof_rx_bytes_per_sec",
    // InfiniBand aggregate (per-host, summed across active devices).
    "ib_total_tx_gbps",
    "ib_total_rx_gbps",
    "ib_active_count",
    "ib_total_link_gbps",
    // Per-IB-device series (one value per active IB device per tick).
    "ibs.device",
    "ibs.port",
    "ibs.tx_gbps",
    "ibs.rx_gbps",
    "ibs.tx_bytes_total",
    "ibs.rx_bytes_total",
    "ibs.tx_packets_total",
    "ibs.rx_packets_total",
    "ibs.state_id",
    "ibs.rate_bytes_per_sec",
    "ibs.link_gbps",
];

/// Accumulates time-series samples over a collection period.
pub struct TimeSeriesCollector {
    pub start: Instant,
    pub samples: Vec<TimeSample>,
    pub node_addrs: Vec<String>,
    /// If set, keep at most this many samples (FIFO — oldest dropped).
    /// `None` means unbounded.
    pub max_samples: Option<usize>,
    /// Total samples dropped due to the cap, reported in progress output.
    pub dropped_samples: usize,
}

impl TimeSeriesCollector {
    pub fn new(node_addrs: Vec<String>) -> Self {
        Self {
            start: Instant::now(),
            samples: Vec::new(),
            node_addrs,
            max_samples: None,
            dropped_samples: 0,
        }
    }

    /// Cap the number of in-memory samples. Once reached, oldest samples are
    /// dropped first. Setting `None` removes the cap.
    pub fn with_max_samples(mut self, max_samples: Option<usize>) -> Self {
        self.max_samples = max_samples;
        self
    }

    pub fn record(&mut self, cluster: &ClusterState) {
        let elapsed = self.start.elapsed().as_secs_f64();
        let timestamp_ms = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map(|d| d.as_millis() as u64)
            .unwrap_or(0);
        let mut nodes = std::collections::HashMap::new();
        for n in &cluster.nodes {
            // Include the node when vLLM is healthy OR when any host-level
            // scrape (GPU/DCGM, RDMA) succeeded. RDMA and DCGM are independent
            // of vLLM health — dropping unhealthy nodes loses host data on
            // peers whose vLLM hasn't started yet or has crashed.
            let has_host_data = n.gpu_scrape.is_some() || n.ib_scrape.is_some();
            if n.is_healthy || has_host_data {
                nodes.insert(n.addr.clone(), NodeSample::from(n));
            }
        }
        self.samples.push(TimeSample {
            elapsed_secs: elapsed,
            timestamp_ms,
            nodes,
            // Scalar `mooncake` stays the first store for backward
            // compatibility; the full list only materializes with ≥2 stores.
            mooncake: cluster.mooncakes.first().map(MooncakeSample::from),
            mooncakes: if cluster.mooncakes.len() >= 2 {
                cluster.mooncakes.iter().map(MooncakeSample::from).collect()
            } else {
                Vec::new()
            },
        });
        if let Some(cap) = self.max_samples {
            if cap == 0 {
                let dropped = self.samples.len();
                self.samples.clear();
                self.dropped_samples = self.dropped_samples.saturating_add(dropped);
            } else if self.samples.len() > cap {
                let excess = self.samples.len() - cap;
                self.samples.drain(..excess);
                self.dropped_samples = self.dropped_samples.saturating_add(excess);
            }
        }
    }

    /// Run collection for the specified duration, reading from a watch channel.
    ///
    /// When `flush_interval` is `Some(non-zero)`, `flush_cb` is invoked with the
    /// current collector roughly every `flush_interval` so the caller can write
    /// a checkpoint to disk mid-run. This makes the report survive a hard kill
    /// (`SIGKILL`, crash, power loss) — Ctrl-C already saves via the normal exit
    /// path. The cadence is bounded below by `interval` (checkpoints can only
    /// happen on a scrape tick). `None` or a zero duration disables checkpoints.
    pub async fn run(
        mut self,
        mut rx: watch::Receiver<ClusterState>,
        duration: Duration,
        interval: Duration,
        flush_interval: Option<Duration>,
        mut progress_cb: impl FnMut(usize, Duration, Duration, usize, usize),
        mut flush_cb: impl FnMut(&Self),
    ) -> Self {
        let deadline = self.start + duration;
        let mut tick = tokio::time::interval(interval);
        let ctrl_c = tokio::signal::ctrl_c();
        tokio::pin!(ctrl_c);
        let flush_interval = flush_interval.filter(|d| !d.is_zero());
        let mut last_flush = self.start;

        loop {
            tokio::select! {
                biased;
                _ = &mut ctrl_c => {
                    eprintln!("\nInterrupted — saving {} samples collected so far.", self.samples.len());
                    break;
                }
                _ = tick.tick() => {}
            }

            let now = Instant::now();
            if now >= deadline {
                break;
            }

            // Use latest data without blocking (scraper updates independently)
            let cluster = rx.borrow_and_update().clone();
            self.record(&cluster);

            let elapsed = now.duration_since(self.start);
            progress_cb(
                self.samples.len(),
                elapsed,
                duration,
                cluster.healthy_count,
                cluster.nodes.len(),
            );

            if let Some(fi) = flush_interval {
                if now.duration_since(last_flush) >= fi {
                    flush_cb(&self);
                    last_flush = now;
                }
            }
        }

        self
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use vmon_core::cluster::ClusterState;
    use vmon_core::gpu::{GpuMetrics, GpuScrape};
    use vmon_core::ib::{IbDevice, IbScrape};
    use vmon_core::node::{NodeInfo, NodeMetrics};

    fn sample_node_with_gpus(addr: &str) -> NodeMetrics {
        let mut m = NodeMetrics::loading(addr.to_string());
        m.is_healthy = true;
        m.kv_cache_usage = 0.42;
        m.generation_tps = 2100.0;
        m.gpu_scrape = Some(GpuScrape {
            gpus: vec![
                GpuMetrics {
                    index: 0,
                    uuid: "GPU-u0".to_string(),
                    name: "Example GPU".to_string(),
                    utilization: 80.0,
                    dram_active: Some(0.52),
                    tensor_active: Some(0.41),
                    gr_engine_active: Some(0.80),
                    power_watts: 161.2,
                    temperature: 25.0,
                    mem_temperature: 25.0,
                    mem_used_bytes: 17_179_869_184,
                    mem_total_bytes: 196_065_034_240,
                    total_energy_mj: 251_885_217_371,
                    ..Default::default()
                },
                GpuMetrics {
                    index: 1,
                    uuid: "GPU-u1".to_string(),
                    name: "Example GPU".to_string(),
                    utilization: 75.0,
                    dram_active: Some(0.47),
                    tensor_active: Some(0.38),
                    power_watts: 163.7,
                    ..Default::default()
                },
            ],
            ..Default::default()
        });
        m
    }

    /// NodeSample carries per-GPU values end-to-end through serde and
    /// TimeSample timestamps are populated. Core acceptance test for the
    /// per-GPU JSON pipeline.
    #[test]
    fn time_sample_round_trip_preserves_per_gpu_fields() {
        let node = sample_node_with_gpus("host:8000");
        let cluster = ClusterState::aggregate(vec![node], {
            let mut m = std::collections::HashMap::new();
            m.insert("host:8000".to_string(), NodeInfo::default());
            m
        });
        let mut c = TimeSeriesCollector::new(vec!["host:8000".to_string()]);
        c.record(&cluster);

        let sample = c.samples.last().expect("at least one sample");
        assert!(sample.timestamp_ms > 0, "wall-clock timestamp populated");

        let json = serde_json::to_string(sample).expect("serialize");
        let back: TimeSample = serde_json::from_str(&json).expect("deserialize");
        let node_sample = back.nodes.get("host:8000").expect("node present");
        assert_eq!(node_sample.gpus.len(), 2);
        assert_eq!(node_sample.gpus[0].index, 0);
        assert_eq!(node_sample.gpus[0].uuid, "GPU-u0");
        assert!((node_sample.gpus[0].dram_active.unwrap() - 0.52).abs() < 1e-6);
        assert!((node_sample.gpus[0].tensor_active.unwrap() - 0.41).abs() < 1e-6);
        assert_eq!(node_sample.gpus[0].total_energy_mj, 251_885_217_371);
        assert_eq!(node_sample.gpus[1].index, 1);
        assert!((node_sample.gpus[1].dram_active.unwrap() - 0.47).abs() < 1e-6);
        // Aggregated fields still populated for backward compat.
        assert!(node_sample.gpu_power_watts.is_some());
    }

    /// NodeSample carries IB per-device values and aggregate fields end-to-end
    /// through serde.
    #[test]
    fn time_sample_round_trip_preserves_ib_fields() {
        let mut node = NodeMetrics::loading("host:8000".to_string());
        node.is_healthy = true;
        node.ib_scrape = Some(IbScrape {
            devices: vec![
                IbDevice {
                    device: "mlx5_0".to_string(),
                    port: 1,
                    tx_bytes_total: 1_000_000,
                    rx_bytes_total: 500_000,
                    state_id: 4,
                    rate_bytes_per_sec: 50_000_000_000,
                    tx_gbps: 60.0,
                    rx_gbps: 40.0,
                    ..Default::default()
                },
                IbDevice {
                    device: "mlx5_1".to_string(),
                    port: 1,
                    state_id: 1, // not active — excluded from aggregates
                    ..Default::default()
                },
            ],
        });
        let cluster = ClusterState::aggregate(vec![node], {
            let mut m = std::collections::HashMap::new();
            m.insert("host:8000".to_string(), NodeInfo::default());
            m
        });
        let mut c = TimeSeriesCollector::new(vec!["host:8000".to_string()]);
        c.record(&cluster);

        let sample = c.samples.last().expect("at least one sample");
        let json = serde_json::to_string(sample).expect("serialize");
        let back: TimeSample = serde_json::from_str(&json).expect("deserialize");
        let n = back.nodes.get("host:8000").expect("node present");
        assert_eq!(n.ibs.len(), 2);
        assert_eq!(n.ibs[0].device, "mlx5_0");
        assert_eq!(n.ibs[0].link_gbps, Some(400.0));
        assert!((n.ibs[0].tx_gbps - 60.0).abs() < 1e-6);
        assert_eq!(n.ib_active_count, Some(1));
        assert!((n.ib_total_tx_gbps.unwrap() - 60.0).abs() < 1e-6);
        assert!((n.ib_total_rx_gbps.unwrap() - 40.0).abs() < 1e-6);
        assert!((n.ib_total_link_gbps.unwrap() - 400.0).abs() < 1e-6);
    }

    /// When a node has no GPU scrape, `gpus` is an empty Vec — shape stays
    /// consistent (no `null`, no missing field).
    #[test]
    fn node_without_gpu_scrape_yields_empty_gpus_vec() {
        let mut m = NodeMetrics::loading("host:8000".to_string());
        m.is_healthy = true;
        // gpu_scrape is None (default).
        let cluster = ClusterState::aggregate(vec![m], {
            let mut map = std::collections::HashMap::new();
            map.insert("host:8000".to_string(), NodeInfo::default());
            map
        });
        let mut c = TimeSeriesCollector::new(vec!["host:8000".to_string()]);
        c.record(&cluster);
        let node_sample = &c.samples.last().unwrap().nodes["host:8000"];
        assert!(node_sample.gpus.is_empty());
        let json = serde_json::to_value(node_sample).unwrap();
        assert!(json.get("gpus").unwrap().is_array());
    }

    /// FIFO cap: once max_samples is reached, oldest samples are dropped and
    /// dropped_samples tracks the total evicted.
    #[test]
    fn max_samples_cap_drops_oldest_samples() {
        let mut m = NodeMetrics::loading("host:8000".to_string());
        m.is_healthy = true;
        let cluster = ClusterState::aggregate(vec![m], {
            let mut map = std::collections::HashMap::new();
            map.insert("host:8000".to_string(), NodeInfo::default());
            map
        });
        let mut c =
            TimeSeriesCollector::new(vec!["host:8000".to_string()]).with_max_samples(Some(3));
        for _ in 0..7 {
            c.record(&cluster);
        }
        assert_eq!(c.samples.len(), 3);
        assert_eq!(c.dropped_samples, 4);
    }

    /// Mooncake sample shape: with one store, only the backward-compatible
    /// scalar `mooncake` is written (format unchanged for existing
    /// consumers); with two, the full `mooncakes` list materializes too.
    #[test]
    fn record_writes_mooncake_scalar_and_list_by_store_count() {
        use vmon_core::mooncake::MooncakeMetrics;
        let store = |addr: &str| MooncakeMetrics {
            addr: addr.to_string(),
            ..Default::default()
        };
        let mut m = NodeMetrics::loading("host:8000".to_string());
        m.is_healthy = true;
        let mut cluster = ClusterState::aggregate(vec![m], std::collections::HashMap::new());
        cluster.mooncakes = vec![store("node01:8702")];
        let mut c = TimeSeriesCollector::new(vec!["host:8000".to_string()]);
        c.record(&cluster);
        let s = c.samples.last().unwrap();
        assert_eq!(s.mooncake.as_ref().unwrap().addr, "node01:8702");
        assert!(s.mooncakes.is_empty());

        cluster.mooncakes = vec![store("node01:8702"), store("node09:8702")];
        c.record(&cluster);
        let s = c.samples.last().unwrap();
        assert_eq!(s.mooncake.as_ref().unwrap().addr, "node01:8702");
        let addrs: Vec<&str> = s.mooncakes.iter().map(|m| m.addr.as_str()).collect();
        assert_eq!(addrs, vec!["node01:8702", "node09:8702"]);
    }

    #[test]
    fn max_samples_unbounded_default_keeps_all() {
        let mut m = NodeMetrics::loading("host:8000".to_string());
        m.is_healthy = true;
        let cluster = ClusterState::aggregate(vec![m], {
            let mut map = std::collections::HashMap::new();
            map.insert("host:8000".to_string(), NodeInfo::default());
            map
        });
        let mut c = TimeSeriesCollector::new(vec!["host:8000".to_string()]);
        for _ in 0..5 {
            c.record(&cluster);
        }
        assert_eq!(c.samples.len(), 5);
        assert_eq!(c.dropped_samples, 0);
    }
}
