// SPDX-License-Identifier: Apache-2.0

use std::collections::{HashMap, HashSet};
use tokio::time::Instant;

use crate::gpu::GpuScrape;
use crate::histogram::HistogramSnapshot;
use crate::ib::IbScrape;
use crate::kv_events::KVEventMetrics;
use crate::metrics::VllmScrape;
use crate::rate::{RateCalc, RatioCalc};

/// Static information about a node, fetched once on first connection.
#[derive(Debug, Clone, Default)]
pub struct NodeInfo {
    pub vllm_version: String,
    pub model_name: Option<String>,
    pub gpu_info: Option<String>,
    /// Raw JSON from /server_info: vllm_config, vllm_env, system_env
    pub server_info: Option<serde_json::Value>,
}

/// Latency percentiles for a single histogram metric (in ms).
#[derive(Debug, Clone, Default)]
pub struct LatencyPct {
    pub p50: f64,
    pub p90: f64,
    pub p99: f64,
    pub mean: f64,
}

/// Computed metrics for a node, derived from consecutive scrapes.
#[derive(Debug, Clone)]
pub struct NodeMetrics {
    pub addr: String,
    pub model_name: String,
    pub is_healthy: bool,
    /// True before the first scrape attempt completes.
    pub is_loading: bool,
    /// Whether the engine is sleeping (dev mode). None if not available.
    pub is_sleeping: Option<bool>,
    /// Server load from /load endpoint. None if not available.
    pub server_load: Option<f64>,
    pub last_updated: Instant,

    // === Gauges (Runtime) ===
    pub requests_running: f64,
    pub requests_waiting: f64,
    pub kv_cache_usage: f64,

    // === Async Remote-KV Fetch Stages (`vllm:num_requests_kv_fetch_by_stage`) ===
    /// Requests with remote-fetch intent but no transfer started yet.
    pub kv_fetch_waiting_to_start: f64,
    /// Requests currently receiving remote KV.
    pub kv_fetch_in_progress: f64,
    /// Requests that finished receiving and are waiting to run.
    pub kv_fetch_completed_waiting: f64,
    /// Whether the kv-fetch stage gauge is exposed (for conditional UI display).
    pub has_kv_fetch: bool,

    // === Rates (from counters) ===
    pub prompt_tokens_total: u64,
    pub generation_tokens_total: u64,
    pub prompt_tps: f64,
    pub generation_tps: f64,
    pub requests_per_sec: f64,
    pub preemptions_per_sec: f64,
    pub preemptions_total: u64,

    // === Cache Stats ===
    pub prefix_cache_hit_rate: f64,
    pub external_cache_hit_rate: f64,
    pub mm_cache_hit_rate: f64,

    // === Token Stats ===
    pub prompt_tokens_by_source: Vec<(String, u64)>,
    pub prompt_tokens_cached_total: u64,
    pub prompt_tokens_recomputed_total: u64,

    // === HTTP ===
    pub http_qps: f64,
    pub http_error_rate: f64,
    pub http_requests_by_status: HashMap<String, u64>,

    // === Latency — cumulative (all-time) ===
    pub cum_ttft: LatencyPct,
    pub cum_itl: LatencyPct,
    pub cum_e2e: LatencyPct,
    pub cum_queue: LatencyPct,
    pub cum_prefill: LatencyPct,
    pub cum_decode: LatencyPct,
    pub cum_inference: LatencyPct,
    pub cum_tpot: LatencyPct,

    // === Latency — recent window (delta between two scrapes) ===
    pub win_ttft: LatencyPct,
    pub win_itl: LatencyPct,
    pub win_e2e: LatencyPct,
    pub win_queue: LatencyPct,
    pub win_prefill: LatencyPct,
    pub win_decode: LatencyPct,
    pub win_inference: LatencyPct,
    pub win_tpot: LatencyPct,

    // === Request Stats ===
    pub request_success_total: u64,
    pub success_by_reason: Vec<(String, u64)>,
    pub avg_prompt_tokens: f64,
    pub avg_generation_tokens: f64,
    pub avg_prefill_kv_computed: f64,
    pub iteration_tokens_mean: f64,
    /// Windowed p50/p99 for prompt token count distribution.
    pub win_prompt_tokens: LatencyPct,
    /// Windowed p50/p99 for generation token count distribution.
    pub win_generation_tokens: LatencyPct,

    // === Speculative Decoding ===
    pub spec_decode_draft_tokens_total: u64,
    pub spec_decode_accepted_tokens_total: u64,
    /// Windowed acceptance rate: delta(accepted) / delta(draft_tokens).
    pub spec_decode_acceptance_rate: f64,
    pub spec_decode_drafts_per_sec: f64,
    /// Per-position acceptance rate (windowed): position → rate.
    pub spec_decode_acceptance_per_pos: Vec<(u32, f64)>,

    // === Performance / MFU ===
    pub estimated_flops_per_gpu_per_sec: f64,
    pub estimated_read_bytes_per_gpu_per_sec: f64,
    pub estimated_write_bytes_per_gpu_per_sec: f64,
    /// MFU% = estimated TFLOP/s per GPU / peak TFLOP/s for the model's compute
    /// dtype (NVFP4 / FP8 / BF16, inferred from model name). Falls back to BF16
    /// peak when the dtype has no native tensor core path on this GPU.
    /// None if neither lookup yields a value.
    pub mfu_percent: Option<f64>,

    // === KV Cache Residency (--kv-cache-metrics) ===
    /// Windowed p50/p99 of KV block lifetime (ms).
    pub kv_block_lifetime: LatencyPct,
    /// Windowed p50/p99 of idle time before eviction (ms).
    pub kv_block_idle_before_evict: LatencyPct,
    /// Windowed p50/p99 of reuse gap (ms).
    pub kv_block_reuse_gap: LatencyPct,
    /// Whether any kv-cache-metrics data is present.
    pub has_kv_block_metrics: bool,

    // === NIXL KV Connector ===
    pub nixl_failed_transfers_total: u64,
    pub nixl_failed_notifications_total: u64,
    pub nixl_kv_expired_reqs_total: u64,
    pub nixl_xfer_time: LatencyPct,
    pub nixl_post_time: LatencyPct,
    /// Windowed average bytes per transfer.
    pub nixl_avg_bytes_transferred: f64,
    /// Windowed average descriptors per transfer.
    pub nixl_avg_descriptors: f64,
    /// Successful transfers per second (from histogram count delta).
    pub nixl_transfers_per_sec: f64,
    /// Throughput in MB/s (bytes_transferred delta / xfer_time delta).
    pub nixl_throughput_mb_per_sec: f64,
    /// Cumulative prompt tokens received via external KV connector (Dynamo
    /// disagg P/D NIXL path), summed across vLLM's `prompt_tokens_by_source`
    /// with `source="external_kv_transfer"`.
    pub external_kv_transfer_tokens_total: u64,
    /// Per-second rate of `external_kv_transfer` prompt tokens. Acts as a
    /// fallback NIXL signal when vLLM's NIXL histogram count stays at 0
    /// despite real disagg traffic.
    pub external_kv_transfer_tokens_per_sec: f64,
    /// Whether any NIXL data is present (for conditional UI display).
    pub has_nixl: bool,

    // === Dynamo Frontend ===
    pub dynamo_context_length: u64,
    pub dynamo_total_kv_blocks: u64,
    pub dynamo_kv_block_size: u64,
    pub dynamo_max_num_seqs: u64,
    pub dynamo_max_num_batched_tokens: u64,
    pub dynamo_disconnected_clients: f64,
    /// Windowed tokenizer encode latency (ms).
    pub dynamo_tokenize_latency: LatencyPct,
    /// Windowed tokenizer decode latency (ms).
    pub dynamo_detokenize_latency: LatencyPct,
    /// Whether Dynamo-specific metrics are present.
    pub has_dynamo_config: bool,
    /// Dynamo worker role from `dynamo_component` label. None if not a Dynamo rank.
    /// Examples: "prefill", "backend" (decode); aggregated hosts may report "P", "D", "P+D".
    pub dynamo_role: Option<String>,
    /// Worker uptime in seconds (from `dynamo_component_uptime_seconds`). 0 if absent.
    pub dynamo_uptime_secs: f64,
    /// Cumulative `generate` request count on this worker.
    pub dynamo_component_requests_total: u64,
    /// Per-second rate of `generate` requests (windowed).
    pub dynamo_component_requests_per_sec: f64,
    /// Currently-inflight `generate` requests on this worker. `None` when
    /// the metric is absent (non-Dynamo deployments).
    pub dynamo_component_inflight: Option<f64>,
    /// Cold-start model load time in seconds. 0 if absent.
    pub dynamo_model_load_secs: f64,

    // === KV Cache Events (ZMQ) ===
    /// Per-DP-rank KV event metrics. Index = DP rank.
    pub kv_events: Option<Vec<KVEventMetrics>>,

    // === Per-engine breakdown (DP mode) ===
    /// Per-engine metrics, one per DP rank. None if single engine.
    pub engine_metrics: Option<Vec<NodeMetrics>>,

    // === GPU hardware metrics (from vmon agent) ===
    pub gpu_scrape: Option<GpuScrape>,

    // === InfiniBand metrics (from node_exporter) ===
    pub ib_scrape: Option<IbScrape>,
}

impl NodeMetrics {
    pub fn loading(addr: String) -> Self {
        let mut m = Self::offline(addr);
        m.is_loading = true;
        m
    }

    pub fn offline(addr: String) -> Self {
        Self {
            addr,
            model_name: String::new(),
            is_healthy: false,
            is_loading: false,
            is_sleeping: None,
            server_load: None,
            last_updated: Instant::now(),
            requests_running: 0.0,
            requests_waiting: 0.0,
            kv_cache_usage: 0.0,
            kv_fetch_waiting_to_start: 0.0,
            kv_fetch_in_progress: 0.0,
            kv_fetch_completed_waiting: 0.0,
            has_kv_fetch: false,
            prompt_tokens_total: 0,
            generation_tokens_total: 0,
            prompt_tps: 0.0,
            generation_tps: 0.0,
            requests_per_sec: 0.0,
            preemptions_per_sec: 0.0,
            preemptions_total: 0,
            prefix_cache_hit_rate: 0.0,
            external_cache_hit_rate: 0.0,
            mm_cache_hit_rate: 0.0,
            prompt_tokens_by_source: Vec::new(),
            prompt_tokens_cached_total: 0,
            prompt_tokens_recomputed_total: 0,
            http_qps: 0.0,
            http_error_rate: 0.0,
            http_requests_by_status: HashMap::new(),
            cum_ttft: LatencyPct::default(),
            cum_itl: LatencyPct::default(),
            cum_e2e: LatencyPct::default(),
            cum_queue: LatencyPct::default(),
            cum_prefill: LatencyPct::default(),
            cum_decode: LatencyPct::default(),
            cum_inference: LatencyPct::default(),
            cum_tpot: LatencyPct::default(),
            win_ttft: LatencyPct::default(),
            win_itl: LatencyPct::default(),
            win_e2e: LatencyPct::default(),
            win_queue: LatencyPct::default(),
            win_prefill: LatencyPct::default(),
            win_decode: LatencyPct::default(),
            win_inference: LatencyPct::default(),
            win_tpot: LatencyPct::default(),
            request_success_total: 0,
            success_by_reason: Vec::new(),
            avg_prompt_tokens: 0.0,
            avg_generation_tokens: 0.0,
            avg_prefill_kv_computed: 0.0,
            iteration_tokens_mean: 0.0,
            win_prompt_tokens: LatencyPct::default(),
            win_generation_tokens: LatencyPct::default(),
            spec_decode_draft_tokens_total: 0,
            spec_decode_accepted_tokens_total: 0,
            spec_decode_acceptance_rate: 0.0,
            spec_decode_drafts_per_sec: 0.0,
            spec_decode_acceptance_per_pos: Vec::new(),
            estimated_flops_per_gpu_per_sec: 0.0,
            estimated_read_bytes_per_gpu_per_sec: 0.0,
            estimated_write_bytes_per_gpu_per_sec: 0.0,
            mfu_percent: None,
            kv_block_lifetime: LatencyPct::default(),
            kv_block_idle_before_evict: LatencyPct::default(),
            kv_block_reuse_gap: LatencyPct::default(),
            has_kv_block_metrics: false,
            nixl_failed_transfers_total: 0,
            nixl_failed_notifications_total: 0,
            nixl_kv_expired_reqs_total: 0,
            nixl_xfer_time: LatencyPct::default(),
            nixl_post_time: LatencyPct::default(),
            nixl_avg_bytes_transferred: 0.0,
            nixl_avg_descriptors: 0.0,
            nixl_transfers_per_sec: 0.0,
            nixl_throughput_mb_per_sec: 0.0,
            external_kv_transfer_tokens_total: 0,
            external_kv_transfer_tokens_per_sec: 0.0,
            has_nixl: false,
            dynamo_context_length: 0,
            dynamo_total_kv_blocks: 0,
            dynamo_kv_block_size: 0,
            dynamo_max_num_seqs: 0,
            dynamo_max_num_batched_tokens: 0,
            dynamo_disconnected_clients: 0.0,
            dynamo_tokenize_latency: LatencyPct::default(),
            dynamo_detokenize_latency: LatencyPct::default(),
            has_dynamo_config: false,
            dynamo_role: None,
            dynamo_uptime_secs: 0.0,
            dynamo_component_requests_total: 0,
            dynamo_component_requests_per_sec: 0.0,
            dynamo_component_inflight: None,
            dynamo_model_load_secs: 0.0,
            kv_events: None,
            engine_metrics: None,
            gpu_scrape: None,
            ib_scrape: None,
        }
    }

    /// Reconstruct NodeMetrics from a serialized NodeSample (for replay mode).
    /// Fields not present in NodeSample are left at defaults.
    pub fn from_sample(addr: String, s: &crate::sample::NodeSample) -> Self {
        let mut m = Self::offline(addr);
        m.is_healthy = true;
        m.is_loading = false;
        m.kv_cache_usage = s.kv_cache;
        m.requests_running = s.running;
        m.requests_waiting = s.waiting;
        m.has_kv_fetch = s.has_kv_fetch;
        m.kv_fetch_waiting_to_start = s.kv_fetch_waiting_to_start;
        m.kv_fetch_in_progress = s.kv_fetch_in_progress;
        m.kv_fetch_completed_waiting = s.kv_fetch_completed_waiting;
        m.generation_tps = s.generation_tps;
        m.prompt_tps = s.prompt_tps;
        m.requests_per_sec = s.requests_per_sec;
        m.preemptions_per_sec = s.preemptions_per_sec;

        m.win_ttft = LatencyPct {
            p50: s.ttft_p50_ms,
            p90: s.ttft_p90_ms,
            p99: s.ttft_p99_ms,
            mean: s.ttft_mean_ms,
        };
        m.win_itl = LatencyPct {
            p50: s.itl_p50_ms,
            p90: s.itl_p90_ms,
            p99: s.itl_p99_ms,
            mean: s.itl_mean_ms,
        };
        m.win_e2e = LatencyPct {
            p50: s.e2e_p50_ms,
            p90: s.e2e_p90_ms,
            p99: s.e2e_p99_ms,
            mean: s.e2e_mean_ms,
        };
        m.win_queue = LatencyPct {
            p50: s.queue_p50_ms,
            p90: s.queue_p90_ms,
            p99: s.queue_p99_ms,
            mean: s.queue_mean_ms,
        };
        m.win_prefill = LatencyPct {
            p50: s.prefill_p50_ms,
            p90: s.prefill_p90_ms,
            p99: s.prefill_p99_ms,
            mean: s.prefill_mean_ms,
        };
        m.win_decode = LatencyPct {
            p50: s.decode_p50_ms,
            p90: s.decode_p90_ms,
            p99: s.decode_p99_ms,
            mean: s.decode_mean_ms,
        };
        m.win_inference = LatencyPct {
            p50: s.inference_p50_ms,
            p90: s.inference_p90_ms,
            p99: s.inference_p99_ms,
            mean: s.inference_mean_ms,
        };
        m.win_tpot = LatencyPct {
            p50: s.tpot_p50_ms,
            p90: s.tpot_p90_ms,
            p99: s.tpot_p99_ms,
            mean: s.tpot_mean_ms,
        };
        // Use windowed as cumulative too for replay (best we can do)
        m.cum_ttft = m.win_ttft.clone();
        m.cum_itl = m.win_itl.clone();
        m.cum_e2e = m.win_e2e.clone();
        m.cum_queue = m.win_queue.clone();
        m.cum_prefill = m.win_prefill.clone();
        m.cum_decode = m.win_decode.clone();
        m.cum_inference = m.win_inference.clone();
        m.cum_tpot = m.win_tpot.clone();

        m.prefix_cache_hit_rate = s.prefix_cache_hit_rate;
        m.external_cache_hit_rate = s.external_cache_hit_rate;
        m.mm_cache_hit_rate = s.mm_cache_hit_rate;
        m.spec_decode_acceptance_rate = s.spec_decode_acceptance_rate;
        m.spec_decode_drafts_per_sec = s.spec_decode_drafts_per_sec;
        m.estimated_flops_per_gpu_per_sec = s.estimated_flops_per_gpu_per_sec;
        m.estimated_read_bytes_per_gpu_per_sec = s.estimated_read_bytes_per_gpu_per_sec;
        m.estimated_write_bytes_per_gpu_per_sec = s.estimated_write_bytes_per_gpu_per_sec;
        m.mfu_percent = s.mfu_percent;
        m.avg_prompt_tokens = s.avg_prompt_tokens;
        m.avg_generation_tokens = s.avg_generation_tokens;
        m.iteration_tokens_mean = s.iteration_tokens_mean;
        m.http_qps = s.http_qps;
        m.http_error_rate = s.http_error_rate;
        m.server_load = s.server_load;

        // Reconstruct GPU data. Modern reports (v0.6+) carry the full per-GPU
        // array in `s.gpus`; we faithfully restore each `GpuMetrics`. Legacy
        // reports only have the aggregate `gpu_utilization` family — fall back
        // to synthesizing a single placeholder GPU at index 0 so the TUI still
        // shows aggregate data instead of nothing.
        if !s.gpus.is_empty() {
            use crate::gpu::{GpuMetrics, GpuScrape};
            let gpus: Vec<GpuMetrics> = s.gpus.iter().map(GpuMetrics::from).collect();
            m.gpu_scrape = Some(GpuScrape {
                gpus,
                cpu_percent: None,
                mem_used_bytes: None,
                mem_total_bytes: None,
            });
        } else if let Some(util) = s.gpu_utilization {
            use crate::gpu::{GpuMetrics, GpuScrape};
            let gpu = GpuMetrics {
                index: 0,
                utilization: util,
                mem_utilization: s.gpu_mem_utilization.unwrap_or(0.0),
                power_watts: s.gpu_power_watts.unwrap_or(0.0),
                temperature: s.gpu_temperature.unwrap_or(0.0),
                mem_used_bytes: s.gpu_vram_used_bytes.unwrap_or(0),
                mem_total_bytes: s.gpu_vram_total_bytes.unwrap_or(0),
                nvlink_tx_kbps: s.gpu_nvlink_tx_kbps.unwrap_or(0),
                nvlink_rx_kbps: s.gpu_nvlink_rx_kbps.unwrap_or(0),
                ..Default::default()
            };
            m.gpu_scrape = Some(GpuScrape {
                gpus: vec![gpu],
                cpu_percent: None,
                mem_used_bytes: None,
                mem_total_bytes: None,
            });
        }

        // Reconstruct InfiniBand data when the sample carried per-device rows.
        // Pre-v0.6 reports lack `ibs[]` entirely; aggregate-only IB fields
        // are not enough to rebuild the device list, so we leave `ib_scrape`
        // as `None` in that case.
        if !s.ibs.is_empty() {
            use crate::ib::{IbDevice, IbScrape};
            let devices: Vec<IbDevice> = s.ibs.iter().map(IbDevice::from).collect();
            m.ib_scrape = Some(IbScrape { devices });
        }

        // Dynamo frontend / per-worker telemetry.
        m.dynamo_role = s.dynamo_role.clone();
        m.has_dynamo_config = s.has_dynamo_config;
        m.dynamo_uptime_secs = s.dynamo_uptime_secs;
        m.dynamo_component_inflight = s.dynamo_component_inflight;
        m.dynamo_component_requests_per_sec = s.dynamo_component_requests_per_sec;
        m.dynamo_context_length = s.dynamo_context_length;
        m.dynamo_total_kv_blocks = s.dynamo_total_kv_blocks;
        m.dynamo_kv_block_size = s.dynamo_kv_block_size;
        m.dynamo_max_num_seqs = s.dynamo_max_num_seqs;
        m.dynamo_max_num_batched_tokens = s.dynamo_max_num_batched_tokens;
        m.dynamo_disconnected_clients = s.dynamo_disconnected_clients;
        m.dynamo_model_load_secs = s.dynamo_model_load_secs;
        m.dynamo_tokenize_latency = LatencyPct {
            p50: s.dynamo_tokenize_p50_ms,
            p90: 0.0,
            p99: s.dynamo_tokenize_p99_ms,
            mean: 0.0,
        };
        m.dynamo_detokenize_latency = LatencyPct {
            p50: s.dynamo_detokenize_p50_ms,
            p90: 0.0,
            p99: s.dynamo_detokenize_p99_ms,
            mean: 0.0,
        };

        // NIXL KV connector.
        m.has_nixl = s.has_nixl;
        m.nixl_failed_transfers_total = s.nixl_failed_transfers_total;
        m.nixl_failed_notifications_total = s.nixl_failed_notifications_total;
        m.nixl_kv_expired_reqs_total = s.nixl_kv_expired_reqs_total;
        m.nixl_xfer_time = LatencyPct {
            p50: s.nixl_xfer_p50_ms,
            p90: 0.0,
            p99: s.nixl_xfer_p99_ms,
            mean: 0.0,
        };
        m.nixl_post_time = LatencyPct {
            p50: s.nixl_post_p50_ms,
            p90: 0.0,
            p99: s.nixl_post_p99_ms,
            mean: 0.0,
        };
        m.nixl_avg_bytes_transferred = s.nixl_avg_bytes_transferred;
        m.nixl_avg_descriptors = s.nixl_avg_descriptors;
        m.nixl_transfers_per_sec = s.nixl_transfers_per_sec;
        m.nixl_throughput_mb_per_sec = s.nixl_throughput_mb_per_sec;
        m.external_kv_transfer_tokens_per_sec = s.external_kv_transfer_tokens_per_sec;

        // KV cache block residency.
        m.has_kv_block_metrics = s.has_kv_block_metrics;
        m.kv_block_lifetime = LatencyPct {
            p50: s.kv_block_lifetime_p50_ms,
            p90: 0.0,
            p99: s.kv_block_lifetime_p99_ms,
            mean: 0.0,
        };
        m.kv_block_idle_before_evict = LatencyPct {
            p50: s.kv_block_idle_before_evict_p50_ms,
            p90: 0.0,
            p99: s.kv_block_idle_before_evict_p99_ms,
            mean: 0.0,
        };
        m.kv_block_reuse_gap = LatencyPct {
            p50: s.kv_block_reuse_gap_p50_ms,
            p90: 0.0,
            p99: s.kv_block_reuse_gap_p99_ms,
            mean: 0.0,
        };

        // KV cache events (ZMQ).
        m.kv_events = s.kv_events.as_ref().map(|v| v.iter().map(KVEventMetrics::from).collect());

        // Per-engine sub-rows (DP rank / Dynamo). Recursively rebuild each
        // sub-row; we don't have a separate addr per rank in the JSON, so
        // reuse the parent host addr (matches the live-mode invariant where
        // each engine's NodeMetrics is computed with the parent's addr).
        m.engine_metrics = s.engine_metrics.as_ref().map(|engines| {
            engines.iter().map(|e| NodeMetrics::from_sample(m.addr.clone(), e)).collect()
        });

        m
    }

    /// Fold per-rank NodeMetrics for one host into a single host-level
    /// NodeMetrics with `engine_metrics = ranks`. Used in Dynamo mode where
    /// each rank port is a separate process on the same host.
    ///
    /// - Rates and counters are summed (parallel workers).
    /// - Gauges and ratios are averaged across healthy ranks.
    /// - Latency p99 is the max across ranks (worst-case); p50 averaged.
    /// - The host-level addr is the bare hostname (port stripped).
    /// - `gpu_scrape` and `ib_scrape` should be attached by the caller after
    ///   folding (they are per-host, not per-rank).
    pub fn aggregate_dynamo_ranks(host: String, ranks: Vec<NodeMetrics>) -> NodeMetrics {
        if ranks.is_empty() {
            return NodeMetrics::offline(host);
        }
        let n_total = ranks.len() as f64;
        let healthy: Vec<&NodeMetrics> = ranks.iter().filter(|r| r.is_healthy).collect();
        let n_healthy = healthy.len();
        let n_h = n_healthy.max(1) as f64;

        let mut m = ranks[0].clone();
        m.addr = host;
        m.is_healthy = n_healthy > 0;
        m.is_loading = ranks.iter().any(|r| r.is_loading) && n_healthy == 0;

        // Backend and co-located frontend inflight counts overlap. Prefer
        // worker counts so each request contributes once to host concurrency.
        let host_has_worker_inflight = ranks.iter().any(|r| r.dynamo_component_inflight.is_some());
        // A tensor-parallel follower exposes runtime uptime but no request
        // handler or model configuration. Its frontend's load-balancer share
        // does not represent local request ownership, so report zero inflight.
        let host_is_follower = !host_has_worker_inflight
            && ranks.iter().any(|r| {
                r.dynamo_uptime_secs > 0.0
                    && r.dynamo_component_inflight.is_none()
                    && r.dynamo_max_num_seqs == 0
            });
        m.requests_running = if host_is_follower {
            0.0
        } else if host_has_worker_inflight {
            ranks
                .iter()
                .filter(|r| r.dynamo_component_inflight.is_some())
                .map(|r| r.requests_running)
                .sum()
        } else {
            ranks.iter().map(|r| r.requests_running).sum()
        };
        m.requests_waiting = if host_is_follower {
            0.0
        } else {
            ranks.iter().map(|r| r.requests_waiting).sum()
        };
        m.kv_fetch_waiting_to_start = ranks.iter().map(|r| r.kv_fetch_waiting_to_start).sum();
        m.kv_fetch_in_progress = ranks.iter().map(|r| r.kv_fetch_in_progress).sum();
        m.kv_fetch_completed_waiting = ranks.iter().map(|r| r.kv_fetch_completed_waiting).sum();
        m.has_kv_fetch = ranks.iter().any(|r| r.has_kv_fetch);
        m.prompt_tps = ranks.iter().map(|r| r.prompt_tps).sum();
        m.generation_tps = ranks.iter().map(|r| r.generation_tps).sum();
        m.requests_per_sec = ranks.iter().map(|r| r.requests_per_sec).sum();
        m.preemptions_per_sec = ranks.iter().map(|r| r.preemptions_per_sec).sum();
        m.http_qps = ranks.iter().map(|r| r.http_qps).sum();
        m.prompt_tokens_total = ranks.iter().map(|r| r.prompt_tokens_total).sum();
        m.generation_tokens_total = ranks.iter().map(|r| r.generation_tokens_total).sum();
        m.prompt_tokens_cached_total = ranks.iter().map(|r| r.prompt_tokens_cached_total).sum();
        m.prompt_tokens_recomputed_total =
            ranks.iter().map(|r| r.prompt_tokens_recomputed_total).sum();
        m.preemptions_total = ranks.iter().map(|r| r.preemptions_total).sum();
        m.request_success_total = ranks.iter().map(|r| r.request_success_total).sum();
        m.spec_decode_draft_tokens_total =
            ranks.iter().map(|r| r.spec_decode_draft_tokens_total).sum();
        m.spec_decode_accepted_tokens_total =
            ranks.iter().map(|r| r.spec_decode_accepted_tokens_total).sum();
        m.spec_decode_drafts_per_sec = ranks.iter().map(|r| r.spec_decode_drafts_per_sec).sum();

        // NIXL: counters sum, throughput sums (per-host total)
        m.nixl_failed_transfers_total = ranks.iter().map(|r| r.nixl_failed_transfers_total).sum();
        m.nixl_failed_notifications_total =
            ranks.iter().map(|r| r.nixl_failed_notifications_total).sum();
        m.nixl_kv_expired_reqs_total = ranks.iter().map(|r| r.nixl_kv_expired_reqs_total).sum();
        m.nixl_transfers_per_sec = ranks.iter().map(|r| r.nixl_transfers_per_sec).sum();
        m.nixl_throughput_mb_per_sec = ranks.iter().map(|r| r.nixl_throughput_mb_per_sec).sum();
        m.external_kv_transfer_tokens_total =
            ranks.iter().map(|r| r.external_kv_transfer_tokens_total).sum();
        m.external_kv_transfer_tokens_per_sec =
            ranks.iter().map(|r| r.external_kv_transfer_tokens_per_sec).sum();
        m.has_nixl = ranks.iter().any(|r| r.has_nixl);

        // Estimated GPU rates: sum across ranks (each rank covers its own GPU)
        m.estimated_flops_per_gpu_per_sec =
            ranks.iter().map(|r| r.estimated_flops_per_gpu_per_sec).sum::<f64>() / n_total;
        m.estimated_read_bytes_per_gpu_per_sec =
            ranks.iter().map(|r| r.estimated_read_bytes_per_gpu_per_sec).sum::<f64>() / n_total;
        m.estimated_write_bytes_per_gpu_per_sec =
            ranks.iter().map(|r| r.estimated_write_bytes_per_gpu_per_sec).sum::<f64>() / n_total;

        // Average: gauges, ratios (over healthy ranks)
        let avg = |f: fn(&NodeMetrics) -> f64| -> f64 {
            if n_healthy == 0 {
                return 0.0;
            }
            healthy.iter().map(|r| f(r)).sum::<f64>() / n_h
        };
        m.kv_cache_usage = avg(|r| r.kv_cache_usage);
        m.prefix_cache_hit_rate = avg(|r| r.prefix_cache_hit_rate);
        m.external_cache_hit_rate = avg(|r| r.external_cache_hit_rate);
        m.mm_cache_hit_rate = avg(|r| r.mm_cache_hit_rate);
        m.http_error_rate = avg(|r| r.http_error_rate);
        m.spec_decode_acceptance_rate = avg(|r| r.spec_decode_acceptance_rate);
        m.iteration_tokens_mean = avg(|r| r.iteration_tokens_mean);
        m.avg_prompt_tokens = avg(|r| r.avg_prompt_tokens);
        m.avg_generation_tokens = avg(|r| r.avg_generation_tokens);
        m.avg_prefill_kv_computed = avg(|r| r.avg_prefill_kv_computed);

        // Optional gauges: average if any rank reports a value
        let avg_opt = |f: fn(&NodeMetrics) -> Option<f64>| -> Option<f64> {
            let vals: Vec<f64> = ranks.iter().filter_map(&f).collect();
            if vals.is_empty() {
                None
            } else {
                Some(vals.iter().sum::<f64>() / vals.len() as f64)
            }
        };
        m.mfu_percent = avg_opt(|r| r.mfu_percent);
        m.server_load = avg_opt(|r| r.server_load);

        // Latencies: p90/p99 = max across ranks (worst case); p50/mean = avg of healthy
        let combine_lat = |get: fn(&NodeMetrics) -> &LatencyPct| -> LatencyPct {
            let mut p90 = 0.0_f64;
            let mut p99 = 0.0_f64;
            let mut p50_sum = 0.0_f64;
            let mut mean_sum = 0.0_f64;
            let mut n_nonzero = 0;
            for r in &healthy {
                let l = get(r);
                if l.p90 > p90 {
                    p90 = l.p90;
                }
                if l.p99 > p99 {
                    p99 = l.p99;
                }
                if l.p50 > 0.0 || l.mean > 0.0 || l.p99 > 0.0 {
                    p50_sum += l.p50;
                    mean_sum += l.mean;
                    n_nonzero += 1;
                }
            }
            let denom = n_nonzero.max(1) as f64;
            LatencyPct {
                p50: p50_sum / denom,
                p90,
                p99,
                mean: mean_sum / denom,
            }
        };
        m.cum_ttft = combine_lat(|r| &r.cum_ttft);
        m.cum_itl = combine_lat(|r| &r.cum_itl);
        m.cum_e2e = combine_lat(|r| &r.cum_e2e);
        m.cum_queue = combine_lat(|r| &r.cum_queue);
        m.cum_prefill = combine_lat(|r| &r.cum_prefill);
        m.cum_decode = combine_lat(|r| &r.cum_decode);
        m.cum_inference = combine_lat(|r| &r.cum_inference);
        m.cum_tpot = combine_lat(|r| &r.cum_tpot);
        m.win_ttft = combine_lat(|r| &r.win_ttft);
        m.win_itl = combine_lat(|r| &r.win_itl);
        m.win_e2e = combine_lat(|r| &r.win_e2e);
        m.win_queue = combine_lat(|r| &r.win_queue);
        m.win_prefill = combine_lat(|r| &r.win_prefill);
        m.win_decode = combine_lat(|r| &r.win_decode);
        m.win_inference = combine_lat(|r| &r.win_inference);
        m.win_tpot = combine_lat(|r| &r.win_tpot);
        m.win_prompt_tokens = combine_lat(|r| &r.win_prompt_tokens);
        m.win_generation_tokens = combine_lat(|r| &r.win_generation_tokens);
        m.nixl_xfer_time = combine_lat(|r| &r.nixl_xfer_time);
        m.nixl_post_time = combine_lat(|r| &r.nixl_post_time);

        // NIXL averages: weight by transfers/sec to keep "average per transfer" semantics
        let total_xfer_per_sec: f64 = ranks.iter().map(|r| r.nixl_transfers_per_sec).sum();
        if total_xfer_per_sec > 0.0 {
            m.nixl_avg_bytes_transferred = ranks
                .iter()
                .map(|r| r.nixl_avg_bytes_transferred * r.nixl_transfers_per_sec)
                .sum::<f64>()
                / total_xfer_per_sec;
            m.nixl_avg_descriptors = ranks
                .iter()
                .map(|r| r.nixl_avg_descriptors * r.nixl_transfers_per_sec)
                .sum::<f64>()
                / total_xfer_per_sec;
        } else {
            m.nixl_avg_bytes_transferred = 0.0;
            m.nixl_avg_descriptors = 0.0;
        }

        // HashMaps: sum by key
        m.http_requests_by_status = HashMap::new();
        for r in &ranks {
            for (k, v) in &r.http_requests_by_status {
                *m.http_requests_by_status.entry(k.clone()).or_insert(0) += v;
            }
        }
        // Vec<(String, u64)>: sum by key, keep stable order from rank0
        let sum_by_key = |get: fn(&NodeMetrics) -> &Vec<(String, u64)>| -> Vec<(String, u64)> {
            let mut acc: HashMap<String, u64> = HashMap::new();
            for r in &ranks {
                for (k, v) in get(r) {
                    *acc.entry(k.clone()).or_insert(0) += v;
                }
            }
            let mut out: Vec<(String, u64)> = acc.into_iter().collect();
            out.sort_by_key(|b| std::cmp::Reverse(b.1));
            out
        };
        m.success_by_reason = sum_by_key(|r| &r.success_by_reason);
        m.prompt_tokens_by_source = sum_by_key(|r| &r.prompt_tokens_by_source);

        // Sleep: any rank sleeping → host shows sleeping; if no rank reports, None
        let sleep_vals: Vec<bool> = ranks.iter().filter_map(|r| r.is_sleeping).collect();
        m.is_sleeping = if sleep_vals.is_empty() {
            None
        } else {
            Some(sleep_vals.iter().any(|&v| v))
        };

        // Role: collapse rank dynamo_component values into a host badge.
        //   all "prefill"  → "P"
        //   all "backend"  → "D"
        //   mixed          → "P+D"
        let mut role_set: std::collections::BTreeSet<&str> = Default::default();
        for r in &ranks {
            if let Some(role) = r.dynamo_role.as_deref() {
                role_set.insert(role);
            }
        }
        // "backend" is ambiguous: Dynamo names the aggregated worker (does
        // both prefill and decode) "backend", but PD deployments reuse the
        // same component name for decode workers — only prefill workers get a
        // distinct label. Prefill ranks on the same host prove this is the
        // decode side → "D"; otherwise keep the raw label so
        // `ClusterState::resolve_backend_roles()` can decide from
        // deployment-wide context.
        let has_prefill = role_set.contains("prefill");
        let mut badges: Vec<&str> = role_set
            .iter()
            .map(|r| match *r {
                "prefill" => "P",
                "decode" => "D",
                "backend" if has_prefill => "D",
                other => other,
            })
            .collect();
        badges.sort_by_key(|b| match *b {
            "P" => 0,
            "D" => 1,
            _ => 2,
        });
        badges.dedup();
        m.dynamo_role = if badges.is_empty() {
            None
        } else {
            Some(badges.join("+"))
        };

        // Component counters: sum across ranks for total/inflight/requests_per_sec;
        // uptime = min (worst-case "youngest" rank, since restarts reset only that rank);
        // model_load = max (slowest cold-start across ranks).
        m.dynamo_component_requests_total =
            ranks.iter().map(|r| r.dynamo_component_requests_total).sum();
        m.dynamo_component_requests_per_sec =
            ranks.iter().map(|r| r.dynamo_component_requests_per_sec).sum();
        // Sum across ranks where present; None if no rank reported it.
        let inflight_present: Vec<f64> =
            ranks.iter().filter_map(|r| r.dynamo_component_inflight).collect();
        m.dynamo_component_inflight = if inflight_present.is_empty() {
            None
        } else {
            Some(inflight_present.iter().sum())
        };
        m.dynamo_uptime_secs = ranks
            .iter()
            .map(|r| r.dynamo_uptime_secs)
            .filter(|v| *v > 0.0)
            .fold(f64::INFINITY, f64::min);
        if !m.dynamo_uptime_secs.is_finite() {
            m.dynamo_uptime_secs = 0.0;
        }
        m.dynamo_model_load_secs =
            ranks.iter().map(|r| r.dynamo_model_load_secs).fold(0.0_f64, f64::max);

        // Set engine_metrics to the worker rank list. Drop frontend ranks
        // (typically port 8180): a frontend exposes `dynamo_frontend_*`
        // gauges (so dynamo_max_num_seqs > 0) and is never assigned a worker
        // role (extraction ignores its KV-router observer labels, so
        // dynamo_role is None). Its stats are already folded into the
        // host-row aggregate above. Keeping it in engine_metrics would inflate
        // the per-engine sub-row count past gpu_count and cause the TUI to
        // render a trailing sub-row with no matching GPU. Drop their own
        // gpu_scrape — the host-level metric holds the full GpuScrape;
        // engine_sub_row reads GPUs from the parent.
        // An endpoint may itself front multiple DP engines (one Dynamo system
        // port serving dp_local_size ranks, e.g. `--data-parallel-size-local 4`
        // behind a single DYN_SYSTEM_PORT). Its own NodeMetrics is the
        // endpoint-level aggregate, and the real per-rank breakdown was
        // already computed into its inner `engine_metrics` from the `engine`
        // labels. Surface those inner entries as the host's sub-rows —
        // keeping the endpoint clone instead would render every per-rank
        // sub-row identical to the host row.
        let mut rank_clones: Vec<NodeMetrics> = Vec::new();
        for mut r in ranks
            .into_iter()
            .filter(|r| !(r.dynamo_role.is_none() && r.dynamo_max_num_seqs > 0))
        {
            match r.engine_metrics.take() {
                Some(inner) if !inner.is_empty() => {
                    for mut e in inner {
                        // Inner engines are filtered to samples carrying the
                        // engine label; Dynamo runtime families may not, so
                        // inherit the endpoint's role for the sub-row badge.
                        if e.dynamo_role.is_none() {
                            e.dynamo_role = r.dynamo_role.clone();
                        }
                        rank_clones.push(e);
                    }
                }
                _ => rank_clones.push(r),
            }
        }
        for r in &mut rank_clones {
            r.gpu_scrape = None;
            r.ib_scrape = None;
        }
        m.engine_metrics = if rank_clones.is_empty() {
            None
        } else {
            Some(rank_clones)
        };
        m.gpu_scrape = None;
        m.ib_scrape = None;

        m
    }

    /// Compute MFU% if we have both a FLOPs rate and a GPU model name.
    /// Peak is chosen by the model's compute dtype (NVFP4 / FP8 / BF16),
    /// inferred from model name. If the dtype has no native path on this
    /// GPU (e.g. H200 + NVFP4), fall back to BF16 peak so the user still
    /// gets a (conservative-upper-bound) MFU% rather than a missing value.
    /// Leaves `mfu_percent` untouched when either input is unavailable.
    pub fn recompute_mfu(&mut self) {
        if self.estimated_flops_per_gpu_per_sec <= 0.0 {
            return;
        }
        let gpu_name = self
            .gpu_scrape
            .as_ref()
            .and_then(|gs| gs.gpus.first())
            .map(|g| g.name.as_str())
            .unwrap_or("");
        let dtype = crate::gpu::infer_compute_dtype(&self.model_name);
        let peak_tflops_opt = crate::gpu::peak_tflops(gpu_name, dtype).or_else(|| {
            if dtype != crate::gpu::ComputeDtype::Bf16 {
                tracing::debug!(
                    gpu = %gpu_name,
                    model = %self.model_name,
                    ?dtype,
                    "MFU%: dtype peak unavailable, falling back to BF16 peak"
                );
                crate::gpu::peak_tflops(gpu_name, crate::gpu::ComputeDtype::Bf16)
            } else {
                None
            }
        });
        if let Some(peak_tflops) = peak_tflops_opt {
            let estimated_tflops = self.estimated_flops_per_gpu_per_sec / 1e12;
            self.mfu_percent = Some(estimated_tflops / peak_tflops * 100.0);
        }
    }
}

/// Re-attribute DP engines that a single "head" endpoint publishes for a
/// whole multi-node data-parallel deployment onto the peer nodes that
/// actually run them.
///
/// vLLM multi-node DP (API server + `--headless` secondaries) centralizes
/// Prometheus metrics on the head node: `engine="0"..="N-1"` all appear on
/// the head's `/metrics`, while secondary nodes expose no HTTP endpoint at
/// all. Without this pass the TUI renders one node with N engine sub-rows
/// and permanently-offline peer rows.
///
/// This detects that shape and slices the head's engines evenly across the
/// group, rebuilding each member's row from its slice (via
/// [`NodeMetrics::aggregate_dynamo_ranks`]) so per-node throughput / KV usage
/// line up with each host's own GPU hardware metrics.
///
/// A group is regrouped only when ALL of the following hold (deliberately
/// conservative — a wrong match would paint a dead node green):
/// - ≥ 2 nodes share the same `:port`;
/// - exactly one of them (the head) is healthy and publishes ≥ 2 engines;
/// - every other member has never scraped OK since vmon started
///   (`ever_healthy`) — a peer that served metrics before is a crashed
///   standalone node, not a headless secondary;
/// - the engine count divides evenly by the group size;
/// - when the head reports GPU hardware, the engine count exceeds its local
///   GPU count (i.e. the engines physically can't all live on the head).
///
/// Engine→node attribution assumes contiguous global ranks in address order
/// with the head first: rank 0 lives on the head, and secondaries take the
/// following slices in sorted-address order — matching how multi-node DP
/// jobs are normally launched (`--data-parallel-start-rank` increasing with
/// node order).
pub fn regroup_dp_engines(nodes: &mut [NodeMetrics], ever_healthy: &HashSet<String>) {
    let mut by_port: HashMap<String, Vec<usize>> = HashMap::new();
    for (i, n) in nodes.iter().enumerate() {
        if let Some((_, port)) = n.addr.rsplit_once(':') {
            by_port.entry(port.to_string()).or_default().push(i);
        }
    }
    for group in by_port.into_values() {
        if group.len() < 2 {
            continue;
        }
        let heads: Vec<usize> = group
            .iter()
            .copied()
            .filter(|&i| {
                nodes[i].is_healthy && nodes[i].engine_metrics.as_ref().map_or(0, |e| e.len()) >= 2
            })
            .collect();
        if heads.len() != 1 {
            continue;
        }
        let head = heads[0];
        if group
            .iter()
            .any(|&i| i != head && (nodes[i].is_healthy || ever_healthy.contains(&nodes[i].addr)))
        {
            continue;
        }
        let e_count = nodes[head].engine_metrics.as_ref().map_or(0, |e| e.len());
        if !e_count.is_multiple_of(group.len()) {
            continue;
        }
        if let Some(gs) = &nodes[head].gpu_scrape {
            if !gs.gpus.is_empty() && e_count <= gs.gpus.len() {
                continue;
            }
        }
        let per_node = e_count / group.len();

        // Head first (owns rank 0), then peers in address order.
        let mut order: Vec<usize> = vec![head];
        let mut peers: Vec<usize> = group.iter().copied().filter(|&i| i != head).collect();
        peers.sort_by(|&a, &b| nodes[a].addr.cmp(&nodes[b].addr));
        order.extend(peers);

        tracing::debug!(
            head = %nodes[head].addr,
            engines = e_count,
            nodes = group.len(),
            per_node,
            "regrouping centralized DP engines across peer nodes"
        );

        let engines = nodes[head].engine_metrics.take().unwrap_or_default();
        let head_row = nodes[head].clone();
        for (slot, &i) in order.iter().enumerate() {
            let slice = engines[slot * per_node..(slot + 1) * per_node].to_vec();
            let mut folded = NodeMetrics::aggregate_dynamo_ranks(nodes[i].addr.clone(), slice);
            folded.gpu_scrape = nodes[i].gpu_scrape.take();
            folded.ib_scrape = nodes[i].ib_scrape.take();
            // ZMQ KV events arrive per DP rank on the head — slice alongside
            // the engines. Each entry keeps its global dp_rank label.
            if let Some(ev) = &head_row.kv_events {
                if ev.len() == e_count {
                    folded.kv_events = Some(ev[slot * per_node..(slot + 1) * per_node].to_vec());
                }
            }
            if folded.model_name.is_empty() {
                folded.model_name = head_row.model_name.clone();
            }
            if i == head {
                // API-server-level signals exist only on the head endpoint;
                // per-engine samples never carry them.
                folded.http_qps = head_row.http_qps;
                folded.http_error_rate = head_row.http_error_rate;
                folded.http_requests_by_status = head_row.http_requests_by_status.clone();
                folded.server_load = head_row.server_load;
                folded.is_sleeping = head_row.is_sleeping;
            }
            folded.recompute_mfu();
            nodes[i] = folded;
        }
    }
}

// ── Rate calculation state ──

/// Bundles all rate calculators and previous scrape for delta computations.
/// Used for both aggregate and per-engine tracking.
struct RateState {
    prev_scrape: Option<VllmScrape>,
    prompt_rate: RateCalc,
    gen_rate: RateCalc,
    req_rate: RateCalc,
    preempt_rate: RateCalc,
    http_rate: RateCalc,
    /// Windowed ratio of non-2xx HTTP responses. Cumulative `1 - ok/total`
    /// hides recent error spikes once total volume is large; this version
    /// reports the delta since the previous scrape.
    http_err_ratio: RatioCalc,
    cache_ratio: RatioCalc,
    ext_cache_ratio: RatioCalc,
    mm_cache_ratio: RatioCalc,
    spec_draft_rate: RateCalc,
    spec_accept_ratio: RatioCalc,
    spec_accept_per_pos: HashMap<u32, RatioCalc>,
    flops_rate: RateCalc,
    read_bw_rate: RateCalc,
    write_bw_rate: RateCalc,
    nixl_xfer_rate: RateCalc,
    external_kv_rate: RateCalc,
    dynamo_component_req_rate: RateCalc,
    /// Cached previous delta_mean values — held when no new samples arrive.
    prev_avg_prompt_tokens: f64,
    prev_avg_generation_tokens: f64,
    prev_iteration_tokens_mean: f64,
    prev_avg_prefill_kv_computed: f64,
}

impl RateState {
    fn new() -> Self {
        Self {
            prev_scrape: None,
            prompt_rate: RateCalc::new(),
            gen_rate: RateCalc::new(),
            req_rate: RateCalc::new(),
            preempt_rate: RateCalc::new(),
            http_rate: RateCalc::new(),
            http_err_ratio: RatioCalc::new(),
            cache_ratio: RatioCalc::new(),
            ext_cache_ratio: RatioCalc::new(),
            mm_cache_ratio: RatioCalc::new(),
            spec_draft_rate: RateCalc::new(),
            spec_accept_ratio: RatioCalc::new(),
            spec_accept_per_pos: HashMap::new(),
            flops_rate: RateCalc::new(),
            read_bw_rate: RateCalc::new(),
            write_bw_rate: RateCalc::new(),
            nixl_xfer_rate: RateCalc::new(),
            external_kv_rate: RateCalc::new(),
            dynamo_component_req_rate: RateCalc::new(),
            prev_avg_prompt_tokens: 0.0,
            prev_avg_generation_tokens: 0.0,
            prev_iteration_tokens_mean: 0.0,
            prev_avg_prefill_kv_computed: 0.0,
        }
    }

    /// Compute NodeMetrics from a scrape, updating rates and delta histograms.
    fn compute(
        &mut self,
        addr: &str,
        scrape: &VllmScrape,
        kv_events: Option<Vec<KVEventMetrics>>,
    ) -> NodeMetrics {
        let now = scrape.scraped_at;

        let prompt_tps = self.prompt_rate.update(scrape.prompt_tokens_total, now).unwrap_or(0.0);
        let generation_tps =
            self.gen_rate.update(scrape.generation_tokens_total, now).unwrap_or(0.0);
        let requests_per_sec_vllm =
            self.req_rate.update(scrape.request_success_total, now).unwrap_or(0.0);
        let preemptions_per_sec =
            self.preempt_rate.update(scrape.num_preemptions_total, now).unwrap_or(0.0);
        let prefix_cache_hit_rate = self
            .cache_ratio
            .update(
                scrape.prefix_cache_hits_total,
                scrape.prefix_cache_queries_total,
            )
            .unwrap_or(0.0);
        let external_cache_hit_rate = self
            .ext_cache_ratio
            .update(
                scrape.external_prefix_cache_hits_total,
                scrape.external_prefix_cache_queries_total,
            )
            .unwrap_or(0.0);
        let mm_cache_hit_rate = self
            .mm_cache_ratio
            .update(scrape.mm_cache_hits_total, scrape.mm_cache_queries_total)
            .unwrap_or(0.0);
        let spec_decode_drafts_per_sec = self
            .spec_draft_rate
            .update(scrape.spec_decode_num_drafts_total, now)
            .unwrap_or(0.0);
        let spec_decode_acceptance_rate = self
            .spec_accept_ratio
            .update(
                scrape.spec_decode_num_accepted_tokens_total,
                scrape.spec_decode_num_draft_tokens_total,
            )
            .unwrap_or(0.0);
        // Per-position acceptance rate: accepted_per_pos[i] / num_drafts
        let mut spec_decode_acceptance_per_pos: Vec<(u32, f64)> = Vec::new();
        let drafts = scrape.spec_decode_num_drafts_total;
        for &(pos, accepted) in &scrape.spec_decode_accepted_per_pos {
            let calc = self.spec_accept_per_pos.entry(pos).or_default();
            if let Some(rate) = calc.update(accepted, drafts) {
                spec_decode_acceptance_per_pos.push((pos, rate));
            }
        }
        let estimated_flops_per_gpu_per_sec =
            self.flops_rate.update(scrape.estimated_flops_per_gpu_total, now).unwrap_or(0.0);
        let estimated_read_bytes_per_gpu_per_sec = self
            .read_bw_rate
            .update(scrape.estimated_read_bytes_per_gpu_total, now)
            .unwrap_or(0.0);
        let estimated_write_bytes_per_gpu_per_sec = self
            .write_bw_rate
            .update(scrape.estimated_write_bytes_per_gpu_total, now)
            .unwrap_or(0.0);
        let http_qps = self.http_rate.update(scrape.http_requests_all, now).unwrap_or(0.0);

        // Windowed error rate: errors / total over the delta since the previous
        // scrape. Use a RatioCalc so RatioCalc handles counter resets and
        // empty-window cases uniformly with the rest of the file.
        let http_error_rate = {
            let ok = scrape.http_requests_by_status.get("2xx").copied().unwrap_or(0);
            let total = scrape.http_requests_all;
            // RatioCalc::update takes (numerator, denominator); we want
            // errors/total, so pass (total - ok, total). Saturating sub guards
            // against the small race where 2xx counter advances ahead of the
            // aggregate (different scrape order in vLLM's instrumentator).
            let errors = total.saturating_sub(ok);
            self.http_err_ratio.update(errors, total).unwrap_or(0.0)
        };

        // Cumulative latency percentiles (all-time)
        let cum_ttft = pct_ms(&scrape.ttft);
        let cum_itl = pct_ms(&scrape.itl);
        let cum_e2e = pct_ms(&scrape.e2e_latency);
        let cum_queue = pct_ms(&scrape.queue_time);
        let cum_prefill = pct_ms(&scrape.prefill_time);
        let cum_decode = pct_ms(&scrape.decode_time);
        let cum_inference = pct_ms(&scrape.inference_time);
        let cum_tpot = pct_ms(&scrape.time_per_output_token);

        // Windowed latency percentiles (delta between two scrapes)
        let win_ttft = self.delta_pct_ms(&scrape.ttft, |s| &s.ttft);
        let win_itl = self.delta_pct_ms(&scrape.itl, |s| &s.itl);
        let win_e2e = self.delta_pct_ms(&scrape.e2e_latency, |s| &s.e2e_latency);
        let win_queue = self.delta_pct_ms(&scrape.queue_time, |s| &s.queue_time);
        let win_prefill = self.delta_pct_ms(&scrape.prefill_time, |s| &s.prefill_time);
        let win_decode = self.delta_pct_ms(&scrape.decode_time, |s| &s.decode_time);
        let win_inference = self.delta_pct_ms(&scrape.inference_time, |s| &s.inference_time);
        let win_tpot =
            self.delta_pct_ms(&scrape.time_per_output_token, |s| &s.time_per_output_token);

        // Request stats from delta histograms (hold previous value when idle)
        let avg_prompt_tokens = self
            .delta_mean(&scrape.request_prompt_tokens, |s| &s.request_prompt_tokens)
            .unwrap_or(self.prev_avg_prompt_tokens);
        self.prev_avg_prompt_tokens = avg_prompt_tokens;
        let avg_generation_tokens = self
            .delta_mean(&scrape.request_generation_tokens, |s| {
                &s.request_generation_tokens
            })
            .unwrap_or(self.prev_avg_generation_tokens);
        self.prev_avg_generation_tokens = avg_generation_tokens;
        let avg_prefill_kv_computed = self
            .delta_mean(&scrape.request_prefill_kv_computed_tokens, |s| {
                &s.request_prefill_kv_computed_tokens
            })
            .unwrap_or(self.prev_avg_prefill_kv_computed);
        self.prev_avg_prefill_kv_computed = avg_prefill_kv_computed;
        let iteration_tokens_mean = self
            .delta_mean(&scrape.iteration_tokens, |s| &s.iteration_tokens)
            .unwrap_or(self.prev_iteration_tokens_mean);
        self.prev_iteration_tokens_mean = iteration_tokens_mean;
        let win_prompt_tokens =
            self.delta_pct_raw(&scrape.request_prompt_tokens, |s| &s.request_prompt_tokens);
        let win_generation_tokens = self.delta_pct_raw(&scrape.request_generation_tokens, |s| {
            &s.request_generation_tokens
        });

        // KV Cache Residency
        let has_kv_block_metrics = scrape.kv_block_lifetime.count > 0;
        let kv_block_lifetime =
            self.delta_pct_ms(&scrape.kv_block_lifetime, |s| &s.kv_block_lifetime);
        let kv_block_idle_before_evict = self
            .delta_pct_ms(&scrape.kv_block_idle_before_evict, |s| {
                &s.kv_block_idle_before_evict
            });
        let kv_block_reuse_gap =
            self.delta_pct_ms(&scrape.kv_block_reuse_gap, |s| &s.kv_block_reuse_gap);

        // NIXL KV Connector
        let external_kv_transfer_tokens_total: u64 = scrape
            .prompt_tokens_by_source
            .iter()
            .find(|(src, _)| src == "external_kv_transfer")
            .map(|(_, v)| *v)
            .unwrap_or(0);
        let external_kv_transfer_tokens_per_sec = self
            .external_kv_rate
            .update(external_kv_transfer_tokens_total, now)
            .unwrap_or(0.0);
        let has_nixl = scrape.nixl_xfer_time.count > 0
            || scrape.nixl_failed_transfers_total > 0
            || scrape.nixl_failed_notifications_total > 0
            || external_kv_transfer_tokens_total > 0;
        let nixl_xfer_time = self.delta_pct_ms(&scrape.nixl_xfer_time, |s| &s.nixl_xfer_time);
        let nixl_post_time = self.delta_pct_ms(&scrape.nixl_post_time, |s| &s.nixl_post_time);
        let nixl_avg_bytes_transferred = self
            .delta_mean(&scrape.nixl_bytes_transferred, |s| {
                &s.nixl_bytes_transferred
            })
            .unwrap_or(0.0);
        let nixl_avg_descriptors = self
            .delta_mean(&scrape.nixl_num_descriptors, |s| &s.nixl_num_descriptors)
            .unwrap_or(0.0);
        let nixl_transfers_per_sec =
            self.nixl_xfer_rate.update(scrape.nixl_xfer_time.count, now).unwrap_or(0.0);
        let nixl_throughput_mb_per_sec = match &self.prev_scrape {
            Some(prev) => {
                let delta_bytes =
                    scrape.nixl_bytes_transferred.sum - prev.nixl_bytes_transferred.sum;
                let delta_time = scrape.nixl_xfer_time.sum - prev.nixl_xfer_time.sum;
                if delta_time > 0.0 && delta_bytes > 0.0 {
                    delta_bytes / delta_time / (1024.0 * 1024.0)
                } else {
                    0.0
                }
            }
            None => 0.0,
        };

        // Dynamo frontend / per-worker component telemetry
        let has_dynamo_config = scrape.dynamo_context_length > 0
            || scrape.dynamo_total_kv_blocks > 0
            || scrape.dynamo_component_uptime_secs > 0.0
            || !scrape.dynamo_component.is_empty();
        let dynamo_tokenize_latency = self.delta_pct_raw(&scrape.dynamo_tokenize_latency, |s| {
            &s.dynamo_tokenize_latency
        });
        let dynamo_detokenize_latency = self
            .delta_pct_raw(&scrape.dynamo_detokenize_latency, |s| {
                &s.dynamo_detokenize_latency
            });
        let dynamo_component_requests_per_sec = self
            .dynamo_component_req_rate
            .update(scrape.dynamo_component_requests_total, now)
            .unwrap_or(0.0);

        // Prefer vLLM's request_success_total rate when present; fall back to
        // Dynamo component rate for worker-only ports (PD-disagg prefill nodes
        // that don't expose a frontend).
        let requests_per_sec = if requests_per_sec_vllm > 0.0 {
            requests_per_sec_vllm
        } else {
            dynamo_component_requests_per_sec
        };

        let metrics = NodeMetrics {
            addr: addr.to_string(),
            model_name: scrape.model_name.clone(),
            is_healthy: true,
            is_loading: false,
            is_sleeping: None,
            server_load: None,
            last_updated: now,
            // Prefer Dynamo's `dynamo_component_inflight_requests` when
            // present: on PD-disaggregated prefill workers, vLLM's
            // `num_requests_running` is transient (a single forward pass)
            // and almost always reads 0 between scrapes. Dynamo's gauge
            // covers the full request lifetime (queue → prefill → KV
            // transfer → decode handoff). Falls back to vLLM's metric for
            // non-Dynamo deployments.
            requests_running: scrape
                .dynamo_component_inflight
                .unwrap_or(scrape.num_requests_running),
            requests_waiting: scrape.num_requests_waiting,
            kv_cache_usage: scrape.kv_cache_usage_perc,
            kv_fetch_waiting_to_start: scrape.kv_fetch_waiting_to_start,
            kv_fetch_in_progress: scrape.kv_fetch_in_progress,
            kv_fetch_completed_waiting: scrape.kv_fetch_completed_waiting,
            has_kv_fetch: scrape.has_kv_fetch,
            prompt_tokens_total: scrape.prompt_tokens_total,
            generation_tokens_total: scrape.generation_tokens_total,
            prompt_tps,
            generation_tps,
            requests_per_sec,
            preemptions_per_sec,
            preemptions_total: scrape.num_preemptions_total,
            prefix_cache_hit_rate,
            external_cache_hit_rate,
            mm_cache_hit_rate,
            prompt_tokens_by_source: scrape.prompt_tokens_by_source.clone(),
            prompt_tokens_cached_total: scrape.prompt_tokens_cached_total,
            prompt_tokens_recomputed_total: scrape.prompt_tokens_recomputed_total,
            http_qps,
            http_error_rate,
            http_requests_by_status: scrape.http_requests_by_status.clone(),
            cum_ttft,
            cum_itl,
            cum_e2e,
            cum_queue,
            cum_prefill,
            cum_decode,
            cum_inference,
            cum_tpot,
            win_ttft,
            win_itl,
            win_e2e,
            win_queue,
            win_prefill,
            win_decode,
            win_inference,
            win_tpot,
            request_success_total: scrape.request_success_total,
            success_by_reason: scrape.request_success_by_reason.clone(),
            avg_prompt_tokens,
            avg_generation_tokens,
            avg_prefill_kv_computed,
            iteration_tokens_mean,
            win_prompt_tokens,
            win_generation_tokens,
            spec_decode_draft_tokens_total: scrape.spec_decode_num_draft_tokens_total,
            spec_decode_accepted_tokens_total: scrape.spec_decode_num_accepted_tokens_total,
            spec_decode_acceptance_rate,
            spec_decode_drafts_per_sec,
            spec_decode_acceptance_per_pos,
            estimated_flops_per_gpu_per_sec,
            estimated_read_bytes_per_gpu_per_sec,
            estimated_write_bytes_per_gpu_per_sec,
            mfu_percent: None, // computed after gpu_scrape is attached
            kv_block_lifetime,
            kv_block_idle_before_evict,
            kv_block_reuse_gap,
            has_kv_block_metrics,
            nixl_failed_transfers_total: scrape.nixl_failed_transfers_total,
            nixl_failed_notifications_total: scrape.nixl_failed_notifications_total,
            nixl_kv_expired_reqs_total: scrape.nixl_kv_expired_reqs_total,
            nixl_xfer_time,
            nixl_post_time,
            nixl_avg_bytes_transferred,
            nixl_avg_descriptors,
            nixl_transfers_per_sec,
            nixl_throughput_mb_per_sec,
            external_kv_transfer_tokens_total,
            external_kv_transfer_tokens_per_sec,
            has_nixl,
            dynamo_context_length: scrape.dynamo_context_length,
            dynamo_total_kv_blocks: scrape.dynamo_total_kv_blocks,
            dynamo_kv_block_size: scrape.dynamo_kv_block_size,
            dynamo_max_num_seqs: scrape.dynamo_max_num_seqs,
            dynamo_max_num_batched_tokens: scrape.dynamo_max_num_batched_tokens,
            dynamo_disconnected_clients: scrape.dynamo_disconnected_clients,
            dynamo_tokenize_latency,
            dynamo_detokenize_latency,
            has_dynamo_config,
            dynamo_role: if scrape.dynamo_component.is_empty() {
                None
            } else {
                Some(scrape.dynamo_component.clone())
            },
            dynamo_uptime_secs: scrape.dynamo_component_uptime_secs,
            dynamo_component_requests_total: scrape.dynamo_component_requests_total,
            dynamo_component_requests_per_sec,
            dynamo_component_inflight: scrape.dynamo_component_inflight,
            dynamo_model_load_secs: scrape.dynamo_component_model_load_secs,
            kv_events,
            engine_metrics: None,
            gpu_scrape: None,
            ib_scrape: None,
        };

        // Store only fields needed for delta histograms; drop heavy unused data.
        let mut prev = scrape.clone();
        prev.engine_scrapes.clear();
        prev.http_requests_by_status.clear();
        prev.prompt_tokens_by_source.clear();
        self.prev_scrape = Some(prev);
        metrics
    }

    fn delta_pct_ms(
        &self,
        current: &HistogramSnapshot,
        get_prev: impl Fn(&VllmScrape) -> &HistogramSnapshot,
    ) -> LatencyPct {
        let delta = match &self.prev_scrape {
            Some(prev) => current.delta(get_prev(prev)),
            None => current.clone(),
        };
        LatencyPct {
            p50: delta.percentile(0.5) * 1000.0,
            p90: delta.percentile(0.9) * 1000.0,
            p99: delta.percentile(0.99) * 1000.0,
            mean: delta.mean() * 1000.0,
        }
    }

    /// Like delta_pct_ms but without the *1000 ms conversion (for token counts).
    fn delta_pct_raw(
        &self,
        current: &HistogramSnapshot,
        get_prev: impl Fn(&VllmScrape) -> &HistogramSnapshot,
    ) -> LatencyPct {
        let delta = match &self.prev_scrape {
            Some(prev) => current.delta(get_prev(prev)),
            None => current.clone(),
        };
        LatencyPct {
            p50: delta.percentile(0.5),
            p90: delta.percentile(0.9),
            p99: delta.percentile(0.99),
            mean: delta.mean(),
        }
    }

    /// Compute delta mean; returns `None` when no new samples arrived (delta count == 0).
    fn delta_mean(
        &self,
        current: &HistogramSnapshot,
        get_prev: impl Fn(&VllmScrape) -> &HistogramSnapshot,
    ) -> Option<f64> {
        let delta = match &self.prev_scrape {
            Some(prev) => current.delta(get_prev(prev)),
            None => current.clone(),
        };
        if delta.count == 0 {
            None
        } else {
            Some(delta.mean())
        }
    }
}

// ── Node state ──

/// How many consecutive scrape failures before marking a node offline.
const OFFLINE_THRESHOLD: u32 = 3;

/// Mutable state for a single node, holding rate calculators and the previous scrape.
pub struct NodeState {
    pub addr: String,
    pub info: NodeInfo,
    pub is_healthy: bool,
    /// Whether server_info has been successfully fetched.
    pub info_fetched: bool,

    /// Last successfully computed metrics (kept across transient failures).
    last_good: Option<NodeMetrics>,
    /// Consecutive scrape failure count.
    fail_count: u32,
    /// Aggregate rate state.
    rates: RateState,
    /// Per-engine rate states keyed by engine label (DP rank ID). Keyed (not
    /// positional) so rank churn — a rank dropping then a sibling rank's
    /// counters showing up in its old slot — doesn't pair the wrong RateState
    /// with the new engine and trigger spurious counter-reset deltas.
    engine_rates: HashMap<String, RateState>,
}

impl NodeState {
    pub fn new(addr: String) -> Self {
        Self {
            addr,
            info: NodeInfo::default(),
            is_healthy: false,
            info_fetched: false,
            last_good: None,
            fail_count: 0,
            rates: RateState::new(),
            engine_rates: HashMap::new(),
        }
    }

    /// Handle a scrape failure. Returns stale metrics if within tolerance,
    /// otherwise returns offline. GPU/IB scrapes are attached regardless (the
    /// host's exporters may still be up even when vLLM is down).
    pub fn on_scrape_error(
        &mut self,
        gpu_scrape: Option<GpuScrape>,
        ib_scrape: Option<IbScrape>,
    ) -> NodeMetrics {
        self.fail_count += 1;
        if self.fail_count < OFFLINE_THRESHOLD {
            if let Some(m) = &mut self.last_good {
                m.gpu_scrape = gpu_scrape;
                m.ib_scrape = ib_scrape;
                return m.clone();
            }
        }
        self.is_healthy = false;
        let mut m = NodeMetrics::offline(self.addr.clone());
        m.gpu_scrape = gpu_scrape;
        m.ib_scrape = ib_scrape;
        m
    }

    /// Update state with a new scrape and compute metrics.
    pub fn update(
        &mut self,
        scrape: VllmScrape,
        kv_events: Option<Vec<KVEventMetrics>>,
        gpu_scrape: Option<GpuScrape>,
        ib_scrape: Option<IbScrape>,
    ) -> NodeMetrics {
        let mut metrics = self.rates.compute(&self.addr, &scrape, kv_events);

        // Per-engine metrics. Key RateState by engine label so a dropped /
        // re-appearing rank keeps its own state instead of inheriting another
        // rank's counters (positional reuse caused spurious "counter reset"
        // deltas and wrong tps on the first tick after churn).
        if !scrape.engine_scrapes.is_empty() {
            let mut seen: std::collections::HashSet<String> = std::collections::HashSet::new();
            let mut engine_metrics: Vec<NodeMetrics> =
                Vec::with_capacity(scrape.engine_scrapes.len());
            for eng in &scrape.engine_scrapes {
                // Engine label is set by extract_engine_metrics; fall back to a
                // synthetic key if absent (shouldn't happen in practice).
                let key = eng
                    .engine_label
                    .clone()
                    .unwrap_or_else(|| format!("#{}", engine_metrics.len()));
                seen.insert(key.clone());
                let state = self.engine_rates.entry(key).or_insert_with(RateState::new);
                engine_metrics.push(state.compute(&self.addr, eng, None));
            }
            // Drop RateState for ranks that have permanently disappeared, so
            // the map doesn't grow without bound across long-running churn.
            self.engine_rates.retain(|k, _| seen.contains(k));
            metrics.engine_metrics = Some(engine_metrics);
        }

        metrics.gpu_scrape = gpu_scrape;
        metrics.ib_scrape = ib_scrape;

        metrics.recompute_mfu();

        self.last_good = Some(metrics.clone());
        self.fail_count = 0;
        metrics
    }
}

/// Compute percentiles from a cumulative histogram snapshot (in ms).
fn pct_ms(h: &HistogramSnapshot) -> LatencyPct {
    LatencyPct {
        p50: h.percentile(0.5) * 1000.0,
        p90: h.percentile(0.9) * 1000.0,
        p99: h.percentile(0.99) * 1000.0,
        mean: h.mean() * 1000.0,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Minimal NodeMetrics fixture; only fields the test cares about.
    fn rank(addr: &str, running: f64, dyn_inflight: Option<f64>) -> NodeMetrics {
        let mut m = NodeMetrics::offline(addr.to_string());
        m.is_healthy = true;
        m.requests_running = running;
        m.dynamo_component_inflight = dyn_inflight;
        m
    }

    fn rank_with_uptime_and_seqs(
        addr: &str,
        running: f64,
        dyn_inflight: Option<f64>,
        uptime_secs: f64,
        max_num_seqs: u64,
    ) -> NodeMetrics {
        let mut m = rank(addr, running, dyn_inflight);
        m.dynamo_uptime_secs = uptime_secs;
        m.dynamo_max_num_seqs = max_num_seqs;
        m
    }

    #[test]
    fn aggregate_prefers_worker_inflight_when_frontend_colocated() {
        // Co-located case: one rank is the frontend (no dynamo_component_inflight,
        // num_requests_running came from dynamo_frontend_inflight_requests = 257);
        // the other is the actual decode worker (dynamo_component_inflight = 512).
        // The 257 and the 512 overlap (frontend tracks the same requests the
        // backend is processing), so the host row should report 512 — not 769.
        let frontend = rank("h1:8180", 257.0, None);
        let worker = rank("h1:8201", 512.0, Some(512.0));
        let agg = NodeMetrics::aggregate_dynamo_ranks("h1".into(), vec![frontend, worker]);
        assert_eq!(agg.requests_running, 512.0);
    }

    #[test]
    fn aggregate_sums_worker_inflight_for_multi_rank_vllm() {
        // vLLM-style: 4 DP rank workers per host, each with its own
        // dynamo_component_inflight. Should sum (each rank handles its own
        // requests) — total = 5 + 7 + 3 + 11 = 26.
        let ranks = vec![
            rank("h1:7500", 5.0, Some(5.0)),
            rank("h1:7501", 7.0, Some(7.0)),
            rank("h1:7502", 3.0, Some(3.0)),
            rank("h1:7503", 11.0, Some(11.0)),
        ];
        let agg = NodeMetrics::aggregate_dynamo_ranks("h1".into(), ranks);
        assert_eq!(agg.requests_running, 26.0);
    }

    #[test]
    fn aggregate_falls_back_to_frontend_view_when_no_worker_inflight() {
        // Follower decode + frontend: worker exposes only uptime (no
        // dynamo_component_inflight); frontend reports 255. With no worker
        // inflight signal, fall back to summing all ranks (= frontend view).
        let frontend = rank("h1:8180", 255.0, None);
        let follower_worker = rank("h1:8202", 0.0, None);
        let agg = NodeMetrics::aggregate_dynamo_ranks("h1".into(), vec![frontend, follower_worker]);
        assert_eq!(agg.requests_running, 255.0);
    }

    /// A tensor-parallel follower participates in compute without owning
    /// requests; its co-located frontend must not inflate the host's load.
    #[test]
    fn aggregate_zeroes_inflight_for_follower_host_with_colocated_frontend() {
        // Frontend rank: dynamo runtime up + has model config (max_num_seqs > 0)
        let frontend = rank_with_uptime_and_seqs("h1:8180", 223.0, None, 100.0, 3072);
        // Follower worker: dynamo runtime up, NO inflight metric, NO model config.
        let follower_worker = rank_with_uptime_and_seqs("h1:8202", 0.0, None, 100.0, 0);
        let agg = NodeMetrics::aggregate_dynamo_ranks("h1".into(), vec![frontend, follower_worker]);
        assert_eq!(agg.requests_running, 0.0);
        assert_eq!(agg.requests_waiting, 0.0);
    }

    /// Frontend-only host (no worker rank at all on this host) should keep
    /// the frontend's view, not be confused with a follower.
    #[test]
    fn aggregate_keeps_frontend_inflight_when_no_worker_rank_present() {
        // Single rank: frontend with model config and request count.
        let frontend = rank_with_uptime_and_seqs("h1:8180", 100.0, None, 50.0, 1024);
        let agg = NodeMetrics::aggregate_dynamo_ranks("h1".into(), vec![frontend]);
        assert_eq!(agg.requests_running, 100.0);
    }

    /// In dynamo discover mode, port 8180 (frontend) is probed alongside the
    /// 8 worker ports — so a host with 8 GPUs gets 9 ranks. The frontend
    /// must NOT show up in engine_metrics, otherwise the TUI renders a 9th
    /// sub-row with no matching GPU (gpus.get(8) is None).
    #[test]
    fn aggregate_excludes_frontend_from_engine_metrics() {
        // Frontend: dynamo_role = None, max_num_seqs > 0 (from
        // dynamo_frontend_model_max_num_seqs).
        let frontend = rank_with_uptime_and_seqs("h1:8180", 50.0, None, 100.0, 3072);
        // 8 workers: dynamo_role = Some("backend"), no frontend gauges.
        let workers: Vec<NodeMetrics> = (0..8)
            .map(|i| {
                let mut r = rank(&format!("h1:820{i}"), 1.0, Some(1.0));
                r.dynamo_role = Some("backend".into());
                r
            })
            .collect();
        let mut all = vec![frontend];
        all.extend(workers);
        let agg = NodeMetrics::aggregate_dynamo_ranks("h1".into(), all);
        let engines = agg.engine_metrics.expect("workers should populate engine_metrics");
        assert_eq!(engines.len(), 8, "frontend rank must be filtered out");
        assert!(
            engines.iter().all(|e| e.dynamo_role.as_deref() == Some("backend")),
            "only worker ranks should appear in engine_metrics"
        );
    }

    /// Host badge from rank roles: prefill ranks co-located with "backend"
    /// ranks prove a PD pair, so the fold resolves to "P+D" locally. A host
    /// whose ranks are all "backend" is ambiguous (aggregated worker vs the
    /// decode side of a multi-host PD deployment) — the fold keeps the raw
    /// label for ClusterState::resolve_backend_roles() to settle.
    #[test]
    fn aggregate_role_badge_resolves_colocated_pd_keeps_lone_backend_raw() {
        let mut p = rank("h1:8201", 1.0, Some(0.0));
        p.dynamo_role = Some("prefill".into());
        let mut d = rank("h1:8202", 1.0, Some(0.0));
        d.dynamo_role = Some("backend".into());
        let mixed = NodeMetrics::aggregate_dynamo_ranks("h1".into(), vec![p, d]);
        assert_eq!(mixed.dynamo_role.as_deref(), Some("P+D"));

        let mut b = rank("h2:8201", 1.0, Some(0.0));
        b.dynamo_role = Some("backend".into());
        let lone = NodeMetrics::aggregate_dynamo_ranks("h2".into(), vec![b]);
        assert_eq!(lone.dynamo_role.as_deref(), Some("backend"));
    }

    /// Frontend-only host: filtering would empty engine_metrics. We set it to
    /// None so the TUI doesn't render any sub-rows for engines (only GPU rows).
    #[test]
    fn aggregate_sets_engine_metrics_none_when_only_frontend() {
        let frontend = rank_with_uptime_and_seqs("h1:8180", 100.0, None, 50.0, 1024);
        let agg = NodeMetrics::aggregate_dynamo_ranks("h1".into(), vec![frontend]);
        assert!(agg.engine_metrics.is_none());
    }

    /// One Dynamo system port fronting dp_local_size DP engines (e.g.
    /// `--data-parallel-size-local 4` behind a single DYN_SYSTEM_PORT): the
    /// endpoint's NodeMetrics is the endpoint-level aggregate, and the real
    /// per-rank breakdown lives in its inner engine_metrics. The fold must
    /// surface those inner entries — not the endpoint clone, which made every
    /// per-rank sub-row identical to the host row.
    #[test]
    fn aggregate_flattens_multi_engine_endpoint_into_per_rank_sub_rows() {
        let mut endpoint = rank("h1:7502", 46.0, Some(46.0));
        endpoint.dynamo_role = Some("backend".into());
        endpoint.engine_metrics =
            Some([7.0, 14.0, 12.0, 13.0].iter().map(|&r| rank("h1:7502", r, None)).collect());
        let agg = NodeMetrics::aggregate_dynamo_ranks("h1".into(), vec![endpoint]);
        assert_eq!(agg.requests_running, 46.0);
        let engines = agg.engine_metrics.expect("inner engines surfaced");
        assert_eq!(engines.len(), 4);
        let running: Vec<f64> = engines.iter().map(|e| e.requests_running).collect();
        assert_eq!(running, vec![7.0, 14.0, 12.0, 13.0]);
        assert!(
            engines.iter().all(|e| e.dynamo_role.as_deref() == Some("backend")),
            "inner engines inherit the endpoint's role"
        );
        assert!(
            engines.iter().all(|e| e.engine_metrics.is_none()),
            "no nesting remains after flattening"
        );
    }

    /// End-to-end replay path: a NodeMetrics with 4 GPUs (full DCGM fields)
    /// and 2 IB devices serializes through NodeSample/JSON, parses back, and
    /// `from_sample` reconstructs both the per-GPU array and the IB devices
    /// — not just an aggregate placeholder. Guards the bug where replay
    /// dropped every GPU after #0 and lost IB data entirely.
    #[test]
    fn from_sample_round_trip_preserves_per_gpu_and_ib_devices() {
        use crate::gpu::{GpuMetrics, GpuScrape};
        use crate::ib::{IB_STATE_ACTIVE, IbDevice, IbScrape};
        use crate::sample::NodeSample;

        let mut n = NodeMetrics::offline("host:8000".to_string());
        n.is_healthy = true;
        n.gpu_scrape = Some(GpuScrape {
            gpus: (0..4)
                .map(|i| GpuMetrics {
                    index: i,
                    uuid: format!("GPU-u{i}"),
                    name: "Example GPU".to_string(),
                    utilization: 100.0 - (i as f64),
                    dram_active: Some(0.5 + (i as f64) * 0.05),
                    gr_engine_active: Some(0.9 - (i as f64) * 0.1),
                    tensor_active: Some(0.4 + (i as f64) * 0.03),
                    power_watts: 900.0 + (i as f64),
                    temperature: 50.0 + (i as f64),
                    mem_used_bytes: 100_000_000_000 + (i as u64) * 1_000_000,
                    mem_total_bytes: 200_000_000_000,
                    xid_errors: i as u64,
                    ..Default::default()
                })
                .collect(),
            ..Default::default()
        });
        n.ib_scrape = Some(IbScrape {
            devices: vec![
                IbDevice {
                    device: "mlx5_0".to_string(),
                    port: 1,
                    state_id: IB_STATE_ACTIVE,
                    rate_bytes_per_sec: 50_000_000_000,
                    tx_gbps: 12.5,
                    rx_gbps: 6.5,
                    ..Default::default()
                },
                IbDevice {
                    device: "mlx5_1".to_string(),
                    port: 1,
                    state_id: IB_STATE_ACTIVE,
                    rate_bytes_per_sec: 50_000_000_000,
                    tx_gbps: 8.0,
                    rx_gbps: 9.0,
                    ..Default::default()
                },
            ],
        });

        // Serialize → JSON → deserialize.
        let sample_in = NodeSample::from(&n);
        let json = serde_json::to_string(&sample_in).expect("serialize");
        let sample_out: NodeSample = serde_json::from_str(&json).expect("deserialize");

        let restored = NodeMetrics::from_sample("host:8000".to_string(), &sample_out);

        // GPUs: count, identity, and DCGM PROF fractions all preserved.
        let gpus = restored.gpu_scrape.as_ref().expect("gpu_scrape rebuilt").gpus.clone();
        assert_eq!(gpus.len(), 4, "all 4 GPUs reconstructed");
        for (i, g) in gpus.iter().enumerate() {
            assert_eq!(g.index, i as u32);
            assert_eq!(g.uuid, format!("GPU-u{i}"));
            assert!((g.dram_active.unwrap() - (0.5 + (i as f64) * 0.05)).abs() < 1e-9);
            assert!((g.tensor_active.unwrap() - (0.4 + (i as f64) * 0.03)).abs() < 1e-9);
            assert_eq!(g.xid_errors, i as u64);
            assert_eq!(g.mem_total_bytes, 200_000_000_000);
        }

        // IB devices: both devices and their port/state/peak survive.
        let ib = restored.ib_scrape.as_ref().expect("ib_scrape rebuilt");
        assert_eq!(ib.devices.len(), 2);
        assert_eq!(ib.devices[0].device, "mlx5_0");
        assert!((ib.devices[0].tx_gbps - 12.5).abs() < 1e-9);
        assert_eq!(ib.devices[0].state_id, IB_STATE_ACTIVE);
        assert_eq!(ib.devices[1].device, "mlx5_1");
    }

    /// Replay path must carry Dynamo / NIXL / KV-block / KV-events and the
    /// per-engine breakdown through JSON. Without this, replay loses every
    /// signal that needs `has_dynamo_config` / `has_nixl` (worker role badge,
    /// PD-disagg KV throughput, the `e`-key engine fold).
    #[test]
    fn from_sample_round_trip_preserves_dynamo_nixl_kv_events_and_engines() {
        use crate::kv_events::KVEventMetrics;
        use crate::sample::NodeSample;

        let mut rank0 = NodeMetrics::offline("h1:7500".to_string());
        rank0.is_healthy = true;
        rank0.has_dynamo_config = true;
        rank0.dynamo_role = Some("backend".into());
        rank0.dynamo_uptime_secs = 123.4;
        rank0.dynamo_component_inflight = Some(7.0);
        rank0.dynamo_max_num_seqs = 64;
        rank0.dynamo_tokenize_latency = LatencyPct {
            p50: 1.2,
            p90: 0.0,
            p99: 4.5,
            mean: 2.0,
        };
        rank0.has_nixl = true;
        rank0.nixl_failed_transfers_total = 3;
        rank0.nixl_xfer_time = LatencyPct {
            p50: 5.0,
            p90: 0.0,
            p99: 22.0,
            mean: 8.0,
        };
        rank0.nixl_transfers_per_sec = 12.5;
        rank0.external_kv_transfer_tokens_per_sec = 800.0;

        let mut host = NodeMetrics::offline("h1".to_string());
        host.is_healthy = true;
        host.has_dynamo_config = true;
        host.dynamo_role = Some("P+D".into());
        host.has_kv_block_metrics = true;
        host.kv_block_lifetime = LatencyPct {
            p50: 100.0,
            p90: 0.0,
            p99: 9000.0,
            mean: 500.0,
        };
        host.kv_block_idle_before_evict = LatencyPct {
            p50: 50.0,
            p90: 0.0,
            p99: 300.0,
            mean: 75.0,
        };
        host.win_ttft = LatencyPct {
            p50: 40.0,
            p90: 95.0,
            p99: 140.0,
            mean: 55.0,
        };
        host.kv_events = Some(vec![KVEventMetrics {
            dp_rank: Some(0),
            blocks_stored: 42,
            blocks_removed: 9,
            tokens_stored: 1024,
            active_blocks: 33,
            total_events: 51,
            seq_gaps: 1,
        }]);
        host.engine_metrics = Some(vec![rank0]);

        // Serialize → JSON → deserialize → from_sample.
        let sample_in = NodeSample::from(&host);
        let json = serde_json::to_string(&sample_in).expect("serialize");
        let sample_out: NodeSample = serde_json::from_str(&json).expect("deserialize");
        let restored = NodeMetrics::from_sample("h1".to_string(), &sample_out);

        // Dynamo flags + role survive on the host row.
        assert!(restored.has_dynamo_config);
        assert_eq!(restored.dynamo_role.as_deref(), Some("P+D"));

        // KV block residency survives.
        assert!(restored.has_kv_block_metrics);
        assert!((restored.kv_block_lifetime.p99 - 9000.0).abs() < 1e-6);
        assert!((restored.kv_block_idle_before_evict.p50 - 50.0).abs() < 1e-6);

        // p90 + mean for main latencies round-trip through NodeSample's
        // new serialized fields (forward-compat: missing fields default to 0).
        assert!((restored.win_ttft.p50 - 40.0).abs() < 1e-6);
        assert!((restored.win_ttft.p90 - 95.0).abs() < 1e-6);
        assert!((restored.win_ttft.p99 - 140.0).abs() < 1e-6);
        assert!((restored.win_ttft.mean - 55.0).abs() < 1e-6);

        // KV events: one rank, fields intact.
        let evs = restored.kv_events.as_ref().expect("kv_events restored");
        assert_eq!(evs.len(), 1);
        assert_eq!(evs[0].blocks_stored, 42);
        assert_eq!(evs[0].active_blocks, 33);
        assert_eq!(evs[0].seq_gaps, 1);

        // engine_metrics: rebuilt recursively, addr inherited from parent.
        let engines = restored.engine_metrics.as_ref().expect("engine_metrics restored");
        assert_eq!(engines.len(), 1);
        let r0 = &engines[0];
        assert_eq!(r0.addr, "h1");
        assert!(r0.has_dynamo_config);
        assert_eq!(r0.dynamo_role.as_deref(), Some("backend"));
        assert!((r0.dynamo_uptime_secs - 123.4).abs() < 1e-6);
        assert_eq!(r0.dynamo_component_inflight, Some(7.0));
        assert_eq!(r0.dynamo_max_num_seqs, 64);
        assert!((r0.dynamo_tokenize_latency.p99 - 4.5).abs() < 1e-6);
        assert!(r0.has_nixl);
        assert_eq!(r0.nixl_failed_transfers_total, 3);
        assert!((r0.nixl_xfer_time.p99 - 22.0).abs() < 1e-6);
        assert!((r0.nixl_transfers_per_sec - 12.5).abs() < 1e-6);
        assert!((r0.external_kv_transfer_tokens_per_sec - 800.0).abs() < 1e-6);
    }

    // ── regroup_dp_engines ──

    /// Healthy DP head at `addr` publishing `n` engines; engine i carries
    /// generation_tps = i so slices are distinguishable after regrouping.
    fn dp_head(addr: &str, n: usize) -> NodeMetrics {
        let mut m = NodeMetrics::offline(addr.to_string());
        m.is_healthy = true;
        m.model_name = "test-model".into();
        m.http_qps = 42.0;
        m.engine_metrics = Some(
            (0..n)
                .map(|i| {
                    let mut e = NodeMetrics::offline(addr.to_string());
                    e.is_healthy = true;
                    e.model_name = "test-model".into();
                    e.generation_tps = i as f64;
                    e
                })
                .collect(),
        );
        m
    }

    #[test]
    fn regroup_splits_engines_across_offline_peers() {
        let mut nodes = vec![
            NodeMetrics::offline("d10:8002".into()),
            dp_head("d08:8002", 16),
            NodeMetrics::offline("d09:8002".into()),
            NodeMetrics::offline("d11:8002".into()),
            // Different port — untouched bystander.
            NodeMetrics::offline("p13:8001".into()),
        ];
        regroup_dp_engines(&mut nodes, &HashSet::new());

        // Every group member is now healthy with 4 engines.
        for (i, expect_ranks) in [(1, 0..4), (2, 4..8), (0, 8..12), (3, 12..16)] {
            let n = &nodes[i];
            assert!(n.is_healthy, "{} should be healthy", n.addr);
            let engines = n.engine_metrics.as_ref().expect("engines attached");
            let tps: Vec<f64> = engines.iter().map(|e| e.generation_tps).collect();
            let expect: Vec<f64> = expect_ranks.map(|r| r as f64).collect();
            assert_eq!(tps, expect, "{} got wrong engine slice", n.addr);
            // Row-level aggregate = sum of its slice.
            assert_eq!(n.generation_tps, expect.iter().sum::<f64>());
            assert_eq!(n.model_name, "test-model");
        }
        // API-server-level HTTP stats stay on the head only.
        assert_eq!(nodes[1].http_qps, 42.0);
        assert_eq!(nodes[0].http_qps, 0.0);
        // Bystander on another port untouched.
        assert!(!nodes[4].is_healthy);
    }

    #[test]
    fn regroup_skips_when_peer_served_metrics_before() {
        // d09 scraped OK earlier — it's a crashed standalone node, not a
        // headless secondary. Regrouping would mask the outage.
        let mut nodes = vec![
            dp_head("d08:8002", 4),
            NodeMetrics::offline("d09:8002".into()),
        ];
        let ever: HashSet<String> = ["d09:8002".to_string()].into();
        regroup_dp_engines(&mut nodes, &ever);
        assert_eq!(nodes[0].engine_metrics.as_ref().unwrap().len(), 4);
        assert!(!nodes[1].is_healthy);
    }

    #[test]
    fn regroup_skips_uneven_engine_count() {
        let mut nodes = vec![
            dp_head("d08:8002", 6),
            NodeMetrics::offline("d09:8002".into()),
            NodeMetrics::offline("d10:8002".into()),
            NodeMetrics::offline("d11:8002".into()),
        ];
        regroup_dp_engines(&mut nodes, &HashSet::new());
        assert_eq!(nodes[0].engine_metrics.as_ref().unwrap().len(), 6);
        assert!(!nodes[1].is_healthy);
    }

    #[test]
    fn regroup_skips_when_engines_fit_local_gpus() {
        // Head has 8 GPUs and only 2 engines — they all live locally
        // (e.g. DP=2 × TP=4 on one node); the offline peer is unrelated.
        let mut head = dp_head("d08:8002", 2);
        head.gpu_scrape = Some(crate::gpu::GpuScrape {
            gpus: vec![crate::gpu::GpuMetrics::default(); 8],
            ..Default::default()
        });
        let mut nodes = vec![head, NodeMetrics::offline("d09:8002".into())];
        regroup_dp_engines(&mut nodes, &HashSet::new());
        assert_eq!(nodes[0].engine_metrics.as_ref().unwrap().len(), 2);
        assert!(!nodes[1].is_healthy);
    }

    #[test]
    fn regroup_proceeds_when_engines_exceed_local_gpus() {
        // 4 GPUs on the head but 8 engines → they can't all be local.
        let mut head = dp_head("d08:8002", 8);
        head.gpu_scrape = Some(crate::gpu::GpuScrape {
            gpus: vec![crate::gpu::GpuMetrics::default(); 4],
            ..Default::default()
        });
        let mut nodes = vec![head, NodeMetrics::offline("d09:8002".into())];
        regroup_dp_engines(&mut nodes, &HashSet::new());
        assert_eq!(nodes[0].engine_metrics.as_ref().unwrap().len(), 4);
        assert_eq!(nodes[1].engine_metrics.as_ref().unwrap().len(), 4);
        assert!(nodes[1].is_healthy);
        // Head keeps its own hardware scrape; the peer has none.
        assert_eq!(nodes[0].gpu_scrape.as_ref().unwrap().gpus.len(), 4);
        assert!(nodes[1].gpu_scrape.is_none());
    }

    #[test]
    fn regroup_skips_multiple_healthy_heads() {
        // Two independent healthy DP nodes on the same port (homogeneous
        // cluster) — nothing to regroup even with an offline third peer.
        let mut nodes = vec![
            dp_head("a:8000", 4),
            dp_head("b:8000", 4),
            NodeMetrics::offline("c:8000".into()),
        ];
        regroup_dp_engines(&mut nodes, &HashSet::new());
        assert_eq!(nodes[0].engine_metrics.as_ref().unwrap().len(), 4);
        assert_eq!(nodes[1].engine_metrics.as_ref().unwrap().len(), 4);
        assert!(!nodes[2].is_healthy);
    }
}
