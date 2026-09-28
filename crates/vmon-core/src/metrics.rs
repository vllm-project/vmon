// SPDX-License-Identifier: Apache-2.0

use std::collections::{BTreeSet, HashMap};
use tokio::time::Instant;

use crate::histogram::HistogramSnapshot;
use crate::parser::{MetricFamily, MetricType, Sample};

/// Raw data extracted from a single `/metrics` scrape.
#[derive(Debug, Clone)]
pub struct VllmScrape {
    pub scraped_at: Instant,
    pub model_name: String,

    // === Gauges (Runtime) ===
    pub num_requests_running: f64,
    pub num_requests_waiting: f64,
    pub kv_cache_usage_perc: f64,

    // === Gauges (Async Remote-KV Fetch, `vllm:num_requests_kv_fetch_by_stage`) ===
    /// Requests with remote-fetch intent but no transfer started yet.
    pub kv_fetch_waiting_to_start: f64,
    /// Requests currently receiving remote KV.
    pub kv_fetch_in_progress: f64,
    /// Requests that finished receiving and are waiting to run.
    pub kv_fetch_completed_waiting: f64,
    /// Whether the kv-fetch stage family is present in the scrape.
    pub has_kv_fetch: bool,

    // === Counters (Token Stats) ===
    pub prompt_tokens_total: u64,
    pub generation_tokens_total: u64,
    pub prompt_tokens_cached_total: u64,
    pub prompt_tokens_recomputed_total: u64,
    /// prompt tokens by source: local_compute, local_cache_hit, external_kv_transfer
    pub prompt_tokens_by_source: Vec<(String, u64)>,

    // === Counters (Request Stats) ===
    pub request_success_total: u64,
    pub request_success_by_reason: Vec<(String, u64)>,
    pub num_preemptions_total: u64,

    // === Counters (Cache Stats) ===
    pub prefix_cache_hits_total: u64,
    pub prefix_cache_queries_total: u64,
    pub external_prefix_cache_hits_total: u64,
    pub external_prefix_cache_queries_total: u64,

    // === Latency Histograms ===
    pub ttft: HistogramSnapshot,
    pub itl: HistogramSnapshot,
    pub e2e_latency: HistogramSnapshot,
    pub queue_time: HistogramSnapshot,
    pub prefill_time: HistogramSnapshot,
    pub decode_time: HistogramSnapshot,
    pub inference_time: HistogramSnapshot,
    pub time_per_output_token: HistogramSnapshot,

    // === Request Stats Histograms ===
    pub request_prompt_tokens: HistogramSnapshot,
    pub request_generation_tokens: HistogramSnapshot,
    pub request_max_num_generation_tokens: HistogramSnapshot,
    pub request_params_max_tokens: HistogramSnapshot,
    pub request_params_n: HistogramSnapshot,
    pub request_prefill_kv_computed_tokens: HistogramSnapshot,

    // === Runtime Histograms ===
    pub iteration_tokens: HistogramSnapshot,

    // === Speculative Decoding Counters ===
    pub spec_decode_num_drafts_total: u64,
    pub spec_decode_num_draft_tokens_total: u64,
    pub spec_decode_num_accepted_tokens_total: u64,
    /// Per-position accepted tokens: position → cumulative count.
    pub spec_decode_accepted_per_pos: Vec<(u32, u64)>,

    // === Performance / MFU Counters (per GPU) ===
    pub estimated_flops_per_gpu_total: u64,
    pub estimated_read_bytes_per_gpu_total: u64,
    pub estimated_write_bytes_per_gpu_total: u64,

    // === Multi-modal Cache Counters ===
    pub mm_cache_hits_total: u64,
    pub mm_cache_queries_total: u64,

    // === KV Cache Residency (--kv-cache-metrics) ===
    pub kv_block_lifetime: HistogramSnapshot,
    pub kv_block_idle_before_evict: HistogramSnapshot,
    pub kv_block_reuse_gap: HistogramSnapshot,

    // === NIXL KV Connector ===
    pub nixl_failed_transfers_total: u64,
    pub nixl_failed_notifications_total: u64,
    pub nixl_kv_expired_reqs_total: u64,
    pub nixl_xfer_time: HistogramSnapshot,
    pub nixl_post_time: HistogramSnapshot,
    pub nixl_bytes_transferred: HistogramSnapshot,
    pub nixl_num_descriptors: HistogramSnapshot,

    // === HTTP Instrumentator ===
    pub http_requests_by_status: HashMap<String, u64>,
    pub http_requests_all: u64,

    // === Dynamo Frontend (model config + tokenizer) ===
    pub dynamo_context_length: u64,
    pub dynamo_total_kv_blocks: u64,
    pub dynamo_kv_block_size: u64,
    pub dynamo_max_num_seqs: u64,
    pub dynamo_max_num_batched_tokens: u64,
    pub dynamo_disconnected_clients: f64,
    pub dynamo_tokenize_latency: HistogramSnapshot,
    pub dynamo_detokenize_latency: HistogramSnapshot,
    /// Dynamo worker role label from `dynamo_component` (e.g. "prefill", "backend"). Empty if absent.
    pub dynamo_component: String,

    // === Dynamo Worker (per-rank component telemetry) ===
    /// Worker uptime in seconds (from `dynamo_component_uptime_seconds`).
    pub dynamo_component_uptime_secs: f64,
    /// Cumulative count of `generate` endpoint requests on this worker.
    pub dynamo_component_requests_total: u64,
    /// Currently-running requests on this worker's `generate` endpoint.
    /// `None` when the metric is absent (non-Dynamo deployments). Used as
    /// a fallback for `num_requests_running`, which is transient on
    /// PD-disaggregated prefill workers.
    pub dynamo_component_inflight: Option<f64>,
    /// Model load (cold-start) time in seconds.
    pub dynamo_component_model_load_secs: f64,

    // === Per-engine breakdown (DP mode) ===
    /// Per-engine scrapes, one per DP rank. Empty if single engine.
    pub engine_scrapes: Vec<VllmScrape>,
    /// Engine label (DP rank ID) when this scrape represents a single engine,
    /// `None` for the host-level aggregate. Used so `NodeState::engine_rates`
    /// can key its per-engine RateState by label, surviving rank churn.
    pub engine_label: Option<String>,
}

impl Default for VllmScrape {
    fn default() -> Self {
        Self {
            scraped_at: Instant::now(),
            model_name: String::new(),
            num_requests_running: 0.0,
            num_requests_waiting: 0.0,
            kv_cache_usage_perc: 0.0,
            kv_fetch_waiting_to_start: 0.0,
            kv_fetch_in_progress: 0.0,
            kv_fetch_completed_waiting: 0.0,
            has_kv_fetch: false,
            prompt_tokens_total: 0,
            generation_tokens_total: 0,
            prompt_tokens_cached_total: 0,
            prompt_tokens_recomputed_total: 0,
            prompt_tokens_by_source: Vec::new(),
            request_success_total: 0,
            request_success_by_reason: Vec::new(),
            num_preemptions_total: 0,
            prefix_cache_hits_total: 0,
            prefix_cache_queries_total: 0,
            external_prefix_cache_hits_total: 0,
            external_prefix_cache_queries_total: 0,
            ttft: HistogramSnapshot::default(),
            itl: HistogramSnapshot::default(),
            e2e_latency: HistogramSnapshot::default(),
            queue_time: HistogramSnapshot::default(),
            prefill_time: HistogramSnapshot::default(),
            decode_time: HistogramSnapshot::default(),
            inference_time: HistogramSnapshot::default(),
            time_per_output_token: HistogramSnapshot::default(),
            request_prompt_tokens: HistogramSnapshot::default(),
            request_generation_tokens: HistogramSnapshot::default(),
            request_max_num_generation_tokens: HistogramSnapshot::default(),
            request_params_max_tokens: HistogramSnapshot::default(),
            request_params_n: HistogramSnapshot::default(),
            request_prefill_kv_computed_tokens: HistogramSnapshot::default(),
            iteration_tokens: HistogramSnapshot::default(),
            spec_decode_num_drafts_total: 0,
            spec_decode_num_draft_tokens_total: 0,
            spec_decode_num_accepted_tokens_total: 0,
            spec_decode_accepted_per_pos: Vec::new(),
            estimated_flops_per_gpu_total: 0,
            estimated_read_bytes_per_gpu_total: 0,
            estimated_write_bytes_per_gpu_total: 0,
            mm_cache_hits_total: 0,
            mm_cache_queries_total: 0,
            kv_block_lifetime: HistogramSnapshot::default(),
            kv_block_idle_before_evict: HistogramSnapshot::default(),
            kv_block_reuse_gap: HistogramSnapshot::default(),
            nixl_failed_transfers_total: 0,
            nixl_failed_notifications_total: 0,
            nixl_kv_expired_reqs_total: 0,
            nixl_xfer_time: HistogramSnapshot::default(),
            nixl_post_time: HistogramSnapshot::default(),
            nixl_bytes_transferred: HistogramSnapshot::default(),
            nixl_num_descriptors: HistogramSnapshot::default(),
            http_requests_by_status: HashMap::new(),
            http_requests_all: 0,
            dynamo_context_length: 0,
            dynamo_total_kv_blocks: 0,
            dynamo_kv_block_size: 0,
            dynamo_max_num_seqs: 0,
            dynamo_max_num_batched_tokens: 0,
            dynamo_disconnected_clients: 0.0,
            dynamo_tokenize_latency: HistogramSnapshot::default(),
            dynamo_detokenize_latency: HistogramSnapshot::default(),
            dynamo_component: String::new(),
            dynamo_component_uptime_secs: 0.0,
            dynamo_component_requests_total: 0,
            dynamo_component_inflight: None,
            dynamo_component_model_load_secs: 0.0,
            engine_scrapes: Vec::new(),
            engine_label: None,
        }
    }
}

/// Extract vLLM metrics from parsed Prometheus families.
pub fn extract_vllm_metrics(families: &[MetricFamily], now: Instant) -> VllmScrape {
    let mut scrape = VllmScrape {
        scraped_at: now,
        ..Default::default()
    };

    for family in families {
        // Newer vLLM versions use "_total" suffix on TYPE lines for counters.
        // Normalize by stripping it so match arms work for both old and new formats.
        let name = family.name.strip_suffix("_total").unwrap_or(&family.name);
        match name {
            // === Gauges (summed across DP engines) ===
            "vllm:num_requests_running" => {
                scrape.num_requests_running = gauge_sum(&family.samples);
                if scrape.model_name.is_empty() {
                    if let Some(s) = family.samples.first() {
                        if let Some(m) = s.label("model_name") {
                            scrape.model_name = m.to_string();
                        }
                    }
                }
            }
            "vllm:num_requests_waiting" => {
                scrape.num_requests_waiting = gauge_sum(&family.samples);
            }
            "vllm:kv_cache_usage_perc" => {
                scrape.kv_cache_usage_perc = gauge_avg(&family.samples);
            }
            // Async remote-KV fetch stage gauge (Dynamo PD / KV-offload
            // deployments). One sample per (engine, stage); sum each stage
            // across DP engines. Guard on non-empty samples so per-engine
            // extraction doesn't flag engines the family never mentions.
            "vllm:num_requests_kv_fetch_by_stage" if !family.samples.is_empty() => {
                scrape.has_kv_fetch = true;
                for (stage, v) in gauge_by_label(&family.samples, "stage") {
                    match stage.as_str() {
                        "waiting_to_start" => scrape.kv_fetch_waiting_to_start = v,
                        "in_progress" => scrape.kv_fetch_in_progress = v,
                        "completed_waiting" => scrape.kv_fetch_completed_waiting = v,
                        _ => {}
                    }
                }
            }

            // === Token Counters ===
            "vllm:prompt_tokens" => {
                scrape.prompt_tokens_total = counter_value(&family.samples);
            }
            "vllm:generation_tokens" => {
                scrape.generation_tokens_total = counter_value(&family.samples);
            }
            "vllm:prompt_tokens_cached" => {
                scrape.prompt_tokens_cached_total = counter_value(&family.samples);
            }
            "vllm:prompt_tokens_recomputed" => {
                scrape.prompt_tokens_recomputed_total = counter_value(&family.samples);
            }
            "vllm:prompt_tokens_by_source" => {
                scrape.prompt_tokens_by_source = counter_by_label(&family.samples, "source");
            }

            // === Request Counters ===
            "vllm:request_success" => {
                scrape.request_success_total = counter_value(&family.samples);
                scrape.request_success_by_reason =
                    counter_by_label(&family.samples, "finished_reason");
            }
            "vllm:num_preemptions" => {
                scrape.num_preemptions_total = counter_value(&family.samples);
            }

            // === Cache Counters ===
            "vllm:prefix_cache_hits" => {
                scrape.prefix_cache_hits_total = counter_value(&family.samples);
            }
            "vllm:prefix_cache_queries" => {
                scrape.prefix_cache_queries_total = counter_value(&family.samples);
            }
            "vllm:external_prefix_cache_hits" => {
                scrape.external_prefix_cache_hits_total = counter_value(&family.samples);
            }
            "vllm:external_prefix_cache_queries" => {
                scrape.external_prefix_cache_queries_total = counter_value(&family.samples);
            }

            // === Latency Histograms ===
            "vllm:time_to_first_token_seconds" => {
                scrape.ttft = extract_histogram(&family.samples);
            }
            "vllm:inter_token_latency_seconds" => {
                scrape.itl = extract_histogram(&family.samples);
            }
            "vllm:e2e_request_latency_seconds" => {
                scrape.e2e_latency = extract_histogram(&family.samples);
            }
            "vllm:request_queue_time_seconds" => {
                scrape.queue_time = extract_histogram(&family.samples);
            }
            "vllm:request_prefill_time_seconds" => {
                scrape.prefill_time = extract_histogram(&family.samples);
            }
            "vllm:request_decode_time_seconds" => {
                scrape.decode_time = extract_histogram(&family.samples);
            }
            "vllm:request_inference_time_seconds" => {
                scrape.inference_time = extract_histogram(&family.samples);
            }
            "vllm:request_time_per_output_token_seconds" => {
                scrape.time_per_output_token = extract_histogram(&family.samples);
            }

            // === Request Stats Histograms ===
            "vllm:request_prompt_tokens" => {
                scrape.request_prompt_tokens = extract_histogram(&family.samples);
            }
            "vllm:request_generation_tokens" => {
                scrape.request_generation_tokens = extract_histogram(&family.samples);
            }
            "vllm:request_max_num_generation_tokens" => {
                scrape.request_max_num_generation_tokens = extract_histogram(&family.samples);
            }
            "vllm:request_params_max_tokens" => {
                scrape.request_params_max_tokens = extract_histogram(&family.samples);
            }
            "vllm:request_params_n" => {
                scrape.request_params_n = extract_histogram(&family.samples);
            }
            "vllm:request_prefill_kv_computed_tokens" => {
                scrape.request_prefill_kv_computed_tokens = extract_histogram(&family.samples);
            }

            // === Speculative Decoding Counters ===
            "vllm:spec_decode_num_drafts" => {
                scrape.spec_decode_num_drafts_total = counter_value(&family.samples);
            }
            "vllm:spec_decode_num_draft_tokens" => {
                scrape.spec_decode_num_draft_tokens_total = counter_value(&family.samples);
            }
            "vllm:spec_decode_num_accepted_tokens" => {
                scrape.spec_decode_num_accepted_tokens_total = counter_value(&family.samples);
            }
            "vllm:spec_decode_num_accepted_tokens_per_pos" => {
                scrape.spec_decode_accepted_per_pos = counter_by_pos(&family.samples, "position");
            }
            // === Performance / MFU Counters ===
            "vllm:estimated_flops_per_gpu" => {
                scrape.estimated_flops_per_gpu_total = counter_value(&family.samples);
            }
            "vllm:estimated_read_bytes_per_gpu" => {
                scrape.estimated_read_bytes_per_gpu_total = counter_value(&family.samples);
            }
            "vllm:estimated_write_bytes_per_gpu" => {
                scrape.estimated_write_bytes_per_gpu_total = counter_value(&family.samples);
            }

            // === Multi-modal Cache Counters ===
            "vllm:mm_cache_hits" => {
                scrape.mm_cache_hits_total = counter_value(&family.samples);
            }
            "vllm:mm_cache_queries" => {
                scrape.mm_cache_queries_total = counter_value(&family.samples);
            }

            // === KV Cache Residency ===
            "vllm:kv_block_lifetime_seconds" => {
                scrape.kv_block_lifetime = extract_histogram(&family.samples);
            }
            "vllm:kv_block_idle_before_evict_seconds" => {
                scrape.kv_block_idle_before_evict = extract_histogram(&family.samples);
            }
            "vllm:kv_block_reuse_gap_seconds" => {
                scrape.kv_block_reuse_gap = extract_histogram(&family.samples);
            }

            // === NIXL KV Connector ===
            "vllm:nixl_num_failed_transfers" => {
                scrape.nixl_failed_transfers_total = counter_value(&family.samples);
            }
            "vllm:nixl_num_failed_notifications" => {
                scrape.nixl_failed_notifications_total = counter_value(&family.samples);
            }
            "vllm:nixl_num_kv_expired_reqs" => {
                scrape.nixl_kv_expired_reqs_total = counter_value(&family.samples);
            }
            "vllm:nixl_xfer_time_seconds" => {
                scrape.nixl_xfer_time = extract_histogram(&family.samples);
            }
            "vllm:nixl_post_time_seconds" => {
                scrape.nixl_post_time = extract_histogram(&family.samples);
            }
            "vllm:nixl_bytes_transferred" => {
                scrape.nixl_bytes_transferred = extract_histogram(&family.samples);
            }
            "vllm:nixl_num_descriptors" => {
                scrape.nixl_num_descriptors = extract_histogram(&family.samples);
            }

            // === Runtime Histograms ===
            "vllm:iteration_tokens_total" | "vllm:iteration_tokens"
                if family.metric_type == MetricType::Histogram =>
            {
                scrape.iteration_tokens = extract_histogram(&family.samples);
            }

            // === HTTP Instrumentator ===
            // Note: HTTP metrics don't have engine labels (they're at the API server level),
            // but we use the same aggregating helpers for safety.
            "http_requests" | "http_requests_total"
                if family.metric_type == MetricType::Counter =>
            {
                scrape.http_requests_all = counter_value(&family.samples);
                for (status, count) in counter_by_label(&family.samples, "status") {
                    *scrape.http_requests_by_status.entry(status).or_default() += count;
                }
            }

            // === NVIDIA Dynamo Frontend ===
            "dynamo_frontend_inflight_requests" => {
                scrape.num_requests_running = gauge_sum(&family.samples);
                if scrape.model_name.is_empty() {
                    if let Some(s) = family.samples.first() {
                        if let Some(m) = s.label("model") {
                            scrape.model_name = m.to_string();
                        }
                    }
                }
            }
            "dynamo_frontend_queued_requests" => {
                scrape.num_requests_waiting = gauge_sum(&family.samples);
            }
            "dynamo_frontend_output_tokens" => {
                scrape.generation_tokens_total = counter_value(&family.samples);
            }
            "dynamo_frontend_requests" => {
                scrape.request_success_total = counter_value(&family.samples);
            }
            "dynamo_frontend_time_to_first_token_seconds" => {
                scrape.ttft = extract_histogram(&family.samples);
            }
            "dynamo_frontend_inter_token_latency_seconds" => {
                scrape.itl = extract_histogram(&family.samples);
            }
            "dynamo_frontend_request_duration_seconds" => {
                scrape.e2e_latency = extract_histogram(&family.samples);
            }
            "dynamo_frontend_input_sequence_tokens" => {
                let h = extract_histogram(&family.samples);
                scrape.prompt_tokens_total = h.sum as u64;
                scrape.request_prompt_tokens = h;
            }
            "dynamo_frontend_output_sequence_tokens" => {
                scrape.request_generation_tokens = extract_histogram(&family.samples);
            }
            "dynamo_frontend_disconnected_clients" => {
                scrape.dynamo_disconnected_clients = gauge_sum(&family.samples);
            }
            "dynamo_frontend_model_context_length" => {
                scrape.dynamo_context_length = gauge_sum(&family.samples) as u64;
            }
            "dynamo_frontend_model_total_kv_blocks" => {
                scrape.dynamo_total_kv_blocks = gauge_sum(&family.samples) as u64;
            }
            "dynamo_frontend_model_kv_cache_block_size" => {
                scrape.dynamo_kv_block_size = gauge_sum(&family.samples) as u64;
            }
            "dynamo_frontend_model_max_num_seqs" => {
                scrape.dynamo_max_num_seqs = gauge_sum(&family.samples) as u64;
            }
            "dynamo_frontend_model_max_num_batched_tokens" => {
                scrape.dynamo_max_num_batched_tokens = gauge_sum(&family.samples) as u64;
            }
            "dynamo_frontend_tokenizer_latency_ms" => {
                let tok: Vec<Sample> = family
                    .samples
                    .iter()
                    .filter(|s| s.label("operation") == Some("tokenize"))
                    .cloned()
                    .collect();
                let detok: Vec<Sample> = family
                    .samples
                    .iter()
                    .filter(|s| s.label("operation") == Some("detokenize"))
                    .cloned()
                    .collect();
                scrape.dynamo_tokenize_latency = extract_histogram(&tok);
                scrape.dynamo_detokenize_latency = extract_histogram(&detok);
            }

            // === Dynamo per-worker component telemetry ===
            "dynamo_component_uptime_seconds" => {
                if let Some(s) = family.samples.first() {
                    scrape.dynamo_component_uptime_secs = s.value;
                }
            }
            "dynamo_component_requests" => {
                // Sum only the `generate` endpoint; `clear_kv_blocks` is housekeeping.
                scrape.dynamo_component_requests_total = family
                    .samples
                    .iter()
                    .filter(|s| s.label("dynamo_endpoint") == Some("generate"))
                    .map(|s| s.value as u64)
                    .sum();
            }
            "dynamo_component_inflight_requests" => {
                let mut saw = false;
                let mut sum = 0.0_f64;
                for s in &family.samples {
                    if s.label("dynamo_endpoint") == Some("generate") {
                        saw = true;
                        sum += s.value;
                    }
                }
                if saw {
                    scrape.dynamo_component_inflight = Some(sum);
                }
            }
            "dynamo_component_model_load_time_seconds" => {
                if let Some(s) = family.samples.first() {
                    scrape.dynamo_component_model_load_secs = s.value;
                }
            }
            // Per-component request duration histogram. Route by dynamo_component
            // label so prefill workers fill prefill_time and decode/backend
            // workers fill decode_time. The value at this stage = full request
            // lifetime AT THIS COMPONENT (not e2e), which matches what
            // prefill_p50_ms / decode_p50_ms semantically mean.
            "dynamo_component_request_duration_seconds" => {
                let role =
                    family.samples.iter().find_map(|s| s.label("dynamo_component")).unwrap_or("");
                let h = extract_histogram(&family.samples);
                match role {
                    "prefill" => scrape.prefill_time = h,
                    "backend" | "decode" => scrape.decode_time = h,
                    _ => {}
                }
            }
            // GPU KV cache usage on this Dynamo worker. HELP text declares
            // 0.0-1.0 (already fractional, despite the "_percent" suffix), so
            // it maps directly to kv_cache_usage_perc.
            "dynamo_component_gpu_cache_usage_percent" if scrape.kv_cache_usage_perc == 0.0 => {
                scrape.kv_cache_usage_perc = gauge_sum(&family.samples);
            }

            _ => {}
        }
    }

    // Detect Dynamo worker role from `dynamo_component_*` metric labels.
    // Frontends never get a role: their KV-router observer families
    // (`dynamo_component_router_*`, `dynamo_component_kv_cache_*`) carry the
    // `dynamo_component` label of the worker they route to / observe, not
    // the frontend's own identity. In multi-frontend PD layouts a frontend
    // co-located with a prefill worker would otherwise tag the host
    // "backend" and fold to a spurious "P+D" badge.
    let is_frontend = families.iter().any(|f| f.name.starts_with("dynamo_frontend_"));
    if scrape.dynamo_component.is_empty() && !is_frontend {
        for f in families {
            if !f.name.starts_with("dynamo_component_") {
                continue;
            }
            // Observer families describe other components — skip them on
            // worker ports too.
            if f.name.starts_with("dynamo_component_router_")
                || f.name.starts_with("dynamo_component_kv_cache_")
            {
                continue;
            }
            if let Some(role) = f.samples.iter().find_map(|s| s.label("dynamo_component")) {
                scrape.dynamo_component = role.to_string();
                break;
            }
        }
    }

    // Fallback model_name from `dynamo_component_*{model="..."}` — Dynamo
    // worker ports (system_status_server) only carry component metrics; the
    // primary vllm:* / dynamo_frontend_* sources don't exist there.
    if scrape.model_name.is_empty() {
        for f in families {
            if !f.name.starts_with("dynamo_component_") {
                continue;
            }
            if let Some(m) = f.samples.iter().find_map(|s| s.label("model")) {
                scrape.model_name = m.to_string();
                break;
            }
        }
    }

    scrape
}

/// Sum a gauge across all engine labels.
fn gauge_sum(samples: &[Sample]) -> f64 {
    samples.iter().filter(|s| !s.name.ends_with("_created")).map(|s| s.value).sum()
}

/// Sum a gauge per label value, aggregating across engine labels.
/// E.g. num_requests_kv_fetch_by_stage with "stage" — engine="0" and
/// engine="1" values for the same stage are summed.
fn gauge_by_label(samples: &[Sample], label_key: &str) -> Vec<(String, f64)> {
    let mut map: HashMap<String, f64> = HashMap::new();
    for s in samples {
        if s.name.ends_with("_created") {
            continue;
        }
        if let Some(label_val) = s.label(label_key) {
            *map.entry(label_val.to_string()).or_default() += s.value;
        }
    }
    let mut result: Vec<_> = map.into_iter().collect();
    result.sort_by(|a, b| a.0.cmp(&b.0));
    result
}

/// Average a gauge across all engine labels.
fn gauge_avg(samples: &[Sample]) -> f64 {
    let vals: Vec<f64> = samples
        .iter()
        .filter(|s| !s.name.ends_with("_created"))
        .map(|s| s.value)
        .collect();
    if vals.is_empty() {
        0.0
    } else {
        vals.iter().sum::<f64>() / vals.len() as f64
    }
}

/// Get the total value of a counter, summing across all engine labels.
fn counter_value(samples: &[Sample]) -> u64 {
    let total: f64 = samples.iter().filter(|s| s.name.ends_with("_total")).map(|s| s.value).sum();
    if total > 0.0 {
        return total as u64;
    }
    // Fallback: sum non-_created samples
    samples
        .iter()
        .filter(|s| !s.name.ends_with("_created"))
        .map(|s| s.value as u64)
        .sum()
}

/// Extract per-label counter values, aggregating across engine labels.
/// E.g. prompt_tokens_by_source with "source" label — if engine="0" and engine="1"
/// both have source="local_compute", their values are summed.
fn counter_by_label(samples: &[Sample], label_key: &str) -> Vec<(String, u64)> {
    let mut map: HashMap<String, u64> = HashMap::new();
    for s in samples {
        if s.name.ends_with("_created") {
            continue;
        }
        if let Some(label_val) = s.label(label_key) {
            *map.entry(label_val.to_string()).or_default() += s.value as u64;
        }
    }
    let mut result: Vec<_> = map.into_iter().collect();
    result.sort_by(|a, b| a.0.cmp(&b.0));
    result
}

/// Extract per-position counter values (numeric label), aggregating across engine labels.
fn counter_by_pos(samples: &[Sample], label_key: &str) -> Vec<(u32, u64)> {
    let mut map: HashMap<u32, u64> = HashMap::new();
    for s in samples {
        if s.name.ends_with("_created") {
            continue;
        }
        if let Some(label_val) = s.label(label_key) {
            if let Ok(pos) = label_val.parse::<u32>() {
                *map.entry(pos).or_default() += s.value as u64;
            }
        }
    }
    let mut result: Vec<_> = map.into_iter().collect();
    result.sort_by_key(|&(pos, _)| pos);
    result
}

/// Extract a HistogramSnapshot from histogram samples, merging across engine labels.
/// Buckets with the same `le` bound are summed; _sum and _count are summed.
fn extract_histogram(samples: &[Sample]) -> HistogramSnapshot {
    let mut bucket_map: HashMap<u64, u64> = HashMap::new(); // le_bits -> count
    let mut sum = 0.0;
    let mut count = 0u64;

    for s in samples {
        if s.name.ends_with("_bucket") {
            if let Some(le) = s.label("le") {
                if let Ok(bound) = if le == "+Inf" {
                    Ok(f64::INFINITY)
                } else {
                    le.parse::<f64>()
                } {
                    *bucket_map.entry(bound.to_bits()).or_default() += s.value as u64;
                }
            }
        } else if s.name.ends_with("_sum") {
            sum += s.value;
        } else if s.name.ends_with("_count") {
            count += s.value as u64;
        }
    }

    let mut buckets: Vec<(f64, u64)> =
        bucket_map.into_iter().map(|(bits, cnt)| (f64::from_bits(bits), cnt)).collect();
    buckets.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal));

    HistogramSnapshot {
        buckets,
        sum,
        count,
    }
}

// ── Per-engine (DP) helpers ──

/// Detect distinct *local* engine (DP rank) label values.
///
/// vLLM exports per-engine metrics with an `engine` label. Under Prometheus
/// multiprocess export, sibling DP ranks that do **not** run on this host leak
/// in as zero-value labels on the handful of aggregated gauges/counters
/// (`num_requests_running`, `kv_cache_usage_perc`, `*_tokens_total`, …) — but
/// never on the per-process histogram families (latency / decode-time buckets),
/// because those are registered only by the local engine processes. So a label
/// is a real local engine iff it appears on at least one histogram family.
///
/// Example: a decode node with `dp_rank=4, dp_local_size=4` exposes real local
/// ranks 4..7 plus phantom labels 0..3 (sibling ranks' values picked up by the
/// multiprocess collector); only 4..7 carry histograms, so 0..3 are dropped.
///
/// Falls back to "all labels" when no per-engine histograms are present at all
/// (older vLLM, or histograms disabled) so we never hide every engine. Returns
/// empty if <= 1 engine survives (nothing to break out into sub-rows).
pub fn detect_engines(families: &[MetricFamily]) -> Vec<String> {
    let mut all: BTreeSet<String> = BTreeSet::new();
    let mut with_hist: BTreeSet<String> = BTreeSet::new();
    for f in families {
        let is_hist = f.metric_type == MetricType::Histogram;
        for s in &f.samples {
            if let Some(e) = s.label("engine") {
                all.insert(e.to_string());
                if is_hist {
                    with_hist.insert(e.to_string());
                }
            }
        }
    }
    let mut engines: Vec<String> = if with_hist.is_empty() {
        all.into_iter().collect()
    } else {
        with_hist.into_iter().collect()
    };
    if engines.len() <= 1 {
        return Vec::new();
    }
    // Numeric-aware order: labels are DP rank numbers, and with hybrid LB
    // they are global ranks (a node may expose 8..11) — lexicographic order
    // would yield 10,11,8,9 and misalign engine[i] with GPU i.
    engines.sort_by(|a, b| match (a.parse::<u64>(), b.parse::<u64>()) {
        (Ok(x), Ok(y)) => x.cmp(&y),
        _ => a.cmp(b),
    });
    engines
}

/// Extract VllmScrape for a specific engine, filtering samples by engine label.
pub fn extract_engine_metrics(families: &[MetricFamily], engine: &str, now: Instant) -> VllmScrape {
    let filtered: Vec<MetricFamily> = families
        .iter()
        .map(|f| MetricFamily {
            name: f.name.clone(),
            help: f.help.clone(),
            metric_type: f.metric_type.clone(),
            samples: f
                .samples
                .iter()
                .filter(|s| s.label("engine") == Some(engine))
                .cloned()
                .collect(),
        })
        .collect();
    let mut scrape = extract_vllm_metrics(&filtered, now);
    scrape.engine_label = Some(engine.to_string());
    scrape
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::parser::parse_prometheus_text;

    // Test fixture uses _total suffix in TYPE/HELP lines to match real
    // prometheus_client output (the Python library appends _total for counters).
    const SAMPLE_METRICS: &str = r#"# HELP vllm:num_requests_running Number of requests currently running
# TYPE vllm:num_requests_running gauge
vllm:num_requests_running{model_name="example/Example-Model"} 15.0
# HELP vllm:num_requests_waiting Number of requests waiting
# TYPE vllm:num_requests_waiting gauge
vllm:num_requests_waiting{model_name="example/Example-Model"} 3.0
# HELP vllm:kv_cache_usage_perc GPU KV-cache usage percent
# TYPE vllm:kv_cache_usage_perc gauge
vllm:kv_cache_usage_perc{model_name="example/Example-Model"} 0.58
# HELP vllm:prompt_tokens_total Number of prefill tokens
# TYPE vllm:prompt_tokens_total counter
vllm:prompt_tokens_total{model_name="example/Example-Model"} 50000.0
vllm:prompt_tokens_created{model_name="example/Example-Model"} 1.7e+09
# HELP vllm:generation_tokens_total Number of generation tokens
# TYPE vllm:generation_tokens_total counter
vllm:generation_tokens_total{model_name="example/Example-Model"} 120000.0
vllm:generation_tokens_created{model_name="example/Example-Model"} 1.7e+09
# HELP vllm:prompt_tokens_cached_total Cached prompt tokens
# TYPE vllm:prompt_tokens_cached_total counter
vllm:prompt_tokens_cached_total{model_name="example/Example-Model"} 8000.0
# HELP vllm:prompt_tokens_recomputed_total Recomputed tokens
# TYPE vllm:prompt_tokens_recomputed_total counter
vllm:prompt_tokens_recomputed_total{model_name="example/Example-Model"} 200.0
# HELP vllm:prompt_tokens_by_source_total Prompt tokens by source
# TYPE vllm:prompt_tokens_by_source_total counter
vllm:prompt_tokens_by_source_total{model_name="example/Example-Model",source="local_compute"} 42000.0
vllm:prompt_tokens_by_source_total{model_name="example/Example-Model",source="local_cache_hit"} 7500.0
vllm:prompt_tokens_by_source_total{model_name="example/Example-Model",source="external_kv_transfer"} 500.0
# HELP vllm:request_success_total Number of successful requests
# TYPE vllm:request_success_total counter
vllm:request_success_total{model_name="example/Example-Model",finished_reason="stop"} 980.0
vllm:request_success_total{model_name="example/Example-Model",finished_reason="length"} 20.0
# HELP vllm:prefix_cache_hits_total Prefix cache hits
# TYPE vllm:prefix_cache_hits_total counter
vllm:prefix_cache_hits_total{model_name="example/Example-Model"} 3400.0
# HELP vllm:prefix_cache_queries_total Prefix cache queries
# TYPE vllm:prefix_cache_queries_total counter
vllm:prefix_cache_queries_total{model_name="example/Example-Model"} 10000.0
# HELP vllm:external_prefix_cache_hits_total External prefix cache hits
# TYPE vllm:external_prefix_cache_hits_total counter
vllm:external_prefix_cache_hits_total{model_name="example/Example-Model"} 500.0
# HELP vllm:external_prefix_cache_queries_total External prefix cache queries
# TYPE vllm:external_prefix_cache_queries_total counter
vllm:external_prefix_cache_queries_total{model_name="example/Example-Model"} 2000.0
# HELP vllm:num_preemptions_total Number of preemptions
# TYPE vllm:num_preemptions_total counter
vllm:num_preemptions_total{model_name="example/Example-Model"} 5.0
# HELP vllm:time_to_first_token_seconds Histogram of TTFT
# TYPE vllm:time_to_first_token_seconds histogram
vllm:time_to_first_token_seconds_bucket{model_name="example/Example-Model",le="0.01"} 5.0
vllm:time_to_first_token_seconds_bucket{model_name="example/Example-Model",le="0.05"} 40.0
vllm:time_to_first_token_seconds_bucket{model_name="example/Example-Model",le="0.1"} 80.0
vllm:time_to_first_token_seconds_bucket{model_name="example/Example-Model",le="0.5"} 95.0
vllm:time_to_first_token_seconds_bucket{model_name="example/Example-Model",le="1.0"} 98.0
vllm:time_to_first_token_seconds_bucket{model_name="example/Example-Model",le="+Inf"} 100.0
vllm:time_to_first_token_seconds_sum{model_name="example/Example-Model"} 8.5
vllm:time_to_first_token_seconds_count{model_name="example/Example-Model"} 100.0
# HELP vllm:inter_token_latency_seconds ITL
# TYPE vllm:inter_token_latency_seconds histogram
vllm:inter_token_latency_seconds_bucket{model_name="example/Example-Model",le="0.01"} 50.0
vllm:inter_token_latency_seconds_bucket{model_name="example/Example-Model",le="0.05"} 90.0
vllm:inter_token_latency_seconds_bucket{model_name="example/Example-Model",le="+Inf"} 100.0
vllm:inter_token_latency_seconds_sum{model_name="example/Example-Model"} 2.0
vllm:inter_token_latency_seconds_count{model_name="example/Example-Model"} 100.0
# HELP vllm:e2e_request_latency_seconds Histogram of E2E latency
# TYPE vllm:e2e_request_latency_seconds histogram
vllm:e2e_request_latency_seconds_bucket{model_name="example/Example-Model",le="0.5"} 10.0
vllm:e2e_request_latency_seconds_bucket{model_name="example/Example-Model",le="1.0"} 40.0
vllm:e2e_request_latency_seconds_bucket{model_name="example/Example-Model",le="5.0"} 90.0
vllm:e2e_request_latency_seconds_bucket{model_name="example/Example-Model",le="10.0"} 98.0
vllm:e2e_request_latency_seconds_bucket{model_name="example/Example-Model",le="+Inf"} 100.0
vllm:e2e_request_latency_seconds_sum{model_name="example/Example-Model"} 250.0
vllm:e2e_request_latency_seconds_count{model_name="example/Example-Model"} 100.0
# HELP vllm:request_inference_time_seconds Inference time
# TYPE vllm:request_inference_time_seconds histogram
vllm:request_inference_time_seconds_bucket{model_name="example/Example-Model",le="1.0"} 30.0
vllm:request_inference_time_seconds_bucket{model_name="example/Example-Model",le="5.0"} 85.0
vllm:request_inference_time_seconds_bucket{model_name="example/Example-Model",le="+Inf"} 100.0
vllm:request_inference_time_seconds_sum{model_name="example/Example-Model"} 230.0
vllm:request_inference_time_seconds_count{model_name="example/Example-Model"} 100.0
# HELP vllm:request_time_per_output_token_seconds TPOT
# TYPE vllm:request_time_per_output_token_seconds histogram
vllm:request_time_per_output_token_seconds_bucket{model_name="example/Example-Model",le="0.01"} 40.0
vllm:request_time_per_output_token_seconds_bucket{model_name="example/Example-Model",le="0.05"} 85.0
vllm:request_time_per_output_token_seconds_bucket{model_name="example/Example-Model",le="+Inf"} 100.0
vllm:request_time_per_output_token_seconds_sum{model_name="example/Example-Model"} 3.5
vllm:request_time_per_output_token_seconds_count{model_name="example/Example-Model"} 100.0
# HELP vllm:request_prompt_tokens Per-request prompt tokens
# TYPE vllm:request_prompt_tokens histogram
vllm:request_prompt_tokens_bucket{model_name="example/Example-Model",le="100.0"} 30.0
vllm:request_prompt_tokens_bucket{model_name="example/Example-Model",le="500.0"} 80.0
vllm:request_prompt_tokens_bucket{model_name="example/Example-Model",le="+Inf"} 100.0
vllm:request_prompt_tokens_sum{model_name="example/Example-Model"} 50000.0
vllm:request_prompt_tokens_count{model_name="example/Example-Model"} 100.0
# HELP http_requests_total Total HTTP requests
# TYPE http_requests_total counter
http_requests_total{handler="/v1/chat/completions",method="POST",status="2xx"} 950.0
http_requests_total{handler="/v1/chat/completions",method="POST",status="4xx"} 40.0
http_requests_total{handler="/v1/chat/completions",method="POST",status="5xx"} 10.0
# HELP vllm:spec_decode_num_drafts_total Number of spec decoding drafts
# TYPE vllm:spec_decode_num_drafts_total counter
vllm:spec_decode_num_drafts_total{model_name="example/Example-Model"} 500.0
# HELP vllm:spec_decode_num_draft_tokens_total Number of draft tokens
# TYPE vllm:spec_decode_num_draft_tokens_total counter
vllm:spec_decode_num_draft_tokens_total{model_name="example/Example-Model"} 2500.0
# HELP vllm:spec_decode_num_accepted_tokens_total Number of accepted tokens
# TYPE vllm:spec_decode_num_accepted_tokens_total counter
vllm:spec_decode_num_accepted_tokens_total{model_name="example/Example-Model"} 2000.0
# HELP vllm:estimated_flops_per_gpu_total Estimated FLOPs per GPU
# TYPE vllm:estimated_flops_per_gpu_total counter
vllm:estimated_flops_per_gpu_total{model_name="example/Example-Model"} 1.5e+15
# HELP vllm:estimated_read_bytes_per_gpu_total Estimated read bytes per GPU
# TYPE vllm:estimated_read_bytes_per_gpu_total counter
vllm:estimated_read_bytes_per_gpu_total{model_name="example/Example-Model"} 5.0e+12
# HELP vllm:estimated_write_bytes_per_gpu_total Estimated write bytes per GPU
# TYPE vllm:estimated_write_bytes_per_gpu_total counter
vllm:estimated_write_bytes_per_gpu_total{model_name="example/Example-Model"} 1.0e+12
# HELP vllm:mm_cache_hits_total Multi-modal cache hits
# TYPE vllm:mm_cache_hits_total counter
vllm:mm_cache_hits_total{model_name="example/Example-Model"} 150.0
# HELP vllm:mm_cache_queries_total Multi-modal cache queries
# TYPE vllm:mm_cache_queries_total counter
vllm:mm_cache_queries_total{model_name="example/Example-Model"} 200.0
"#;

    #[test]
    fn test_extract_vllm_metrics() {
        let families = parse_prometheus_text(SAMPLE_METRICS).unwrap();
        let scrape = extract_vllm_metrics(&families, Instant::now());

        assert_eq!(scrape.model_name, "example/Example-Model");
        assert_eq!(scrape.num_requests_running, 15.0);
        assert_eq!(scrape.num_requests_waiting, 3.0);
        assert!((scrape.kv_cache_usage_perc - 0.58).abs() < 0.001);

        // Token counters
        assert_eq!(scrape.prompt_tokens_total, 50000);
        assert_eq!(scrape.generation_tokens_total, 120000);
        assert_eq!(scrape.prompt_tokens_cached_total, 8000);
        assert_eq!(scrape.prompt_tokens_recomputed_total, 200);
        assert_eq!(scrape.prompt_tokens_by_source.len(), 3);

        // Request counters
        assert_eq!(scrape.request_success_total, 1000);
        assert_eq!(scrape.num_preemptions_total, 5);

        // Cache counters
        assert_eq!(scrape.prefix_cache_hits_total, 3400);
        assert_eq!(scrape.prefix_cache_queries_total, 10000);
        assert_eq!(scrape.external_prefix_cache_hits_total, 500);
        assert_eq!(scrape.external_prefix_cache_queries_total, 2000);

        // Latency histograms
        assert_eq!(scrape.ttft.count, 100);
        assert!((scrape.ttft.sum - 8.5).abs() < 0.001);
        assert_eq!(scrape.itl.count, 100);
        assert_eq!(scrape.e2e_latency.count, 100);
        assert_eq!(scrape.inference_time.count, 100);
        assert_eq!(scrape.time_per_output_token.count, 100);

        // Request stats histograms
        assert_eq!(scrape.request_prompt_tokens.count, 100);
        assert!((scrape.request_prompt_tokens.mean() - 500.0).abs() < 0.001);

        // HTTP
        assert_eq!(scrape.http_requests_all, 1000);
        assert_eq!(scrape.http_requests_by_status.get("2xx"), Some(&950));
        assert_eq!(scrape.http_requests_by_status.get("4xx"), Some(&40));

        // Speculative decoding
        assert_eq!(scrape.spec_decode_num_drafts_total, 500);
        assert_eq!(scrape.spec_decode_num_draft_tokens_total, 2500);
        assert_eq!(scrape.spec_decode_num_accepted_tokens_total, 2000);

        // Performance / MFU
        assert_eq!(scrape.estimated_flops_per_gpu_total, 1_500_000_000_000_000);
        assert_eq!(scrape.estimated_read_bytes_per_gpu_total, 5_000_000_000_000);
        assert_eq!(
            scrape.estimated_write_bytes_per_gpu_total,
            1_000_000_000_000
        );

        // Multi-modal cache
        assert_eq!(scrape.mm_cache_hits_total, 150);
        assert_eq!(scrape.mm_cache_queries_total, 200);
    }

    /// Test that metrics with multiple engine labels (DP mode) are aggregated correctly.
    #[test]
    fn test_dp_multi_engine_aggregation() {
        let input = r#"# HELP vllm:num_requests_running Number of requests running
# TYPE vllm:num_requests_running gauge
vllm:num_requests_running{model_name="example-model",engine="0"} 5.0
vllm:num_requests_running{model_name="example-model",engine="1"} 3.0
# HELP vllm:num_requests_waiting Number of requests waiting
# TYPE vllm:num_requests_waiting gauge
vllm:num_requests_waiting{model_name="example-model",engine="0"} 2.0
vllm:num_requests_waiting{model_name="example-model",engine="1"} 1.0
# HELP vllm:kv_cache_usage_perc KV cache usage
# TYPE vllm:kv_cache_usage_perc gauge
vllm:kv_cache_usage_perc{model_name="example-model",engine="0"} 0.60
vllm:kv_cache_usage_perc{model_name="example-model",engine="1"} 0.40
# HELP vllm:prompt_tokens_total Number of prefill tokens
# TYPE vllm:prompt_tokens_total counter
vllm:prompt_tokens_total{model_name="example-model",engine="0"} 30000.0
vllm:prompt_tokens_total{model_name="example-model",engine="1"} 20000.0
# HELP vllm:generation_tokens_total Number of generation tokens
# TYPE vllm:generation_tokens_total counter
vllm:generation_tokens_total{model_name="example-model",engine="0"} 70000.0
vllm:generation_tokens_total{model_name="example-model",engine="1"} 50000.0
# HELP vllm:request_success_total Successful requests
# TYPE vllm:request_success_total counter
vllm:request_success_total{model_name="example-model",finished_reason="stop",engine="0"} 400.0
vllm:request_success_total{model_name="example-model",finished_reason="length",engine="0"} 10.0
vllm:request_success_total{model_name="example-model",finished_reason="stop",engine="1"} 380.0
vllm:request_success_total{model_name="example-model",finished_reason="length",engine="1"} 5.0
# HELP vllm:time_to_first_token_seconds TTFT
# TYPE vllm:time_to_first_token_seconds histogram
vllm:time_to_first_token_seconds_bucket{model_name="example-model",engine="0",le="0.1"} 30.0
vllm:time_to_first_token_seconds_bucket{model_name="example-model",engine="0",le="0.5"} 45.0
vllm:time_to_first_token_seconds_bucket{model_name="example-model",engine="0",le="+Inf"} 50.0
vllm:time_to_first_token_seconds_sum{model_name="example-model",engine="0"} 5.0
vllm:time_to_first_token_seconds_count{model_name="example-model",engine="0"} 50.0
vllm:time_to_first_token_seconds_bucket{model_name="example-model",engine="1",le="0.1"} 25.0
vllm:time_to_first_token_seconds_bucket{model_name="example-model",engine="1",le="0.5"} 38.0
vllm:time_to_first_token_seconds_bucket{model_name="example-model",engine="1",le="+Inf"} 40.0
vllm:time_to_first_token_seconds_sum{model_name="example-model",engine="1"} 3.5
vllm:time_to_first_token_seconds_count{model_name="example-model",engine="1"} 40.0
"#;
        let families = parse_prometheus_text(input).unwrap();
        let scrape = extract_vllm_metrics(&families, Instant::now());

        // Gauges: running/waiting summed, kv_cache averaged
        assert_eq!(scrape.num_requests_running, 8.0); // 5 + 3
        assert_eq!(scrape.num_requests_waiting, 3.0); // 2 + 1
        assert!((scrape.kv_cache_usage_perc - 0.50).abs() < 0.001); // avg(0.6, 0.4)

        // Counters: summed across engines
        assert_eq!(scrape.prompt_tokens_total, 50000); // 30k + 20k
        assert_eq!(scrape.generation_tokens_total, 120000); // 70k + 50k

        // Request success: summed, by_reason aggregated
        assert_eq!(scrape.request_success_total, 795); // 400+10+380+5
        let stop: u64 = scrape
            .request_success_by_reason
            .iter()
            .filter(|(r, _)| r == "stop")
            .map(|(_, c)| *c)
            .sum();
        let length: u64 = scrape
            .request_success_by_reason
            .iter()
            .filter(|(r, _)| r == "length")
            .map(|(_, c)| *c)
            .sum();
        assert_eq!(stop, 780); // 400 + 380
        assert_eq!(length, 15); // 10 + 5
        // No duplicate reasons
        let reason_count = scrape.request_success_by_reason.len();
        assert_eq!(reason_count, 2);

        // Histogram: buckets merged, sum/count summed
        assert_eq!(scrape.ttft.count, 90); // 50 + 40
        assert!((scrape.ttft.sum - 8.5).abs() < 0.001); // 5.0 + 3.5
        assert_eq!(scrape.ttft.buckets.len(), 3); // 3 unique le values
        // le=0.1: 30+25=55, le=0.5: 45+38=83, +Inf: 50+40=90
        assert_eq!(scrape.ttft.buckets[0], (0.1, 55));
        assert_eq!(scrape.ttft.buckets[1], (0.5, 83));
        assert_eq!(scrape.ttft.buckets[2].1, 90);
    }

    /// Stage gauges from `vllm:num_requests_kv_fetch_by_stage` (Dynamo PD /
    /// KV-offload remote fetch) sum per stage across DP engines, and the
    /// per-engine extraction keeps only that engine's values.
    #[test]
    fn test_kv_fetch_by_stage_extraction() {
        let input = r#"# HELP vllm:num_requests_kv_fetch_by_stage Number of requests by asynchronous remote-KV fetch stage.
# TYPE vllm:num_requests_kv_fetch_by_stage gauge
vllm:num_requests_kv_fetch_by_stage{dynamo_component="prefill",engine="0",model_name="example-model",stage="waiting_to_start"} 2.0
vllm:num_requests_kv_fetch_by_stage{dynamo_component="prefill",engine="1",model_name="example-model",stage="waiting_to_start"} 1.0
vllm:num_requests_kv_fetch_by_stage{dynamo_component="prefill",engine="0",model_name="example-model",stage="in_progress"} 6.0
vllm:num_requests_kv_fetch_by_stage{dynamo_component="prefill",engine="1",model_name="example-model",stage="in_progress"} 1.0
vllm:num_requests_kv_fetch_by_stage{dynamo_component="prefill",engine="0",model_name="example-model",stage="completed_waiting"} 0.0
vllm:num_requests_kv_fetch_by_stage{dynamo_component="prefill",engine="1",model_name="example-model",stage="completed_waiting"} 5.0
"#;
        let families = parse_prometheus_text(input).unwrap();
        let scrape = extract_vllm_metrics(&families, Instant::now());
        assert!(scrape.has_kv_fetch);
        assert_eq!(scrape.kv_fetch_waiting_to_start, 3.0); // 2 + 1
        assert_eq!(scrape.kv_fetch_in_progress, 7.0); // 6 + 1
        assert_eq!(scrape.kv_fetch_completed_waiting, 5.0); // 0 + 5

        let e1 = extract_engine_metrics(&families, "1", Instant::now());
        assert!(e1.has_kv_fetch);
        assert_eq!(e1.kv_fetch_waiting_to_start, 1.0);
        assert_eq!(e1.kv_fetch_in_progress, 1.0);
        assert_eq!(e1.kv_fetch_completed_waiting, 5.0);

        // Engine absent from the family → flag stays off.
        let e9 = extract_engine_metrics(&families, "9", Instant::now());
        assert!(!e9.has_kv_fetch);

        // Metric absent entirely → flag stays off.
        let none = extract_vllm_metrics(
            &parse_prometheus_text("# TYPE vllm:num_requests_running gauge\n").unwrap(),
            Instant::now(),
        );
        assert!(!none.has_kv_fetch);
    }

    #[test]
    fn test_detect_engines_drops_multiproc_phantom_labels() {
        // Mirrors a decode node with dp_rank=6, dp_local_size=2: real local
        // ranks 6,7 carry the full metric set incl. a histogram, while sibling
        // ranks 0,1 leak in as zero-value labels on aggregated gauges only
        // (Prometheus multiprocess export). Only 6,7 should be detected.
        let input = r#"# HELP vllm:num_requests_running Number of requests running
# TYPE vllm:num_requests_running gauge
vllm:num_requests_running{model_name="example-model",engine="0"} 0.0
vllm:num_requests_running{model_name="example-model",engine="1"} 0.0
vllm:num_requests_running{model_name="example-model",engine="6"} 4.0
vllm:num_requests_running{model_name="example-model",engine="7"} 2.0
# HELP vllm:kv_cache_usage_perc KV cache usage
# TYPE vllm:kv_cache_usage_perc gauge
vllm:kv_cache_usage_perc{model_name="example-model",engine="0"} 0.0
vllm:kv_cache_usage_perc{model_name="example-model",engine="1"} 0.0
vllm:kv_cache_usage_perc{model_name="example-model",engine="6"} 0.55
vllm:kv_cache_usage_perc{model_name="example-model",engine="7"} 0.45
# HELP vllm:time_to_first_token_seconds TTFT
# TYPE vllm:time_to_first_token_seconds histogram
vllm:time_to_first_token_seconds_bucket{model_name="example-model",engine="6",le="0.1"} 30.0
vllm:time_to_first_token_seconds_bucket{model_name="example-model",engine="6",le="+Inf"} 50.0
vllm:time_to_first_token_seconds_sum{model_name="example-model",engine="6"} 5.0
vllm:time_to_first_token_seconds_count{model_name="example-model",engine="6"} 50.0
vllm:time_to_first_token_seconds_bucket{model_name="example-model",engine="7",le="0.1"} 25.0
vllm:time_to_first_token_seconds_bucket{model_name="example-model",engine="7",le="+Inf"} 40.0
vllm:time_to_first_token_seconds_sum{model_name="example-model",engine="7"} 3.5
vllm:time_to_first_token_seconds_count{model_name="example-model",engine="7"} 40.0
"#;
        let families = parse_prometheus_text(input).unwrap();
        let engines = detect_engines(&families);
        assert_eq!(engines, vec!["6".to_string(), "7".to_string()]);
    }

    #[test]
    fn test_detect_engines_numeric_order() {
        // Hybrid-LB nodes expose global DP ranks: a node hosting ranks 8..11
        // must come out 8,9,10,11 (numeric), not 10,11,8,9 (lexicographic),
        // so engine_metrics[i] pairs with GPU i.
        let mut input = String::from("# TYPE vllm:num_requests_running gauge\n");
        for e in [8, 9, 10, 11] {
            input.push_str(&format!(
                "vllm:num_requests_running{{model_name=\"m\",engine=\"{e}\"}} 1.0\n"
            ));
        }
        let families = parse_prometheus_text(&input).unwrap();
        let engines = detect_engines(&families);
        assert_eq!(engines, vec!["8", "9", "10", "11"]);
    }

    #[test]
    fn test_detect_engines_fallback_when_no_histograms() {
        // No per-engine histograms exported at all → keep every engine label
        // rather than hiding them (older vLLM / histograms disabled).
        let input = r#"# HELP vllm:num_requests_running Number of requests running
# TYPE vllm:num_requests_running gauge
vllm:num_requests_running{model_name="example-model",engine="0"} 5.0
vllm:num_requests_running{model_name="example-model",engine="1"} 3.0
"#;
        let families = parse_prometheus_text(input).unwrap();
        let mut engines = detect_engines(&families);
        engines.sort();
        assert_eq!(engines, vec!["0".to_string(), "1".to_string()]);
    }

    #[test]
    fn test_extract_dynamo_metrics() {
        let input = r#"# HELP dynamo_frontend_inflight_requests Number of inflight requests
# TYPE dynamo_frontend_inflight_requests gauge
dynamo_frontend_inflight_requests{model="example/example-model-nvfp4"} 12
# HELP dynamo_frontend_queued_requests Number of requests in HTTP processing queue
# TYPE dynamo_frontend_queued_requests gauge
dynamo_frontend_queued_requests{model="example/example-model-nvfp4"} 3
# HELP dynamo_frontend_output_tokens_total Total number of output tokens generated
# TYPE dynamo_frontend_output_tokens_total counter
dynamo_frontend_output_tokens_total{model="example/example-model-nvfp4"} 500000
# HELP dynamo_frontend_requests_total Total number of LLM requests processed
# TYPE dynamo_frontend_requests_total counter
dynamo_frontend_requests_total{endpoint="completions",error_type="",model="example/example-model-nvfp4",request_type="stream",status="success"} 6000
dynamo_frontend_requests_total{endpoint="chat_completions",error_type="",model="example/example-model-nvfp4",request_type="unary",status="success"} 100
# HELP dynamo_frontend_time_to_first_token_seconds Time to first token in seconds
# TYPE dynamo_frontend_time_to_first_token_seconds histogram
dynamo_frontend_time_to_first_token_seconds_bucket{model="example/example-model-nvfp4",le="1"} 400
dynamo_frontend_time_to_first_token_seconds_bucket{model="example/example-model-nvfp4",le="10"} 900
dynamo_frontend_time_to_first_token_seconds_bucket{model="example/example-model-nvfp4",le="+Inf"} 1000
dynamo_frontend_time_to_first_token_seconds_sum{model="example/example-model-nvfp4"} 5200.0
dynamo_frontend_time_to_first_token_seconds_count{model="example/example-model-nvfp4"} 1000
# HELP dynamo_frontend_inter_token_latency_seconds Inter-token latency in seconds
# TYPE dynamo_frontend_inter_token_latency_seconds histogram
dynamo_frontend_inter_token_latency_seconds_bucket{model="example/example-model-nvfp4",le="0.024"} 80000
dynamo_frontend_inter_token_latency_seconds_bucket{model="example/example-model-nvfp4",le="0.045"} 95000
dynamo_frontend_inter_token_latency_seconds_bucket{model="example/example-model-nvfp4",le="+Inf"} 100000
dynamo_frontend_inter_token_latency_seconds_sum{model="example/example-model-nvfp4"} 2800.0
dynamo_frontend_inter_token_latency_seconds_count{model="example/example-model-nvfp4"} 100000
# HELP dynamo_frontend_request_duration_seconds Duration of LLM requests
# TYPE dynamo_frontend_request_duration_seconds histogram
dynamo_frontend_request_duration_seconds_bucket{model="example/example-model-nvfp4",le="22"} 300
dynamo_frontend_request_duration_seconds_bucket{model="example/example-model-nvfp4",le="40"} 800
dynamo_frontend_request_duration_seconds_bucket{model="example/example-model-nvfp4",le="+Inf"} 1000
dynamo_frontend_request_duration_seconds_sum{model="example/example-model-nvfp4"} 32000.0
dynamo_frontend_request_duration_seconds_count{model="example/example-model-nvfp4"} 1000
# HELP dynamo_frontend_input_sequence_tokens Input sequence length in tokens
# TYPE dynamo_frontend_input_sequence_tokens histogram
dynamo_frontend_input_sequence_tokens_bucket{model="example/example-model-nvfp4",le="7400"} 600
dynamo_frontend_input_sequence_tokens_bucket{model="example/example-model-nvfp4",le="15000"} 1000
dynamo_frontend_input_sequence_tokens_bucket{model="example/example-model-nvfp4",le="+Inf"} 1000
dynamo_frontend_input_sequence_tokens_sum{model="example/example-model-nvfp4"} 8000000
dynamo_frontend_input_sequence_tokens_count{model="example/example-model-nvfp4"} 1000
# HELP dynamo_frontend_output_sequence_tokens Output sequence length in tokens
# TYPE dynamo_frontend_output_sequence_tokens histogram
dynamo_frontend_output_sequence_tokens_bucket{model="example/example-model-nvfp4",le="880"} 500
dynamo_frontend_output_sequence_tokens_bucket{model="example/example-model-nvfp4",le="1800"} 1000
dynamo_frontend_output_sequence_tokens_bucket{model="example/example-model-nvfp4",le="+Inf"} 1000
dynamo_frontend_output_sequence_tokens_sum{model="example/example-model-nvfp4"} 500000
dynamo_frontend_output_sequence_tokens_count{model="example/example-model-nvfp4"} 1000
"#;
        let families = parse_prometheus_text(input).unwrap();
        let scrape = extract_vllm_metrics(&families, Instant::now());

        assert_eq!(scrape.model_name, "example/example-model-nvfp4");
        assert_eq!(scrape.num_requests_running, 12.0);
        assert_eq!(scrape.num_requests_waiting, 3.0);

        // Token counters
        assert_eq!(scrape.generation_tokens_total, 500000);
        assert_eq!(scrape.prompt_tokens_total, 8000000); // from histogram sum

        // Request counter (all status=success summed)
        assert_eq!(scrape.request_success_total, 6100);

        // Latency histograms
        assert_eq!(scrape.ttft.count, 1000);
        assert!((scrape.ttft.sum - 5200.0).abs() < 0.001);
        assert_eq!(scrape.itl.count, 100000);
        assert_eq!(scrape.e2e_latency.count, 1000);

        // Request stats histograms
        assert_eq!(scrape.request_prompt_tokens.count, 1000);
        assert!((scrape.request_prompt_tokens.sum - 8000000.0).abs() < 0.001);
        assert_eq!(scrape.request_generation_tokens.count, 1000);

        // Dynamo-specific fields not in this fixture
        assert_eq!(scrape.dynamo_context_length, 0);
    }

    #[test]
    fn test_extract_dynamo_config_metrics() {
        let input = r#"# HELP dynamo_frontend_model_context_length Max context length
# TYPE dynamo_frontend_model_context_length gauge
dynamo_frontend_model_context_length{model="example/Example-Model-NVFP4"} 10240
# HELP dynamo_frontend_model_kv_cache_block_size KV cache block size
# TYPE dynamo_frontend_model_kv_cache_block_size gauge
dynamo_frontend_model_kv_cache_block_size{model="example/Example-Model-NVFP4"} 64
# HELP dynamo_frontend_model_max_num_batched_tokens Max batched tokens
# TYPE dynamo_frontend_model_max_num_batched_tokens gauge
dynamo_frontend_model_max_num_batched_tokens{model="example/Example-Model-NVFP4"} 10240
# HELP dynamo_frontend_model_max_num_seqs Max number of sequences
# TYPE dynamo_frontend_model_max_num_seqs gauge
dynamo_frontend_model_max_num_seqs{model="example/Example-Model-NVFP4"} 512
# HELP dynamo_frontend_model_total_kv_blocks Total KV cache blocks
# TYPE dynamo_frontend_model_total_kv_blocks gauge
dynamo_frontend_model_total_kv_blocks{model="example/Example-Model-NVFP4"} 41713
# HELP dynamo_frontend_disconnected_clients Disconnected clients
# TYPE dynamo_frontend_disconnected_clients gauge
dynamo_frontend_disconnected_clients 0
# HELP dynamo_frontend_tokenizer_latency_ms Tokenizer latency
# TYPE dynamo_frontend_tokenizer_latency_ms histogram
dynamo_frontend_tokenizer_latency_ms_bucket{operation="tokenize",le="4"} 342
dynamo_frontend_tokenizer_latency_ms_bucket{operation="tokenize",le="8"} 6098
dynamo_frontend_tokenizer_latency_ms_bucket{operation="tokenize",le="+Inf"} 6146
dynamo_frontend_tokenizer_latency_ms_sum{operation="tokenize"} 31817.0
dynamo_frontend_tokenizer_latency_ms_count{operation="tokenize"} 6146
dynamo_frontend_tokenizer_latency_ms_bucket{operation="detokenize",le="0.5"} 6146
dynamo_frontend_tokenizer_latency_ms_bucket{operation="detokenize",le="+Inf"} 6146
dynamo_frontend_tokenizer_latency_ms_sum{operation="detokenize"} 7.48
dynamo_frontend_tokenizer_latency_ms_count{operation="detokenize"} 6146
"#;
        let families = parse_prometheus_text(input).unwrap();
        let scrape = extract_vllm_metrics(&families, Instant::now());

        assert_eq!(scrape.dynamo_context_length, 10240);
        assert_eq!(scrape.dynamo_kv_block_size, 64);
        assert_eq!(scrape.dynamo_max_num_batched_tokens, 10240);
        assert_eq!(scrape.dynamo_max_num_seqs, 512);
        assert_eq!(scrape.dynamo_total_kv_blocks, 41713);
        assert_eq!(scrape.dynamo_disconnected_clients, 0.0);

        // Tokenizer histograms split by operation label
        assert_eq!(scrape.dynamo_tokenize_latency.count, 6146);
        assert!((scrape.dynamo_tokenize_latency.sum - 31817.0).abs() < 0.1);
        assert_eq!(scrape.dynamo_detokenize_latency.count, 6146);
        assert!((scrape.dynamo_detokenize_latency.sum - 7.48).abs() < 0.01);
    }

    /// Per-component request duration on the worker port routes to
    /// prefill_time when dynamo_component="prefill"; that is the value the
    /// TUI reads for prefill_p50/p99_ms.
    #[test]
    fn dynamo_component_request_duration_routes_to_prefill_time_for_prefill() {
        let input = r#"# TYPE dynamo_component_request_duration_seconds histogram
dynamo_component_request_duration_seconds_bucket{dynamo_component="prefill",dynamo_endpoint="generate",le="1"} 0
dynamo_component_request_duration_seconds_bucket{dynamo_component="prefill",dynamo_endpoint="generate",le="10"} 100
dynamo_component_request_duration_seconds_bucket{dynamo_component="prefill",dynamo_endpoint="generate",le="+Inf"} 100
dynamo_component_request_duration_seconds_sum{dynamo_component="prefill",dynamo_endpoint="generate"} 500
dynamo_component_request_duration_seconds_count{dynamo_component="prefill",dynamo_endpoint="generate"} 100
"#;
        let families = parse_prometheus_text(input).unwrap();
        let scrape = extract_vllm_metrics(&families, Instant::now());
        assert_eq!(scrape.prefill_time.count, 100);
        assert!((scrape.prefill_time.sum - 500.0).abs() < 0.01);
        // Decode untouched.
        assert_eq!(scrape.decode_time.count, 0);
    }

    #[test]
    fn dynamo_component_request_duration_routes_to_decode_time_for_backend() {
        let input = r#"# TYPE dynamo_component_request_duration_seconds histogram
dynamo_component_request_duration_seconds_bucket{dynamo_component="backend",dynamo_endpoint="generate",le="+Inf"} 50
dynamo_component_request_duration_seconds_sum{dynamo_component="backend",dynamo_endpoint="generate"} 200
dynamo_component_request_duration_seconds_count{dynamo_component="backend",dynamo_endpoint="generate"} 50
"#;
        let families = parse_prometheus_text(input).unwrap();
        let scrape = extract_vllm_metrics(&families, Instant::now());
        assert_eq!(scrape.decode_time.count, 50);
        assert_eq!(scrape.prefill_time.count, 0);
    }

    /// `gpu_cache_usage_percent` is fractional (0.0-1.0) per Dynamo's HELP
    /// text. Maps directly to kv_cache_usage_perc without scaling.
    #[test]
    fn dynamo_component_gpu_cache_usage_percent_fills_kv_cache_usage() {
        let input = r#"# HELP dynamo_component_gpu_cache_usage_percent GPU cache usage as a percentage (0.0-1.0).
# TYPE dynamo_component_gpu_cache_usage_percent gauge
dynamo_component_gpu_cache_usage_percent{dp_rank="0",dynamo_component="prefill"} 0.42
"#;
        let families = parse_prometheus_text(input).unwrap();
        let scrape = extract_vllm_metrics(&families, Instant::now());
        assert!((scrape.kv_cache_usage_perc - 0.42).abs() < 1e-6);
    }

    /// Multi-frontend PD layouts run a frontend (:8180) on worker hosts. The
    /// frontend's KV-router metrics carry `dynamo_component="backend"` labels
    /// describing the *observed* workers — the frontend itself must stay
    /// role-less, or a co-located prefill host folds to a spurious "P+D".
    #[test]
    fn frontend_router_observer_labels_do_not_assign_worker_role() {
        let input = r#"# TYPE dynamo_frontend_model_max_num_seqs gauge
dynamo_frontend_model_max_num_seqs{model="example/example-model"} 120
# TYPE dynamo_component_uptime_seconds gauge
dynamo_component_uptime_seconds{worker_id="694d9f4ee4c59407"} 3248.8
# TYPE dynamo_component_router_requests counter
dynamo_component_router_requests_total{dynamo_component="backend",dynamo_namespace="dynamo",router_id="758",worker_id="694d9f4ee4c59407"} 5186
# TYPE dynamo_component_kv_cache_events_applied counter
dynamo_component_kv_cache_events_applied{dynamo_component="backend",dynamo_namespace="dynamo",event_type="stored",status="ok",worker_id="694d9f4ee4c59407"} 295
"#;
        let families = parse_prometheus_text(input).unwrap();
        let scrape = extract_vllm_metrics(&families, Instant::now());
        assert!(scrape.dynamo_component.is_empty());
    }

    /// A worker port exposing observer families alongside its own identity
    /// families must take the role from the latter.
    #[test]
    fn worker_role_skips_observer_families() {
        let input = r#"# TYPE dynamo_component_kv_cache_events_applied counter
dynamo_component_kv_cache_events_applied{dynamo_component="backend",event_type="stored",status="ok"} 10
# TYPE dynamo_component_requests counter
dynamo_component_requests_total{dynamo_component="prefill",dynamo_endpoint="generate"} 42
"#;
        let families = parse_prometheus_text(input).unwrap();
        let scrape = extract_vllm_metrics(&families, Instant::now());
        assert_eq!(scrape.dynamo_component, "prefill");
    }

    /// Dynamo PD-disagg prefill workers expose only `dynamo_component_*`
    /// (their `system_status_server` carries no vllm:* or
    /// dynamo_frontend_*). Model name should still surface from the
    /// `model` label on those component metrics.
    #[test]
    fn model_name_falls_back_to_dynamo_component_label() {
        let input = r#"# TYPE dynamo_component_inflight_requests gauge
dynamo_component_inflight_requests{dynamo_component="prefill",dynamo_endpoint="generate",model="example/Example-Model"} 333
# TYPE dynamo_component_uptime_seconds gauge
dynamo_component_uptime_seconds{worker_id="abc"} 100.0
"#;
        let families = parse_prometheus_text(input).unwrap();
        let scrape = extract_vllm_metrics(&families, Instant::now());
        assert_eq!(scrape.model_name, "example/Example-Model");
        assert_eq!(scrape.dynamo_component, "prefill");
    }

    /// Dynamo prefill workers report `inflight=0` on `generate` while
    /// `clear_kv_blocks` is housekeeping — the field should still parse as
    /// `Some(0.0)` (metric *was* observed, value happens to be 0) so the
    /// fallback wins over `vllm:num_requests_running`.
    #[test]
    fn dynamo_inflight_filters_clear_kv_blocks_and_records_presence() {
        let input = r#"# TYPE dynamo_component_inflight_requests gauge
dynamo_component_inflight_requests{dynamo_component="prefill",dynamo_endpoint="generate"} 0
dynamo_component_inflight_requests{dynamo_component="prefill",dynamo_endpoint="clear_kv_blocks"} 7
# TYPE vllm:num_requests_running gauge
vllm:num_requests_running{model_name="x"} 0
"#;
        let families = parse_prometheus_text(input).unwrap();
        let scrape = extract_vllm_metrics(&families, Instant::now());
        // Filtered by dynamo_endpoint=generate, so the housekeeping 7 is dropped.
        assert_eq!(scrape.dynamo_component_inflight, Some(0.0));
    }

    #[test]
    fn dynamo_inflight_sums_generate_endpoint_only() {
        let input = r#"# TYPE dynamo_component_inflight_requests gauge
dynamo_component_inflight_requests{dynamo_component="backend",dynamo_endpoint="generate"} 5
dynamo_component_inflight_requests{dynamo_component="backend",dynamo_endpoint="generate"} 3
dynamo_component_inflight_requests{dynamo_component="backend",dynamo_endpoint="clear_kv_blocks"} 99
"#;
        let families = parse_prometheus_text(input).unwrap();
        let scrape = extract_vllm_metrics(&families, Instant::now());
        assert_eq!(scrape.dynamo_component_inflight, Some(8.0));
    }

    #[test]
    fn dynamo_inflight_absent_for_pure_vllm() {
        // No dynamo_component_inflight_requests family at all.
        let families = parse_prometheus_text(SAMPLE_METRICS).unwrap();
        let scrape = extract_vllm_metrics(&families, Instant::now());
        assert_eq!(scrape.dynamo_component_inflight, None);
    }
}
