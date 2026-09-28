// SPDX-License-Identifier: Apache-2.0

//! Serializable snapshot types for the collect → JSON → replay pipeline.
//!
//! These mirror `NodeMetrics` / `GpuMetrics` / `IbDevice` for serde, and are the
//! single source of truth for the on-disk JSON report format. Both
//! `vmon-report` (the writer side) and `vmon-core::node::NodeMetrics::from_sample`
//! (the replay reader side) use the same struct definitions so the two halves
//! cannot drift.

use std::collections::HashMap;

use serde::{Deserialize, Serialize};

use crate::gpu::GpuMetrics;
use crate::ib::IbDevice;
use crate::kv_events::KVEventMetrics;
use crate::mooncake::{MooncakeHealth, MooncakeMetrics, MooncakeSegment};
use crate::node::NodeMetrics;

/// Per-GPU values for one scrape tick. Mirrors `crate::gpu::GpuMetrics` for
/// serialization into the collect report.
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct GpuSample {
    // Identity
    #[serde(default)]
    pub index: u32,
    #[serde(default)]
    pub uuid: String,
    #[serde(default)]
    pub name: String,

    // Utilization / activity
    #[serde(default)]
    pub utilization: f64,
    #[serde(default)]
    pub mem_utilization: f64,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub dram_active: Option<f64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub gr_engine_active: Option<f64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub tensor_active: Option<f64>,
    #[serde(default)]
    pub enc_utilization: f64,
    #[serde(default)]
    pub dec_utilization: f64,

    // Power / thermal
    #[serde(default)]
    pub power_watts: f64,
    #[serde(default)]
    pub power_limit_watts: f64,
    #[serde(default)]
    pub temperature: f64,
    #[serde(default)]
    pub mem_temperature: f64,

    // Memory
    #[serde(default)]
    pub mem_used_bytes: u64,
    #[serde(default)]
    pub mem_total_bytes: u64,

    // Clocks
    #[serde(default)]
    pub clock_mhz: u32,
    #[serde(default)]
    pub mem_clock_mhz: u32,

    // I/O bandwidth
    #[serde(default)]
    pub pcie_tx_kbps: u64,
    #[serde(default)]
    pub pcie_rx_kbps: u64,
    #[serde(default)]
    pub pcie_prof_tx_bytes_per_sec: u64,
    #[serde(default)]
    pub pcie_prof_rx_bytes_per_sec: u64,
    #[serde(default)]
    pub nvlink_tx_kbps: u64,
    #[serde(default)]
    pub nvlink_rx_kbps: u64,

    // Counters / error state
    #[serde(default)]
    pub throttle_reasons: u64,
    #[serde(default)]
    pub ecc_sbe_volatile: u64,
    #[serde(default)]
    pub ecc_dbe_volatile: u64,
    #[serde(default)]
    pub total_energy_mj: u64,
    #[serde(default)]
    pub pcie_replay_count: u64,
    #[serde(default)]
    pub xid_errors: u64,
    #[serde(default)]
    pub remapped_rows_correctable: u64,
    #[serde(default)]
    pub remapped_rows_uncorrectable: u64,
    #[serde(default)]
    pub row_remap_failure: u64,
    #[serde(default)]
    pub vgpu_license_status: u64,
}

impl From<&GpuMetrics> for GpuSample {
    fn from(g: &GpuMetrics) -> Self {
        Self {
            index: g.index,
            uuid: g.uuid.clone(),
            name: g.name.clone(),
            utilization: g.utilization,
            mem_utilization: g.mem_utilization,
            dram_active: g.dram_active,
            gr_engine_active: g.gr_engine_active,
            tensor_active: g.tensor_active,
            enc_utilization: g.enc_utilization,
            dec_utilization: g.dec_utilization,
            power_watts: g.power_watts,
            power_limit_watts: g.power_limit_watts,
            temperature: g.temperature,
            mem_temperature: g.mem_temperature,
            mem_used_bytes: g.mem_used_bytes,
            mem_total_bytes: g.mem_total_bytes,
            clock_mhz: g.clock_mhz,
            mem_clock_mhz: g.mem_clock_mhz,
            pcie_tx_kbps: g.pcie_tx_kbps,
            pcie_rx_kbps: g.pcie_rx_kbps,
            pcie_prof_tx_bytes_per_sec: g.pcie_prof_tx_bytes_per_sec,
            pcie_prof_rx_bytes_per_sec: g.pcie_prof_rx_bytes_per_sec,
            nvlink_tx_kbps: g.nvlink_tx_kbps,
            nvlink_rx_kbps: g.nvlink_rx_kbps,
            throttle_reasons: g.throttle_reasons,
            ecc_sbe_volatile: g.ecc_sbe_volatile,
            ecc_dbe_volatile: g.ecc_dbe_volatile,
            total_energy_mj: g.total_energy_mj,
            pcie_replay_count: g.pcie_replay_count,
            xid_errors: g.xid_errors,
            remapped_rows_correctable: g.remapped_rows_correctable,
            remapped_rows_uncorrectable: g.remapped_rows_uncorrectable,
            row_remap_failure: g.row_remap_failure,
            vgpu_license_status: g.vgpu_license_status,
        }
    }
}

impl From<&GpuSample> for GpuMetrics {
    fn from(s: &GpuSample) -> Self {
        Self {
            index: s.index,
            name: s.name.clone(),
            uuid: s.uuid.clone(),
            utilization: s.utilization,
            mem_utilization: s.mem_utilization,
            power_watts: s.power_watts,
            temperature: s.temperature,
            mem_used_bytes: s.mem_used_bytes,
            mem_total_bytes: s.mem_total_bytes,
            clock_mhz: s.clock_mhz,
            mem_clock_mhz: s.mem_clock_mhz,
            pcie_tx_kbps: s.pcie_tx_kbps,
            pcie_rx_kbps: s.pcie_rx_kbps,
            nvlink_tx_kbps: s.nvlink_tx_kbps,
            nvlink_rx_kbps: s.nvlink_rx_kbps,
            power_limit_watts: s.power_limit_watts,
            throttle_reasons: s.throttle_reasons,
            ecc_sbe_volatile: s.ecc_sbe_volatile,
            ecc_dbe_volatile: s.ecc_dbe_volatile,
            dram_active: s.dram_active,
            gr_engine_active: s.gr_engine_active,
            tensor_active: s.tensor_active,
            enc_utilization: s.enc_utilization,
            dec_utilization: s.dec_utilization,
            mem_temperature: s.mem_temperature,
            total_energy_mj: s.total_energy_mj,
            pcie_replay_count: s.pcie_replay_count,
            xid_errors: s.xid_errors,
            remapped_rows_correctable: s.remapped_rows_correctable,
            remapped_rows_uncorrectable: s.remapped_rows_uncorrectable,
            row_remap_failure: s.row_remap_failure,
            vgpu_license_status: s.vgpu_license_status,
            pcie_prof_tx_bytes_per_sec: s.pcie_prof_tx_bytes_per_sec,
            pcie_prof_rx_bytes_per_sec: s.pcie_prof_rx_bytes_per_sec,
        }
    }
}

/// Per-IB-device values for one scrape tick. Mirrors `crate::ib::IbDevice` for
/// serialization into the collect report.
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct IbSample {
    // Identity
    #[serde(default)]
    pub device: String,
    #[serde(default)]
    pub port: u32,

    // Computed rates
    #[serde(default)]
    pub tx_gbps: f64,
    #[serde(default)]
    pub rx_gbps: f64,

    // Cumulative counters
    #[serde(default)]
    pub tx_bytes_total: u64,
    #[serde(default)]
    pub rx_bytes_total: u64,
    #[serde(default)]
    pub tx_packets_total: u64,
    #[serde(default)]
    pub rx_packets_total: u64,

    // Link state
    #[serde(default)]
    pub state_id: u8,
    #[serde(default)]
    pub rate_bytes_per_sec: u64,
    /// Theoretical one-direction peak link Gbps (None if not reported).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub link_gbps: Option<f64>,
}

impl From<&IbDevice> for IbSample {
    fn from(d: &IbDevice) -> Self {
        Self {
            device: d.device.clone(),
            port: d.port,
            tx_gbps: d.tx_gbps,
            rx_gbps: d.rx_gbps,
            tx_bytes_total: d.tx_bytes_total,
            rx_bytes_total: d.rx_bytes_total,
            tx_packets_total: d.tx_packets_total,
            rx_packets_total: d.rx_packets_total,
            state_id: d.state_id,
            rate_bytes_per_sec: d.rate_bytes_per_sec,
            link_gbps: d.link_gbps(),
        }
    }
}

impl From<&IbSample> for IbDevice {
    fn from(s: &IbSample) -> Self {
        Self {
            device: s.device.clone(),
            port: s.port,
            tx_bytes_total: s.tx_bytes_total,
            rx_bytes_total: s.rx_bytes_total,
            tx_packets_total: s.tx_packets_total,
            rx_packets_total: s.rx_packets_total,
            state_id: s.state_id,
            rate_bytes_per_sec: s.rate_bytes_per_sec,
            tx_gbps: s.tx_gbps,
            rx_gbps: s.rx_gbps,
        }
    }
}

/// A single time-series sample for one node.
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct NodeSample {
    // Runtime
    #[serde(default)]
    pub kv_cache: f64,
    #[serde(default)]
    pub running: f64,
    #[serde(default)]
    pub waiting: f64,
    #[serde(default)]
    pub generation_tps: f64,
    #[serde(default)]
    pub prompt_tps: f64,

    // Latency — windowed (ms)
    #[serde(default)]
    pub ttft_p50_ms: f64,
    #[serde(default)]
    pub ttft_p90_ms: f64,
    #[serde(default)]
    pub ttft_p99_ms: f64,
    #[serde(default)]
    pub ttft_mean_ms: f64,
    #[serde(default)]
    pub itl_p50_ms: f64,
    #[serde(default)]
    pub itl_p90_ms: f64,
    #[serde(default)]
    pub itl_p99_ms: f64,
    #[serde(default)]
    pub itl_mean_ms: f64,
    #[serde(default)]
    pub e2e_p50_ms: f64,
    #[serde(default)]
    pub e2e_p90_ms: f64,
    #[serde(default)]
    pub e2e_p99_ms: f64,
    #[serde(default)]
    pub e2e_mean_ms: f64,
    #[serde(default)]
    pub queue_p50_ms: f64,
    #[serde(default)]
    pub queue_p90_ms: f64,
    #[serde(default)]
    pub queue_p99_ms: f64,
    #[serde(default)]
    pub queue_mean_ms: f64,
    #[serde(default)]
    pub prefill_p50_ms: f64,
    #[serde(default)]
    pub prefill_p90_ms: f64,
    #[serde(default)]
    pub prefill_p99_ms: f64,
    #[serde(default)]
    pub prefill_mean_ms: f64,
    #[serde(default)]
    pub decode_p50_ms: f64,
    #[serde(default)]
    pub decode_p90_ms: f64,
    #[serde(default)]
    pub decode_p99_ms: f64,
    #[serde(default)]
    pub decode_mean_ms: f64,
    #[serde(default)]
    pub inference_p50_ms: f64,
    #[serde(default)]
    pub inference_p90_ms: f64,
    #[serde(default)]
    pub inference_p99_ms: f64,
    #[serde(default)]
    pub inference_mean_ms: f64,
    #[serde(default)]
    pub tpot_p50_ms: f64,
    #[serde(default)]
    pub tpot_p90_ms: f64,
    #[serde(default)]
    pub tpot_p99_ms: f64,
    #[serde(default)]
    pub tpot_mean_ms: f64,

    // Cache
    #[serde(default)]
    pub prefix_cache_hit_rate: f64,
    #[serde(default)]
    pub external_cache_hit_rate: f64,
    #[serde(default)]
    pub mm_cache_hit_rate: f64,

    // Speculative decoding
    #[serde(default)]
    pub spec_decode_acceptance_rate: f64,
    #[serde(default)]
    pub spec_decode_drafts_per_sec: f64,

    // Performance (per GPU)
    #[serde(default)]
    pub estimated_flops_per_gpu_per_sec: f64,
    #[serde(default)]
    pub estimated_read_bytes_per_gpu_per_sec: f64,
    #[serde(default)]
    pub estimated_write_bytes_per_gpu_per_sec: f64,
    pub mfu_percent: Option<f64>,

    // Request stats
    #[serde(default)]
    pub avg_prompt_tokens: f64,
    #[serde(default)]
    pub avg_generation_tokens: f64,
    #[serde(default)]
    pub iteration_tokens_mean: f64,
    #[serde(default)]
    pub preemptions_per_sec: f64,
    #[serde(default)]
    pub requests_per_sec: f64,

    // HTTP
    #[serde(default)]
    pub http_qps: f64,
    #[serde(default)]
    pub http_error_rate: f64,

    // Server load
    pub server_load: Option<f64>,

    // GPU hardware (aggregated across all GPUs on the node)
    pub gpu_utilization: Option<f64>,
    pub gpu_mem_utilization: Option<f64>,
    pub gpu_power_watts: Option<f64>,
    pub gpu_temperature: Option<f64>,
    pub gpu_vram_used_bytes: Option<u64>,
    pub gpu_vram_total_bytes: Option<u64>,
    pub gpu_nvlink_tx_kbps: Option<u64>,
    pub gpu_nvlink_rx_kbps: Option<u64>,

    /// Per-GPU values for this tick. Empty when no GPU scrape succeeded on this
    /// node; otherwise one entry per GPU visible to this node (partitioned by
    /// the scraper when multiple vLLM processes share a host).
    #[serde(default)]
    pub gpus: Vec<GpuSample>,

    // InfiniBand aggregate (per-host, summed across active devices).
    pub ib_total_tx_gbps: Option<f64>,
    pub ib_total_rx_gbps: Option<f64>,
    pub ib_active_count: Option<u32>,
    pub ib_total_link_gbps: Option<f64>,

    /// Per-IB-device values for this tick. Empty when no IB scrape succeeded
    /// on this host. IB is per-host shared infrastructure: when multiple vLLM
    /// processes share a host, each peer carries the same `ibs[]`.
    #[serde(default)]
    pub ibs: Vec<IbSample>,

    // === Dynamo Frontend ===
    /// Dynamo worker role from `dynamo_component` label. None for non-Dynamo.
    /// Examples: "prefill", "backend" (decode); aggregated hosts may report
    /// "P", "D", "P+D".
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub dynamo_role: Option<String>,
    #[serde(default)]
    pub has_dynamo_config: bool,
    #[serde(default)]
    pub dynamo_uptime_secs: f64,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub dynamo_component_inflight: Option<f64>,
    #[serde(default)]
    pub dynamo_component_requests_per_sec: f64,
    #[serde(default)]
    pub dynamo_context_length: u64,
    #[serde(default)]
    pub dynamo_total_kv_blocks: u64,
    #[serde(default)]
    pub dynamo_kv_block_size: u64,
    #[serde(default)]
    pub dynamo_max_num_seqs: u64,
    #[serde(default)]
    pub dynamo_max_num_batched_tokens: u64,
    #[serde(default)]
    pub dynamo_disconnected_clients: f64,
    #[serde(default)]
    pub dynamo_model_load_secs: f64,
    #[serde(default)]
    pub dynamo_tokenize_p50_ms: f64,
    #[serde(default)]
    pub dynamo_tokenize_p99_ms: f64,
    #[serde(default)]
    pub dynamo_detokenize_p50_ms: f64,
    #[serde(default)]
    pub dynamo_detokenize_p99_ms: f64,

    // === NIXL KV Connector ===
    #[serde(default)]
    pub has_nixl: bool,
    #[serde(default)]
    pub nixl_failed_transfers_total: u64,
    #[serde(default)]
    pub nixl_failed_notifications_total: u64,
    #[serde(default)]
    pub nixl_kv_expired_reqs_total: u64,
    #[serde(default)]
    pub nixl_xfer_p50_ms: f64,
    #[serde(default)]
    pub nixl_xfer_p99_ms: f64,
    #[serde(default)]
    pub nixl_post_p50_ms: f64,
    #[serde(default)]
    pub nixl_post_p99_ms: f64,
    #[serde(default)]
    pub nixl_avg_bytes_transferred: f64,
    #[serde(default)]
    pub nixl_avg_descriptors: f64,
    #[serde(default)]
    pub nixl_transfers_per_sec: f64,
    #[serde(default)]
    pub nixl_throughput_mb_per_sec: f64,
    #[serde(default)]
    pub external_kv_transfer_tokens_per_sec: f64,

    // === Async Remote-KV Fetch Stages ===
    #[serde(default)]
    pub has_kv_fetch: bool,
    #[serde(default)]
    pub kv_fetch_waiting_to_start: f64,
    #[serde(default)]
    pub kv_fetch_in_progress: f64,
    #[serde(default)]
    pub kv_fetch_completed_waiting: f64,

    // === KV Cache Block Residency (--kv-cache-metrics) ===
    #[serde(default)]
    pub has_kv_block_metrics: bool,
    #[serde(default)]
    pub kv_block_lifetime_p50_ms: f64,
    #[serde(default)]
    pub kv_block_lifetime_p99_ms: f64,
    #[serde(default)]
    pub kv_block_idle_before_evict_p50_ms: f64,
    #[serde(default)]
    pub kv_block_idle_before_evict_p99_ms: f64,
    #[serde(default)]
    pub kv_block_reuse_gap_p50_ms: f64,
    #[serde(default)]
    pub kv_block_reuse_gap_p99_ms: f64,

    // === KV Cache Events (ZMQ) ===
    /// Per-DP-rank KV event aggregates. None if ZMQ events aren't being
    /// subscribed; otherwise one entry per DP rank (index = rank).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub kv_events: Option<Vec<KvEventSample>>,

    // === Per-engine breakdown (DP / Dynamo per-rank) ===
    /// Per-engine (DP rank) sub-rows. `None` for non-Dynamo single-engine
    /// hosts. When present, ToggleEngines (`e`) in the TUI expands the host
    /// row into one sub-row per entry.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub engine_metrics: Option<Vec<NodeSample>>,
}

/// Mirror of `crate::kv_events::KVEventMetrics` for serde round-trip in the
/// replay JSON.
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct KvEventSample {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub dp_rank: Option<u16>,
    #[serde(default)]
    pub blocks_stored: u64,
    #[serde(default)]
    pub blocks_removed: u64,
    #[serde(default)]
    pub tokens_stored: u64,
    #[serde(default)]
    pub active_blocks: i64,
    #[serde(default)]
    pub total_events: u64,
    #[serde(default)]
    pub seq_gaps: u64,
}

impl From<&KVEventMetrics> for KvEventSample {
    fn from(m: &KVEventMetrics) -> Self {
        Self {
            dp_rank: m.dp_rank,
            blocks_stored: m.blocks_stored,
            blocks_removed: m.blocks_removed,
            tokens_stored: m.tokens_stored,
            active_blocks: m.active_blocks,
            total_events: m.total_events,
            seq_gaps: m.seq_gaps,
        }
    }
}

impl From<&KvEventSample> for KVEventMetrics {
    fn from(s: &KvEventSample) -> Self {
        Self {
            dp_rank: s.dp_rank,
            blocks_stored: s.blocks_stored,
            blocks_removed: s.blocks_removed,
            tokens_stored: s.tokens_stored,
            active_blocks: s.active_blocks,
            total_events: s.total_events,
            seq_gaps: s.seq_gaps,
        }
    }
}

impl From<&NodeMetrics> for NodeSample {
    fn from(n: &NodeMetrics) -> Self {
        Self {
            kv_cache: n.kv_cache_usage,
            running: n.requests_running,
            waiting: n.requests_waiting,
            generation_tps: n.generation_tps,
            prompt_tps: n.prompt_tps,
            ttft_p50_ms: n.win_ttft.p50,
            ttft_p90_ms: n.win_ttft.p90,
            ttft_p99_ms: n.win_ttft.p99,
            ttft_mean_ms: n.win_ttft.mean,
            itl_p50_ms: n.win_itl.p50,
            itl_p90_ms: n.win_itl.p90,
            itl_p99_ms: n.win_itl.p99,
            itl_mean_ms: n.win_itl.mean,
            e2e_p50_ms: n.win_e2e.p50,
            e2e_p90_ms: n.win_e2e.p90,
            e2e_p99_ms: n.win_e2e.p99,
            e2e_mean_ms: n.win_e2e.mean,
            queue_p50_ms: n.win_queue.p50,
            queue_p90_ms: n.win_queue.p90,
            queue_p99_ms: n.win_queue.p99,
            queue_mean_ms: n.win_queue.mean,
            prefill_p50_ms: n.win_prefill.p50,
            prefill_p90_ms: n.win_prefill.p90,
            prefill_p99_ms: n.win_prefill.p99,
            prefill_mean_ms: n.win_prefill.mean,
            decode_p50_ms: n.win_decode.p50,
            decode_p90_ms: n.win_decode.p90,
            decode_p99_ms: n.win_decode.p99,
            decode_mean_ms: n.win_decode.mean,
            inference_p50_ms: n.win_inference.p50,
            inference_p90_ms: n.win_inference.p90,
            inference_p99_ms: n.win_inference.p99,
            inference_mean_ms: n.win_inference.mean,
            tpot_p50_ms: n.win_tpot.p50,
            tpot_p90_ms: n.win_tpot.p90,
            tpot_p99_ms: n.win_tpot.p99,
            tpot_mean_ms: n.win_tpot.mean,
            prefix_cache_hit_rate: n.prefix_cache_hit_rate,
            external_cache_hit_rate: n.external_cache_hit_rate,
            mm_cache_hit_rate: n.mm_cache_hit_rate,
            spec_decode_acceptance_rate: n.spec_decode_acceptance_rate,
            spec_decode_drafts_per_sec: n.spec_decode_drafts_per_sec,
            estimated_flops_per_gpu_per_sec: n.estimated_flops_per_gpu_per_sec,
            estimated_read_bytes_per_gpu_per_sec: n.estimated_read_bytes_per_gpu_per_sec,
            estimated_write_bytes_per_gpu_per_sec: n.estimated_write_bytes_per_gpu_per_sec,
            mfu_percent: n.mfu_percent,
            avg_prompt_tokens: n.avg_prompt_tokens,
            avg_generation_tokens: n.avg_generation_tokens,
            iteration_tokens_mean: n.iteration_tokens_mean,
            preemptions_per_sec: n.preemptions_per_sec,
            requests_per_sec: n.requests_per_sec,
            http_qps: n.http_qps,
            http_error_rate: n.http_error_rate,
            server_load: n.server_load,
            gpu_utilization: n.gpu_scrape.as_ref().map(|g| g.avg_utilization()),
            gpu_mem_utilization: n.gpu_scrape.as_ref().map(|g| g.avg_mem_utilization()),
            gpu_power_watts: n.gpu_scrape.as_ref().map(|g| g.total_power()),
            gpu_temperature: n.gpu_scrape.as_ref().map(|g| g.avg_temperature()),
            gpu_vram_used_bytes: n.gpu_scrape.as_ref().map(|g| g.mem_used_total().0),
            gpu_vram_total_bytes: n.gpu_scrape.as_ref().map(|g| g.mem_used_total().1),
            gpu_nvlink_tx_kbps: n.gpu_scrape.as_ref().map(|g| g.total_nvlink_tx_kbps()),
            gpu_nvlink_rx_kbps: n.gpu_scrape.as_ref().map(|g| g.total_nvlink_rx_kbps()),
            gpus: n
                .gpu_scrape
                .as_ref()
                .map(|g| g.gpus.iter().map(GpuSample::from).collect())
                .unwrap_or_default(),
            ib_total_tx_gbps: n.ib_scrape.as_ref().map(|i| i.total_tx_gbps()),
            ib_total_rx_gbps: n.ib_scrape.as_ref().map(|i| i.total_rx_gbps()),
            ib_active_count: n.ib_scrape.as_ref().map(|i| i.active_count() as u32),
            ib_total_link_gbps: n.ib_scrape.as_ref().map(|i| i.total_link_gbps()),
            ibs: n
                .ib_scrape
                .as_ref()
                .map(|i| i.devices.iter().map(IbSample::from).collect())
                .unwrap_or_default(),
            // Dynamo Frontend
            dynamo_role: n.dynamo_role.clone(),
            has_dynamo_config: n.has_dynamo_config,
            dynamo_uptime_secs: n.dynamo_uptime_secs,
            dynamo_component_inflight: n.dynamo_component_inflight,
            dynamo_component_requests_per_sec: n.dynamo_component_requests_per_sec,
            dynamo_context_length: n.dynamo_context_length,
            dynamo_total_kv_blocks: n.dynamo_total_kv_blocks,
            dynamo_kv_block_size: n.dynamo_kv_block_size,
            dynamo_max_num_seqs: n.dynamo_max_num_seqs,
            dynamo_max_num_batched_tokens: n.dynamo_max_num_batched_tokens,
            dynamo_disconnected_clients: n.dynamo_disconnected_clients,
            dynamo_model_load_secs: n.dynamo_model_load_secs,
            dynamo_tokenize_p50_ms: n.dynamo_tokenize_latency.p50,
            dynamo_tokenize_p99_ms: n.dynamo_tokenize_latency.p99,
            dynamo_detokenize_p50_ms: n.dynamo_detokenize_latency.p50,
            dynamo_detokenize_p99_ms: n.dynamo_detokenize_latency.p99,
            // NIXL
            has_nixl: n.has_nixl,
            nixl_failed_transfers_total: n.nixl_failed_transfers_total,
            nixl_failed_notifications_total: n.nixl_failed_notifications_total,
            nixl_kv_expired_reqs_total: n.nixl_kv_expired_reqs_total,
            nixl_xfer_p50_ms: n.nixl_xfer_time.p50,
            nixl_xfer_p99_ms: n.nixl_xfer_time.p99,
            nixl_post_p50_ms: n.nixl_post_time.p50,
            nixl_post_p99_ms: n.nixl_post_time.p99,
            nixl_avg_bytes_transferred: n.nixl_avg_bytes_transferred,
            nixl_avg_descriptors: n.nixl_avg_descriptors,
            nixl_transfers_per_sec: n.nixl_transfers_per_sec,
            nixl_throughput_mb_per_sec: n.nixl_throughput_mb_per_sec,
            external_kv_transfer_tokens_per_sec: n.external_kv_transfer_tokens_per_sec,
            // Async remote-KV fetch stages
            has_kv_fetch: n.has_kv_fetch,
            kv_fetch_waiting_to_start: n.kv_fetch_waiting_to_start,
            kv_fetch_in_progress: n.kv_fetch_in_progress,
            kv_fetch_completed_waiting: n.kv_fetch_completed_waiting,
            // KV block residency
            has_kv_block_metrics: n.has_kv_block_metrics,
            kv_block_lifetime_p50_ms: n.kv_block_lifetime.p50,
            kv_block_lifetime_p99_ms: n.kv_block_lifetime.p99,
            kv_block_idle_before_evict_p50_ms: n.kv_block_idle_before_evict.p50,
            kv_block_idle_before_evict_p99_ms: n.kv_block_idle_before_evict.p99,
            kv_block_reuse_gap_p50_ms: n.kv_block_reuse_gap.p50,
            kv_block_reuse_gap_p99_ms: n.kv_block_reuse_gap.p99,
            // KV cache events
            kv_events: n.kv_events.as_ref().map(|v| v.iter().map(KvEventSample::from).collect()),
            // Per-engine breakdown (recursive)
            engine_metrics: n
                .engine_metrics
                .as_ref()
                .map(|engines| engines.iter().map(NodeSample::from).collect()),
        }
    }
}

/// Replace a non-finite f64 with 0.0. Used by the sanitize helpers below.
/// serde_json rejects NaN / +Inf / -Inf and would error on the entire
/// sample, so we coerce upstream and lose only the offending field.
#[inline]
fn fix_f64(v: &mut f64) {
    if !v.is_finite() {
        *v = 0.0;
    }
}

#[inline]
fn fix_opt_f64(v: &mut Option<f64>) {
    if let Some(x) = v.as_mut() {
        if !x.is_finite() {
            *x = 0.0;
        }
    }
}

impl NodeSample {
    /// Replace every non-finite f64 field (NaN / ±Inf) — including those in
    /// nested `gpus[]` / `ibs[]` — with `0.0`. Required before
    /// `serde_json::to_value`, which errors on any non-finite number and
    /// would otherwise drop the entire NodeSample.
    pub fn sanitize(&mut self) {
        fix_f64(&mut self.kv_cache);
        fix_f64(&mut self.running);
        fix_f64(&mut self.waiting);
        fix_f64(&mut self.generation_tps);
        fix_f64(&mut self.prompt_tps);
        fix_f64(&mut self.ttft_p50_ms);
        fix_f64(&mut self.ttft_p99_ms);
        fix_f64(&mut self.itl_p50_ms);
        fix_f64(&mut self.itl_p99_ms);
        fix_f64(&mut self.e2e_p50_ms);
        fix_f64(&mut self.e2e_p99_ms);
        fix_f64(&mut self.queue_p50_ms);
        fix_f64(&mut self.queue_p99_ms);
        fix_f64(&mut self.prefill_p50_ms);
        fix_f64(&mut self.prefill_p99_ms);
        fix_f64(&mut self.decode_p50_ms);
        fix_f64(&mut self.decode_p99_ms);
        fix_f64(&mut self.inference_p50_ms);
        fix_f64(&mut self.inference_p99_ms);
        fix_f64(&mut self.tpot_p50_ms);
        fix_f64(&mut self.tpot_p99_ms);
        fix_f64(&mut self.prefix_cache_hit_rate);
        fix_f64(&mut self.external_cache_hit_rate);
        fix_f64(&mut self.mm_cache_hit_rate);
        fix_f64(&mut self.spec_decode_acceptance_rate);
        fix_f64(&mut self.spec_decode_drafts_per_sec);
        fix_f64(&mut self.estimated_flops_per_gpu_per_sec);
        fix_f64(&mut self.estimated_read_bytes_per_gpu_per_sec);
        fix_f64(&mut self.estimated_write_bytes_per_gpu_per_sec);
        fix_opt_f64(&mut self.mfu_percent);
        fix_f64(&mut self.avg_prompt_tokens);
        fix_f64(&mut self.avg_generation_tokens);
        fix_f64(&mut self.iteration_tokens_mean);
        fix_f64(&mut self.preemptions_per_sec);
        fix_f64(&mut self.requests_per_sec);
        fix_f64(&mut self.http_qps);
        fix_f64(&mut self.http_error_rate);
        fix_opt_f64(&mut self.server_load);
        fix_opt_f64(&mut self.gpu_utilization);
        fix_opt_f64(&mut self.gpu_mem_utilization);
        fix_opt_f64(&mut self.gpu_power_watts);
        fix_opt_f64(&mut self.gpu_temperature);
        for g in &mut self.gpus {
            g.sanitize();
        }
        fix_opt_f64(&mut self.ib_total_tx_gbps);
        fix_opt_f64(&mut self.ib_total_rx_gbps);
        fix_opt_f64(&mut self.ib_total_link_gbps);
        // Dynamo
        fix_f64(&mut self.dynamo_uptime_secs);
        fix_opt_f64(&mut self.dynamo_component_inflight);
        fix_f64(&mut self.dynamo_component_requests_per_sec);
        fix_f64(&mut self.dynamo_disconnected_clients);
        fix_f64(&mut self.dynamo_model_load_secs);
        fix_f64(&mut self.dynamo_tokenize_p50_ms);
        fix_f64(&mut self.dynamo_tokenize_p99_ms);
        fix_f64(&mut self.dynamo_detokenize_p50_ms);
        fix_f64(&mut self.dynamo_detokenize_p99_ms);
        // NIXL
        fix_f64(&mut self.nixl_xfer_p50_ms);
        fix_f64(&mut self.nixl_xfer_p99_ms);
        fix_f64(&mut self.nixl_post_p50_ms);
        fix_f64(&mut self.nixl_post_p99_ms);
        fix_f64(&mut self.nixl_avg_bytes_transferred);
        fix_f64(&mut self.nixl_avg_descriptors);
        fix_f64(&mut self.nixl_transfers_per_sec);
        fix_f64(&mut self.nixl_throughput_mb_per_sec);
        fix_f64(&mut self.external_kv_transfer_tokens_per_sec);
        // Async remote-KV fetch stages
        fix_f64(&mut self.kv_fetch_waiting_to_start);
        fix_f64(&mut self.kv_fetch_in_progress);
        fix_f64(&mut self.kv_fetch_completed_waiting);
        // KV block residency
        fix_f64(&mut self.kv_block_lifetime_p50_ms);
        fix_f64(&mut self.kv_block_lifetime_p99_ms);
        fix_f64(&mut self.kv_block_idle_before_evict_p50_ms);
        fix_f64(&mut self.kv_block_idle_before_evict_p99_ms);
        fix_f64(&mut self.kv_block_reuse_gap_p50_ms);
        fix_f64(&mut self.kv_block_reuse_gap_p99_ms);
        // Per-engine sub-rows: recurse so DP-rank rows get sanitized too.
        if let Some(engines) = self.engine_metrics.as_mut() {
            for e in engines {
                e.sanitize();
            }
        }
        for d in &mut self.ibs {
            d.sanitize();
        }
    }
}

impl GpuSample {
    pub fn sanitize(&mut self) {
        fix_f64(&mut self.utilization);
        fix_f64(&mut self.mem_utilization);
        fix_opt_f64(&mut self.dram_active);
        fix_opt_f64(&mut self.gr_engine_active);
        fix_opt_f64(&mut self.tensor_active);
        fix_f64(&mut self.enc_utilization);
        fix_f64(&mut self.dec_utilization);
        fix_f64(&mut self.power_watts);
        fix_f64(&mut self.power_limit_watts);
        fix_f64(&mut self.temperature);
        fix_f64(&mut self.mem_temperature);
    }
}

impl IbSample {
    pub fn sanitize(&mut self) {
        fix_f64(&mut self.tx_gbps);
        fix_f64(&mut self.rx_gbps);
        fix_opt_f64(&mut self.link_gbps);
    }
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct MooncakeSegmentSample {
    #[serde(default)]
    pub segment: String,
    #[serde(default)]
    pub allocated_bytes: u64,
    #[serde(default)]
    pub total_bytes: u64,
}

impl From<&MooncakeSegment> for MooncakeSegmentSample {
    fn from(s: &MooncakeSegment) -> Self {
        Self {
            segment: s.segment.clone(),
            allocated_bytes: s.allocated_bytes,
            total_bytes: s.total_bytes,
        }
    }
}

impl From<&MooncakeSegmentSample> for MooncakeSegment {
    fn from(s: &MooncakeSegmentSample) -> Self {
        Self {
            segment: s.segment.clone(),
            allocated_bytes: s.allocated_bytes,
            total_bytes: s.total_bytes,
        }
    }
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct MooncakeHealthSample {
    #[serde(default)]
    pub role: String,
    #[serde(default)]
    pub ha_state: String,
    #[serde(default)]
    pub service_ready: bool,
}

impl From<&MooncakeHealth> for MooncakeHealthSample {
    fn from(h: &MooncakeHealth) -> Self {
        Self {
            role: h.role.clone(),
            ha_state: h.ha_state.clone(),
            service_ready: h.service_ready,
        }
    }
}

impl From<&MooncakeHealthSample> for MooncakeHealth {
    fn from(h: &MooncakeHealthSample) -> Self {
        Self {
            role: h.role.clone(),
            ha_state: h.ha_state.clone(),
            service_ready: h.service_ready,
        }
    }
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct MooncakeSample {
    #[serde(default)]
    pub addr: String,
    #[serde(default)]
    pub health: Option<MooncakeHealthSample>,
    #[serde(default)]
    pub mem_allocated_bytes: u64,
    #[serde(default)]
    pub mem_total_bytes: u64,
    #[serde(default)]
    pub mem_util: f64,
    #[serde(default)]
    pub key_count: u64,
    #[serde(default)]
    pub soft_pin_key_count: u64,
    #[serde(default)]
    pub active_clients: u64,
    #[serde(default)]
    pub total_request_rate: f64,
    #[serde(default)]
    pub failure_rate: f64,
    #[serde(default)]
    pub get_rate: f64,
    #[serde(default)]
    pub put_rate: f64,
    #[serde(default)]
    pub exist_rate: f64,
    #[serde(default)]
    pub remove_rate: f64,
    #[serde(default)]
    pub ping_rate: f64,
    #[serde(default)]
    pub eviction_rate: f64,
    #[serde(default)]
    pub eviction_bytes_rate: f64,
    #[serde(default)]
    pub segments: Vec<MooncakeSegmentSample>,
    #[serde(default)]
    pub segment_count: usize,
    #[serde(default)]
    pub segment_fill_min: f64,
    #[serde(default)]
    pub segment_fill_max: f64,
    #[serde(default)]
    pub ha_oplog_standby_lag: i64,
    #[serde(default)]
    pub ha_oplog_pending_entries: i64,
    #[serde(default)]
    pub ha_standby_state: u8,
}

impl From<&MooncakeMetrics> for MooncakeSample {
    fn from(m: &MooncakeMetrics) -> Self {
        Self {
            addr: m.addr.clone(),
            health: m.health.as_ref().map(MooncakeHealthSample::from),
            mem_allocated_bytes: m.mem_allocated_bytes,
            mem_total_bytes: m.mem_total_bytes,
            mem_util: m.mem_util,
            key_count: m.key_count,
            soft_pin_key_count: m.soft_pin_key_count,
            active_clients: m.active_clients,
            total_request_rate: m.total_request_rate,
            failure_rate: m.failure_rate,
            get_rate: m.get_rate,
            put_rate: m.put_rate,
            exist_rate: m.exist_rate,
            remove_rate: m.remove_rate,
            ping_rate: m.ping_rate,
            eviction_rate: m.eviction_rate,
            eviction_bytes_rate: m.eviction_bytes_rate,
            segments: m.segments.iter().map(MooncakeSegmentSample::from).collect(),
            segment_count: m.segment_count,
            segment_fill_min: m.segment_fill_min,
            segment_fill_max: m.segment_fill_max,
            ha_oplog_standby_lag: m.ha_oplog_standby_lag,
            ha_oplog_pending_entries: m.ha_oplog_pending_entries,
            ha_standby_state: m.ha_standby_state,
        }
    }
}

impl From<&MooncakeSample> for MooncakeMetrics {
    fn from(s: &MooncakeSample) -> Self {
        Self {
            addr: s.addr.clone(),
            health: s.health.as_ref().map(MooncakeHealth::from),
            mem_allocated_bytes: s.mem_allocated_bytes,
            mem_total_bytes: s.mem_total_bytes,
            mem_util: s.mem_util,
            key_count: s.key_count,
            soft_pin_key_count: s.soft_pin_key_count,
            active_clients: s.active_clients,
            total_request_rate: s.total_request_rate,
            failure_rate: s.failure_rate,
            get_rate: s.get_rate,
            put_rate: s.put_rate,
            exist_rate: s.exist_rate,
            remove_rate: s.remove_rate,
            ping_rate: s.ping_rate,
            eviction_rate: s.eviction_rate,
            eviction_bytes_rate: s.eviction_bytes_rate,
            segments: s.segments.iter().map(MooncakeSegment::from).collect(),
            segment_count: s.segment_count,
            segment_fill_min: s.segment_fill_min,
            segment_fill_max: s.segment_fill_max,
            ha_oplog_standby_lag: s.ha_oplog_standby_lag,
            ha_oplog_pending_entries: s.ha_oplog_pending_entries,
            ha_standby_state: s.ha_standby_state,
        }
    }
}

/// A time-stamped cluster sample.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TimeSample {
    pub elapsed_secs: f64,
    /// Unix epoch milliseconds at which this sample was recorded. Both the
    /// vLLM-side and GPU/DCGM-side numbers in this sample share this
    /// timestamp — they were scraped within the same tick of the scrape loop.
    #[serde(default)]
    pub timestamp_ms: u64,
    pub nodes: HashMap<String, NodeSample>,
    /// First (or only) Mooncake store — kept as a scalar for backward
    /// compatibility with existing consumers of recorded JSON.
    #[serde(default)]
    pub mooncake: Option<MooncakeSample>,
    /// Full store list when more than one is tracked (multi-job auto mode).
    /// Includes the first store again; readers should prefer this field when
    /// non-empty and fall back to `mooncake` otherwise. Omitted (empty) on
    /// single-store recordings so their format is unchanged.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub mooncakes: Vec<MooncakeSample>,
}
