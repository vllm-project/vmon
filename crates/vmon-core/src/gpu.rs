// SPDX-License-Identifier: Apache-2.0

use std::collections::HashMap;

use crate::parser::MetricFamily;

/// Per-GPU metrics from a single scrape.
#[derive(Debug, Clone, Default)]
pub struct GpuMetrics {
    pub index: u32,
    pub name: String,
    /// GPU UUID (populated from DCGM `UUID` label; empty for `vmon agent` source).
    pub uuid: String,
    pub utilization: f64,
    pub mem_utilization: f64,
    pub power_watts: f64,
    pub temperature: f64,
    pub mem_used_bytes: u64,
    pub mem_total_bytes: u64,
    pub clock_mhz: u32,
    pub mem_clock_mhz: u32,
    /// PCIe TX throughput in KB/s
    pub pcie_tx_kbps: u64,
    /// PCIe RX throughput in KB/s
    pub pcie_rx_kbps: u64,
    /// NVLink TX throughput in KB/s
    pub nvlink_tx_kbps: u64,
    /// NVLink RX throughput in KB/s
    pub nvlink_rx_kbps: u64,
    /// Power management limit in watts
    pub power_limit_watts: f64,
    /// Clock throttle reasons bitmask
    pub throttle_reasons: u64,
    /// Corrected (single-bit) ECC errors since boot
    pub ecc_sbe_volatile: u64,
    /// Uncorrected (double-bit) ECC errors since boot
    pub ecc_dbe_volatile: u64,

    // ── DCGM profile activity fractions (0.0 – 1.0), populated directly from
    // DCGM_FI_PROF_* metrics. None when DCGM Profiling is disabled cluster-wide.
    /// DRAM bus active fraction (DCGM_FI_PROF_DRAM_ACTIVE).
    pub dram_active: Option<f64>,
    /// Graphics engine active fraction (DCGM_FI_PROF_GR_ENGINE_ACTIVE).
    pub gr_engine_active: Option<f64>,
    /// Tensor pipe active fraction (DCGM_FI_PROF_PIPE_TENSOR_ACTIVE).
    pub tensor_active: Option<f64>,

    /// Video encoder utilization (DCGM_FI_DEV_ENC_UTIL), percent.
    pub enc_utilization: f64,
    /// Video decoder utilization (DCGM_FI_DEV_DEC_UTIL), percent.
    pub dec_utilization: f64,

    /// HBM memory temperature (DCGM_FI_DEV_MEMORY_TEMP), °C.
    pub mem_temperature: f64,

    /// Cumulative energy consumption since boot (DCGM_FI_DEV_TOTAL_ENERGY_CONSUMPTION), mJ.
    pub total_energy_mj: u64,
    /// Cumulative PCIe replay count (DCGM_FI_DEV_PCIE_REPLAY_COUNTER).
    pub pcie_replay_count: u64,
    /// Cumulative XID error count (DCGM_FI_DEV_XID_ERRORS).
    pub xid_errors: u64,
    /// Correctable remapped row count (DCGM_FI_DEV_CORRECTABLE_REMAPPED_ROWS).
    pub remapped_rows_correctable: u64,
    /// Uncorrectable remapped row count (DCGM_FI_DEV_UNCORRECTABLE_REMAPPED_ROWS).
    pub remapped_rows_uncorrectable: u64,
    /// Row-remap failure flag (DCGM_FI_DEV_ROW_REMAP_FAILURE), 0 or 1.
    pub row_remap_failure: u64,
    /// vGPU license status (DCGM_FI_DEV_VGPU_LICENSE_STATUS).
    pub vgpu_license_status: u64,

    /// PCIe profile TX throughput in bytes/sec (DCGM_FI_PROF_PCIE_TX_BYTES).
    pub pcie_prof_tx_bytes_per_sec: u64,
    /// PCIe profile RX throughput in bytes/sec (DCGM_FI_PROF_PCIE_RX_BYTES).
    pub pcie_prof_rx_bytes_per_sec: u64,
}

impl GpuMetrics {
    /// Per-GPU PCIe TX in GB/s (binary), preferring DCGM PROF over DEV.
    pub fn pcie_tx_gbps(&self) -> f64 {
        if self.pcie_prof_tx_bytes_per_sec > 0 {
            self.pcie_prof_tx_bytes_per_sec as f64 / (1024.0 * 1024.0 * 1024.0)
        } else {
            self.pcie_tx_kbps as f64 / (1024.0 * 1024.0)
        }
    }

    /// Per-GPU PCIe RX in GB/s (binary), preferring DCGM PROF over DEV.
    pub fn pcie_rx_gbps(&self) -> f64 {
        if self.pcie_prof_rx_bytes_per_sec > 0 {
            self.pcie_prof_rx_bytes_per_sec as f64 / (1024.0 * 1024.0 * 1024.0)
        } else {
            self.pcie_rx_kbps as f64 / (1024.0 * 1024.0)
        }
    }

    /// GPU activity percent (0-100), preferring DCGM PROF GR_ENGINE_ACTIVE
    /// (cycle-accurate) over DCGM_FI_DEV_GPU_UTIL (sample jitter, can mislead).
    /// Falls back to the DEV value when Profiling is disabled.
    pub fn gpu_pct(&self) -> f64 {
        match self.gr_engine_active {
            Some(v) => v * 100.0,
            None => self.utilization,
        }
    }

    /// Memory bandwidth percent (0-100), preferring DCGM PROF DRAM_ACTIVE
    /// over DCGM_FI_DEV_MEM_COPY_UTIL. Falls back when Profiling is disabled.
    pub fn mbw_pct(&self) -> f64 {
        match self.dram_active {
            Some(v) => v * 100.0,
            None => self.mem_utilization,
        }
    }

    /// Tensor pipe active percent (0-100). None when DCGM Profiling is
    /// disabled — there's no DEV-side fallback for tensor activity.
    pub fn tc_pct(&self) -> Option<f64> {
        self.tensor_active.map(|v| v * 100.0)
    }
}

/// All GPUs on one node, plus host-level CPU/memory metrics.
#[derive(Debug, Clone, Default)]
pub struct GpuScrape {
    pub gpus: Vec<GpuMetrics>,
    /// Host CPU utilization 0-100%.
    pub cpu_percent: Option<f64>,
    /// Host memory used in bytes.
    pub mem_used_bytes: Option<u64>,
    /// Host memory total in bytes.
    pub mem_total_bytes: Option<u64>,
}

impl GpuScrape {
    pub fn avg_utilization(&self) -> f64 {
        if self.gpus.is_empty() {
            return 0.0;
        }
        self.gpus.iter().map(|g| g.utilization).sum::<f64>() / self.gpus.len() as f64
    }

    pub fn avg_mem_utilization(&self) -> f64 {
        if self.gpus.is_empty() {
            return 0.0;
        }
        self.gpus.iter().map(|g| g.mem_utilization).sum::<f64>() / self.gpus.len() as f64
    }

    /// Average GPU% across GPUs, preferring DCGM PROF GR_ENGINE_ACTIVE over
    /// the unreliable DCGM_FI_DEV_GPU_UTIL. Falls back per-GPU when Profiling
    /// is disabled.
    pub fn avg_gpu_pct(&self) -> f64 {
        if self.gpus.is_empty() {
            return 0.0;
        }
        self.gpus.iter().map(|g| g.gpu_pct()).sum::<f64>() / self.gpus.len() as f64
    }

    /// Average MBW% across GPUs, preferring DCGM PROF DRAM_ACTIVE.
    pub fn avg_mbw_pct(&self) -> f64 {
        if self.gpus.is_empty() {
            return 0.0;
        }
        self.gpus.iter().map(|g| g.mbw_pct()).sum::<f64>() / self.gpus.len() as f64
    }

    /// Average TC% across GPUs that report tensor activity. Returns None when
    /// DCGM Profiling is disabled (no GPU has the metric).
    pub fn avg_tc_pct(&self) -> Option<f64> {
        let vals: Vec<f64> = self.gpus.iter().filter_map(|g| g.tc_pct()).collect();
        if vals.is_empty() {
            None
        } else {
            Some(vals.iter().sum::<f64>() / vals.len() as f64)
        }
    }

    pub fn total_power(&self) -> f64 {
        self.gpus.iter().map(|g| g.power_watts).sum()
    }

    pub fn avg_temperature(&self) -> f64 {
        if self.gpus.is_empty() {
            return 0.0;
        }
        self.gpus.iter().map(|g| g.temperature).sum::<f64>() / self.gpus.len() as f64
    }

    pub fn mem_used_total(&self) -> (u64, u64) {
        let used: u64 = self.gpus.iter().map(|g| g.mem_used_bytes).sum();
        let total: u64 = self.gpus.iter().map(|g| g.mem_total_bytes).sum();
        (used, total)
    }

    pub fn total_pcie_tx_kbps(&self) -> u64 {
        self.gpus.iter().map(|g| g.pcie_tx_kbps).sum()
    }

    pub fn total_pcie_rx_kbps(&self) -> u64 {
        self.gpus.iter().map(|g| g.pcie_rx_kbps).sum()
    }

    /// Sum PCIe TX across all GPUs in GB/s (binary). Prefers the DCGM PROF
    /// counter (`DCGM_FI_PROF_PCIE_TX_BYTES`, exact bytes/sec) and falls back
    /// to the NVML/DEV throughput counter (KB/s) when PROF isn't reported.
    pub fn total_pcie_tx_gbps(&self) -> f64 {
        let prof: u64 = self.gpus.iter().map(|g| g.pcie_prof_tx_bytes_per_sec).sum();
        if prof > 0 {
            prof as f64 / (1024.0 * 1024.0 * 1024.0)
        } else {
            self.total_pcie_tx_kbps() as f64 / (1024.0 * 1024.0)
        }
    }

    /// Sum PCIe RX across all GPUs in GB/s (binary). Same source preference
    /// as `total_pcie_tx_gbps`.
    pub fn total_pcie_rx_gbps(&self) -> f64 {
        let prof: u64 = self.gpus.iter().map(|g| g.pcie_prof_rx_bytes_per_sec).sum();
        if prof > 0 {
            prof as f64 / (1024.0 * 1024.0 * 1024.0)
        } else {
            self.total_pcie_rx_kbps() as f64 / (1024.0 * 1024.0)
        }
    }

    pub fn total_nvlink_tx_kbps(&self) -> u64 {
        self.gpus.iter().map(|g| g.nvlink_tx_kbps).sum()
    }

    pub fn total_nvlink_rx_kbps(&self) -> u64 {
        self.gpus.iter().map(|g| g.nvlink_rx_kbps).sum()
    }

    /// OR of all GPUs' throttle reasons.
    pub fn combined_throttle_reasons(&self) -> u64 {
        self.gpus.iter().fold(0u64, |acc, g| acc | g.throttle_reasons)
    }

    pub fn total_ecc_sbe(&self) -> u64 {
        self.gpus.iter().map(|g| g.ecc_sbe_volatile).sum()
    }

    pub fn total_ecc_dbe(&self) -> u64 {
        self.gpus.iter().map(|g| g.ecc_dbe_volatile).sum()
    }

    pub fn total_power_limit(&self) -> f64 {
        self.gpus.iter().map(|g| g.power_limit_watts).sum()
    }

    /// Host memory usage as a percentage (0-100).
    pub fn mem_percent(&self) -> f64 {
        match (self.mem_used_bytes, self.mem_total_bytes) {
            (Some(used), Some(total)) if total > 0 => used as f64 / total as f64 * 100.0,
            _ => 0.0,
        }
    }

    /// Average NVLink utilization for a direction as a percentage of theoretical peak.
    /// Skips GPUs with no NVLink data or unknown model. Returns None if no data.
    fn avg_nvlink_utilization(&self, get_kbps: impl Fn(&GpuMetrics) -> u64) -> Option<f64> {
        let mut total_util = 0.0;
        let mut count = 0;
        for g in &self.gpus {
            let kbps = get_kbps(g);
            if kbps == 0 {
                continue;
            }
            let Some(peak) = peak_nvlink_bandwidth_gbps(&g.name) else {
                continue;
            };
            let actual_gbps = kbps as f64 / (1024.0 * 1024.0);
            // Each direction is half of bidirectional peak
            let util = (actual_gbps / (peak / 2.0)) * 100.0;
            total_util += util;
            count += 1;
        }
        if count > 0 {
            Some(total_util / count as f64)
        } else {
            None
        }
    }

    /// Average NVLink TX utilization as a percentage of theoretical peak.
    pub fn avg_nvlink_tx_utilization(&self) -> Option<f64> {
        self.avg_nvlink_utilization(|g| g.nvlink_tx_kbps)
    }

    /// Average NVLink RX utilization as a percentage of theoretical peak.
    pub fn avg_nvlink_rx_utilization(&self) -> Option<f64> {
        self.avg_nvlink_utilization(|g| g.nvlink_rx_kbps)
    }
}

/// Theoretical peak NVLink bandwidth in GB/s (bidirectional) for known GPU models.
/// Used to compute NVLink utilization% = actual_bandwidth / peak_bandwidth.
pub fn peak_nvlink_bandwidth_gbps(gpu_name: &str) -> Option<f64> {
    let name = gpu_name.to_uppercase();
    // Match from most specific to least specific
    // Peak bandwidth is bidirectional (TX + RX combined)
    if name.contains("B200") {
        return Some(1800.0); // 18 links × 100 GB/s per link (bidirectional)
    }
    if name.contains("B100") {
        return Some(1800.0); // 18 links × 100 GB/s per link
    }
    if name.contains("H200") || name.contains("H100") || name.contains("H800") {
        return Some(900.0); // 18 links × 50 GB/s per link (NVLink 4.0)
    }
    if name.contains("A100") || name.contains("A800") {
        return Some(600.0); // 12 links × 50 GB/s per link (NVLink 3.0)
    }
    if name.contains("A30") {
        return Some(400.0); // 8 links × 50 GB/s per link
    }
    None
}

/// Theoretical peak BF16 TFLOP/s for known GPU models.
/// Used to compute MFU% = estimated_flops_per_gpu / peak_flops.
pub fn peak_bf16_tflops(gpu_name: &str) -> Option<f64> {
    let name = gpu_name.to_uppercase();
    // Match from most specific to least specific
    if name.contains("B200") {
        return Some(4500.0);
    }
    if name.contains("B100") {
        return Some(3500.0);
    }
    if name.contains("H200") {
        return Some(989.5);
    }
    if name.contains("H100") {
        return Some(if name.contains("PCIE") || name.contains("PCI") {
            756.0
        } else {
            989.5
        });
    }
    if name.contains("H800") {
        return Some(989.5);
    }
    if name.contains("A100") {
        return Some(312.0);
    }
    if name.contains("A800") {
        return Some(312.0);
    }
    if name.contains("L40S") {
        return Some(362.0);
    }
    if name.contains("L40") {
        return Some(181.0);
    }
    if name.contains("L20") {
        return Some(119.5);
    }
    if name.contains("A10G") {
        return Some(70.0);
    }
    if name.contains("A10") && !name.contains("A100") {
        return Some(125.0);
    }
    if name.contains("A30") {
        return Some(165.0);
    }
    if name.contains("A6000") {
        return Some(155.2);
    }
    if name.contains("4090") {
        return Some(330.3);
    }
    if name.contains("3090") {
        return Some(142.0);
    }
    None
}

/// Compute precision for MFU% peak lookup. Inferred from the model name by
/// `infer_compute_dtype`. INT4 / AWQ / GPTQ models dequantize to BF16 for the
/// actual GEMM, so they fall under `Bf16` — only native low-precision tensor
/// core paths (FP8 / NVFP4 / MXFP4) get their own dtype.
#[derive(Copy, Clone, Eq, PartialEq, Debug)]
pub enum ComputeDtype {
    Bf16,
    Fp8,
    Fp4,
}

/// Theoretical peak TFLOP/s per GPU for a given compute dtype.
///
/// Returns `None` when the GPU is known but the dtype has no native tensor
/// core path (e.g. Hopper + FP4). Callers should fall back to `Bf16`.
///
/// **Convention:** all peaks here are NVIDIA's published numbers **with 2:4
/// structured sparsity** — i.e. the marketing-headline values. Dense workloads
/// (the common case) will compute MFU% relative to the sparse peak and thus
/// land at half the value of a "dense MFU%". Keep this consistent across all
/// rows so cross-GPU comparison stays meaningful; mixing dense and sparse here
/// makes the metric incomparable. B200 numbers are inherited from
/// `peak_bf16_tflops` which already used sparse, so B300 follows suit.
///
/// For BF16, falls back to the broader `peak_bf16_tflops` table when the GPU
/// isn't in this dtype-aware list — so H100/B100/A100 still get an MFU%.
pub fn peak_tflops(gpu_name: &str, dtype: ComputeDtype) -> Option<f64> {
    let name = gpu_name.to_uppercase();
    if name.contains("B300") {
        // Sparse peaks per the NVIDIA datasheet.
        return Some(match dtype {
            ComputeDtype::Bf16 => 7000.0,
            ComputeDtype::Fp8 => 15000.0,
            ComputeDtype::Fp4 => 30000.0,
        });
    }
    if name.contains("B200") {
        return Some(match dtype {
            ComputeDtype::Bf16 => 4500.0,
            ComputeDtype::Fp8 => 9000.0,
            ComputeDtype::Fp4 => 18000.0,
        });
    }
    if name.contains("H200") {
        return match dtype {
            ComputeDtype::Bf16 => Some(989.5),
            ComputeDtype::Fp8 => Some(1979.0),
            // Hopper has no native FP4 tensor core path.
            ComputeDtype::Fp4 => None,
        };
    }
    // GPU not in the dtype-aware table. For BF16, defer to the broader
    // lookup so H100 / B100 / A100 / H800 keep getting an MFU%.
    if matches!(dtype, ComputeDtype::Bf16) {
        return peak_bf16_tflops(gpu_name);
    }
    None
}

/// Infer the model's compute dtype from its name. Used for MFU% peak lookup.
///
/// Matches HF-style suffixes. Only native low-precision GEMM paths
/// (FP8 / NVFP4 / MXFP4) map to non-BF16 — INT4 / AWQ / GPTQ stay BF16 because
/// they dequantize at matmul time.
pub fn infer_compute_dtype(model_name: &str) -> ComputeDtype {
    let name = model_name.to_uppercase();
    if name.contains("NVFP4") || name.contains("MXFP4") || name.contains("FP4") {
        return ComputeDtype::Fp4;
    }
    if name.contains("FP8") {
        return ComputeDtype::Fp8;
    }
    ComputeDtype::Bf16
}

/// Decode throttle reasons bitmask to human-readable labels.
///
/// Bitmask values follow NVML `nvmlClocksThrottleReasons` / DCGM `DCGM_FI_DEV_CLOCK_THROTTLE_REASONS`.
pub fn decode_throttle_reasons(mask: u64) -> Vec<&'static str> {
    const REASONS: &[(u64, &str)] = &[
        (0x0000_0000_0000_0001, "GpuIdle"),
        (0x0000_0000_0000_0002, "AppClocks"),
        (0x0000_0000_0000_0008, "SwPowerCap"),
        (0x0000_0000_0000_0020, "HwThermal"),
        (0x0000_0000_0000_0080, "HwPowerBrake"),
        (0x0000_0000_0000_0100, "DisplayClocks"),
    ];
    let mut out = Vec::new();
    for &(bit, label) in REASONS {
        if mask & bit != 0 {
            out.push(label);
        }
    }
    out
}

/// Parse GPU metrics from Prometheus metric families.
/// Supports both vmon agent (`vmon_gpu_*`) and DCGM exporter (`DCGM_FI_DEV_*`).
pub fn extract_gpu_metrics(families: &[MetricFamily]) -> GpuScrape {
    let mut gpus_map: HashMap<u32, GpuMetrics> = HashMap::new();
    // DCGM FB values per GPU (in MiB) — needed to compute total.
    let mut fb_free: HashMap<u32, f64> = HashMap::new();
    let mut fb_total: HashMap<u32, f64> = HashMap::new();
    let mut fb_reserved: HashMap<u32, f64> = HashMap::new();
    // Host-level CPU/memory metrics (no gpu label)
    let mut cpu_percent: Option<f64> = None;
    let mut host_mem_used: Option<u64> = None;
    let mut host_mem_total: Option<u64> = None;

    for family in families {
        let name = family.name.as_str();

        // Host-level metrics (no gpu label required)
        match name {
            "vmon_cpu_usage_percent" => {
                if let Some(s) = family.samples.first() {
                    cpu_percent = Some(s.value);
                }
                continue;
            }
            "vmon_memory_used_bytes" => {
                if let Some(s) = family.samples.first() {
                    host_mem_used = Some(s.value as u64);
                }
                continue;
            }
            "vmon_memory_total_bytes" => {
                if let Some(s) = family.samples.first() {
                    host_mem_total = Some(s.value as u64);
                }
                continue;
            }
            _ => {}
        }

        for sample in &family.samples {
            let gpu_idx = match sample.label("gpu").and_then(|v| v.parse::<u32>().ok()) {
                Some(idx) => idx,
                None => continue,
            };
            let g = gpus_map.entry(gpu_idx).or_insert_with(|| GpuMetrics {
                index: gpu_idx,
                ..Default::default()
            });
            // Pick up GPU name from label (first non-empty wins)
            if g.name.is_empty() {
                if let Some(n) = sample.label("name") {
                    g.name = n.to_string();
                } else if let Some(n) = sample.label("modelName") {
                    // DCGM uses modelName label
                    g.name = n.to_string();
                }
            }
            // Pick up UUID from DCGM's UUID label (vmon agent does not emit this).
            if g.uuid.is_empty() {
                if let Some(u) = sample.label("UUID") {
                    g.uuid = u.to_string();
                }
            }
            match name {
                // ── vmon agent metrics ──
                "vmon_gpu_utilization" => g.utilization = sample.value,
                "vmon_gpu_memory_utilization" => g.mem_utilization = sample.value,
                "vmon_gpu_power_watts" => g.power_watts = sample.value,
                "vmon_gpu_temperature_celsius" => g.temperature = sample.value,
                "vmon_gpu_memory_used_bytes" => g.mem_used_bytes = sample.value as u64,
                "vmon_gpu_memory_total_bytes" => g.mem_total_bytes = sample.value as u64,
                "vmon_gpu_clock_mhz" => g.clock_mhz = sample.value as u32,
                "vmon_gpu_mem_clock_mhz" => g.mem_clock_mhz = sample.value as u32,
                "vmon_gpu_pcie_tx_kbps" => g.pcie_tx_kbps = sample.value as u64,
                "vmon_gpu_pcie_rx_kbps" => g.pcie_rx_kbps = sample.value as u64,
                "vmon_gpu_nvlink_tx_kbps" => g.nvlink_tx_kbps = sample.value as u64,
                "vmon_gpu_nvlink_rx_kbps" => g.nvlink_rx_kbps = sample.value as u64,
                "vmon_gpu_power_limit_watts" => g.power_limit_watts = sample.value,
                "vmon_gpu_throttle_reasons" => g.throttle_reasons = sample.value as u64,
                "vmon_gpu_ecc_sbe_volatile" => g.ecc_sbe_volatile = sample.value as u64,
                "vmon_gpu_ecc_dbe_volatile" => g.ecc_dbe_volatile = sample.value as u64,

                // ── DCGM exporter metrics ──
                "DCGM_FI_DEV_GPU_UTIL" => g.utilization = sample.value,
                "DCGM_FI_DEV_MEM_COPY_UTIL" => g.mem_utilization = sample.value,
                "DCGM_FI_DEV_POWER_USAGE" => g.power_watts = sample.value,
                "DCGM_FI_DEV_GPU_TEMP" => g.temperature = sample.value,
                // DCGM reports framebuffer in MiB
                "DCGM_FI_DEV_FB_USED" => {
                    g.mem_used_bytes = (sample.value * 1024.0 * 1024.0) as u64;
                }
                "DCGM_FI_DEV_FB_FREE" => {
                    fb_free.insert(gpu_idx, sample.value);
                }
                "DCGM_FI_DEV_FB_TOTAL" => {
                    fb_total.insert(gpu_idx, sample.value);
                }
                "DCGM_FI_DEV_FB_RESERVED" => {
                    fb_reserved.insert(gpu_idx, sample.value);
                }
                // Profiling activity fractions — stored directly, and also used
                // as a fallback to populate `mem_utilization` / `utilization`
                // when the DCGM_FI_DEV_* util metrics return 0 (keeps existing
                // HTML charts meaningful on hosts where only PROF metrics fire).
                "DCGM_FI_PROF_DRAM_ACTIVE" => {
                    g.dram_active = Some(sample.value);
                    if g.mem_utilization == 0.0 {
                        g.mem_utilization = sample.value * 100.0;
                    }
                }
                "DCGM_FI_PROF_GR_ENGINE_ACTIVE" => {
                    g.gr_engine_active = Some(sample.value);
                    if g.utilization == 0.0 {
                        g.utilization = sample.value * 100.0;
                    }
                }
                "DCGM_FI_PROF_PIPE_TENSOR_ACTIVE" => g.tensor_active = Some(sample.value),
                "DCGM_FI_DEV_ENC_UTIL" => g.enc_utilization = sample.value,
                "DCGM_FI_DEV_DEC_UTIL" => g.dec_utilization = sample.value,
                "DCGM_FI_DEV_MEMORY_TEMP" => g.mem_temperature = sample.value,
                "DCGM_FI_DEV_TOTAL_ENERGY_CONSUMPTION" => {
                    g.total_energy_mj = sample.value as u64;
                }
                "DCGM_FI_DEV_PCIE_REPLAY_COUNTER" => {
                    g.pcie_replay_count = sample.value as u64;
                }
                "DCGM_FI_DEV_XID_ERRORS" => g.xid_errors = sample.value as u64,
                "DCGM_FI_DEV_CORRECTABLE_REMAPPED_ROWS" => {
                    g.remapped_rows_correctable = sample.value as u64;
                }
                "DCGM_FI_DEV_UNCORRECTABLE_REMAPPED_ROWS" => {
                    g.remapped_rows_uncorrectable = sample.value as u64;
                }
                "DCGM_FI_DEV_ROW_REMAP_FAILURE" => g.row_remap_failure = sample.value as u64,
                "DCGM_FI_DEV_VGPU_LICENSE_STATUS" => {
                    g.vgpu_license_status = sample.value as u64;
                }
                // PCIe profile bandwidth (bytes/sec). Also populate pcie_tx/rx_kbps
                // when the DEV throughput fields are absent so existing consumers
                // keep working on hosts where only PROF_PCIE_* is exposed.
                "DCGM_FI_PROF_PCIE_TX_BYTES" => {
                    g.pcie_prof_tx_bytes_per_sec = sample.value as u64;
                    if g.pcie_tx_kbps == 0 {
                        g.pcie_tx_kbps = (sample.value / 1024.0) as u64;
                    }
                }
                "DCGM_FI_PROF_PCIE_RX_BYTES" => {
                    g.pcie_prof_rx_bytes_per_sec = sample.value as u64;
                    if g.pcie_rx_kbps == 0 {
                        g.pcie_rx_kbps = (sample.value / 1024.0) as u64;
                    }
                }
                "DCGM_FI_DEV_SM_CLOCK" => g.clock_mhz = sample.value as u32,
                "DCGM_FI_DEV_MEM_CLOCK" => g.mem_clock_mhz = sample.value as u32,
                // DCGM reports PCIe throughput in KB/s
                "DCGM_FI_DEV_PCIE_TX_THROUGHPUT" => g.pcie_tx_kbps = sample.value as u64,
                "DCGM_FI_DEV_PCIE_RX_THROUGHPUT" => g.pcie_rx_kbps = sample.value as u64,
                // DCGM NVLink bandwidth (bytes/s → KB/s)
                "DCGM_FI_DEV_NVLINK_BANDWIDTH_TX" => {
                    g.nvlink_tx_kbps = (sample.value / 1024.0) as u64;
                }
                "DCGM_FI_DEV_NVLINK_BANDWIDTH_RX" => {
                    g.nvlink_rx_kbps = (sample.value / 1024.0) as u64;
                }
                // DCGM NVLink bandwidth (KiB/s gauge despite "counter" type label).
                // This field reports a sampled rate rather than a cumulative counter.
                // Use as direct rate, split evenly into TX/RX when no per-direction data.
                "DCGM_FI_DEV_NVLINK_BANDWIDTH_TOTAL"
                    if g.nvlink_tx_kbps == 0 && g.nvlink_rx_kbps == 0 =>
                {
                    let rate_kbps = sample.value as u64; // KiB/s ≈ KB/s
                    g.nvlink_tx_kbps = rate_kbps / 2;
                    g.nvlink_rx_kbps = rate_kbps / 2;
                }
                // DCGM profiling NVLink metrics (bytes/sec gauge → KB/s)
                "DCGM_FI_PROF_NVLINK_TX_BYTES" => {
                    g.nvlink_tx_kbps = (sample.value / 1024.0) as u64;
                }
                "DCGM_FI_PROF_NVLINK_RX_BYTES" => {
                    g.nvlink_rx_kbps = (sample.value / 1024.0) as u64;
                }
                "DCGM_FI_DEV_POWER_MGMT_LIMIT" => g.power_limit_watts = sample.value,
                "DCGM_FI_DEV_CLOCK_THROTTLE_REASONS" => g.throttle_reasons = sample.value as u64,
                "DCGM_FI_DEV_ECC_SBE_VOL_TOTAL" => g.ecc_sbe_volatile = sample.value as u64,
                "DCGM_FI_DEV_ECC_DBE_VOL_TOTAL" => g.ecc_dbe_volatile = sample.value as u64,

                _ => {}
            }
        }
    }

    // Compute mem_total for DCGM GPUs using best available data:
    // 1. FB_TOTAL (direct, most reliable)
    // 2. FB_USED + FB_FREE
    // 3. FB_USED + FB_FREE + FB_RESERVED
    for (&idx, g) in gpus_map.iter_mut() {
        if g.mem_total_bytes > 0 {
            continue; // already set (e.g. by vmon agent)
        }
        let mib_to_bytes = |mib: f64| (mib * 1024.0 * 1024.0) as u64;
        if let Some(&total_mib) = fb_total.get(&idx) {
            g.mem_total_bytes = mib_to_bytes(total_mib);
        } else if let Some(&free_mib) = fb_free.get(&idx) {
            let reserved = fb_reserved.get(&idx).copied().unwrap_or(0.0);
            g.mem_total_bytes = g.mem_used_bytes + mib_to_bytes(free_mib) + mib_to_bytes(reserved);
        }
    }

    let mut gpus: Vec<GpuMetrics> = gpus_map.into_values().collect();
    gpus.sort_by_key(|g| g.index);
    GpuScrape {
        gpus,
        cpu_percent,
        mem_used_bytes: host_mem_used,
        mem_total_bytes: host_mem_total,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::parser::parse_prometheus_text;

    const GPU_METRICS: &str = r#"# HELP vmon_cpu_usage_percent Host CPU utilization percentage
# TYPE vmon_cpu_usage_percent gauge
vmon_cpu_usage_percent 72.5
# HELP vmon_memory_used_bytes Host memory used in bytes
# TYPE vmon_memory_used_bytes gauge
vmon_memory_used_bytes 198432120832
# HELP vmon_memory_total_bytes Host memory total in bytes
# TYPE vmon_memory_total_bytes gauge
vmon_memory_total_bytes 270582939648
# HELP vmon_gpu_utilization GPU compute utilization percentage
# TYPE vmon_gpu_utilization gauge
vmon_gpu_utilization{gpu="0"} 85
vmon_gpu_utilization{gpu="1"} 92
# HELP vmon_gpu_memory_utilization GPU memory bandwidth utilization percentage
# TYPE vmon_gpu_memory_utilization gauge
vmon_gpu_memory_utilization{gpu="0"} 45
vmon_gpu_memory_utilization{gpu="1"} 50
# HELP vmon_gpu_power_watts GPU power draw in watts
# TYPE vmon_gpu_power_watts gauge
vmon_gpu_power_watts{gpu="0"} 312.5
vmon_gpu_power_watts{gpu="1"} 313.0
# HELP vmon_gpu_temperature_celsius GPU temperature in degrees Celsius
# TYPE vmon_gpu_temperature_celsius gauge
vmon_gpu_temperature_celsius{gpu="0"} 71
vmon_gpu_temperature_celsius{gpu="1"} 73
# HELP vmon_gpu_memory_used_bytes GPU memory used in bytes
# TYPE vmon_gpu_memory_used_bytes gauge
vmon_gpu_memory_used_bytes{gpu="0"} 81604378624
vmon_gpu_memory_used_bytes{gpu="1"} 81604378624
# HELP vmon_gpu_memory_total_bytes GPU memory total in bytes
# TYPE vmon_gpu_memory_total_bytes gauge
vmon_gpu_memory_total_bytes{gpu="0"} 85899345920
vmon_gpu_memory_total_bytes{gpu="1"} 85899345920
# HELP vmon_gpu_clock_mhz GPU SM clock speed in MHz
# TYPE vmon_gpu_clock_mhz gauge
vmon_gpu_clock_mhz{gpu="0"} 1980
vmon_gpu_clock_mhz{gpu="1"} 1995
# HELP vmon_gpu_mem_clock_mhz GPU memory clock speed in MHz
# TYPE vmon_gpu_mem_clock_mhz gauge
vmon_gpu_mem_clock_mhz{gpu="0"} 1593
vmon_gpu_mem_clock_mhz{gpu="1"} 1593
# HELP vmon_gpu_pcie_tx_kbps GPU PCIe TX throughput in KB/s
# TYPE vmon_gpu_pcie_tx_kbps gauge
vmon_gpu_pcie_tx_kbps{gpu="0"} 512000
vmon_gpu_pcie_tx_kbps{gpu="1"} 480000
# HELP vmon_gpu_pcie_rx_kbps GPU PCIe RX throughput in KB/s
# TYPE vmon_gpu_pcie_rx_kbps gauge
vmon_gpu_pcie_rx_kbps{gpu="0"} 256000
vmon_gpu_pcie_rx_kbps{gpu="1"} 240000
# HELP vmon_gpu_power_limit_watts GPU power management limit in watts
# TYPE vmon_gpu_power_limit_watts gauge
vmon_gpu_power_limit_watts{gpu="0"} 400.0
vmon_gpu_power_limit_watts{gpu="1"} 400.0
# HELP vmon_gpu_throttle_reasons GPU clock throttle reasons bitmask
# TYPE vmon_gpu_throttle_reasons gauge
vmon_gpu_throttle_reasons{gpu="0"} 0
vmon_gpu_throttle_reasons{gpu="1"} 8
# HELP vmon_gpu_ecc_sbe_volatile GPU corrected ECC errors since boot
# TYPE vmon_gpu_ecc_sbe_volatile gauge
vmon_gpu_ecc_sbe_volatile{gpu="0"} 3
vmon_gpu_ecc_sbe_volatile{gpu="1"} 0
# HELP vmon_gpu_ecc_dbe_volatile GPU uncorrected ECC errors since boot
# TYPE vmon_gpu_ecc_dbe_volatile gauge
vmon_gpu_ecc_dbe_volatile{gpu="0"} 0
vmon_gpu_ecc_dbe_volatile{gpu="1"} 0
"#;

    #[test]
    fn test_extract_gpu_metrics() {
        let families = parse_prometheus_text(GPU_METRICS).unwrap();
        let scrape = extract_gpu_metrics(&families);

        // Host-level metrics
        assert!((scrape.cpu_percent.unwrap() - 72.5).abs() < 0.001);
        assert_eq!(scrape.mem_used_bytes.unwrap(), 198432120832);
        assert_eq!(scrape.mem_total_bytes.unwrap(), 270582939648);

        assert_eq!(scrape.gpus.len(), 2);

        let g0 = &scrape.gpus[0];
        assert_eq!(g0.index, 0);
        assert!((g0.utilization - 85.0).abs() < 0.001);
        assert!((g0.mem_utilization - 45.0).abs() < 0.001);
        assert!((g0.power_watts - 312.5).abs() < 0.001);
        assert!((g0.temperature - 71.0).abs() < 0.001);
        assert_eq!(g0.mem_used_bytes, 81604378624);
        assert_eq!(g0.mem_total_bytes, 85899345920);
        assert_eq!(g0.clock_mhz, 1980);
        assert_eq!(g0.mem_clock_mhz, 1593);
        assert_eq!(g0.pcie_tx_kbps, 512000);
        assert_eq!(g0.pcie_rx_kbps, 256000);
        assert!((g0.power_limit_watts - 400.0).abs() < 0.001);
        assert_eq!(g0.throttle_reasons, 0);
        assert_eq!(g0.ecc_sbe_volatile, 3);
        assert_eq!(g0.ecc_dbe_volatile, 0);

        let g1 = &scrape.gpus[1];
        assert_eq!(g1.index, 1);
        assert!((g1.utilization - 92.0).abs() < 0.001);
        assert_eq!(g1.pcie_tx_kbps, 480000);
        assert_eq!(g1.pcie_rx_kbps, 240000);
        assert_eq!(g1.throttle_reasons, 8); // SwPowerCap
    }

    #[test]
    fn test_gpu_scrape_aggregation() {
        let families = parse_prometheus_text(GPU_METRICS).unwrap();
        let scrape = extract_gpu_metrics(&families);

        assert!((scrape.avg_utilization() - 88.5).abs() < 0.001);
        assert!((scrape.total_power() - 625.5).abs() < 0.001);
        assert!((scrape.avg_temperature() - 72.0).abs() < 0.001);

        let (used, total) = scrape.mem_used_total();
        assert_eq!(used, 81604378624 * 2);
        assert_eq!(total, 85899345920 * 2);

        assert_eq!(scrape.total_pcie_tx_kbps(), 992000);
        assert_eq!(scrape.total_pcie_rx_kbps(), 496000);
        assert!((scrape.total_power_limit() - 800.0).abs() < 0.001);
        assert_eq!(scrape.combined_throttle_reasons(), 8); // gpu1 has SwPowerCap
        assert_eq!(scrape.total_ecc_sbe(), 3);
        assert_eq!(scrape.total_ecc_dbe(), 0);
    }

    #[test]
    fn test_empty_families() {
        let scrape = extract_gpu_metrics(&[]);
        assert!(scrape.gpus.is_empty());
        assert_eq!(scrape.avg_utilization(), 0.0);
        assert_eq!(scrape.total_power(), 0.0);
    }

    const DCGM_METRICS: &str = r#"# HELP DCGM_FI_DEV_GPU_UTIL GPU utilization
# TYPE DCGM_FI_DEV_GPU_UTIL gauge
DCGM_FI_DEV_GPU_UTIL{gpu="0",UUID="GPU-abc",device="nvidia0"} 78
DCGM_FI_DEV_GPU_UTIL{gpu="1",UUID="GPU-def",device="nvidia1"} 91
# HELP DCGM_FI_DEV_MEM_COPY_UTIL Memory utilization
# TYPE DCGM_FI_DEV_MEM_COPY_UTIL gauge
DCGM_FI_DEV_MEM_COPY_UTIL{gpu="0",UUID="GPU-abc",device="nvidia0"} 42
DCGM_FI_DEV_MEM_COPY_UTIL{gpu="1",UUID="GPU-def",device="nvidia1"} 55
# HELP DCGM_FI_DEV_POWER_USAGE Power draw in watts
# TYPE DCGM_FI_DEV_POWER_USAGE gauge
DCGM_FI_DEV_POWER_USAGE{gpu="0",UUID="GPU-abc",device="nvidia0"} 295.5
DCGM_FI_DEV_POWER_USAGE{gpu="1",UUID="GPU-def",device="nvidia1"} 310.0
# HELP DCGM_FI_DEV_GPU_TEMP GPU temperature
# TYPE DCGM_FI_DEV_GPU_TEMP gauge
DCGM_FI_DEV_GPU_TEMP{gpu="0",UUID="GPU-abc",device="nvidia0"} 68
DCGM_FI_DEV_GPU_TEMP{gpu="1",UUID="GPU-def",device="nvidia1"} 72
# HELP DCGM_FI_DEV_FB_USED Framebuffer used MiB
# TYPE DCGM_FI_DEV_FB_USED gauge
DCGM_FI_DEV_FB_USED{gpu="0",UUID="GPU-abc",device="nvidia0"} 76000
DCGM_FI_DEV_FB_USED{gpu="1",UUID="GPU-def",device="nvidia1"} 76000
# HELP DCGM_FI_DEV_FB_FREE Framebuffer free MiB
# TYPE DCGM_FI_DEV_FB_FREE gauge
DCGM_FI_DEV_FB_FREE{gpu="0",UUID="GPU-abc",device="nvidia0"} 5904
DCGM_FI_DEV_FB_FREE{gpu="1",UUID="GPU-def",device="nvidia1"} 5904
# HELP DCGM_FI_DEV_SM_CLOCK SM clock MHz
# TYPE DCGM_FI_DEV_SM_CLOCK gauge
DCGM_FI_DEV_SM_CLOCK{gpu="0",UUID="GPU-abc",device="nvidia0"} 1980
DCGM_FI_DEV_SM_CLOCK{gpu="1",UUID="GPU-def",device="nvidia1"} 1995
# HELP DCGM_FI_DEV_MEM_CLOCK Memory clock MHz
# TYPE DCGM_FI_DEV_MEM_CLOCK gauge
DCGM_FI_DEV_MEM_CLOCK{gpu="0",UUID="GPU-abc",device="nvidia0"} 1593
DCGM_FI_DEV_MEM_CLOCK{gpu="1",UUID="GPU-def",device="nvidia1"} 1593
# HELP DCGM_FI_DEV_PCIE_TX_THROUGHPUT PCIe TX KB/s
# TYPE DCGM_FI_DEV_PCIE_TX_THROUGHPUT gauge
DCGM_FI_DEV_PCIE_TX_THROUGHPUT{gpu="0",UUID="GPU-abc",device="nvidia0"} 650000
DCGM_FI_DEV_PCIE_TX_THROUGHPUT{gpu="1",UUID="GPU-def",device="nvidia1"} 620000
# HELP DCGM_FI_DEV_PCIE_RX_THROUGHPUT PCIe RX KB/s
# TYPE DCGM_FI_DEV_PCIE_RX_THROUGHPUT gauge
DCGM_FI_DEV_PCIE_RX_THROUGHPUT{gpu="0",UUID="GPU-abc",device="nvidia0"} 300000
DCGM_FI_DEV_PCIE_RX_THROUGHPUT{gpu="1",UUID="GPU-def",device="nvidia1"} 280000
# HELP DCGM_FI_DEV_NVLINK_BANDWIDTH_TX NVLink TX bytes/s
# TYPE DCGM_FI_DEV_NVLINK_BANDWIDTH_TX gauge
DCGM_FI_DEV_NVLINK_BANDWIDTH_TX{gpu="0",UUID="GPU-abc",device="nvidia0"} 52428800
DCGM_FI_DEV_NVLINK_BANDWIDTH_TX{gpu="1",UUID="GPU-def",device="nvidia1"} 41943040
# HELP DCGM_FI_DEV_NVLINK_BANDWIDTH_RX NVLink RX bytes/s
# TYPE DCGM_FI_DEV_NVLINK_BANDWIDTH_RX gauge
DCGM_FI_DEV_NVLINK_BANDWIDTH_RX{gpu="0",UUID="GPU-abc",device="nvidia0"} 26214400
DCGM_FI_DEV_NVLINK_BANDWIDTH_RX{gpu="1",UUID="GPU-def",device="nvidia1"} 20971520
# HELP DCGM_FI_DEV_POWER_MGMT_LIMIT Power limit watts
# TYPE DCGM_FI_DEV_POWER_MGMT_LIMIT gauge
DCGM_FI_DEV_POWER_MGMT_LIMIT{gpu="0",UUID="GPU-abc",device="nvidia0"} 400
DCGM_FI_DEV_POWER_MGMT_LIMIT{gpu="1",UUID="GPU-def",device="nvidia1"} 400
# HELP DCGM_FI_DEV_CLOCK_THROTTLE_REASONS Clock throttle reasons bitmask
# TYPE DCGM_FI_DEV_CLOCK_THROTTLE_REASONS gauge
DCGM_FI_DEV_CLOCK_THROTTLE_REASONS{gpu="0",UUID="GPU-abc",device="nvidia0"} 32
DCGM_FI_DEV_CLOCK_THROTTLE_REASONS{gpu="1",UUID="GPU-def",device="nvidia1"} 0
# HELP DCGM_FI_DEV_ECC_SBE_VOL_TOTAL Single-bit ECC errors volatile
# TYPE DCGM_FI_DEV_ECC_SBE_VOL_TOTAL gauge
DCGM_FI_DEV_ECC_SBE_VOL_TOTAL{gpu="0",UUID="GPU-abc",device="nvidia0"} 5
DCGM_FI_DEV_ECC_SBE_VOL_TOTAL{gpu="1",UUID="GPU-def",device="nvidia1"} 2
# HELP DCGM_FI_DEV_ECC_DBE_VOL_TOTAL Double-bit ECC errors volatile
# TYPE DCGM_FI_DEV_ECC_DBE_VOL_TOTAL gauge
DCGM_FI_DEV_ECC_DBE_VOL_TOTAL{gpu="0",UUID="GPU-abc",device="nvidia0"} 0
DCGM_FI_DEV_ECC_DBE_VOL_TOTAL{gpu="1",UUID="GPU-def",device="nvidia1"} 0
"#;

    /// Test profiling-only exporters (no DEV_GPU_UTIL/MEM_COPY_UTIL,
    /// uses PROF_GR_ENGINE_ACTIVE / PROF_DRAM_ACTIVE instead).
    #[test]
    fn test_extract_dcgm_prof_metrics() {
        let input = r#"# HELP DCGM_FI_DEV_FB_USED Framebuffer memory used (in MiB).
# TYPE DCGM_FI_DEV_FB_USED gauge
DCGM_FI_DEV_FB_USED{gpu="0",UUID="GPU-aaa",device="nvidia0",modelName="Example GPU",Hostname="node01.example",DCGM_FI_DRIVER_VERSION="0.0.0"} 4762
DCGM_FI_DEV_FB_USED{gpu="1",UUID="GPU-bbb",device="nvidia1",modelName="Example GPU",Hostname="node01.example",DCGM_FI_DRIVER_VERSION="0.0.0"} 9716
# HELP DCGM_FI_DEV_FB_FREE Framebuffer memory free (in MiB).
# TYPE DCGM_FI_DEV_FB_FREE gauge
DCGM_FI_DEV_FB_FREE{gpu="0",UUID="GPU-aaa",device="nvidia0",modelName="Example GPU",Hostname="node01.example",DCGM_FI_DRIVER_VERSION="0.0.0"} 183967
DCGM_FI_DEV_FB_FREE{gpu="1",UUID="GPU-bbb",device="nvidia1",modelName="Example GPU",Hostname="node01.example",DCGM_FI_DRIVER_VERSION="0.0.0"} 179013
# HELP DCGM_FI_PROF_DRAM_ACTIVE Ratio of cycles the device memory interface is active.
# TYPE DCGM_FI_PROF_DRAM_ACTIVE gauge
DCGM_FI_PROF_DRAM_ACTIVE{gpu="0",UUID="GPU-aaa",device="nvidia0",modelName="Example GPU",Hostname="node01.example",DCGM_FI_DRIVER_VERSION="0.0.0"} 0.250000
DCGM_FI_PROF_DRAM_ACTIVE{gpu="1",UUID="GPU-bbb",device="nvidia1",modelName="Example GPU",Hostname="node01.example",DCGM_FI_DRIVER_VERSION="0.0.0"} 0.450000
# HELP DCGM_FI_PROF_GR_ENGINE_ACTIVE Ratio of time the graphics engine is active.
# TYPE DCGM_FI_PROF_GR_ENGINE_ACTIVE gauge
DCGM_FI_PROF_GR_ENGINE_ACTIVE{gpu="0",UUID="GPU-aaa",device="nvidia0",modelName="Example GPU",Hostname="node01.example",DCGM_FI_DRIVER_VERSION="0.0.0"} 0.780000
DCGM_FI_PROF_GR_ENGINE_ACTIVE{gpu="1",UUID="GPU-bbb",device="nvidia1",modelName="Example GPU",Hostname="node01.example",DCGM_FI_DRIVER_VERSION="0.0.0"} 0.910000
"#;
        let families = parse_prometheus_text(input).unwrap();
        let scrape = extract_gpu_metrics(&families);

        assert_eq!(scrape.gpus.len(), 2);

        let g0 = &scrape.gpus[0];
        assert_eq!(g0.name, "Example GPU");
        // PROF_GR_ENGINE_ACTIVE 0.78 → 78%
        assert!((g0.utilization - 78.0).abs() < 0.001);
        // PROF_DRAM_ACTIVE 0.25 → 25%
        assert!((g0.mem_utilization - 25.0).abs() < 0.001);
        // FB_USED = 4762 MiB
        assert_eq!(g0.mem_used_bytes, (4762.0 * 1024.0 * 1024.0) as u64);
        // FB total = 4762 + 183967 = 188729 MiB
        assert_eq!(
            g0.mem_total_bytes,
            ((4762.0 + 183967.0) * 1024.0 * 1024.0) as u64
        );

        let g1 = &scrape.gpus[1];
        assert!((g1.utilization - 91.0).abs() < 0.001);
        assert!((g1.mem_utilization - 45.0).abs() < 0.001);
        assert_eq!(g1.mem_used_bytes, (9716.0 * 1024.0 * 1024.0) as u64);
        assert_eq!(
            g1.mem_total_bytes,
            ((9716.0 + 179013.0) * 1024.0 * 1024.0) as u64
        );

        // VRAM percentage: 4762/188729 ≈ 2.5%
        let (used, total) = scrape.mem_used_total();
        let pct = used as f64 / total as f64 * 100.0;
        assert!(pct > 2.0 && pct < 5.0, "VRAM% should be ~2-5%, got {pct}");
    }

    #[test]
    fn test_extract_dcgm_metrics() {
        let families = parse_prometheus_text(DCGM_METRICS).unwrap();
        let scrape = extract_gpu_metrics(&families);

        assert_eq!(scrape.gpus.len(), 2);

        let g0 = &scrape.gpus[0];
        assert_eq!(g0.index, 0);
        assert!((g0.utilization - 78.0).abs() < 0.001);
        assert!((g0.mem_utilization - 42.0).abs() < 0.001);
        assert!((g0.power_watts - 295.5).abs() < 0.001);
        assert!((g0.temperature - 68.0).abs() < 0.001);
        assert_eq!(g0.clock_mhz, 1980);
        assert_eq!(g0.mem_clock_mhz, 1593);
        assert_eq!(g0.pcie_tx_kbps, 650000);
        assert_eq!(g0.pcie_rx_kbps, 300000);
        // NVLink: 52428800 bytes/s → 51200 KB/s
        assert_eq!(g0.nvlink_tx_kbps, 51200);
        assert_eq!(g0.nvlink_rx_kbps, 25600);
        assert!((g0.power_limit_watts - 400.0).abs() < 0.001);
        assert_eq!(g0.throttle_reasons, 32); // HwThermal
        assert_eq!(g0.ecc_sbe_volatile, 5);
        assert_eq!(g0.ecc_dbe_volatile, 0);

        // FB_USED = 76000 MiB, FB_FREE = 5904 MiB → total = 81904 MiB
        let used_bytes = (76000.0 * 1024.0 * 1024.0) as u64;
        let total_bytes = ((76000.0 + 5904.0) * 1024.0 * 1024.0) as u64;
        assert_eq!(g0.mem_used_bytes, used_bytes);
        assert_eq!(g0.mem_total_bytes, total_bytes);

        let g1 = &scrape.gpus[1];
        assert!((g1.utilization - 91.0).abs() < 0.001);
        assert_eq!(g1.clock_mhz, 1995);
        assert_eq!(g1.pcie_tx_kbps, 620000);
        assert_eq!(g1.pcie_rx_kbps, 280000);
        // NVLink: 41943040 bytes/s → 40960 KB/s
        assert_eq!(g1.nvlink_tx_kbps, 40960);
        assert_eq!(g1.nvlink_rx_kbps, 20480);
        assert_eq!(g1.ecc_sbe_volatile, 2);
    }

    /// Exhaustive coverage of every DCGM field added in the per-GPU expansion:
    /// identity (UUID), profile activity fractions, encoder/decoder util, memory
    /// temp, cumulative counters, and PCIe profile throughput.
    #[test]
    fn test_extract_dcgm_extended_fields() {
        let input = r#"# HELP DCGM_FI_PROF_DRAM_ACTIVE DRAM active fraction
# TYPE DCGM_FI_PROF_DRAM_ACTIVE gauge
DCGM_FI_PROF_DRAM_ACTIVE{gpu="0",UUID="GPU-u0",modelName="Example GPU"} 0.33
DCGM_FI_PROF_DRAM_ACTIVE{gpu="1",UUID="GPU-u1",modelName="Example GPU"} 0.66
# HELP DCGM_FI_PROF_GR_ENGINE_ACTIVE GR engine active fraction
# TYPE DCGM_FI_PROF_GR_ENGINE_ACTIVE gauge
DCGM_FI_PROF_GR_ENGINE_ACTIVE{gpu="0",UUID="GPU-u0",modelName="Example GPU"} 0.55
DCGM_FI_PROF_GR_ENGINE_ACTIVE{gpu="1",UUID="GPU-u1",modelName="Example GPU"} 0.88
# HELP DCGM_FI_PROF_PIPE_TENSOR_ACTIVE Tensor pipe active fraction
# TYPE DCGM_FI_PROF_PIPE_TENSOR_ACTIVE gauge
DCGM_FI_PROF_PIPE_TENSOR_ACTIVE{gpu="0",UUID="GPU-u0",modelName="Example GPU"} 0.41
DCGM_FI_PROF_PIPE_TENSOR_ACTIVE{gpu="1",UUID="GPU-u1",modelName="Example GPU"} 0.72
# HELP DCGM_FI_DEV_ENC_UTIL Encoder utilization
# TYPE DCGM_FI_DEV_ENC_UTIL gauge
DCGM_FI_DEV_ENC_UTIL{gpu="0",UUID="GPU-u0"} 3
DCGM_FI_DEV_ENC_UTIL{gpu="1",UUID="GPU-u1"} 4
# HELP DCGM_FI_DEV_DEC_UTIL Decoder utilization
# TYPE DCGM_FI_DEV_DEC_UTIL gauge
DCGM_FI_DEV_DEC_UTIL{gpu="0",UUID="GPU-u0"} 5
DCGM_FI_DEV_DEC_UTIL{gpu="1",UUID="GPU-u1"} 6
# HELP DCGM_FI_DEV_MEMORY_TEMP HBM temperature C
# TYPE DCGM_FI_DEV_MEMORY_TEMP gauge
DCGM_FI_DEV_MEMORY_TEMP{gpu="0",UUID="GPU-u0"} 55
DCGM_FI_DEV_MEMORY_TEMP{gpu="1",UUID="GPU-u1"} 58
# HELP DCGM_FI_DEV_TOTAL_ENERGY_CONSUMPTION Energy mJ
# TYPE DCGM_FI_DEV_TOTAL_ENERGY_CONSUMPTION counter
DCGM_FI_DEV_TOTAL_ENERGY_CONSUMPTION{gpu="0",UUID="GPU-u0"} 251885217371
DCGM_FI_DEV_TOTAL_ENERGY_CONSUMPTION{gpu="1",UUID="GPU-u1"} 260537897279
# HELP DCGM_FI_DEV_PCIE_REPLAY_COUNTER PCIe replays
# TYPE DCGM_FI_DEV_PCIE_REPLAY_COUNTER counter
DCGM_FI_DEV_PCIE_REPLAY_COUNTER{gpu="0",UUID="GPU-u0"} 7
DCGM_FI_DEV_PCIE_REPLAY_COUNTER{gpu="1",UUID="GPU-u1"} 0
# HELP DCGM_FI_DEV_XID_ERRORS XID errors
# TYPE DCGM_FI_DEV_XID_ERRORS counter
DCGM_FI_DEV_XID_ERRORS{gpu="0",UUID="GPU-u0"} 2
DCGM_FI_DEV_XID_ERRORS{gpu="1",UUID="GPU-u1"} 0
# HELP DCGM_FI_DEV_CORRECTABLE_REMAPPED_ROWS Correctable remapped rows
# TYPE DCGM_FI_DEV_CORRECTABLE_REMAPPED_ROWS counter
DCGM_FI_DEV_CORRECTABLE_REMAPPED_ROWS{gpu="0",UUID="GPU-u0"} 1
DCGM_FI_DEV_CORRECTABLE_REMAPPED_ROWS{gpu="1",UUID="GPU-u1"} 0
# HELP DCGM_FI_DEV_UNCORRECTABLE_REMAPPED_ROWS Uncorrectable remapped rows
# TYPE DCGM_FI_DEV_UNCORRECTABLE_REMAPPED_ROWS counter
DCGM_FI_DEV_UNCORRECTABLE_REMAPPED_ROWS{gpu="0",UUID="GPU-u0"} 0
DCGM_FI_DEV_UNCORRECTABLE_REMAPPED_ROWS{gpu="1",UUID="GPU-u1"} 0
# HELP DCGM_FI_DEV_ROW_REMAP_FAILURE Row remap failure
# TYPE DCGM_FI_DEV_ROW_REMAP_FAILURE gauge
DCGM_FI_DEV_ROW_REMAP_FAILURE{gpu="0",UUID="GPU-u0"} 0
DCGM_FI_DEV_ROW_REMAP_FAILURE{gpu="1",UUID="GPU-u1"} 0
# HELP DCGM_FI_DEV_VGPU_LICENSE_STATUS vGPU license
# TYPE DCGM_FI_DEV_VGPU_LICENSE_STATUS gauge
DCGM_FI_DEV_VGPU_LICENSE_STATUS{gpu="0",UUID="GPU-u0"} 1
DCGM_FI_DEV_VGPU_LICENSE_STATUS{gpu="1",UUID="GPU-u1"} 1
# HELP DCGM_FI_PROF_PCIE_TX_BYTES PCIe TX bytes/sec
# TYPE DCGM_FI_PROF_PCIE_TX_BYTES gauge
DCGM_FI_PROF_PCIE_TX_BYTES{gpu="0",UUID="GPU-u0"} 12582912
DCGM_FI_PROF_PCIE_TX_BYTES{gpu="1",UUID="GPU-u1"} 10485760
# HELP DCGM_FI_PROF_PCIE_RX_BYTES PCIe RX bytes/sec
# TYPE DCGM_FI_PROF_PCIE_RX_BYTES gauge
DCGM_FI_PROF_PCIE_RX_BYTES{gpu="0",UUID="GPU-u0"} 8388608
DCGM_FI_PROF_PCIE_RX_BYTES{gpu="1",UUID="GPU-u1"} 6291456
"#;
        let families = parse_prometheus_text(input).unwrap();
        let scrape = extract_gpu_metrics(&families);
        assert_eq!(scrape.gpus.len(), 2);

        let g0 = &scrape.gpus[0];
        assert_eq!(g0.uuid, "GPU-u0");
        // Profile fractions stored directly (independent of utilization fallback).
        assert!((g0.dram_active.unwrap() - 0.33).abs() < 1e-6);
        assert!((g0.gr_engine_active.unwrap() - 0.55).abs() < 1e-6);
        assert!((g0.tensor_active.unwrap() - 0.41).abs() < 1e-6);
        // Encoder/decoder percentages.
        assert!((g0.enc_utilization - 3.0).abs() < 1e-6);
        assert!((g0.dec_utilization - 5.0).abs() < 1e-6);
        assert!((g0.mem_temperature - 55.0).abs() < 1e-6);
        // Cumulative counters stored raw.
        assert_eq!(g0.total_energy_mj, 251_885_217_371);
        assert_eq!(g0.pcie_replay_count, 7);
        assert_eq!(g0.xid_errors, 2);
        assert_eq!(g0.remapped_rows_correctable, 1);
        assert_eq!(g0.remapped_rows_uncorrectable, 0);
        assert_eq!(g0.row_remap_failure, 0);
        assert_eq!(g0.vgpu_license_status, 1);
        // PCIe profile bytes/sec stored directly.
        assert_eq!(g0.pcie_prof_tx_bytes_per_sec, 12_582_912);
        assert_eq!(g0.pcie_prof_rx_bytes_per_sec, 8_388_608);
        // Also populated pcie_tx/rx_kbps fallback (12582912 / 1024 = 12288 KB/s).
        assert_eq!(g0.pcie_tx_kbps, 12_288);
        assert_eq!(g0.pcie_rx_kbps, 8_192);

        let g1 = &scrape.gpus[1];
        assert_eq!(g1.uuid, "GPU-u1");
        assert!((g1.dram_active.unwrap() - 0.66).abs() < 1e-6);
        assert!((g1.gr_engine_active.unwrap() - 0.88).abs() < 1e-6);
        assert!((g1.tensor_active.unwrap() - 0.72).abs() < 1e-6);
        assert_eq!(g1.total_energy_mj, 260_537_897_279);
        assert_eq!(g1.pcie_replay_count, 0);
        assert_eq!(g1.xid_errors, 0);
    }

    /// When both DCGM_FI_DEV_GPU_UTIL and DCGM_FI_PROF_GR_ENGINE_ACTIVE are
    /// present, `utilization` takes the DEV value (exact) while
    /// `gr_engine_active` captures the PROF fraction independently. Same for
    /// mem_utilization vs dram_active.
    #[test]
    fn test_prof_and_dev_utils_independent() {
        let input = r#"# TYPE DCGM_FI_DEV_GPU_UTIL gauge
DCGM_FI_DEV_GPU_UTIL{gpu="0",UUID="GPU-x"} 73
# TYPE DCGM_FI_DEV_MEM_COPY_UTIL gauge
DCGM_FI_DEV_MEM_COPY_UTIL{gpu="0",UUID="GPU-x"} 22
# TYPE DCGM_FI_PROF_GR_ENGINE_ACTIVE gauge
DCGM_FI_PROF_GR_ENGINE_ACTIVE{gpu="0",UUID="GPU-x"} 0.85
# TYPE DCGM_FI_PROF_DRAM_ACTIVE gauge
DCGM_FI_PROF_DRAM_ACTIVE{gpu="0",UUID="GPU-x"} 0.50
"#;
        let families = parse_prometheus_text(input).unwrap();
        let scrape = extract_gpu_metrics(&families);
        let g = &scrape.gpus[0];
        // DEV values win for the legacy scalar fields.
        assert!((g.utilization - 73.0).abs() < 1e-6);
        assert!((g.mem_utilization - 22.0).abs() < 1e-6);
        // PROF fractions are captured independently.
        assert!((g.gr_engine_active.unwrap() - 0.85).abs() < 1e-6);
        assert!((g.dram_active.unwrap() - 0.50).abs() < 1e-6);
    }

    #[test]
    fn test_gpu_pct_prefers_prof() {
        // When both DEV (jitter-prone 25) and PROF (cycle-accurate 0.92 → 92%)
        // are present, gpu_pct() returns PROF.
        let input = r#"# TYPE DCGM_FI_DEV_GPU_UTIL gauge
DCGM_FI_DEV_GPU_UTIL{gpu="0",UUID="GPU-x"} 25
# TYPE DCGM_FI_DEV_MEM_COPY_UTIL gauge
DCGM_FI_DEV_MEM_COPY_UTIL{gpu="0",UUID="GPU-x"} 30
# TYPE DCGM_FI_PROF_GR_ENGINE_ACTIVE gauge
DCGM_FI_PROF_GR_ENGINE_ACTIVE{gpu="0",UUID="GPU-x"} 0.92
# TYPE DCGM_FI_PROF_DRAM_ACTIVE gauge
DCGM_FI_PROF_DRAM_ACTIVE{gpu="0",UUID="GPU-x"} 0.85
# TYPE DCGM_FI_PROF_PIPE_TENSOR_ACTIVE gauge
DCGM_FI_PROF_PIPE_TENSOR_ACTIVE{gpu="0",UUID="GPU-x"} 0.71
"#;
        let families = parse_prometheus_text(input).unwrap();
        let scrape = extract_gpu_metrics(&families);
        let g = &scrape.gpus[0];
        assert!((g.gpu_pct() - 92.0).abs() < 1e-6);
        assert!((g.mbw_pct() - 85.0).abs() < 1e-6);
        assert!((g.tc_pct().unwrap() - 71.0).abs() < 1e-6);
    }

    #[test]
    fn test_gpu_pct_falls_back_to_dev_when_prof_missing() {
        // No PROF metrics — gpu_pct() falls back to DEV, tc_pct() returns None.
        let input = r#"# TYPE DCGM_FI_DEV_GPU_UTIL gauge
DCGM_FI_DEV_GPU_UTIL{gpu="0",UUID="GPU-x"} 78
# TYPE DCGM_FI_DEV_MEM_COPY_UTIL gauge
DCGM_FI_DEV_MEM_COPY_UTIL{gpu="0",UUID="GPU-x"} 42
"#;
        let families = parse_prometheus_text(input).unwrap();
        let scrape = extract_gpu_metrics(&families);
        let g = &scrape.gpus[0];
        assert!((g.gpu_pct() - 78.0).abs() < 1e-6);
        assert!((g.mbw_pct() - 42.0).abs() < 1e-6);
        assert!(g.tc_pct().is_none());
        assert!(scrape.avg_tc_pct().is_none());
    }

    #[test]
    fn test_peak_nvlink_bandwidth() {
        assert_eq!(peak_nvlink_bandwidth_gbps("NVIDIA B200"), Some(1800.0));
        assert_eq!(peak_nvlink_bandwidth_gbps("NVIDIA B100"), Some(1800.0));
        assert_eq!(peak_nvlink_bandwidth_gbps("NVIDIA H100 SXM"), Some(900.0));
        assert_eq!(peak_nvlink_bandwidth_gbps("NVIDIA H200"), Some(900.0));
        assert_eq!(peak_nvlink_bandwidth_gbps("NVIDIA H800"), Some(900.0));
        assert_eq!(
            peak_nvlink_bandwidth_gbps("NVIDIA A100-SXM4-80GB"),
            Some(600.0)
        );
        assert_eq!(peak_nvlink_bandwidth_gbps("NVIDIA A800"), Some(600.0));
        assert_eq!(peak_nvlink_bandwidth_gbps("NVIDIA A30"), Some(400.0));
        // Unknown GPU
        assert_eq!(peak_nvlink_bandwidth_gbps("NVIDIA RTX 4090"), None);
        assert_eq!(peak_nvlink_bandwidth_gbps(""), None);
    }

    #[test]
    fn test_peak_tflops_dtype() {
        // H200 — FP4 not supported (Hopper)
        assert_eq!(peak_tflops("NVIDIA H200", ComputeDtype::Bf16), Some(989.5));
        assert_eq!(peak_tflops("NVIDIA H200", ComputeDtype::Fp8), Some(1979.0));
        assert_eq!(peak_tflops("NVIDIA H200", ComputeDtype::Fp4), None);

        // B200 — sparse Blackwell numbers
        assert_eq!(peak_tflops("NVIDIA B200", ComputeDtype::Bf16), Some(4500.0));
        assert_eq!(peak_tflops("NVIDIA B200", ComputeDtype::Fp8), Some(9000.0));
        assert_eq!(peak_tflops("NVIDIA B200", ComputeDtype::Fp4), Some(18000.0));

        // B300 — sparse Blackwell Ultra numbers (consistent convention w/ B200)
        assert_eq!(peak_tflops("NVIDIA B300", ComputeDtype::Bf16), Some(7000.0));
        assert_eq!(peak_tflops("NVIDIA B300", ComputeDtype::Fp8), Some(15000.0));
        assert_eq!(peak_tflops("NVIDIA B300", ComputeDtype::Fp4), Some(30000.0));

        // Out-of-dtype-table GPUs: BF16 falls back to peak_bf16_tflops so
        // H100/A100 still resolve. Low-precision dtypes return None — there's
        // no FP8/FP4 entry in the broader table to use.
        assert_eq!(
            peak_tflops("NVIDIA H100 SXM", ComputeDtype::Bf16),
            Some(989.5)
        );
        assert_eq!(peak_tflops("NVIDIA H100 SXM", ComputeDtype::Fp8), None);
        assert_eq!(
            peak_tflops("NVIDIA A100-SXM4-80GB", ComputeDtype::Bf16),
            Some(312.0)
        );
        assert_eq!(peak_tflops("", ComputeDtype::Bf16), None);
    }

    #[test]
    fn test_infer_compute_dtype() {
        assert_eq!(
            infer_compute_dtype("example/Example-Model-NVFP4"),
            ComputeDtype::Fp4
        );
        assert_eq!(
            infer_compute_dtype("some-org/some-model-MXFP4"),
            ComputeDtype::Fp4
        );
        assert_eq!(
            infer_compute_dtype("example/Example-Model-FP8"),
            ComputeDtype::Fp8
        );
        // lowercase suffix is normalized
        assert_eq!(
            infer_compute_dtype("example/example-model-fp8"),
            ComputeDtype::Fp8
        );
        assert_eq!(
            infer_compute_dtype("example/Example-Model"),
            ComputeDtype::Bf16
        );
        // INT4 / AWQ / GPTQ dequant to BF16 at matmul — peak should be BF16.
        assert_eq!(
            infer_compute_dtype("example/Example-Model-GPTQ-Int4"),
            ComputeDtype::Bf16
        );
        assert_eq!(
            infer_compute_dtype("example/Example-Model-AWQ"),
            ComputeDtype::Bf16
        );
    }

    #[test]
    fn test_nvlink_utilization_known_gpu() {
        // H100: peak 900 GB/s bidirectional → 450 GB/s per direction
        // TX = 450 GB/s = 450 * 1024 * 1024 KB/s = 471859200 KB/s → 100% util
        let scrape = GpuScrape {
            gpus: vec![GpuMetrics {
                index: 0,
                name: "NVIDIA H100 SXM".to_string(),
                nvlink_tx_kbps: 471859200, // 450 GB/s
                nvlink_rx_kbps: 235929600, // 225 GB/s → 50%
                ..Default::default()
            }],
            ..Default::default()
        };
        let tx = scrape.avg_nvlink_tx_utilization().unwrap();
        assert!(
            (tx - 100.0).abs() < 0.1,
            "TX util should be ~100%, got {tx}"
        );
        let rx = scrape.avg_nvlink_rx_utilization().unwrap();
        assert!((rx - 50.0).abs() < 0.1, "RX util should be ~50%, got {rx}");
    }

    #[test]
    fn test_nvlink_utilization_unknown_gpu_skipped() {
        // Unknown GPU model should be skipped, not abort the whole calculation
        let scrape = GpuScrape {
            gpus: vec![
                GpuMetrics {
                    index: 0,
                    name: "NVIDIA H100 SXM".to_string(),
                    nvlink_tx_kbps: 471859200,
                    ..Default::default()
                },
                GpuMetrics {
                    index: 1,
                    name: "Unknown GPU".to_string(),
                    nvlink_tx_kbps: 100000,
                    ..Default::default()
                },
            ],
            ..Default::default()
        };
        // Should return Some (H100 data), not None (unknown GPU shouldn't abort)
        let tx = scrape.avg_nvlink_tx_utilization();
        assert!(
            tx.is_some(),
            "Should return Some even with unknown GPU mixed in"
        );
    }

    #[test]
    fn test_nvlink_utilization_no_data() {
        let scrape = GpuScrape {
            gpus: vec![GpuMetrics {
                index: 0,
                name: "NVIDIA H100 SXM".to_string(),
                nvlink_tx_kbps: 0,
                nvlink_rx_kbps: 0,
                ..Default::default()
            }],
            ..Default::default()
        };
        assert!(scrape.avg_nvlink_tx_utilization().is_none());
        assert!(scrape.avg_nvlink_rx_utilization().is_none());
    }

    #[test]
    fn test_extract_vmon_nvlink_metrics() {
        let input = r#"# HELP vmon_gpu_nvlink_tx_kbps GPU NVLink TX throughput in KB/s
# TYPE vmon_gpu_nvlink_tx_kbps gauge
vmon_gpu_nvlink_tx_kbps{gpu="0",name="NVIDIA H100 SXM"} 102400
vmon_gpu_nvlink_tx_kbps{gpu="1",name="NVIDIA H100 SXM"} 98304
# HELP vmon_gpu_nvlink_rx_kbps GPU NVLink RX throughput in KB/s
# TYPE vmon_gpu_nvlink_rx_kbps gauge
vmon_gpu_nvlink_rx_kbps{gpu="0",name="NVIDIA H100 SXM"} 51200
vmon_gpu_nvlink_rx_kbps{gpu="1",name="NVIDIA H100 SXM"} 49152
"#;
        let families = parse_prometheus_text(input).unwrap();
        let scrape = extract_gpu_metrics(&families);
        assert_eq!(scrape.gpus.len(), 2);
        assert_eq!(scrape.gpus[0].nvlink_tx_kbps, 102400);
        assert_eq!(scrape.gpus[0].nvlink_rx_kbps, 51200);
        assert_eq!(scrape.gpus[1].nvlink_tx_kbps, 98304);
        assert_eq!(scrape.gpus[1].nvlink_rx_kbps, 49152);
        assert_eq!(scrape.total_nvlink_tx_kbps(), 200704);
        assert_eq!(scrape.total_nvlink_rx_kbps(), 100352);
    }
}

#[cfg(test)]
mod throttle_tests {
    use super::*;

    #[test]
    fn test_decode_throttle_reasons() {
        assert!(decode_throttle_reasons(0).is_empty());
        assert_eq!(decode_throttle_reasons(1), vec!["GpuIdle"]);
        assert_eq!(decode_throttle_reasons(8), vec!["SwPowerCap"]);
        assert_eq!(decode_throttle_reasons(0x20), vec!["HwThermal"]);
        assert_eq!(
            decode_throttle_reasons(0x20 | 0x08),
            vec!["SwPowerCap", "HwThermal"]
        );
    }
}
