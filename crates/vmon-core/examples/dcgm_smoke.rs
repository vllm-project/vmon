// SPDX-License-Identifier: Apache-2.0

//! Smoke test: read Prometheus text from stdin, parse through our pipeline,
//! print a summary of parsed per-GPU fields. Used to validate against a live
//! DCGM exporter:
//!
//!   curl -s http://localhost:9400/metrics | cargo run --example dcgm_smoke -p vmon-core

use std::io::Read;

use vmon_core::gpu::extract_gpu_metrics;
use vmon_core::parser::parse_prometheus_text;

fn main() {
    let mut text = String::new();
    std::io::stdin().read_to_string(&mut text).expect("read stdin");
    let families = parse_prometheus_text(&text).expect("parse Prometheus text");
    let scrape = extract_gpu_metrics(&families);
    println!("Parsed {} GPU(s):", scrape.gpus.len());
    for g in &scrape.gpus {
        println!(
            "\nGPU {} ({}) uuid={}\n  util={:.1}% mem_util={:.1}% power={:.1}W temp={}°C mem_temp={}°C\n  \
             dram_active={:.3} gr_engine_active={:.3} tensor_active={:.3}\n  \
             enc_util={:.1}% dec_util={:.1}%\n  \
             fb_used={} fb_total={} clock={}MHz mem_clock={}MHz\n  \
             pcie_kbps tx={} rx={}  pcie_prof_bps tx={} rx={}\n  \
             nvlink_kbps tx={} rx={}\n  \
             throttle=0x{:x} ecc_sbe={} ecc_dbe={}\n  \
             total_energy_mj={} pcie_replay={} xid={} remap(c/u/fail)={}/{}/{} vgpu_lic={}",
            g.index,
            g.name,
            g.uuid,
            g.utilization,
            g.mem_utilization,
            g.power_watts,
            g.temperature,
            g.mem_temperature,
            g.dram_active.unwrap_or(0.0),
            g.gr_engine_active.unwrap_or(0.0),
            g.tensor_active.unwrap_or(0.0),
            g.enc_utilization,
            g.dec_utilization,
            g.mem_used_bytes,
            g.mem_total_bytes,
            g.clock_mhz,
            g.mem_clock_mhz,
            g.pcie_tx_kbps,
            g.pcie_rx_kbps,
            g.pcie_prof_tx_bytes_per_sec,
            g.pcie_prof_rx_bytes_per_sec,
            g.nvlink_tx_kbps,
            g.nvlink_rx_kbps,
            g.throttle_reasons,
            g.ecc_sbe_volatile,
            g.ecc_dbe_volatile,
            g.total_energy_mj,
            g.pcie_replay_count,
            g.xid_errors,
            g.remapped_rows_correctable,
            g.remapped_rows_uncorrectable,
            g.row_remap_failure,
            g.vgpu_license_status,
        );
    }
}
