// SPDX-License-Identifier: Apache-2.0

//! Smoke test: read Prometheus text (vLLM/Dynamo /metrics) from stdin,
//! parse through our extractor, print what vmon would render.
//!
//!   curl -s http://gpu-node.example:8081/metrics | cargo run --example dynamo_smoke -p vmon-core

use std::io::Read;

use tokio::time::Instant;

use vmon_core::metrics::{detect_engines, extract_engine_metrics, extract_vllm_metrics};
use vmon_core::parser::parse_prometheus_text;

fn main() {
    let mut text = String::new();
    std::io::stdin().read_to_string(&mut text).expect("read stdin");

    let families = match parse_prometheus_text(&text) {
        Ok(f) => f,
        Err(e) => {
            eprintln!("PARSE ERROR: {e}");
            std::process::exit(2);
        }
    };
    println!("Parsed {} metric families", families.len());

    let now = Instant::now();
    let scrape = extract_vllm_metrics(&families, now);

    println!("\n-- Aggregate scrape --");
    println!(
        "  num_requests_running     = {}",
        scrape.num_requests_running
    );
    println!(
        "  num_requests_waiting     = {}",
        scrape.num_requests_waiting
    );
    println!(
        "  kv_cache_usage_perc      = {}",
        scrape.kv_cache_usage_perc
    );
    println!(
        "  prompt_tokens_total      = {}",
        scrape.prompt_tokens_total
    );
    println!(
        "  generation_tokens_total  = {}",
        scrape.generation_tokens_total
    );
    println!(
        "  request_success_total    = {}",
        scrape.request_success_total
    );
    println!("  ttft.count               = {}", scrape.ttft.count);
    println!("  itl.count                = {}", scrape.itl.count);
    println!("  e2e_latency.count        = {}", scrape.e2e_latency.count);
    println!(
        "  engine_scrapes len       = {}",
        scrape.engine_scrapes.len()
    );
    println!("  dynamo_component         = {:?}", scrape.dynamo_component);
    println!(
        "  dynamo_uptime_secs       = {}",
        scrape.dynamo_component_uptime_secs
    );
    println!(
        "  dynamo_requests_total    = {}",
        scrape.dynamo_component_requests_total
    );
    println!(
        "  dynamo_inflight          = {:?}",
        scrape.dynamo_component_inflight
    );
    println!(
        "  dynamo_model_load_secs   = {}",
        scrape.dynamo_component_model_load_secs
    );

    let engines = detect_engines(&families);
    println!("\nDetected engines: {:?}", engines);

    if !engines.is_empty() {
        for eng in &engines {
            let s = extract_engine_metrics(&families, eng, now);
            println!(
                "  engine={eng}: running={} waiting={} kv={} ttft.cnt={}",
                s.num_requests_running, s.num_requests_waiting, s.kv_cache_usage_perc, s.ttft.count
            );
        }
    }
}
