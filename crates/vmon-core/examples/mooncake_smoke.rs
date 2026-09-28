// SPDX-License-Identifier: Apache-2.0

//! Smoke test: read Prometheus text from stdin, parse through our pipeline,
//! print a summary of parsed Mooncake Store fields:
//!
//!   curl -s http://gpu-node.example:9003/metrics | \
//!     cargo run --example mooncake_smoke -p vmon-core

use std::io::Read;

use vmon_core::mooncake::parse_mooncake_metrics;
use vmon_core::parser::parse_prometheus_text;

fn main() {
    let mut text = String::new();
    std::io::stdin().read_to_string(&mut text).expect("read stdin");
    let families = parse_prometheus_text(&text).expect("parse Prometheus text");
    let s = parse_mooncake_metrics(&families);
    println!("mem_allocated_bytes      = {}", s.mem_allocated_bytes);
    println!("mem_total_bytes          = {}", s.mem_total_bytes);
    println!("key_count                = {}", s.key_count);
    println!("active_clients           = {}", s.active_clients);
    println!("total_requests           = {}", s.total_requests);
    println!("total_failures           = {}", s.total_failures);
    println!(
        "get_replica_list         = {}",
        s.get_replica_list_requests_total
    );
    println!("put_start                = {}", s.put_start_requests_total);
    println!("put_end                  = {}", s.put_end_requests_total);
    println!("exist_key                = {}", s.exist_key_requests_total);
    println!(
        "batch_get_replica_list   = {}",
        s.batch_get_replica_list_requests_total
    );
    println!(
        "batch_put_start          = {}",
        s.batch_put_start_requests_total
    );
    println!(
        "batch_put_end            = {}",
        s.batch_put_end_requests_total
    );
    println!(
        "batch_exist_key          = {}",
        s.batch_exist_key_requests_total
    );
    println!("segments                 = {}", s.segments.len());
    for seg in &s.segments {
        println!(
            "  {:<32} alloc={} total={}",
            seg.segment, seg.allocated_bytes, seg.total_bytes
        );
    }
}
