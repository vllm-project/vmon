// SPDX-License-Identifier: Apache-2.0

mod agent;
mod cli;
mod report_io;

use agent::{daemonize, run_agent};
use cli::{Cli, Command};
use report_io::write_report_atomic;
use std::collections::{HashMap, HashSet};
use std::path::{Path, PathBuf};
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

use clap::Parser;
use regex::Regex;
use tokio::io::AsyncWriteExt;
use vmon_core::parser::{MetricFamily, MetricType, Sample, parse_prometheus_text};

/// Render a collected report as JSON or HTML based on the output extension.
/// `.json` → JSON (honoring the optional metric filter); anything else → HTML.
/// Used for both periodic checkpoints and the final write so they stay in sync.
fn render_report(
    collector: &vmon_report::collector::TimeSeriesCollector,
    output: &std::path::Path,
    metrics: Option<&[String]>,
) -> String {
    match output.extension().and_then(|e| e.to_str()) {
        Some("json") => vmon_report::json::render(collector, metrics),
        _ => vmon_report::html::render(collector),
    }
}

fn default_true() -> bool {
    true
}

#[derive(Debug, Clone, serde::Deserialize)]
#[serde(deny_unknown_fields)]
struct RawCaptureConfig {
    capture: Vec<RawCaptureRule>,
}

#[derive(Debug, Clone, serde::Deserialize)]
#[serde(deny_unknown_fields)]
struct RawCaptureRule {
    name: String,
    output: PathBuf,
    mode: RawMatchMode,
    patterns: Vec<String>,
    #[serde(default = "default_true")]
    include_help: bool,
    #[serde(default = "default_true")]
    include_type: bool,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Deserialize)]
#[serde(rename_all = "lowercase")]
enum RawMatchMode {
    Prefix,
    Regex,
}

#[derive(Debug, Clone)]
enum RawMatcher {
    Prefix(Vec<String>),
    Regex(Vec<Regex>),
}

#[derive(Debug, Clone)]
struct CompiledRawCaptureRule {
    name: String,
    output: PathBuf,
    include_help: bool,
    include_type: bool,
    matcher: RawMatcher,
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct RawCaptureTarget {
    node: String,
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct RawCaptureSummary {
    name: String,
    output: PathBuf,
    written: usize,
}

struct RawCaptureSink {
    rule: CompiledRawCaptureRule,
    file: tokio::fs::File,
    written: usize,
}

/// Expand range patterns like `example-[01-18]` into individual node names.
/// Supports zero-padded ranges and port suffixes: `node[1-4]:8000`.
fn expand_nodes(nodes: &[String]) -> Vec<String> {
    nodes.iter().flat_map(|n| expand_range(n)).collect()
}

fn expand_range(node: &str) -> Vec<String> {
    let Some(open) = node.find('[') else {
        return vec![node.to_string()];
    };
    let Some(close) = node[open..].find(']').map(|i| open + i) else {
        return vec![node.to_string()];
    };
    let prefix = &node[..open];
    let suffix = &node[close + 1..];
    let range_str = &node[open + 1..close];

    // Support comma-separated values and ranges: [01-04,07,10-12]
    let mut result = Vec::new();
    for part in range_str.split(',') {
        let part = part.trim();
        if let Some(dash_pos) = part.find('-') {
            let start_str = &part[..dash_pos];
            let end_str = &part[dash_pos + 1..];
            let Ok(start) = start_str.parse::<u32>() else {
                result.push(node.to_string());
                return result;
            };
            let Ok(end) = end_str.parse::<u32>() else {
                result.push(node.to_string());
                return result;
            };
            let width = start_str.len();
            for i in start..=end {
                result.push(format!("{prefix}{i:0>width$}{suffix}"));
            }
        } else if let Ok(i) = part.parse::<u32>() {
            let width = part.len();
            result.push(format!("{prefix}{i:0>width$}{suffix}"));
        } else {
            return vec![node.to_string()];
        }
    }
    result
}

/// Detect ranks per host for Dynamo mode.
/// Priority: explicit override → SLURM_GPUS_ON_NODE → SLURM_NTASKS_PER_NODE → 1.
fn detect_ranks_per_host(override_val: Option<u16>) -> u16 {
    if let Some(n) = override_val {
        return n.max(1);
    }
    for var in ["SLURM_GPUS_ON_NODE", "SLURM_NTASKS_PER_NODE"] {
        if let Ok(s) = std::env::var(var) {
            if let Ok(n) = s.trim().parse::<u16>() {
                if n > 0 {
                    return n;
                }
            }
        }
    }
    1
}

/// Expand bare hosts into `host:port` entries with globally-incrementing
/// rank ports starting at `base_port`. Each host gets `ranks_per_host`
/// entries; ports continue incrementing across hosts.
///
/// Example: hosts=[node01, node02], base_port=8081, ranks_per_host=4 →
///   node01:8081..8084, node02:8085..8088.
///
/// If a host already has an explicit `:port` suffix, it is preserved as-is
/// (treated as a single rank).
fn dynamo_expand_ports(hosts: &[String], base_port: u16, ranks_per_host: u16) -> Vec<String> {
    let ranks = ranks_per_host.max(1);
    let mut result = Vec::with_capacity(hosts.len() * ranks as usize);
    let mut port: u32 = base_port as u32;
    for h in hosts {
        if let Some((bare, p)) = h.rsplit_once(':') {
            if p.parse::<u16>().is_ok() && !bare.is_empty() {
                result.push(h.clone());
                continue;
            }
        }
        for _ in 0..ranks {
            let p = port.min(u16::MAX as u32) as u16;
            result.push(format!("{h}:{p}"));
            port += 1;
        }
    }
    result
}

/// Expand `expanded` hosts for Dynamo mode, resetting the rank port counter
/// per SLURM job so each job's ranks start at `base_port`. When no jobs are
/// known, falls back to a single-group expansion (the legacy behavior).
///
/// Hosts not owned by any of the given jobs (e.g. SLURM unavailable, or a
/// node added manually) are folded into a final group that also starts at
/// `base_port` — this matches "everything was launched independently".
fn dynamo_expand_grouped(
    expanded: &[String],
    slurm_jobs: &[vmon_core::slurm::SlurmJobInfo],
    base_port: u16,
    ranks_per_host: u16,
) -> Vec<String> {
    if slurm_jobs.is_empty() {
        let entries = dynamo_expand_ports(expanded, base_port, ranks_per_host);
        tracing::info!(
            hosts = expanded.len(),
            ranks_per_host,
            endpoints = entries.len(),
            port_lo = base_port,
            port_hi = base_port as u32 + entries.len().saturating_sub(1) as u32,
            "Dynamo: expanded host × rank endpoints"
        );
        return entries;
    }

    let mut entries: Vec<String> = Vec::new();
    let mut covered: std::collections::HashSet<String> = std::collections::HashSet::new();

    for job in slurm_jobs {
        let group: Vec<String> =
            expanded.iter().filter(|h| job.nodes.iter().any(|n| n == *h)).cloned().collect();
        if group.is_empty() {
            continue;
        }
        let group_entries = dynamo_expand_ports(&group, base_port, ranks_per_host);
        tracing::info!(
            job_id = %job.job_id,
            hosts = group.len(),
            ranks_per_host,
            endpoints = group_entries.len(),
            port_lo = base_port,
            port_hi = base_port as u32 + group_entries.len().saturating_sub(1) as u32,
            "Dynamo job: expanded host × rank endpoints"
        );
        for h in &group {
            covered.insert(h.clone());
        }
        entries.extend(group_entries);
    }

    let orphans: Vec<String> = expanded.iter().filter(|h| !covered.contains(*h)).cloned().collect();
    if !orphans.is_empty() {
        let orphan_entries = dynamo_expand_ports(&orphans, base_port, ranks_per_host);
        tracing::info!(
            hosts = orphans.len(),
            ranks_per_host,
            endpoints = orphan_entries.len(),
            port_lo = base_port,
            port_hi = base_port as u32 + orphan_entries.len().saturating_sub(1) as u32,
            "Dynamo (no SLURM job): expanded host × rank endpoints"
        );
        entries.extend(orphan_entries);
    }

    entries
}

/// Normalize node addresses: append default port if not specified.
fn normalize_addrs(nodes: &[String], default_port: u16) -> Vec<String> {
    nodes
        .iter()
        .map(|n| {
            if n.contains(':') {
                n.clone()
            } else {
                format!("{n}:{default_port}")
            }
        })
        .collect()
}

/// Split a `host:port` addr into its host slice and parsed port. Unparseable
/// ports collate before numeric ones (port = 0) so malformed addrs still get
/// a deterministic position.
fn split_host_port(addr: &str) -> (&str, u16) {
    match addr.rsplit_once(':') {
        Some((host, port)) => (host, port.parse().unwrap_or(0)),
        None => (addr, 0),
    }
}

/// Default Mooncake Store leader port. Used as
/// the auto-target when the user passes a bare `--mooncake` (no address).
const DEFAULT_MOONCAKE_PORT: u16 = 8702;

/// Wire the `--mooncake` flag into the scraper:
///   - `None`            → flag omitted: monitoring disabled.
///   - `Some(Some(addr))`→ explicit `--mooncake=host:port`, single target.
///   - `Some(None)`      → bare `--mooncake`: auto-target one store per
///     tracked SLURM job (the job's first node on [`DEFAULT_MOONCAKE_PORT`]),
///     following jobs as they come and go. Without SLURM context the scraper
///     falls back to the first scrape target's host.
fn apply_mooncake_flag(
    scraper: vmon_core::scraper::Scraper,
    flag: Option<Option<String>>,
) -> vmon_core::scraper::Scraper {
    match flag {
        None => scraper,
        Some(Some(addr)) => scraper.with_mooncake(addr),
        Some(None) => {
            eprintln!(
                "Mooncake: auto-targeting each tracked job's first node \
                 :{DEFAULT_MOONCAKE_PORT} (use --mooncake=host:port to override)"
            );
            scraper.with_mooncake_auto(DEFAULT_MOONCAKE_PORT)
        }
    }
}

/// Resolve node list: merge CLI nodes with Slurm discovery results.
/// When no nodes are given and no explicit flags, auto-detects from
/// `SLURM_JOB_NODELIST` environment variable.
fn resolve_nodes(
    nodes: Vec<String>,
    slurm_jobs: Vec<String>,
    slurm_job_name: Option<String>,
) -> Vec<String> {
    let mut result = nodes;

    // Explicit Slurm flags: query squeue per job spec
    // Each entry can be "job_id" or "job_id:port"
    if !slurm_jobs.is_empty() {
        for spec in &slurm_jobs {
            let (job_id, port) = match spec.rsplit_once(':') {
                Some((id, p)) if p.parse::<u16>().is_ok() => (id, Some(p.to_string())),
                _ => (spec.as_str(), None),
            };
            match discover_slurm_nodes(&[job_id.to_string()], None) {
                Ok(hosts) => {
                    if hosts.is_empty() {
                        eprintln!("Warning: job {job_id} returned no nodes");
                    } else {
                        let label = port.as_deref().unwrap_or("default");
                        eprintln!("Slurm job {job_id}: {} node(s) (port {label})", hosts.len());
                        for h in hosts {
                            if let Some(ref p) = port {
                                result.push(format!("{h}:{p}"));
                            } else {
                                result.push(h);
                            }
                        }
                    }
                }
                Err(e) => eprintln!("Slurm job {job_id}: {e}"),
            }
        }
    }

    if slurm_job_name.is_some() {
        match discover_slurm_nodes(&[], slurm_job_name.as_deref()) {
            Ok(discovered) => {
                if discovered.is_empty() {
                    eprintln!("Warning: Slurm discovery returned no nodes");
                } else {
                    eprintln!("Slurm: discovered {} node(s)", discovered.len());
                    result.extend(discovered);
                }
            }
            Err(e) => eprintln!("Slurm discovery failed: {e}"),
        }
    }

    // Auto-detect from Slurm. Source-of-truth ordering:
    //   1) `squeue -t RUNNING -u $USER` — authoritative live state.
    //   2) `SLURM_JOB_NODELIST` env — only when squeue is unavailable.
    //      The env var persists in interactive shells after the
    //      originating allocation ends, so trusting it on a login node
    //      pins the TUI to stale hosts; we only fall back to it when
    //      squeue itself can't be reached (no SLURM on this machine).
    //   3) localhost
    if result.is_empty() {
        match discover_slurm_nodes(&[], None) {
            Ok(discovered) if !discovered.is_empty() => {
                eprintln!(
                    "Slurm: discovered {} node(s) from running jobs",
                    discovered.len()
                );
                result.extend(discovered);
            }
            Ok(_) => {
                if let Ok(nodelist) = std::env::var("SLURM_JOB_NODELIST") {
                    if !nodelist.is_empty() {
                        eprintln!(
                            "Slurm: ignoring stale SLURM_JOB_NODELIST={nodelist} \
                             (squeue shows no running job for this user)"
                        );
                    }
                }
            }
            Err(_) => {
                // squeue unavailable (e.g. running outside a SLURM cluster).
                // Last-ditch: use env if set.
                if let Ok(nodelist) = std::env::var("SLURM_JOB_NODELIST") {
                    if !nodelist.is_empty() {
                        eprintln!(
                            "Auto-detected SLURM_JOB_NODELIST={nodelist} (squeue unavailable)"
                        );
                        let expanded = expand_slurm_nodelist(&nodelist);
                        if !expanded.is_empty() {
                            eprintln!("Slurm: {} node(s)", expanded.len());
                            result.extend(expanded);
                        }
                    }
                }
            }
        }

        if result.is_empty() {
            eprintln!("No nodes specified, monitoring localhost");
            result.push("localhost".to_string());
        }
    }

    // Deduplicate
    let mut seen = std::collections::HashSet::new();
    result.retain(|n| seen.insert(n.clone()));
    result
}

/// Expand a Slurm compact nodelist (e.g. "gpu-node-[01-04],other-[1-2]").
fn expand_slurm_nodelist(nodelist: &str) -> Vec<String> {
    // Try scontrol first (most reliable)
    if let Ok(output) = std::process::Command::new("scontrol")
        .args(["show", "hostnames", nodelist])
        .output()
    {
        if output.status.success() {
            let hosts = String::from_utf8_lossy(&output.stdout);
            let nodes: Vec<String> =
                hosts.lines().map(|l| l.trim().to_string()).filter(|l| !l.is_empty()).collect();
            if !nodes.is_empty() {
                return nodes;
            }
        }
    }

    // Fallback: split on commas outside brackets and use our expand_range
    split_slurm_hostlist(nodelist).iter().flat_map(|n| expand_range(n)).collect()
}

/// Split a Slurm hostlist like "gpu[01-04],other[01-02]" into individual patterns.
/// Commas inside brackets are range separators, not group separators.
fn split_slurm_hostlist(hostlist: &str) -> Vec<String> {
    let mut result = Vec::new();
    let mut current = String::new();
    let mut depth = 0;
    for c in hostlist.chars() {
        match c {
            '[' => {
                depth += 1;
                current.push(c);
            }
            ']' => {
                depth -= 1;
                current.push(c);
            }
            ',' if depth == 0 => {
                let trimmed = current.trim().to_string();
                if !trimmed.is_empty() {
                    result.push(trimmed);
                }
                current.clear();
            }
            _ => current.push(c),
        }
    }
    let trimmed = current.trim().to_string();
    if !trimmed.is_empty() {
        result.push(trimmed);
    }
    result
}

/// Discover nodes via Slurm commands.
///
/// Uses `squeue` to find running jobs, then `scontrol show hostnames` to
/// expand the compact hostlist into individual hostnames.
fn discover_slurm_nodes(job_ids: &[String], job_name: Option<&str>) -> Result<Vec<String>, String> {
    // Build squeue command to get compact node lists
    let mut cmd = std::process::Command::new("squeue");
    cmd.args(["-h", "-t", "RUNNING", "-o", "%N"]);

    if !job_ids.is_empty() {
        cmd.args(["-j", &job_ids.join(",")]);
    } else {
        // Filter by current user unless specific job IDs are given
        let user = std::env::var("USER").unwrap_or_default();
        if !user.is_empty() {
            cmd.args(["-u", &user]);
        }
        if let Some(name) = job_name {
            cmd.args(["-n", name]);
        }
    }

    let output = cmd
        .output()
        .map_err(|e| format!("failed to run squeue: {e} (is Slurm installed?)"))?;

    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        return Err(format!("squeue failed: {}", stderr.trim()));
    }

    let stdout = String::from_utf8_lossy(&output.stdout);
    let compact_lists: Vec<&str> = stdout.lines().filter(|l| !l.trim().is_empty()).collect();

    if compact_lists.is_empty() {
        return Ok(Vec::new());
    }

    // Expand each compact hostlist via scontrol show hostnames
    let mut all_nodes = Vec::new();
    for hostlist in &compact_lists {
        let expand_output = std::process::Command::new("scontrol")
            .args(["show", "hostnames", hostlist.trim()])
            .output()
            .map_err(|e| format!("failed to run scontrol: {e}"))?;

        if !expand_output.status.success() {
            // Fallback: try to expand via our own expand_range
            let expanded = expand_nodes(&[hostlist.trim().to_string()]);
            all_nodes.extend(expanded);
            continue;
        }

        let hosts = String::from_utf8_lossy(&expand_output.stdout);
        for host in hosts.lines() {
            let h = host.trim();
            if !h.is_empty() {
                all_nodes.push(h.to_string());
            }
        }
    }

    // Deduplicate (multiple jobs may share nodes)
    let mut seen = std::collections::HashSet::new();
    all_nodes.retain(|n| seen.insert(n.clone()));

    Ok(all_nodes)
}

/// Find every user-owned running SLURM job whose nodelist overlaps `targets`.
///
/// Returns an empty Vec when SLURM is unavailable or nothing matches. Targets
/// are bare hostnames (no port). Compact lists from `squeue %N` are expanded
/// via `scontrol show hostnames`.
fn discover_slurm_jobs_for_hosts(targets: &[String]) -> Vec<vmon_core::slurm::SlurmJobInfo> {
    if targets.is_empty() {
        return Vec::new();
    }

    // Strip ports from targets so we compare bare hostnames.
    let target_hosts: std::collections::HashSet<String> =
        targets.iter().map(|t| t.split(':').next().unwrap_or(t).to_string()).collect();

    let user = std::env::var("USER").unwrap_or_default();
    let mut cmd = std::process::Command::new("squeue");
    cmd.args(["-h", "-t", "RUNNING", "-o", "%i|%j|%M|%N"]);
    if !user.is_empty() {
        cmd.args(["-u", &user]);
    }

    let output = match cmd.output() {
        Ok(o) if o.status.success() => o,
        _ => return Vec::new(),
    };
    let stdout = String::from_utf8_lossy(&output.stdout);

    let mut matches: Vec<vmon_core::slurm::SlurmJobInfo> = Vec::new();
    for line in stdout.lines() {
        let parts: Vec<&str> = line.splitn(4, '|').collect();
        if parts.len() < 4 {
            continue;
        }
        let job_id = parts[0].trim().to_string();
        let job_name = parts[1].trim().to_string();
        let elapsed = parts[2].trim();
        let nodelist_compact = parts[3].trim().to_string();
        if job_id.is_empty() || nodelist_compact.is_empty() {
            continue;
        }

        let hosts = expand_slurm_nodelist(&nodelist_compact);
        if !hosts.iter().any(|h| target_hosts.contains(h)) {
            continue;
        }

        let Some(elapsed_secs) = parse_slurm_elapsed(elapsed) else {
            continue;
        };
        let Some(start_time) =
            std::time::SystemTime::now().checked_sub(std::time::Duration::from_secs(elapsed_secs))
        else {
            continue;
        };

        matches.push(vmon_core::slurm::SlurmJobInfo {
            job_id,
            job_name,
            start_time,
            end_time: None,
            nodelist_compact,
            nodes: hosts,
        });
    }

    // Newest job first (most recent start_time); job_id breaks ties for stability.
    matches.sort_by(|a, b| b.start_time.cmp(&a.start_time).then_with(|| a.job_id.cmp(&b.job_id)));
    matches
}

/// Re-poll `squeue` and freeze any tracked job whose id no longer appears
/// in the running set by stamping `end_time = now`. Once frozen, a job's
/// uptime stops walking; the TUI dims it and renders an `(ended)` marker.
///
/// Failures (squeue missing, non-zero exit, transient hiccups) are silent
/// no-ops so we never produce false "ended" reports — accuracy over speed.
async fn refresh_slurm_end_times(handle: &vmon_core::scraper::SharedSlurmJobs) {
    let user = std::env::var("USER").unwrap_or_default();
    let mut cmd = tokio::process::Command::new("squeue");
    cmd.args(["-h", "-t", "RUNNING", "-o", "%i"]);
    if !user.is_empty() {
        cmd.args(["-u", &user]);
    }
    let output = match cmd.output().await {
        Ok(o) if o.status.success() => o,
        _ => return,
    };
    let stdout = String::from_utf8_lossy(&output.stdout);
    let running: std::collections::HashSet<String> =
        stdout.lines().map(|l| l.trim().to_string()).filter(|s| !s.is_empty()).collect();

    let mut jobs = handle.lock().expect("slurm_jobs mutex poisoned");
    let now = std::time::SystemTime::now();
    for job in jobs.iter_mut() {
        if job.end_time.is_none() && !running.contains(&job.job_id) {
            job.end_time = Some(now);
            tracing::info!(job_id = %job.job_id, "SLURM job left squeue, freezing uptime");
        }
    }
}

/// Spawn a background task that re-polls `squeue` every 30s and freezes
/// uptime for any tracked job that has left the running set.
fn spawn_slurm_refresh_task(handle: vmon_core::scraper::SharedSlurmJobs) {
    tokio::spawn(async move {
        let mut ticker = tokio::time::interval(std::time::Duration::from_secs(30));
        ticker.tick().await; // skip immediate fire
        loop {
            ticker.tick().await;
            refresh_slurm_end_times(&handle).await;
        }
    });
}

/// Poll `sinfo` once and update the shared cluster-wide node counts.
///
/// `-N` prints one line per node per partition; parsing dedupes by
/// hostname. Failures (sinfo missing, non-zero exit) are silent no-ops
/// that keep the last-known value — a transient hiccup shouldn't blank
/// the header, and outside a SLURM cluster the value simply stays `None`
/// so the TUI never shows the segment.
async fn refresh_slurm_cluster_stats(handle: &vmon_core::scraper::SharedSlurmCluster) {
    let output = match tokio::process::Command::new("sinfo")
        .args(["-h", "-N", "-o", "%N %t"])
        .output()
        .await
    {
        Ok(o) if o.status.success() => o,
        _ => return,
    };
    let stdout = String::from_utf8_lossy(&output.stdout);
    let stats = vmon_core::slurm::parse_sinfo_node_states(&stdout);
    if stats.total_nodes == 0 {
        return;
    }
    *handle.lock().expect("slurm_cluster mutex poisoned") = Some(stats);
}

/// Spawn a background task that polls `sinfo` every 30s (with an immediate
/// first fire) and publishes cluster-wide total/idle node counts.
fn spawn_slurm_cluster_stats_task(handle: vmon_core::scraper::SharedSlurmCluster) {
    tokio::spawn(async move {
        let mut ticker = tokio::time::interval(std::time::Duration::from_secs(30));
        loop {
            ticker.tick().await; // first tick fires immediately
            refresh_slurm_cluster_stats(&handle).await;
        }
    });
}

/// Filter applied by the SLURM follow task on every `squeue` poll.
///
/// `User` matches every RUNNING job owned by `$USER` (the default when
/// `vmon watch` is invoked with no positional nodes and no slurm flags).
/// `JobIds` and `JobName` mirror the existing `--slurm-job` / `--slurm-job-name`
/// CLI flags, applied on every poll instead of just at startup.
#[derive(Debug, Clone)]
enum SlurmFollowFilter {
    User,
    JobIds(Vec<String>),
    JobName(String),
}

/// Configuration for how to expand SLURM-discovered hosts into scrape addresses
/// in NVIDIA Dynamo mode. `None` is the vLLM default (host:port via `port`).
#[derive(Debug, Clone)]
enum DynamoFollowMode {
    /// Fixed per-rank base port; `dynamo_expand_grouped` resets ports per job.
    Explicit { base_port: u16, ranks_per_host: u16 },
    /// Probe candidate ports per host via `dynamo_discover::discover_addrs`.
    Discover,
}

/// Run `squeue` once and parse the user's currently-RUNNING jobs that match
/// `filter`.
///
/// Returns:
/// - `Some(jobs)` — squeue ran and parsed (may be empty if no jobs match).
/// - `None` — squeue failed (binary missing, non-zero exit, ...); the caller
///   skips the merge so a transient failure doesn't spuriously stamp
///   `end_time` on every tracked job.
async fn query_slurm_jobs(
    filter: &SlurmFollowFilter,
) -> Option<Vec<vmon_core::slurm::SlurmJobInfo>> {
    let mut cmd = tokio::process::Command::new("squeue");
    cmd.args(["-h", "-t", "RUNNING", "-o", "%i|%j|%M|%N"]);
    match filter {
        SlurmFollowFilter::User => {
            let user = std::env::var("USER").unwrap_or_default();
            if !user.is_empty() {
                cmd.args(["-u", &user]);
            }
        }
        SlurmFollowFilter::JobIds(ids) => {
            cmd.args(["-j", &ids.join(",")]);
        }
        SlurmFollowFilter::JobName(name) => {
            let user = std::env::var("USER").unwrap_or_default();
            if !user.is_empty() {
                cmd.args(["-u", &user]);
            }
            cmd.args(["-n", name]);
        }
    }

    let output = match cmd.output().await {
        Ok(o) if o.status.success() => o,
        Ok(o) => {
            tracing::warn!(
                exit = ?o.status,
                stderr = %String::from_utf8_lossy(&o.stderr),
                "squeue exited non-zero — skipping merge this tick"
            );
            return None;
        }
        Err(e) => {
            tracing::warn!(error = %e, "squeue invocation failed — skipping merge this tick");
            return None;
        }
    };
    let stdout = String::from_utf8_lossy(&output.stdout);

    let mut jobs = Vec::new();
    for line in stdout.lines() {
        let parts: Vec<&str> = line.splitn(4, '|').collect();
        if parts.len() < 4 {
            continue;
        }
        let job_id = parts[0].trim().to_string();
        let job_name = parts[1].trim().to_string();
        let elapsed = parts[2].trim();
        let nodelist_compact = parts[3].trim().to_string();
        if job_id.is_empty() || nodelist_compact.is_empty() {
            continue;
        }
        let Some(elapsed_secs) = parse_slurm_elapsed(elapsed) else {
            continue;
        };
        let Some(start_time) =
            std::time::SystemTime::now().checked_sub(std::time::Duration::from_secs(elapsed_secs))
        else {
            continue;
        };
        // expand_slurm_nodelist invokes scontrol synchronously; offload to
        // the blocking pool so it doesn't stall the async runtime when SLURM
        // is slow to respond.
        let nodelist_for_blocking = nodelist_compact.clone();
        let nodes =
            tokio::task::spawn_blocking(move || expand_slurm_nodelist(&nodelist_for_blocking))
                .await
                .unwrap_or_default();
        jobs.push(vmon_core::slurm::SlurmJobInfo {
            job_id,
            job_name,
            start_time,
            end_time: None,
            nodelist_compact,
            nodes,
        });
    }
    // Newest job first (most recent start_time); job_id breaks ties for stability.
    jobs.sort_by(|a, b| b.start_time.cmp(&a.start_time).then_with(|| a.job_id.cmp(&b.job_id)));
    Some(jobs)
}

/// Merge a fresh squeue snapshot into the existing `slurm_jobs` list.
///
/// Returns the list of job IDs that just transitioned from running → ended on
/// this poll (logged by the caller), plus mutates `existing` in place:
/// - Existing IDs that are still in `fresh` keep their `start_time` and
///   `end_time = None`.
/// - Existing IDs that have left `fresh` get `end_time = Some(now)` if not
///   already set.
/// - Existing IDs whose `end_time` is older than `grace` are dropped.
/// - New IDs in `fresh` are appended.
fn merge_slurm_jobs(
    existing: &mut Vec<vmon_core::slurm::SlurmJobInfo>,
    fresh: Vec<vmon_core::slurm::SlurmJobInfo>,
    now: std::time::SystemTime,
    grace: Duration,
) -> Vec<String> {
    use std::collections::HashSet;

    let fresh_ids: HashSet<String> = fresh.iter().map(|j| j.job_id.clone()).collect();

    let mut newly_ended = Vec::new();
    for job in existing.iter_mut() {
        if !fresh_ids.contains(&job.job_id) && job.end_time.is_none() {
            job.end_time = Some(now);
            newly_ended.push(job.job_id.clone());
        }
    }
    // Drop ended jobs whose grace has elapsed. On time error (clock
    // jumped backwards), drop rather than retain — otherwise an `(ended)`
    // row could stick forever.
    existing.retain(|job| match job.end_time {
        Some(end) => now.duration_since(end).map(|d| d < grace).unwrap_or(false),
        None => true,
    });
    for fresh_job in fresh {
        if let Some(existing_job) = existing.iter_mut().find(|j| j.job_id == fresh_job.job_id) {
            // Job re-appeared in squeue. Update mutable fields so requeues
            // / nodelist edits propagate, and clear any end_time stamp
            // from a prior poll where it was briefly missing. Keep
            // `start_time` so uptime doesn't reset.
            existing_job.nodes = fresh_job.nodes;
            existing_job.nodelist_compact = fresh_job.nodelist_compact;
            existing_job.job_name = fresh_job.job_name;
            existing_job.end_time = None;
        } else {
            existing.push(fresh_job);
        }
    }
    // Newest job first (most recent start_time); job_id breaks ties for stability.
    existing.sort_by(|a, b| b.start_time.cmp(&a.start_time).then_with(|| a.job_id.cmp(&b.job_id)));
    newly_ended
}

/// Compute the desired scrape address list from a set of jobs (vLLM mode).
/// Hosts come from each job's `nodes`, expanded via `expand_nodes` (range
/// patterns) and normalized with `normalize_addrs`. Order is stable: jobs
/// in the order given, hosts in `nodes` order, deduped.
///
/// Includes ended jobs that are still in the list (i.e. within the grace
/// window — `merge_slurm_jobs` drops them once `grace` has elapsed). The
/// row needs to remain in the scrape set so the TUI can render `(ended)`
/// against it instead of having the row disappear instantly.
fn desired_addrs_vllm(jobs: &[vmon_core::slurm::SlurmJobInfo], default_port: u16) -> Vec<String> {
    let mut hosts: Vec<String> = Vec::new();
    let mut seen = std::collections::HashSet::new();
    for job in jobs {
        for h in &job.nodes {
            if seen.insert(h.clone()) {
                hosts.push(h.clone());
            }
        }
    }
    let expanded = expand_nodes(&hosts);
    normalize_addrs(&expanded, default_port)
}

/// Compute the desired scrape address list from a set of jobs (Dynamo,
/// explicit base port). Each job's hosts are expanded with port counter
/// resetting per job (`dynamo_expand_grouped`).
///
/// Includes ended jobs that are still in the list (within grace) so the
/// `(ended)` overlay is visible before the row disappears.
fn desired_addrs_dynamo_explicit(
    jobs: &[vmon_core::slurm::SlurmJobInfo],
    base_port: u16,
    ranks_per_host: u16,
) -> Vec<String> {
    let mut hosts: Vec<String> = Vec::new();
    let mut seen = std::collections::HashSet::new();
    for job in jobs {
        for h in &job.nodes {
            if seen.insert(h.clone()) {
                hosts.push(h.clone());
            }
        }
    }
    dynamo_expand_grouped(&hosts, jobs, base_port, ranks_per_host)
}

/// Decide which SLURM-tracked hosts need a fresh Dynamo `/metrics` probe and
/// which existing addrs to carry over. Pure function — no I/O — so it can be
/// unit-tested without spinning up real workers.
///
/// `highwater` is the caller-maintained "max real ports ever seen" per host;
/// pass an empty map on the first tick. The caller is responsible for updating
/// it via [`update_highwater`] once a new addr set is settled.
///
/// `healthy_hosts` / `dead_hosts` carry scrape-health context from the watch
/// channel: hosts whose folded row is currently healthy, and hosts whose
/// every endpoint has been failing scrapes for several consecutive follow
/// ticks (see `DEAD_HOST_TICKS`). Pass empty sets when health is unknown —
/// that reproduces the health-blind behavior.
///
/// Returns `(kept_addrs, hosts_to_probe)`:
/// - `kept_addrs` retains every real (non-placeholder) addr whose host is
///   still in the SLURM snapshot. Placeholder addrs (unless the host is
///   healthy — then something really is serving on the placeholder port),
///   addrs of dead hosts, and addrs for hosts no longer tracked are
///   dropped — the caller will re-add probe results.
/// - `hosts_to_probe` is the set of hosts to send through `discover_addrs`
///   again this tick. Five classes:
///   (a) tracked hosts with no current addrs (newly joined);
///   (b) tracked hosts whose only addrs are placeholders (workers were not
///   responding when the previous probe ran) — skipped when the host
///   scrapes healthy, i.e. a real worker happens to sit on the
///   placeholder port (7500);
///   (c) tracked hosts whose real-addr count is below the max of their
///   SLURM job siblings — recovers from a partial initial probe where some
///   ranks were still loading the model;
///   (d) tracked hosts whose real-addr count is below their own historical
///   high-water — covers the "every host in the job is equally partial"
///   case that class (c) cannot detect, and the case where a host loses a
///   rank mid-job (e.g. the worker crashed and respawned on a new port);
///   (e) tracked hosts in `dead_hosts` — every endpoint has been down long
///   enough that the ports are presumed stale. Happens when a job is
///   requeued on the same host with different worker ports; without this
///   the old dead ports would be scraped forever and the host would show
///   DOWN despite a live worker.
fn dynamo_discover_reconcile(
    snapshot: &[vmon_core::slurm::SlurmJobInfo],
    current: &[String],
    highwater: &std::collections::HashMap<String, usize>,
    healthy_hosts: &std::collections::HashSet<String>,
    dead_hosts: &std::collections::HashSet<String>,
) -> (Vec<String>, Vec<String>) {
    use vmon_core::dynamo_discover::is_placeholder_addr;

    let tracked_hosts: std::collections::HashSet<String> =
        snapshot.iter().flat_map(|j| j.nodes.iter().cloned()).collect();

    let mut by_host: std::collections::HashMap<String, Vec<String>> =
        std::collections::HashMap::new();
    for addr in current {
        let h = addr.split(':').next().unwrap_or(addr).to_string();
        by_host.entry(h).or_default().push(addr.clone());
    }
    // A placeholder-port addr on a healthy host is a real endpoint (the
    // worker bound port 7500), not a placeholder.
    let is_ph = |a: &str| -> bool {
        let h = a.split(':').next().unwrap_or(a);
        is_placeholder_addr(a) && !healthy_hosts.contains(h)
    };
    let real_count = |h: &str| -> usize {
        by_host.get(h).map(|a| a.iter().filter(|s| !is_ph(s)).count()).unwrap_or(0)
    };

    let mut hosts_to_probe: Vec<String> = Vec::new();
    let mut probe_set: std::collections::HashSet<String> = std::collections::HashSet::new();
    let mut queue = |h: String, list: &mut Vec<String>| {
        if probe_set.insert(h.clone()) {
            list.push(h);
        }
    };

    // (a) hosts in slurm but not in current
    for h in &tracked_hosts {
        if !by_host.contains_key(h) {
            queue(h.clone(), &mut hosts_to_probe);
        }
    }
    // (b) hosts tracked but only via placeholder
    for (h, addrs) in &by_host {
        if tracked_hosts.contains(h) && !addrs.is_empty() && addrs.iter().all(|a| is_ph(a)) {
            queue(h.clone(), &mut hosts_to_probe);
        }
    }
    // (c) hosts below their job's sibling max real-port count
    for job in snapshot {
        let job_max = job.nodes.iter().map(|h| real_count(h)).max().unwrap_or(0);
        if job_max == 0 {
            continue;
        }
        for h in &job.nodes {
            let cnt = real_count(h);
            if cnt > 0 && cnt < job_max {
                tracing::info!(
                    host = %h,
                    have = cnt,
                    job_max,
                    "Dynamo follow: re-probing host with partial discovery (vs siblings)"
                );
                queue(h.clone(), &mut hosts_to_probe);
            }
        }
    }
    // (d) hosts below their own historical high-water
    for h in &tracked_hosts {
        let cnt = real_count(h);
        let hw = highwater.get(h).copied().unwrap_or(0);
        if cnt > 0 && cnt < hw {
            tracing::info!(
                host = %h,
                have = cnt,
                highwater = hw,
                "Dynamo follow: re-probing host below high-water"
            );
            queue(h.clone(), &mut hosts_to_probe);
        }
    }
    // (e) hosts whose every endpoint has been dead for several ticks —
    // presume the tracked ports are stale (job requeued with a shifted
    // port layout) and rediscover from scratch.
    for h in dead_hosts {
        if tracked_hosts.contains(h) && by_host.contains_key(h) {
            tracing::info!(
                host = %h,
                addrs = ?by_host.get(h),
                "Dynamo follow: all endpoints dead, dropping and re-probing host"
            );
            queue(h.clone(), &mut hosts_to_probe);
        }
    }

    let kept: Vec<String> = current
        .iter()
        .filter(|a| {
            let h = a.split(':').next().unwrap_or(a);
            tracked_hosts.contains(h) && !is_ph(a) && !dead_hosts.contains(h)
        })
        .cloned()
        .collect();

    (kept, hosts_to_probe)
}

/// Fold a settled addr list into the high-water map: bump each host's mark to
/// the new real-port count, and drop entries for hosts no longer in
/// `tracked_hosts` so a torn-down job doesn't keep its mark forever (which
/// would cause spurious re-probes if the host later rejoins a smaller job).
fn update_highwater(
    highwater: &mut std::collections::HashMap<String, usize>,
    settled: &[String],
    tracked_hosts: &std::collections::HashSet<String>,
) {
    use vmon_core::dynamo_discover::is_placeholder_addr;
    let mut counts: std::collections::HashMap<String, usize> = std::collections::HashMap::new();
    for addr in settled {
        if is_placeholder_addr(addr) {
            continue;
        }
        let h = addr.split(':').next().unwrap_or(addr).to_string();
        *counts.entry(h).or_insert(0) += 1;
    }
    for (h, cnt) in counts {
        let entry = highwater.entry(h).or_insert(0);
        if cnt > *entry {
            *entry = cnt;
        }
    }
    highwater.retain(|h, _| tracked_hosts.contains(h));
}

/// Spawn a background task that polls `squeue` on `poll_interval`, merges the
/// result into `jobs_handle`, and rewrites `nodes_handle` to match the new
/// alive set. End-of-job rows survive `grace` after leaving squeue so the
/// TUI shows `(ended)` for one cycle before the row disappears.
/// Sort scrape addresses so rows group by SLURM job in `jobs` order (newest job
/// first, since `jobs` is sorted newest-first) and stay stable within a job by
/// (host, numeric port). Addresses whose host isn't owned by any job sort last,
/// ordered by (host, port). With 0 or 1 jobs this degrades to the old
/// host-then-port sort, so single-task row order is unchanged.
fn sort_addrs_by_job(addrs: &mut [String], jobs: &[vmon_core::slurm::SlurmJobInfo]) {
    use std::collections::HashMap;
    // host -> rank (index in the newest-first `jobs` slice). When a host shows
    // up in more than one job, `or_insert` keeps the earliest (newest) job.
    let mut rank: HashMap<&str, usize> = HashMap::new();
    for (i, job) in jobs.iter().enumerate() {
        for h in &job.nodes {
            rank.entry(h.as_str()).or_insert(i);
        }
    }
    addrs.sort_by(|a, b| {
        let (ha, pa) = split_host_port(a);
        let (hb, pb) = split_host_port(b);
        let ra = rank.get(ha).copied().unwrap_or(usize::MAX);
        let rb = rank.get(hb).copied().unwrap_or(usize::MAX);
        ra.cmp(&rb).then(ha.cmp(hb)).then(pa.cmp(&pb))
    });
}

/// Consecutive follow ticks (15s each) a host must scrape fully unhealthy
/// before its tracked ports are presumed stale and rediscovered — long
/// enough to ride out a transient scrape blip, short enough that a requeued
/// job's shifted ports are picked up within a minute.
const DEAD_HOST_TICKS: u32 = 2;

#[allow(clippy::too_many_arguments)] // internal spawn helper, args are all distinct handles/config
fn spawn_slurm_follow_task(
    jobs_handle: vmon_core::scraper::SharedSlurmJobs,
    nodes_handle: vmon_core::scraper::SharedNodes,
    state_rx: tokio::sync::watch::Receiver<vmon_core::cluster::ClusterState>,
    filter: SlurmFollowFilter,
    default_port: u16,
    dynamo: Option<DynamoFollowMode>,
    grace: Duration,
    poll_interval: Duration,
) {
    tokio::spawn(async move {
        let mut ticker = tokio::time::interval(poll_interval);
        ticker.set_missed_tick_behavior(tokio::time::MissedTickBehavior::Delay);
        // Per-host high-water mark for Dynamo discover mode. Carried across
        // ticks so we can detect "this host used to have 4 ranks, now only
        // shows 1" — both the initial-partial case (every sibling equally
        // partial, so class (c) can't fire) and the mid-job regression case
        // (a worker crashed and respawned on a different port).
        let mut highwater: std::collections::HashMap<String, usize> =
            std::collections::HashMap::new();
        // Consecutive fully-unhealthy tick count per host, for reconcile
        // class (e) — stale-port detection.
        let mut dead_ticks: std::collections::HashMap<String, u32> =
            std::collections::HashMap::new();
        // Don't skip the immediate first tick: startup-time `addrs` can
        // include stale hosts (e.g. SLURM_JOB_NODELIST left over from a
        // prior allocation), and the user-filtered `query_slurm_jobs`
        // call can see jobs the host-scoped seed missed. Reconciling on
        // tick 0 closes the window where the TUI scrapes hosts that
        // squeue doesn't actually own.
        loop {
            ticker.tick().await;
            // If squeue failed (binary missing / transient error), skip the
            // merge entirely — better to keep the previous snapshot than to
            // pass an empty Vec and have merge_slurm_jobs mark every tracked
            // job as ended (which would then clear nodes_handle).
            let Some(fresh) = query_slurm_jobs(&filter).await else {
                continue;
            };

            // Take a snapshot for the desired-addr computation while holding
            // the jobs lock briefly.
            let snapshot = {
                let mut jobs = jobs_handle.lock().expect("slurm_jobs mutex poisoned");
                let now = std::time::SystemTime::now();
                let newly_ended = merge_slurm_jobs(&mut jobs, fresh, now, grace);
                for id in &newly_ended {
                    tracing::info!(job_id = %id, "SLURM job left squeue, freezing uptime");
                }
                jobs.clone()
            };

            // Recompute the desired scrape address list from the surviving
            // (non-ended) jobs.
            let desired: Vec<String> = match &dynamo {
                None => desired_addrs_vllm(&snapshot, default_port),
                Some(DynamoFollowMode::Explicit {
                    base_port,
                    ranks_per_host,
                }) => desired_addrs_dynamo_explicit(&snapshot, *base_port, *ranks_per_host),
                Some(DynamoFollowMode::Discover) => {
                    let current = nodes_handle.lock().expect("nodes mutex poisoned").clone();
                    // Scrape-health context from the latest ClusterState: in
                    // dynamo mode nodes are folded one row per host (addr =
                    // bare hostname), healthy when any rank scraped OK.
                    let (healthy_hosts, unhealthy_hosts) = {
                        let st = state_rx.borrow();
                        let mut healthy = std::collections::HashSet::new();
                        let mut unhealthy = std::collections::HashSet::new();
                        for n in &st.nodes {
                            let h = n.addr.split(':').next().unwrap_or(&n.addr).to_string();
                            if n.is_healthy {
                                healthy.insert(h);
                            } else {
                                unhealthy.insert(h);
                            }
                        }
                        (healthy, unhealthy)
                    };
                    // Bump the dead-tick counter for hosts that are present
                    // in the state and fully unhealthy; reset everyone else
                    // (healthy, or not scraped yet — e.g. just added).
                    dead_ticks.retain(|h, _| unhealthy_hosts.contains(h));
                    for h in &unhealthy_hosts {
                        if !healthy_hosts.contains(h) {
                            *dead_ticks.entry(h.clone()).or_insert(0) += 1;
                        }
                    }
                    let dead_hosts: std::collections::HashSet<String> = dead_ticks
                        .iter()
                        .filter(|(_, c)| **c >= DEAD_HOST_TICKS)
                        .map(|(h, _)| h.clone())
                        .collect();
                    let (mut next, hosts_to_probe) = dynamo_discover_reconcile(
                        &snapshot,
                        &current,
                        &highwater,
                        &healthy_hosts,
                        &dead_hosts,
                    );
                    if !hosts_to_probe.is_empty() {
                        let probed =
                            vmon_core::dynamo_discover::discover_addrs(&hosts_to_probe).await;
                        for addr in probed {
                            if !next.contains(&addr) {
                                next.push(addr);
                            }
                        }
                    }
                    let tracked_hosts: std::collections::HashSet<String> =
                        snapshot.iter().flat_map(|j| j.nodes.iter().cloned()).collect();
                    update_highwater(&mut highwater, &next, &tracked_hosts);
                    next
                }
            };

            // Group rows by SLURM job (newest job first, since `snapshot` is
            // sorted newest-first) and order stably by (host, port) within each
            // job. This keeps multi-task monitoring sorted by task with the
            // newest task on top, while still pinning the order across ticks —
            // without it the discover branch builds addrs from a HashSet
            // (random iteration order) and rows would shuffle every refresh.
            let mut desired = desired;
            sort_addrs_by_job(&mut desired, &snapshot);

            // Write the new address list. Only swap if changed to avoid
            // needless lock/clone churn.
            {
                let mut current = nodes_handle.lock().expect("nodes mutex poisoned");
                if *current != desired {
                    tracing::info!(
                        before = current.len(),
                        after = desired.len(),
                        "follow: scrape addr set updated"
                    );
                    *current = desired;
                }
            }
        }
    });
}

/// Parse SLURM `%M` elapsed time string into seconds.
///
/// Formats: `SS`, `MM:SS`, `HH:MM:SS`, `D-HH:MM:SS`.
fn parse_slurm_elapsed(s: &str) -> Option<u64> {
    let (days, rest) = match s.split_once('-') {
        Some((d, r)) => (d.parse::<u64>().ok()?, r),
        None => (0u64, s),
    };
    let parts: Vec<&str> = rest.split(':').collect();
    let parse_u64 = |s: &str| s.parse::<u64>().ok();
    let (h, m, sec) = match parts.as_slice() {
        [s] => (0u64, 0u64, parse_u64(s)?),
        [m, s] => (0u64, parse_u64(m)?, parse_u64(s)?),
        [h, m, s] => (parse_u64(h)?, parse_u64(m)?, parse_u64(s)?),
        _ => return None,
    };
    Some(days * 86_400 + h * 3_600 + m * 60 + sec)
}

#[cfg(test)]
mod slurm_elapsed_tests {
    use super::parse_slurm_elapsed;

    #[test]
    fn parses_seconds_only() {
        assert_eq!(parse_slurm_elapsed("45"), Some(45));
    }

    #[test]
    fn parses_minutes_seconds() {
        assert_eq!(parse_slurm_elapsed("2:23"), Some(143));
    }

    #[test]
    fn parses_hours_minutes_seconds() {
        assert_eq!(parse_slurm_elapsed("3:12:05"), Some(3 * 3600 + 12 * 60 + 5));
    }

    #[test]
    fn parses_days_hours_minutes_seconds() {
        assert_eq!(
            parse_slurm_elapsed("2-07:30:00"),
            Some(2 * 86_400 + 7 * 3600 + 30 * 60)
        );
    }

    #[test]
    fn rejects_garbage() {
        assert_eq!(parse_slurm_elapsed("infinity"), None);
        assert_eq!(parse_slurm_elapsed(""), None);
    }
}

fn build_cluster_from_sample(
    sample: &vmon_core::sample::TimeSample,
    node_order: &[String],
) -> vmon_core::cluster::ClusterState {
    let mut nodes: Vec<vmon_core::node::NodeMetrics> = sample
        .nodes
        .iter()
        .map(|(addr, s)| vmon_core::node::NodeMetrics::from_sample(addr.clone(), s))
        .collect();
    // `sample.nodes` is a HashMap whose iteration order is randomized per
    // instance, so each tick would otherwise reorder the rows and they would
    // jump around. Sort by the original capture order (`meta.nodes`), falling
    // back to address for any node not listed, to keep rows stable.
    let rank = |addr: &str| node_order.iter().position(|a| a == addr);
    nodes.sort_by(|a, b| match (rank(&a.addr), rank(&b.addr)) {
        (Some(x), Some(y)) => x.cmp(&y),
        (Some(_), None) => std::cmp::Ordering::Less,
        (None, Some(_)) => std::cmp::Ordering::Greater,
        (None, None) => a.addr.cmp(&b.addr),
    });
    let mut cluster = vmon_core::cluster::ClusterState::aggregate(nodes, HashMap::new());
    // Live captures store already-resolved badges, but recordings from older
    // versions may carry raw "backend" roles — resolve them (one cluster-wide
    // scope, since replays have no SLURM job info).
    cluster.resolve_backend_roles();
    // Prefer the full store list (multi-job recordings); fall back to the
    // backward-compatible scalar written by single-store recordings.
    let mooncakes: Vec<vmon_core::mooncake::MooncakeMetrics> = if !sample.mooncakes.is_empty() {
        sample
            .mooncakes
            .iter()
            .map(vmon_core::mooncake::MooncakeMetrics::from)
            .collect()
    } else {
        sample
            .mooncake
            .as_ref()
            .map(vmon_core::mooncake::MooncakeMetrics::from)
            .into_iter()
            .collect()
    };
    cluster.mooncake_addrs = mooncakes.iter().map(|m| m.addr.clone()).collect();
    cluster.mooncakes = mooncakes;
    cluster
}

/// Fill no-traffic latency gaps during replay by carrying forward each node's
/// last non-empty windowed latency. Updates `mem` with any group that has data,
/// and substitutes the remembered value for any group that's currently blank.
/// Cumulative is mirrored from windowed since replays only store windowed.
fn carry_forward_latency(
    node: &mut vmon_core::node::NodeMetrics,
    mem: &mut std::collections::HashMap<String, [vmon_core::node::LatencyPct; 8]>,
) {
    use vmon_core::node::LatencyPct;
    let has = |p: &LatencyPct| p.p50 > 0.0 || p.p90 > 0.0 || p.p99 > 0.0 || p.mean > 0.0;
    let slot = mem.entry(node.addr.clone()).or_default();
    let groups: [&mut LatencyPct; 8] = [
        &mut node.win_ttft,
        &mut node.win_itl,
        &mut node.win_e2e,
        &mut node.win_queue,
        &mut node.win_prefill,
        &mut node.win_decode,
        &mut node.win_inference,
        &mut node.win_tpot,
    ];
    for (i, g) in groups.into_iter().enumerate() {
        if has(g) {
            slot[i] = g.clone();
        } else if has(&slot[i]) {
            *g = slot[i].clone();
        }
    }
    node.cum_ttft = node.win_ttft.clone();
    node.cum_itl = node.win_itl.clone();
    node.cum_e2e = node.win_e2e.clone();
    node.cum_queue = node.win_queue.clone();
    node.cum_prefill = node.win_prefill.clone();
    node.cum_decode = node.win_decode.clone();
    node.cum_inference = node.win_inference.clone();
    node.cum_tpot = node.win_tpot.clone();
}

async fn run_replay_task(
    samples: Vec<vmon_core::sample::TimeSample>,
    node_order: Vec<String>,
    tx: tokio::sync::watch::Sender<vmon_core::cluster::ClusterState>,
    state: std::sync::Arc<vmon_tui::ui::ReplayState>,
) {
    use std::sync::atomic::Ordering::Relaxed;

    let mut idx = 0usize;
    let total = samples.len();

    // Per-node last-known latency, used to fill no-traffic intervals. Recordings
    // store *windowed* latency (delta-histogram percentiles over one scrape
    // interval), so an interval with no completed requests carries all-zero
    // latency. The detail panel shows these as cumulative, so without
    // carry-forward the values flicker between present and blank.
    let mut lat_mem: std::collections::HashMap<String, [vmon_core::node::LatencyPct; 8]> =
        std::collections::HashMap::new();

    while idx < total {
        // Update state
        state.current_sample.store(idx, Relaxed);
        state.set_elapsed(samples[idx].elapsed_secs);

        // Send current sample to TUI
        let mut cluster = build_cluster_from_sample(&samples[idx], &node_order);
        for node in &mut cluster.nodes {
            carry_forward_latency(node, &mut lat_mem);
        }
        if tx.send(cluster).is_err() {
            break; // TUI closed
        }

        // Wait for the appropriate interval
        if idx + 1 < total {
            let dt = samples[idx + 1].elapsed_secs - samples[idx].elapsed_secs;

            // Wait loop: check pause/speed every 50ms
            let mut remaining = dt;
            while remaining > 0.0 {
                if state.paused.load(Relaxed) {
                    // When paused, sleep and check if unpaused
                    tokio::time::sleep(Duration::from_millis(50)).await;

                    // Check for step-forward (sample index changed externally)
                    // We detect this by checking if the sample counter has been
                    // set to something different than our current idx
                    continue;
                }

                let speed = state.speed.load(Relaxed) as f64 / 10.0;
                let sleep_ms = (remaining * 1000.0 / speed).min(50.0);
                // Floor at 1 ms: at high speed (e.g. 10x) with small remaining,
                // (sleep_ms as u64) rounds to 0 and the loop becomes a busy
                // spin that burns CPU and starves the event loop.
                let sleep_u = (sleep_ms as u64).max(1);
                tokio::time::sleep(Duration::from_millis(sleep_u)).await;
                remaining -= sleep_u as f64 * speed / 1000.0;
            }
        } else {
            // Last sample: pause at end
            state.paused.store(true, Relaxed);
            state.current_sample.store(total - 1, Relaxed);

            // Wait until TUI closes
            loop {
                tokio::time::sleep(Duration::from_millis(100)).await;
                if tx.is_closed() {
                    break;
                }
            }
            break;
        }

        idx += 1;
    }
}

fn raw_capture_targets(nodes: &[String]) -> Vec<RawCaptureTarget> {
    nodes.iter().cloned().map(|node| RawCaptureTarget { node }).collect()
}

fn resolve_capture_path(path: &Path, cwd: &Path) -> PathBuf {
    if path.is_absolute() {
        path.to_path_buf()
    } else {
        cwd.join(path)
    }
}

fn load_raw_capture_config(path: &Path) -> Result<Vec<CompiledRawCaptureRule>, String> {
    let content = std::fs::read_to_string(path)
        .map_err(|e| format!("Failed to read raw capture config {}: {e}", path.display()))?;
    let cwd = std::env::current_dir().map_err(|e| format!("Failed to resolve cwd: {e}"))?;
    parse_raw_capture_config_str(&content, &cwd)
}

fn parse_raw_capture_config_str(
    content: &str,
    cwd: &Path,
) -> Result<Vec<CompiledRawCaptureRule>, String> {
    let config: RawCaptureConfig =
        toml::from_str(content).map_err(|e| format!("Invalid raw capture config: {e}"))?;
    if config.capture.is_empty() {
        return Err("Raw capture config must define at least one [[capture]] rule".to_string());
    }

    let mut rules = Vec::with_capacity(config.capture.len());
    let mut names = HashSet::new();
    let mut outputs = HashSet::new();

    for rule in config.capture {
        let name = rule.name.trim().to_string();
        if name.is_empty() {
            return Err("Raw capture rule name cannot be empty".to_string());
        }
        if !names.insert(name.clone()) {
            return Err(format!("Duplicate raw capture rule name: {name}"));
        }
        if rule.patterns.is_empty() {
            return Err(format!(
                "Raw capture rule {name} must define at least one pattern"
            ));
        }

        let output = resolve_capture_path(&rule.output, cwd);
        if !outputs.insert(output.clone()) {
            return Err(format!(
                "Duplicate raw capture output path: {}",
                output.display()
            ));
        }

        let matcher = match rule.mode {
            RawMatchMode::Prefix => {
                let mut prefixes = Vec::with_capacity(rule.patterns.len());
                for pattern in rule.patterns {
                    if pattern.is_empty() {
                        return Err(format!(
                            "Raw capture rule {name} contains an empty prefix pattern"
                        ));
                    }
                    prefixes.push(pattern);
                }
                RawMatcher::Prefix(prefixes)
            }
            RawMatchMode::Regex => {
                let mut regexes = Vec::with_capacity(rule.patterns.len());
                for pattern in rule.patterns {
                    let regex = Regex::new(&pattern).map_err(|e| {
                        format!("Invalid regex pattern for raw capture rule {name}: {e}")
                    })?;
                    regexes.push(regex);
                }
                RawMatcher::Regex(regexes)
            }
        };

        rules.push(CompiledRawCaptureRule {
            name,
            output,
            include_help: rule.include_help,
            include_type: rule.include_type,
            matcher,
        });
    }

    Ok(rules)
}

fn validate_raw_capture_outputs(
    rules: &[CompiledRawCaptureRule],
    report_output: &Path,
) -> Result<(), String> {
    let cwd = std::env::current_dir().map_err(|e| format!("Failed to resolve cwd: {e}"))?;
    let report_output = resolve_capture_path(report_output, &cwd);
    for rule in rules {
        if rule.output == report_output {
            return Err(format!(
                "Raw capture output {} must be different from the main --output file",
                rule.output.display()
            ));
        }
    }
    Ok(())
}

fn raw_matcher_matches(matcher: &RawMatcher, metric_name: &str) -> bool {
    match matcher {
        RawMatcher::Prefix(prefixes) => {
            prefixes.iter().any(|prefix| metric_name.starts_with(prefix))
        }
        RawMatcher::Regex(regexes) => regexes.iter().any(|regex| regex.is_match(metric_name)),
    }
}

fn rule_matches_family(rule: &CompiledRawCaptureRule, family: &MetricFamily) -> bool {
    raw_matcher_matches(&rule.matcher, &family.name)
        || (family.metric_type == MetricType::Counter
            && raw_matcher_matches(&rule.matcher, &format!("{}_total", family.name)))
}

fn select_matching_families<'a>(
    families: &'a [MetricFamily],
    rule: &CompiledRawCaptureRule,
) -> Vec<&'a MetricFamily> {
    families.iter().filter(|family| rule_matches_family(rule, family)).collect()
}

fn render_labels(sample: &Sample) -> String {
    if sample.labels.is_empty() {
        return String::new();
    }

    let labels = sample
        .labels
        .iter()
        .map(|(key, value)| format!(r#"{key}="{}""#, escape_label_value(value)))
        .collect::<Vec<_>>()
        .join(",");
    format!("{{{labels}}}")
}

fn escape_label_value(value: &str) -> String {
    let mut out = String::with_capacity(value.len());
    for ch in value.chars() {
        match ch {
            '\\' => out.push_str(r#"\\"#),
            '"' => out.push_str(r#"\""#),
            '\n' => out.push_str(r#"\n"#),
            _ => out.push(ch),
        }
    }
    out
}

fn render_metric_family_header_name(family: &MetricFamily) -> String {
    if family.metric_type == MetricType::Counter {
        format!("{}_total", family.name)
    } else {
        family.name.clone()
    }
}

fn render_metric_families(
    families: &[&MetricFamily],
    include_help: bool,
    include_type: bool,
) -> String {
    let mut lines = Vec::new();
    for family in families {
        let header_name = render_metric_family_header_name(family);
        if include_help && !family.help.is_empty() {
            lines.push(format!("# HELP {header_name} {}", family.help));
        }
        if include_type {
            lines.push(format!("# TYPE {header_name} {}", family.metric_type));
        }
        for sample in &family.samples {
            lines.push(format!(
                "{}{} {}",
                sample.name,
                render_labels(sample),
                sample.value
            ));
        }
    }
    lines.join("\n")
}

fn raw_capture_entry(
    unix_secs: u64,
    node: &str,
    capture: &str,
    metrics: &str,
) -> serde_json::Value {
    serde_json::json!({
        "unix_secs": unix_secs,
        "node": node,
        "capture": capture,
        "metrics": metrics,
    })
}

async fn collect_raw_captures(
    targets: Vec<RawCaptureTarget>,
    rules: Vec<CompiledRawCaptureRule>,
    duration: Duration,
    interval: Duration,
    mut cancel_rx: tokio::sync::watch::Receiver<bool>,
) -> std::io::Result<Vec<RawCaptureSummary>> {
    let client = reqwest::Client::builder()
        .timeout(Duration::from_secs(15))
        .build()
        .expect("failed to build HTTP client");
    let mut sinks = Vec::with_capacity(rules.len());
    for rule in rules {
        let file = tokio::fs::File::create(&rule.output).await?;
        sinks.push(RawCaptureSink {
            rule,
            file,
            written: 0,
        });
    }

    let start = Instant::now();
    let deadline = start + duration;
    let mut tick = tokio::time::interval(interval);

    loop {
        tokio::select! {
            _ = tick.tick() => {}
            changed = cancel_rx.changed() => {
                if changed.is_ok() && *cancel_rx.borrow() {
                    break;
                }
            }
        }

        let now = Instant::now();
        let unix_secs = SystemTime::now().duration_since(UNIX_EPOCH).unwrap_or_default().as_secs();
        if now >= deadline {
            break;
        }

        let mut scrape_tasks = tokio::task::JoinSet::new();
        for target in &targets {
            let client = client.clone();
            let target = target.clone();
            scrape_tasks.spawn(async move {
                let url = format!("http://{}/metrics", target.node);
                let body = vmon_core::http::text(client.get(url).send().await.ok()?).await.ok()?;
                Some((target, body))
            });
        }

        while let Some(result) = scrape_tasks.join_next().await {
            let Ok(Some((target, body))) = result else {
                continue;
            };
            let families = match parse_prometheus_text(&body) {
                Ok(families) => families,
                Err(e) => {
                    tracing::warn!(node = %target.node, %e, "raw capture parse failed");
                    continue;
                }
            };

            for sink in &mut sinks {
                let matching = select_matching_families(&families, &sink.rule);
                if matching.is_empty() {
                    continue;
                }

                let metrics = render_metric_families(
                    &matching,
                    sink.rule.include_help,
                    sink.rule.include_type,
                );
                if metrics.is_empty() {
                    continue;
                }

                let entry = raw_capture_entry(unix_secs, &target.node, &sink.rule.name, &metrics);
                sink.file.write_all(entry.to_string().as_bytes()).await?;
                sink.file.write_all(b"\n").await?;
                sink.written += 1;
            }
        }
    }

    let mut summaries = Vec::with_capacity(sinks.len());
    for mut sink in sinks {
        sink.file.flush().await?;
        summaries.push(RawCaptureSummary {
            name: sink.rule.name,
            output: sink.rule.output,
            written: sink.written,
        });
    }

    Ok(summaries)
}

fn init_tracing_stderr() {
    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::from_default_env()
                .add_directive("vmon=info".parse().unwrap()),
        )
        .with_target(false)
        .init();
}

fn init_tracing_file() {
    let log_file = std::fs::OpenOptions::new()
        .create(true)
        .append(true)
        .open("vmon.log")
        .expect("failed to open vmon.log");
    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::from_default_env()
                .add_directive("vmon=info".parse().unwrap()),
        )
        .with_target(false)
        .with_writer(std::sync::Mutex::new(log_file))
        .with_ansi(false)
        .init();
}

#[tokio::main]
async fn main() {
    let cli = Cli::parse();

    match cli.command {
        Command::Watch {
            nodes,
            slurm_job,
            slurm_job_name,
            port,
            interval,
            zmq_port,
            zmq_topic,
            zmq_dp_ranks,
            gpu_port,
            ib_port,
            mooncake,
            dynamo,
            dynamo_base_port,
            dynamo_ranks_per_host,
        } => {
            init_tracing_file();
            // Follow mode = the user gave no positional nodes. The CLI then
            // tracks SLURM jobs continuously (poll squeue, add new jobs' nodes,
            // drop jobs after a grace period) instead of a one-shot snapshot.
            let follow_mode = nodes.is_empty();
            let resolved = resolve_nodes(nodes, slurm_job.clone(), slurm_job_name.clone());
            let expanded = expand_nodes(&resolved);
            let slurm_jobs = discover_slurm_jobs_for_hosts(&expanded);
            for j in &slurm_jobs {
                eprintln!(
                    "Slurm: matched job {} ({}) on {}",
                    j.job_id, j.job_name, j.nodelist_compact
                );
            }
            let addrs = if dynamo {
                match dynamo_base_port {
                    Some(base) => {
                        let ranks = detect_ranks_per_host(dynamo_ranks_per_host);
                        dynamo_expand_grouped(&expanded, &slurm_jobs, base, ranks)
                    }
                    None => vmon_core::dynamo_discover::discover_addrs(&expanded).await,
                }
            } else {
                normalize_addrs(&expanded, port)
            };
            let mut scraper = vmon_core::scraper::Scraper::new(addrs, interval);
            if let Some(zport) = zmq_port {
                scraper = scraper.with_zmq(zport, zmq_topic.clone(), zmq_dp_ranks);
            }
            if gpu_port > 0 {
                scraper = scraper.with_gpu(gpu_port);
            }
            if ib_port > 0 {
                scraper = scraper.with_ib(ib_port);
            }
            scraper = apply_mooncake_flag(scraper, mooncake);
            if dynamo {
                scraper = scraper.with_dynamo();
            }
            let jobs_handle: vmon_core::scraper::SharedSlurmJobs =
                std::sync::Arc::new(std::sync::Mutex::new(slurm_jobs));
            scraper = scraper.with_slurm_jobs_handle(jobs_handle.clone());
            // Cluster-wide sinfo totals for the header (total / idle nodes).
            // Harmless outside SLURM: the poll silently no-ops and the TUI
            // omits the segment.
            let cluster_stats_handle: vmon_core::scraper::SharedSlurmCluster =
                std::sync::Arc::new(std::sync::Mutex::new(None));
            scraper = scraper.with_slurm_cluster_handle(cluster_stats_handle.clone());
            spawn_slurm_cluster_stats_task(cluster_stats_handle);
            let nodes_handle = scraper.nodes_handle();
            // Start the scrape loop before the follow task: the follow task
            // reads the ClusterState watch channel for scrape-health context
            // (stale-port detection needs to know which hosts are down).
            let rx = scraper.run_loop().await;

            if follow_mode {
                let filter = if !slurm_job.is_empty() {
                    SlurmFollowFilter::JobIds(
                        slurm_job
                            .iter()
                            .map(|s| {
                                s.rsplit_once(':')
                                    .filter(|(_, p)| p.parse::<u16>().is_ok())
                                    .map(|(id, _)| id.to_string())
                                    .unwrap_or_else(|| s.clone())
                            })
                            .collect(),
                    )
                } else if let Some(name) = slurm_job_name.clone() {
                    SlurmFollowFilter::JobName(name)
                } else {
                    SlurmFollowFilter::User
                };
                let dynamo_mode = if dynamo {
                    Some(match dynamo_base_port {
                        Some(base) => DynamoFollowMode::Explicit {
                            base_port: base,
                            ranks_per_host: detect_ranks_per_host(dynamo_ranks_per_host),
                        },
                        None => DynamoFollowMode::Discover,
                    })
                } else {
                    None
                };
                eprintln!("Slurm: follow mode (poll every 15s, drop ended after 60s grace)");
                spawn_slurm_follow_task(
                    jobs_handle,
                    nodes_handle,
                    rx.clone(),
                    filter,
                    port,
                    dynamo_mode,
                    Duration::from_secs(60),
                    Duration::from_secs(15),
                );
            } else {
                // Static mode: keep the existing freeze-only refresh task so
                // jobs that end mid-session render `(ended)` without the row
                // disappearing.
                let has_jobs = !jobs_handle.lock().expect("slurm_jobs mutex poisoned").is_empty();
                if has_jobs {
                    spawn_slurm_refresh_task(jobs_handle);
                }
            }

            if let Err(e) = vmon_tui::app::run(rx, interval).await {
                eprintln!("TUI error: {e}");
                std::process::exit(1);
            }
        }
        Command::Collect {
            nodes,
            slurm_job,
            slurm_job_name,
            duration,
            interval,
            port,
            output,
            gpu_port,
            ib_port,
            mooncake,
            metrics,
            list_metrics,
            dynamo,
            dynamo_base_port,
            dynamo_ranks_per_host,
            raw_capture_config,
            max_samples,
            flush_interval,
        } => {
            if list_metrics {
                for name in vmon_report::collector::METRIC_NAMES {
                    println!("{name}");
                }
                return;
            }
            init_tracing_stderr();
            let resolved = resolve_nodes(nodes, slurm_job, slurm_job_name);
            let expanded = expand_nodes(&resolved);
            let addrs = if dynamo {
                let slurm_jobs = discover_slurm_jobs_for_hosts(&expanded);
                match dynamo_base_port {
                    Some(base) => {
                        let ranks = detect_ranks_per_host(dynamo_ranks_per_host);
                        dynamo_expand_grouped(&expanded, &slurm_jobs, base, ranks)
                    }
                    None => vmon_core::dynamo_discover::discover_addrs(&expanded).await,
                }
            } else {
                normalize_addrs(&expanded, port)
            };
            let raw_rules = raw_capture_config.as_ref().map(|path| {
                let rules = load_raw_capture_config(path).unwrap_or_else(|e| {
                    eprintln!("{e}");
                    std::process::exit(1);
                });
                validate_raw_capture_outputs(&rules, &output).unwrap_or_else(|e| {
                    eprintln!("{e}");
                    std::process::exit(1);
                });
                rules
            });
            let collector = vmon_report::collector::TimeSeriesCollector::new(addrs.clone())
                .with_max_samples(max_samples);
            let mut scraper = vmon_core::scraper::Scraper::new(addrs, interval);
            if gpu_port > 0 {
                scraper = scraper.with_gpu(gpu_port);
            }
            if ib_port > 0 {
                scraper = scraper.with_ib(ib_port);
            }
            let collect_slurm_jobs = discover_slurm_jobs_for_hosts(&expanded);
            if matches!(mooncake, Some(None)) {
                // Auto targets derive from the SLURM job list; collect has no
                // follow task updating it, so attach a startup snapshot.
                scraper = scraper.with_slurm_jobs(collect_slurm_jobs.clone());
            }
            scraper = apply_mooncake_flag(scraper, mooncake);
            if dynamo {
                scraper = scraper.with_dynamo();
            }
            let raw_targets = raw_capture_targets(&collector.node_addrs);
            let rx = scraper.run_loop().await;

            eprintln!(
                "vmon collect: {} nodes, duration {:?}, interval {:?}",
                expanded.len(),
                duration,
                interval
            );

            let (raw_cancel_tx, raw_cancel_rx) = tokio::sync::watch::channel(false);
            let raw_handle = raw_rules.clone().map(|rules| {
                tokio::spawn(async move {
                    collect_raw_captures(raw_targets, rules, duration, interval, raw_cancel_rx)
                        .await
                })
            });

            // Best-effort periodic checkpoint: re-render and atomically write the
            // report mid-run so a hard kill still leaves the latest data on disk.
            // Owns clones of output/metrics so the originals remain usable for the
            // authoritative final write below.
            let checkpoint_output = output.clone();
            let checkpoint_metrics = metrics.clone();
            let checkpoint = move |c: &vmon_report::collector::TimeSeriesCollector| {
                let content = render_report(c, &checkpoint_output, checkpoint_metrics.as_deref());
                if let Err(e) = write_report_atomic(&checkpoint_output, &content) {
                    eprintln!(
                        "\nCheckpoint write to {} failed: {e}",
                        checkpoint_output.display()
                    );
                }
            };

            let collector = collector
                .run(
                    rx,
                    duration,
                    interval,
                    Some(flush_interval),
                    |samples, elapsed, total, online, total_nodes| {
                        let pct = elapsed.as_secs_f64() / total.as_secs_f64() * 100.0;
                        let remaining = total.saturating_sub(elapsed);
                        eprint!(
                            "\rCollecting... {pct:3.0}% ({remaining:.0?} left) | {samples} samples | {online}/{total_nodes} online  "
                        );
                    },
                    checkpoint,
                )
                .await;

            let _ = raw_cancel_tx.send(true);

            if collector.dropped_samples > 0 {
                eprintln!(
                    "\nDone. {} samples retained, {} dropped by --max-samples cap.",
                    collector.samples.len(),
                    collector.dropped_samples
                );
            } else {
                eprintln!("\nDone. {} samples collected.", collector.samples.len());
            }

            let content = render_report(&collector, &output, metrics.as_deref());

            write_report_atomic(&output, &content).unwrap_or_else(|e| {
                eprintln!("Failed to write {}: {e}", output.display());
                std::process::exit(1);
            });

            eprintln!("Report written to {}", output.display());
            if let Some(handle) = raw_handle {
                match handle.await {
                    Ok(Ok(summaries)) => {
                        for summary in summaries {
                            eprintln!(
                                "Raw capture {} written to {} ({} scrape entries)",
                                summary.name,
                                summary.output.display(),
                                summary.written
                            );
                        }
                    }
                    Ok(Err(e)) => {
                        eprintln!("Failed to write raw capture output: {e}");
                        std::process::exit(1);
                    }
                    Err(e) => {
                        eprintln!("Raw capture collection task failed: {e}");
                        std::process::exit(1);
                    }
                }
            }
        }
        Command::Replay { file, speed } => {
            init_tracing_file();
            let content = std::fs::read_to_string(&file).unwrap_or_else(|e| {
                eprintln!("Failed to read {}: {e}", file.display());
                std::process::exit(1);
            });
            let replay_file: vmon_core::replay::ReplayFile = serde_json::from_str(&content)
                .unwrap_or_else(|e| {
                    eprintln!("Failed to parse JSON: {e}");
                    std::process::exit(1);
                });

            if replay_file.samples.is_empty() {
                eprintln!("No samples in report file");
                std::process::exit(1);
            }

            let total_duration = replay_file.meta.duration_secs;
            let total_samples = replay_file.samples.len();
            let replay_state = std::sync::Arc::new(vmon_tui::ui::ReplayState::new(
                total_samples,
                total_duration,
            ));

            // Set initial speed
            let speed_x10 = (speed * 10.0).round() as u32;
            replay_state.speed.store(speed_x10.max(1), std::sync::atomic::Ordering::Relaxed);

            eprintln!(
                "Replay: {} samples, {:.1}s duration, {speed:.0}x speed",
                total_samples, total_duration
            );

            // Stable row order: capture order from meta, used to sort each
            // tick's nodes (the per-sample HashMap order is randomized).
            let node_order = replay_file.meta.nodes;

            // Build initial ClusterState from first sample
            let initial = build_cluster_from_sample(&replay_file.samples[0], &node_order);
            let (tx, rx) = tokio::sync::watch::channel(initial);

            // Spawn replay task
            let rs = replay_state.clone();
            let samples = replay_file.samples;
            tokio::spawn(async move {
                run_replay_task(samples, node_order, tx, rs).await;
            });

            if let Err(e) = vmon_tui::app::run_replay(rx, replay_state).await {
                eprintln!("TUI error: {e}");
                std::process::exit(1);
            }
        }
        Command::Agent {
            bind,
            port,
            interval,
            forward,
            daemon,
            pid_file,
            log_file,
        } => {
            if daemon {
                if let Err(e) = daemonize(
                    bind,
                    port,
                    interval,
                    forward.as_deref(),
                    pid_file.as_deref(),
                    log_file.as_deref(),
                ) {
                    eprintln!("Failed to start agent daemon: {e}");
                    std::process::exit(1);
                }
                return;
            }
            init_tracing_stderr();
            run_agent(bind, port, interval, forward).await;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use clap::Parser;

    #[test]
    fn test_expand_range_basic() {
        let r = expand_range("example-[01-04]");
        assert_eq!(
            r,
            vec!["example-01", "example-02", "example-03", "example-04"]
        );
    }

    #[test]
    fn test_expand_range_no_padding() {
        let r = expand_range("node[1-3]");
        assert_eq!(r, vec!["node1", "node2", "node3"]);
    }

    #[test]
    fn test_expand_range_with_port() {
        let r = expand_range("192.0.2.[1-3]:8000");
        assert_eq!(
            r,
            vec!["192.0.2.1:8000", "192.0.2.2:8000", "192.0.2.3:8000"]
        );
    }

    #[test]
    fn test_expand_range_comma_list() {
        let r = expand_range("gpu[01-02,05,08-09]");
        assert_eq!(r, vec!["gpu01", "gpu02", "gpu05", "gpu08", "gpu09"]);
    }

    #[test]
    fn test_expand_range_no_bracket() {
        let r = expand_range("192.0.2.1");
        assert_eq!(r, vec!["192.0.2.1"]);
    }

    #[test]
    fn test_expand_nodes_mixed() {
        let nodes = vec!["example-[01-03]".to_string(), "192.0.2.5".to_string()];
        let r = expand_nodes(&nodes);
        assert_eq!(
            r,
            vec!["example-01", "example-02", "example-03", "192.0.2.5"]
        );
    }

    #[test]
    fn test_dynamo_expand_ports_single_host_four_ranks() {
        let hosts = vec!["node03".to_string()];
        let r = dynamo_expand_ports(&hosts, 8081, 4);
        assert_eq!(
            r,
            vec!["node03:8081", "node03:8082", "node03:8083", "node03:8084"]
        );
    }

    #[test]
    fn test_dynamo_expand_ports_global_increment() {
        let hosts = vec!["node03".to_string(), "node04".to_string()];
        let r = dynamo_expand_ports(&hosts, 8081, 4);
        assert_eq!(
            r,
            vec![
                "node03:8081",
                "node03:8082",
                "node03:8083",
                "node03:8084",
                "node04:8085",
                "node04:8086",
                "node04:8087",
                "node04:8088",
            ]
        );
    }

    #[test]
    fn test_dynamo_expand_ports_preserves_explicit_port() {
        let hosts = vec!["node03:9001".to_string(), "node04".to_string()];
        let r = dynamo_expand_ports(&hosts, 8081, 2);
        assert_eq!(r, vec!["node03:9001", "node04:8081", "node04:8082"]);
    }

    #[test]
    fn test_dynamo_expand_ports_zero_ranks_treated_as_one() {
        let hosts = vec!["node03".to_string()];
        let r = dynamo_expand_ports(&hosts, 8081, 0);
        assert_eq!(r, vec!["node03:8081"]);
    }

    fn test_slurm_job(job_id: &str, nodes: &[&str]) -> vmon_core::slurm::SlurmJobInfo {
        vmon_core::slurm::SlurmJobInfo {
            job_id: job_id.to_string(),
            job_name: format!("job-{job_id}"),
            start_time: std::time::SystemTime::now(),
            end_time: None,
            nodelist_compact: nodes.join(","),
            nodes: nodes.iter().map(|s| s.to_string()).collect(),
        }
    }

    #[test]
    fn test_dynamo_expand_grouped_resets_port_per_job() {
        let expanded = vec![
            "node03".to_string(),
            "node04".to_string(),
            "node09".to_string(),
            "node10".to_string(),
        ];
        let jobs = vec![
            test_slurm_job("1001", &["node03", "node04"]),
            test_slurm_job("1002", &["node09", "node10"]),
        ];
        let r = dynamo_expand_grouped(&expanded, &jobs, 8081, 4);
        assert_eq!(
            r,
            vec![
                // First job
                "node03:8081",
                "node03:8082",
                "node03:8083",
                "node03:8084",
                "node04:8085",
                "node04:8086",
                "node04:8087",
                "node04:8088",
                // Second job: its port counter restarts at 8081
                "node09:8081",
                "node09:8082",
                "node09:8083",
                "node09:8084",
                "node10:8085",
                "node10:8086",
                "node10:8087",
                "node10:8088",
            ]
        );
    }

    #[test]
    fn test_dynamo_expand_grouped_no_jobs_uses_global_counter() {
        let expanded = vec!["node03".to_string(), "node04".to_string()];
        let r = dynamo_expand_grouped(&expanded, &[], 8081, 2);
        assert_eq!(
            r,
            vec!["node03:8081", "node03:8082", "node04:8083", "node04:8084",]
        );
    }

    #[test]
    fn test_dynamo_expand_grouped_orphan_hosts_get_their_own_group() {
        let expanded = vec![
            "node03".to_string(),
            "node04".to_string(),
            "manual-host".to_string(),
        ];
        let jobs = vec![test_slurm_job("1001", &["node03", "node04"])];
        let r = dynamo_expand_grouped(&expanded, &jobs, 8081, 2);
        assert_eq!(
            r,
            vec![
                // First job
                "node03:8081",
                "node03:8082",
                "node04:8083",
                "node04:8084",
                // Orphan group resets
                "manual-host:8081",
                "manual-host:8082",
            ]
        );
    }

    #[test]
    fn test_dynamo_expand_grouped_skips_jobs_with_no_matching_hosts() {
        let expanded = vec!["node09".to_string(), "node10".to_string()];
        let jobs = vec![
            test_slurm_job("1001", &["node03", "node04"]),
            test_slurm_job("1002", &["node09", "node10"]),
        ];
        let r = dynamo_expand_grouped(&expanded, &jobs, 8081, 2);
        assert_eq!(
            r,
            vec!["node09:8081", "node09:8082", "node10:8083", "node10:8084",]
        );
    }

    #[test]
    fn test_dynamo_discover_reconcile_partial_host_gets_reprobed() {
        // node04 was discovered with just one rank (7502) while node05 found all 4.
        // The partial-discovery class (c) must queue node04 for re-probe so the
        // missing 7500/7501/7503 can be picked up.
        let jobs = vec![test_slurm_job("1003", &["node04", "node05"])];
        let current = vec![
            "node04:7502".to_string(),
            "node05:7504".to_string(),
            "node05:7505".to_string(),
            "node05:7506".to_string(),
            "node05:7507".to_string(),
        ];
        let hw = std::collections::HashMap::new();
        let (kept, to_probe) = dynamo_discover_reconcile(
            &jobs,
            &current,
            &hw,
            &Default::default(),
            &Default::default(),
        );
        assert_eq!(kept, current, "all real addrs must be kept verbatim");
        assert_eq!(to_probe, vec!["node04".to_string()]);
    }

    #[test]
    fn test_dynamo_discover_reconcile_placeholder_only_host() {
        // Class (b): host has only the discovery placeholder → re-probe.
        let jobs = vec![test_slurm_job("1003", &["node04", "node05"])];
        let current = vec![
            "node04:7500".to_string(), // PLACEHOLDER_PORT
            "node05:7504".to_string(),
        ];
        let hw = std::collections::HashMap::new();
        let (kept, to_probe) = dynamo_discover_reconcile(
            &jobs,
            &current,
            &hw,
            &Default::default(),
            &Default::default(),
        );
        assert_eq!(kept, vec!["node05:7504".to_string()]);
        assert!(to_probe.contains(&"node04".to_string()));
    }

    #[test]
    fn test_dynamo_discover_reconcile_new_host() {
        // Class (a): host appeared in slurm but no addrs yet.
        let jobs = vec![test_slurm_job("1003", &["node04", "node05"])];
        let current = vec!["node04:7501".to_string()];
        let hw = std::collections::HashMap::new();
        let (kept, to_probe) = dynamo_discover_reconcile(
            &jobs,
            &current,
            &hw,
            &Default::default(),
            &Default::default(),
        );
        assert_eq!(kept, vec!["node04:7501".to_string()]);
        assert!(to_probe.contains(&"node05".to_string()));
    }

    #[test]
    fn test_dynamo_discover_reconcile_balanced_no_reprobe() {
        // All hosts have the same port count → nothing to re-probe.
        let jobs = vec![test_slurm_job("1003", &["node04", "node05"])];
        let current = vec![
            "node04:7501".to_string(),
            "node04:7502".to_string(),
            "node05:7503".to_string(),
            "node05:7504".to_string(),
        ];
        let hw = std::collections::HashMap::new();
        let (kept, to_probe) = dynamo_discover_reconcile(
            &jobs,
            &current,
            &hw,
            &Default::default(),
            &Default::default(),
        );
        assert_eq!(kept.len(), 4);
        assert!(to_probe.is_empty());
    }

    #[test]
    fn test_dynamo_discover_reconcile_drops_untracked_host() {
        // Host no longer in any SLURM job → its addrs are dropped.
        let jobs = vec![test_slurm_job("1003", &["node05"])];
        let current = vec!["node04:7501".to_string(), "node05:7504".to_string()];
        let hw = std::collections::HashMap::new();
        let (kept, to_probe) = dynamo_discover_reconcile(
            &jobs,
            &current,
            &hw,
            &Default::default(),
            &Default::default(),
        );
        assert_eq!(kept, vec!["node05:7504".to_string()]);
        assert!(!to_probe.contains(&"node04".to_string()));
    }

    #[test]
    fn test_dynamo_discover_reconcile_below_highwater_gets_reprobed() {
        // Class (d): every host in the job is equally partial (1 port each),
        // so the sibling-max check can't help — but high-water knows each
        // host had 4 ranks earlier and queues all of them for re-probe.
        let jobs = vec![test_slurm_job("1003", &["node04", "node05"])];
        let current = vec!["node04:7501".to_string(), "node05:7504".to_string()];
        let mut hw = std::collections::HashMap::new();
        hw.insert("node04".to_string(), 4);
        hw.insert("node05".to_string(), 4);
        let (_kept, to_probe) = dynamo_discover_reconcile(
            &jobs,
            &current,
            &hw,
            &Default::default(),
            &Default::default(),
        );
        assert!(to_probe.contains(&"node04".to_string()));
        assert!(to_probe.contains(&"node05".to_string()));
    }

    #[test]
    fn test_dynamo_discover_reconcile_at_highwater_no_reprobe() {
        // Host's current real-port count equals its high-water → nothing to
        // do. Avoid port 7500 because that's PLACEHOLDER_PORT and would be
        // filtered out of the real count.
        let jobs = vec![test_slurm_job("1003", &["node04"])];
        let current = vec![
            "node04:7501".to_string(),
            "node04:7502".to_string(),
            "node04:7503".to_string(),
            "node04:7504".to_string(),
        ];
        let mut hw = std::collections::HashMap::new();
        hw.insert("node04".to_string(), 4);
        let (_kept, to_probe) = dynamo_discover_reconcile(
            &jobs,
            &current,
            &hw,
            &Default::default(),
            &Default::default(),
        );
        assert!(to_probe.is_empty());
    }

    /// Requeued jobs can change worker ports. Discard stale endpoints and
    /// probe the host again when every tracked endpoint is unavailable.
    #[test]
    fn test_dynamo_discover_reconcile_dead_host_dropped_and_reprobed() {
        let jobs = vec![test_slurm_job("1004", &["node06", "node17"])];
        let current = vec!["node06:7501".to_string(), "node17:7503".to_string()];
        let hw = std::collections::HashMap::new();
        let dead: std::collections::HashSet<String> = ["node06".to_string()].into();
        let (kept, to_probe) =
            dynamo_discover_reconcile(&jobs, &current, &hw, &Default::default(), &dead);
        assert_eq!(kept, vec!["node17:7503".to_string()]);
        assert_eq!(to_probe, vec!["node06".to_string()]);
    }

    /// A healthy host whose worker really listens on the placeholder port
    /// (7500) is a real endpoint: keep the
    /// addr and don't re-probe every tick (class (b) log spam).
    #[test]
    fn test_dynamo_discover_reconcile_healthy_placeholder_port_is_real() {
        let jobs = vec![test_slurm_job("9092", &["node01", "node02"])];
        let current = vec!["node01:7500".to_string(), "node02:7501".to_string()];
        let hw = std::collections::HashMap::new();
        let healthy: std::collections::HashSet<String> =
            ["node01".to_string(), "node02".to_string()].into();
        let (kept, to_probe) =
            dynamo_discover_reconcile(&jobs, &current, &hw, &healthy, &Default::default());
        assert_eq!(kept, current);
        assert!(to_probe.is_empty());
    }

    /// An unhealthy placeholder addr still re-probes every tick (class (b)) —
    /// the not-ready-yet path must not gain hysteresis from the dead-host
    /// machinery.
    #[test]
    fn test_dynamo_discover_reconcile_unhealthy_placeholder_still_probes() {
        let jobs = vec![test_slurm_job("1004", &["node17"])];
        let current = vec!["node17:7500".to_string()];
        let hw = std::collections::HashMap::new();
        let (kept, to_probe) = dynamo_discover_reconcile(
            &jobs,
            &current,
            &hw,
            &Default::default(),
            &Default::default(),
        );
        assert!(kept.is_empty());
        assert_eq!(to_probe, vec!["node17".to_string()]);
    }

    #[test]
    fn test_update_highwater_bumps_and_evicts() {
        let mut hw = std::collections::HashMap::new();
        hw.insert("node04".to_string(), 2);
        hw.insert("node99".to_string(), 4); // stale host, no longer tracked
        let settled = vec![
            "node04:7501".to_string(),
            "node04:7502".to_string(),
            "node04:7503".to_string(), // new high of 3
            "node05:7504".to_string(),
        ];
        let mut tracked = std::collections::HashSet::new();
        tracked.insert("node04".to_string());
        tracked.insert("node05".to_string());
        update_highwater(&mut hw, &settled, &tracked);
        assert_eq!(hw.get("node04"), Some(&3));
        assert_eq!(hw.get("node05"), Some(&1));
        assert_eq!(hw.get("node99"), None, "untracked host must be evicted");
    }

    #[test]
    fn test_update_highwater_does_not_regress() {
        // Once we've seen 4 ports on a host, a tick that only sees 1 must NOT
        // lower the high-water — otherwise class (d) loses its memory of the
        // "true" peak and stops re-probing.
        let mut hw = std::collections::HashMap::new();
        hw.insert("node04".to_string(), 4);
        let settled = vec!["node04:7501".to_string()];
        let mut tracked = std::collections::HashSet::new();
        tracked.insert("node04".to_string());
        update_highwater(&mut hw, &settled, &tracked);
        assert_eq!(hw.get("node04"), Some(&4));
    }

    fn sample_families() -> Vec<MetricFamily> {
        parse_prometheus_text(
            r#"# HELP vllm:mooncake_store_operation_time_seconds Histogram
# TYPE vllm:mooncake_store_operation_time_seconds histogram
vllm:mooncake_store_operation_time_seconds_bucket{operation="save_put",status="ok",le="0.1"} 3
vllm:mooncake_store_operation_time_seconds_sum{operation="save_put",status="ok"} 0.2
# HELP vllm:router_queue_depth Router queue depth
# TYPE vllm:router_queue_depth gauge
vllm:router_queue_depth{kind="shared"} 12
# HELP vllm:request_success_total Count of successful requests
# TYPE vllm:request_success_total counter
vllm:request_success_total{finished_reason="stop"} 980
"#,
        )
        .unwrap()
    }

    fn prefix_rule(prefix: &str) -> CompiledRawCaptureRule {
        CompiledRawCaptureRule {
            name: "prefix".to_string(),
            output: PathBuf::from("/tmp/prefix.jsonl"),
            include_help: true,
            include_type: true,
            matcher: RawMatcher::Prefix(vec![prefix.to_string()]),
        }
    }

    fn regex_rule(pattern: &str) -> CompiledRawCaptureRule {
        CompiledRawCaptureRule {
            name: "regex".to_string(),
            output: PathBuf::from("/tmp/regex.jsonl"),
            include_help: true,
            include_type: true,
            matcher: RawMatcher::Regex(vec![Regex::new(pattern).unwrap()]),
        }
    }

    #[test]
    fn test_select_matching_families_prefix_matches_only_matching_family_names() {
        let families = sample_families();
        let matching = select_matching_families(&families, &prefix_rule("vllm:mooncake_store_"));
        assert_eq!(matching.len(), 1);
        assert_eq!(
            matching[0].name,
            "vllm:mooncake_store_operation_time_seconds"
        );
    }

    #[test]
    fn test_select_matching_families_regex_matches_only_matching_family_names() {
        let families = sample_families();
        let matching = select_matching_families(&families, &regex_rule(r"^vllm:router_.*"));
        assert_eq!(matching.len(), 1);
        assert_eq!(matching[0].name, "vllm:router_queue_depth");
    }

    #[test]
    fn test_select_matching_families_matches_counter_total_name() {
        let families = sample_families();
        let matching =
            select_matching_families(&families, &prefix_rule("vllm:request_success_total"));
        assert_eq!(matching.len(), 1);
        assert_eq!(matching[0].name, "vllm:request_success");
    }

    #[test]
    fn test_raw_capture_targets_preserve_host_and_port() {
        let targets = raw_capture_targets(&["node-a:8000".to_string(), "node-b:8001".to_string()]);
        assert_eq!(
            targets,
            vec![
                RawCaptureTarget {
                    node: "node-a:8000".to_string(),
                },
                RawCaptureTarget {
                    node: "node-b:8001".to_string(),
                },
            ]
        );
    }

    #[test]
    fn test_render_metric_families_includes_help_type_and_samples() {
        let families = sample_families();
        let matching = select_matching_families(&families, &prefix_rule("vllm:mooncake_store_"));
        let rendered = render_metric_families(&matching, true, true);
        assert!(rendered.contains("# HELP vllm:mooncake_store_operation_time_seconds Histogram"));
        assert!(rendered.contains("# TYPE vllm:mooncake_store_operation_time_seconds histogram"));
        assert!(rendered.contains(
            r#"vllm:mooncake_store_operation_time_seconds_bucket{operation="save_put",status="ok",le="0.1"} 3"#
        ));
        assert!(rendered.contains(
            r#"vllm:mooncake_store_operation_time_seconds_sum{operation="save_put",status="ok"} 0.2"#
        ));
    }

    #[test]
    fn test_render_metric_families_omits_help_and_type_when_disabled() {
        let families = sample_families();
        let matching = select_matching_families(&families, &regex_rule(r"^vllm:router_.*"));
        let rendered = render_metric_families(&matching, false, false);
        assert!(!rendered.contains("# HELP"));
        assert!(!rendered.contains("# TYPE"));
        assert!(rendered.contains(r#"vllm:router_queue_depth{kind="shared"} 12"#));
    }

    #[test]
    fn test_render_metric_families_preserves_counter_headers_and_samples() {
        let families = sample_families();
        let matching =
            select_matching_families(&families, &prefix_rule("vllm:request_success_total"));
        let rendered = render_metric_families(&matching, true, true);
        assert!(
            rendered.contains("# HELP vllm:request_success_total Count of successful requests")
        );
        assert!(rendered.contains("# TYPE vllm:request_success_total counter"));
        assert!(rendered.contains(r#"vllm:request_success_total{finished_reason="stop"} 980"#));
    }

    #[test]
    fn test_parse_raw_capture_config_applies_defaults() {
        let rules = parse_raw_capture_config_str(
            r#"
[[capture]]
name = "mooncake"
output = "mooncake.jsonl"
mode = "prefix"
patterns = ["vllm:mooncake_store_"]
"#,
            Path::new("/tmp"),
        )
        .unwrap();
        assert_eq!(rules.len(), 1);
        assert!(rules[0].include_help);
        assert!(rules[0].include_type);
        assert_eq!(rules[0].output, PathBuf::from("/tmp/mooncake.jsonl"));
    }

    #[test]
    fn test_parse_raw_capture_config_rejects_empty_capture_list() {
        let err = parse_raw_capture_config_str("capture = []", Path::new("/tmp")).unwrap_err();
        assert!(err.contains("at least one"));
    }

    #[test]
    fn test_parse_raw_capture_config_rejects_invalid_regex() {
        let err = parse_raw_capture_config_str(
            r#"
[[capture]]
name = "router"
output = "router.jsonl"
mode = "regex"
patterns = ["("]
"#,
            Path::new("/tmp"),
        )
        .unwrap_err();
        assert!(err.contains("Invalid regex pattern"));
    }

    #[test]
    fn test_parse_raw_capture_config_rejects_duplicate_names() {
        let err = parse_raw_capture_config_str(
            r#"
[[capture]]
name = "dup"
output = "a.jsonl"
mode = "prefix"
patterns = ["a"]

[[capture]]
name = "dup"
output = "b.jsonl"
mode = "prefix"
patterns = ["b"]
"#,
            Path::new("/tmp"),
        )
        .unwrap_err();
        assert!(err.contains("Duplicate raw capture rule name"));
    }

    #[test]
    fn test_parse_raw_capture_config_rejects_duplicate_outputs() {
        let err = parse_raw_capture_config_str(
            r#"
[[capture]]
name = "a"
output = "same.jsonl"
mode = "prefix"
patterns = ["a"]

[[capture]]
name = "b"
output = "same.jsonl"
mode = "prefix"
patterns = ["b"]
"#,
            Path::new("/tmp"),
        )
        .unwrap_err();
        assert!(err.contains("Duplicate raw capture output path"));
    }

    #[test]
    fn test_validate_raw_capture_outputs_rejects_main_output_collision() {
        let cwd = std::env::current_dir().unwrap();
        let rules = parse_raw_capture_config_str(
            r#"
[[capture]]
name = "mooncake"
output = "report.json"
mode = "prefix"
patterns = ["vllm:mooncake_store_"]
"#,
            cwd.as_path(),
        )
        .unwrap();
        let err = validate_raw_capture_outputs(&rules, Path::new("report.json")).unwrap_err();
        assert!(err.contains("must be different"));
    }

    #[test]
    fn test_raw_capture_entry_uses_unix_seconds_and_capture_name() {
        let entry = raw_capture_entry(1_746_000_123, "node-a:8000", "mooncake", "metric 1");
        assert_eq!(entry["unix_secs"], 1_746_000_123);
        assert_eq!(entry["node"], "node-a:8000");
        assert_eq!(entry["capture"], "mooncake");
        assert_eq!(entry["metrics"], "metric 1");
        assert!(entry.get("elapsed_secs").is_none());
        assert!(entry.get("scrape_addr").is_none());
    }

    #[test]
    fn test_cli_accepts_raw_capture_config_and_rejects_removed_mooncake_flag() {
        assert!(
            Cli::try_parse_from([
                "vmon",
                "collect",
                "node-a",
                "-d",
                "1s",
                "--raw-capture-config",
                "captures.toml",
            ])
            .is_ok()
        );

        assert!(
            Cli::try_parse_from([
                "vmon",
                "collect",
                "node-a",
                "-d",
                "1s",
                "--mooncake-raw-output",
                "mooncake.jsonl",
            ])
            .is_err()
        );
    }

    fn follow_test_job(
        job_id: &str,
        nodes: &[&str],
        start_offset_secs: u64,
        end_offset_secs: Option<u64>,
    ) -> vmon_core::slurm::SlurmJobInfo {
        let now = std::time::SystemTime::now();
        vmon_core::slurm::SlurmJobInfo {
            job_id: job_id.to_string(),
            job_name: format!("job-{job_id}"),
            start_time: now - std::time::Duration::from_secs(start_offset_secs),
            end_time: end_offset_secs.map(|s| now - std::time::Duration::from_secs(s)),
            nodelist_compact: nodes.join(","),
            nodes: nodes.iter().map(|s| s.to_string()).collect(),
        }
    }

    #[test]
    fn merge_slurm_jobs_appends_new_jobs() {
        let mut existing = vec![follow_test_job("100", &["node01"], 60, None)];
        let fresh = vec![
            follow_test_job("100", &["node01"], 70, None),
            follow_test_job("200", &["node02"], 5, None),
        ];
        let now = std::time::SystemTime::now();
        let ended = merge_slurm_jobs(&mut existing, fresh, now, Duration::from_secs(60));
        assert!(ended.is_empty());
        // Newest job (200, started 5s ago) sorts above the older job (100).
        let ids: Vec<&str> = existing.iter().map(|j| j.job_id.as_str()).collect();
        assert_eq!(ids, vec!["200", "100"]);
        // Existing entry preserved (not replaced) — start_time unchanged.
        let job100 = existing.iter().find(|j| j.job_id == "100").unwrap();
        assert!(job100.end_time.is_none());
    }

    #[test]
    fn merge_slurm_jobs_stamps_end_time_when_job_disappears() {
        let mut existing = vec![
            follow_test_job("100", &["node01"], 60, None),
            follow_test_job("200", &["node02"], 30, None),
        ];
        let fresh = vec![follow_test_job("100", &["node01"], 65, None)];
        let now = std::time::SystemTime::now();
        let ended = merge_slurm_jobs(&mut existing, fresh, now, Duration::from_secs(60));
        assert_eq!(ended, vec!["200"]);
        let job200 = existing.iter().find(|j| j.job_id == "200").unwrap();
        assert!(job200.end_time.is_some());
        // Still in the list — within grace.
        assert_eq!(existing.len(), 2);
    }

    #[test]
    fn merge_slurm_jobs_drops_jobs_past_grace() {
        // Job ended 120s ago, grace 60s → drop.
        let mut existing = vec![follow_test_job("100", &["node01"], 300, Some(120))];
        let now = std::time::SystemTime::now();
        let ended = merge_slurm_jobs(&mut existing, vec![], now, Duration::from_secs(60));
        assert!(ended.is_empty()); // already ended in a prior poll, not "newly ended"
        assert!(existing.is_empty());
    }

    #[test]
    fn merge_slurm_jobs_keeps_recently_ended_within_grace() {
        // Job ended 10s ago, grace 60s → keep.
        let mut existing = vec![follow_test_job("100", &["node01"], 300, Some(10))];
        let now = std::time::SystemTime::now();
        merge_slurm_jobs(&mut existing, vec![], now, Duration::from_secs(60));
        assert_eq!(existing.len(), 1);
        assert!(existing[0].is_ended());
    }

    #[test]
    fn desired_addrs_vllm_keeps_ended_during_grace_and_dedups() {
        // Ended-but-still-in-list jobs MUST stay in the scrape set so the TUI
        // can render `(ended)` against their rows during the grace window.
        // `merge_slurm_jobs` is responsible for dropping them after grace.
        let jobs = vec![
            follow_test_job("100", &["node01", "node02"], 60, None),
            follow_test_job("200", &["node02", "node03"], 30, None),
            follow_test_job("300", &["node04"], 300, Some(10)), // ended, within grace
        ];
        let addrs = desired_addrs_vllm(&jobs, 8000);
        assert_eq!(
            addrs,
            vec!["node01:8000", "node02:8000", "node03:8000", "node04:8000"]
        );
    }

    #[test]
    fn split_host_port_parses_and_falls_back() {
        assert_eq!(split_host_port("example-01:8000"), ("example-01", 8000));
        assert_eq!(split_host_port("nova-node03:8081"), ("nova-node03", 8081));
        // Unparseable / missing port keeps the addr collatable.
        assert_eq!(split_host_port("hostonly"), ("hostonly", 0));
        assert_eq!(split_host_port("host:bogus"), ("host", 0));
    }

    /// Parser wiring for the optional-value flag: absent / bare / `=value`.
    #[test]
    fn cli_parses_mooncake_optional_value() {
        fn mc(args: &[&str]) -> Option<Option<String>> {
            match Cli::try_parse_from(args).unwrap().command {
                Command::Watch { mooncake, .. } => mooncake,
                _ => unreachable!(),
            }
        }
        assert_eq!(mc(&["vmon", "watch", "node-a"]), None);
        assert_eq!(mc(&["vmon", "watch", "node-a", "--mooncake"]), Some(None));
        assert_eq!(
            mc(&["vmon", "watch", "node-a", "--mooncake=h:9003"]),
            Some(Some("h:9003".into()))
        );
        // require_equals: space-separated value must NOT be swallowed as the
        // address — it stays a positional node instead.
        assert_eq!(
            mc(&["vmon", "watch", "node-a", "--mooncake", "node-b"]),
            Some(None)
        );
    }

    /// With no jobs to group by, the address list sorts by host first, then
    /// numeric port, so the TUI row order is stable regardless of how the
    /// follow task discovered hosts. Without this, the discover branch builds
    /// new addrs from a HashSet (random iteration order) and rows would shuffle
    /// every refresh.
    #[test]
    fn addr_sort_orders_by_host_then_numeric_port() {
        let mut addrs = vec![
            "example-02:7500".to_string(),
            "example-01:7501".to_string(),
            "example-01:7500".to_string(),
            "example-01:7400".to_string(), // numerically smaller port
            "example-02:7400".to_string(),
        ];
        sort_addrs_by_job(&mut addrs, &[]);
        assert_eq!(
            addrs,
            vec![
                "example-01:7400",
                "example-01:7500",
                "example-01:7501",
                "example-02:7400",
                "example-02:7500",
            ]
        );
    }

    /// Multi-task monitoring groups rows by SLURM job with the newest job's
    /// hosts on top, and sorts (host, port) within each job. `follow_test_job`'s
    /// third arg is "seconds ago it started", so a smaller offset is newer.
    #[test]
    fn sort_addrs_groups_by_job_newest_first() {
        // Newest-first job order, as produced by merge_slurm_jobs.
        let jobs = vec![
            follow_test_job("200", &["node09", "node10"], 5, None), // newer
            follow_test_job("100", &["node03", "node04"], 600, None), // older
        ];
        // Interleaved + out-of-order ports to prove the sort regroups them.
        let mut addrs = vec![
            "node04:8001".to_string(),
            "node10:8000".to_string(),
            "node03:8000".to_string(),
            "node09:8001".to_string(),
            "node09:8000".to_string(),
            "node04:8000".to_string(),
        ];
        sort_addrs_by_job(&mut addrs, &jobs);
        assert_eq!(
            addrs,
            vec![
                // Newest job (200) first, (host, port) ordered within it.
                "node09:8000",
                "node09:8001",
                "node10:8000",
                // Older job (100) second.
                "node03:8000",
                "node04:8000",
                "node04:8001",
            ]
        );
    }

    /// Hosts not owned by any tracked job sink below all job-owned rows,
    /// still (host, port) ordered among themselves.
    #[test]
    fn sort_addrs_places_orphan_hosts_last() {
        let jobs = vec![follow_test_job("100", &["node03"], 60, None)];
        let mut addrs = vec![
            "orphan:8000".to_string(),
            "node03:8000".to_string(),
            "another:8000".to_string(),
        ];
        sort_addrs_by_job(&mut addrs, &jobs);
        assert_eq!(addrs, vec!["node03:8000", "another:8000", "orphan:8000"]);
    }

    #[test]
    fn desired_addrs_dynamo_explicit_groups_per_job() {
        let jobs = vec![
            follow_test_job("100", &["node01", "node02"], 60, None),
            follow_test_job("200", &["node03"], 30, None),
        ];
        let addrs = desired_addrs_dynamo_explicit(&jobs, 8081, 2);
        assert_eq!(
            addrs,
            vec![
                // Job 100
                "node01:8081",
                "node01:8082",
                "node02:8083",
                "node02:8084",
                // Job 200 — port counter resets to 8081
                "node03:8081",
                "node03:8082",
            ]
        );
    }
}
