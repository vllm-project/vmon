// SPDX-License-Identifier: Apache-2.0

use std::time::SystemTime;

/// Cluster-wide node counts from `sinfo`, attached to a `ClusterState`.
///
/// Unlike the scrape targets (which only cover the user's tracked jobs),
/// this reflects the whole SLURM cluster: how many nodes exist and how
/// many are currently idle (free to allocate).
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct SlurmClusterStats {
    /// Unique nodes across all partitions.
    pub total_nodes: usize,
    /// Nodes whose base state is `idle` (allocatable right now).
    pub idle_nodes: usize,
}

/// Parse `sinfo -h -N -o "%N %t"` output into cluster-wide node counts.
///
/// `-N` prints one line per node per partition, so a node shared by
/// several partitions appears more than once — deduplicate by hostname
/// (first occurrence wins). State flags appended by SLURM (`*` not
/// responding, `~` powered down, `#` powering up, ...) are stripped
/// before comparing, so `idle~` still counts as idle.
pub fn parse_sinfo_node_states(output: &str) -> SlurmClusterStats {
    let mut seen = std::collections::HashSet::new();
    let mut stats = SlurmClusterStats::default();
    for line in output.lines() {
        let mut parts = line.split_whitespace();
        let (Some(node), Some(state)) = (parts.next(), parts.next()) else {
            continue;
        };
        if !seen.insert(node.to_string()) {
            continue;
        }
        stats.total_nodes += 1;
        let base = state.trim_end_matches(|c: char| !c.is_ascii_alphanumeric());
        if base.eq_ignore_ascii_case("idle") {
            stats.idle_nodes += 1;
        }
    }
    stats
}

/// SLURM job metadata attached to a `ClusterState`.
///
/// Initially captured at CLI startup by intersecting the user's running
/// jobs (via `squeue`) with the resolved scrape targets, then refreshed
/// periodically; when a tracked job_id stops appearing in `squeue`, the
/// CLI sets `end_time = Some(now)` so uptime stops walking.
#[derive(Debug, Clone)]
pub struct SlurmJobInfo {
    pub job_id: String,
    pub job_name: String,
    /// Wall-clock job start time (from `squeue %V` / `scontrol StartTime`).
    pub start_time: SystemTime,
    /// When the CLI first observed the job leaving `squeue`. Once set,
    /// `format_uptime()` / `uptime_secs()` freeze at this moment.
    pub end_time: Option<SystemTime>,
    /// Compact node list as reported by SLURM (e.g. `gpu-node[01-04]`).
    pub nodelist_compact: String,
    /// Expanded bare hostnames covered by this job (no port).
    pub nodes: Vec<String>,
}

impl SlurmJobInfo {
    /// Whether this job owns the given bare hostname.
    pub fn owns_host(&self, host: &str) -> bool {
        self.nodes.iter().any(|n| n == host)
    }

    /// True once the CLI has observed the job leaving `squeue`.
    pub fn is_ended(&self) -> bool {
        self.end_time.is_some()
    }

    /// Format the job runtime as `2d 3h`, `4h 12m`, `2m 23s`, or `45s`,
    /// matching `squeue %M` style. Frozen once `end_time` is set.
    pub fn format_uptime(&self) -> String {
        format_duration(self.uptime_secs())
    }

    /// Job runtime in seconds. Frozen at `end_time` once the job has ended;
    /// otherwise computed against `now`. Returns 0 if `start_time` is in
    /// the future relative to the reference instant.
    pub fn uptime_secs(&self) -> u64 {
        let reference = self.end_time.unwrap_or_else(SystemTime::now);
        reference.duration_since(self.start_time).map(|d| d.as_secs()).unwrap_or(0)
    }
}

fn format_duration(secs: u64) -> String {
    let d = secs / 86_400;
    let h = (secs % 86_400) / 3_600;
    let m = (secs % 3_600) / 60;
    let s = secs % 60;

    if d > 0 {
        format!("{d}d {h}h")
    } else if h > 0 {
        format!("{h}h {m}m")
    } else if m > 0 {
        format!("{m}m {s}s")
    } else {
        format!("{s}s")
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn duration_formatting_chooses_two_largest_units() {
        assert_eq!(format_duration(45), "45s");
        assert_eq!(format_duration(143), "2m 23s");
        assert_eq!(format_duration(3 * 3600 + 12 * 60 + 5), "3h 12m");
        assert_eq!(format_duration(2 * 86_400 + 7 * 3600), "2d 7h");
    }

    #[test]
    fn uptime_freezes_after_end_time_is_set() {
        let start = SystemTime::now() - std::time::Duration::from_secs(120);
        let end = start + std::time::Duration::from_secs(60);
        let job = SlurmJobInfo {
            job_id: "1".into(),
            job_name: "j".into(),
            start_time: start,
            end_time: Some(end),
            nodelist_compact: String::new(),
            nodes: vec![],
        };
        // Frozen at end - start = 60s, regardless of wall clock.
        assert_eq!(job.uptime_secs(), 60);
        assert!(job.is_ended());
        // And stays frozen on a second call after some real time elapses.
        std::thread::sleep(std::time::Duration::from_millis(20));
        assert_eq!(job.uptime_secs(), 60);
    }

    #[test]
    fn sinfo_parse_counts_totals_and_idle() {
        let out = "node01 alloc\nnode02 idle\nnode03 mix\nnode04 idle\nnode05 down*\n";
        let stats = parse_sinfo_node_states(out);
        assert_eq!(stats.total_nodes, 5);
        assert_eq!(stats.idle_nodes, 2);
    }

    #[test]
    fn sinfo_parse_dedupes_nodes_shared_across_partitions() {
        // Same node listed under two partitions — count it once.
        let out = "node01 idle\nnode01 idle\nnode02 alloc\n";
        let stats = parse_sinfo_node_states(out);
        assert_eq!(stats.total_nodes, 2);
        assert_eq!(stats.idle_nodes, 1);
    }

    #[test]
    fn sinfo_parse_strips_state_flag_suffixes() {
        // `idle~` (powered down) and `idle*` (not responding) still have
        // base state idle; `drain` and `down` do not.
        let out = "node01 idle~\nnode02 idle*\nnode03 drain\nnode04 down~\n";
        let stats = parse_sinfo_node_states(out);
        assert_eq!(stats.total_nodes, 4);
        assert_eq!(stats.idle_nodes, 2);
    }

    #[test]
    fn sinfo_parse_tolerates_blank_and_malformed_lines() {
        let out = "\nnode01\n  \nnode02 idle\n";
        let stats = parse_sinfo_node_states(out);
        assert_eq!(stats.total_nodes, 1);
        assert_eq!(stats.idle_nodes, 1);
    }

    #[test]
    fn uptime_walks_when_end_time_is_none() {
        let start = SystemTime::now() - std::time::Duration::from_secs(5);
        let job = SlurmJobInfo {
            job_id: "1".into(),
            job_name: "j".into(),
            start_time: start,
            end_time: None,
            nodelist_compact: String::new(),
            nodes: vec![],
        };
        assert!(!job.is_ended());
        assert!(job.uptime_secs() >= 5);
    }
}
