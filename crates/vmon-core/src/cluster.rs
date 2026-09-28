// SPDX-License-Identifier: Apache-2.0

use std::collections::HashMap;

use crate::mooncake::MooncakeMetrics;
use crate::node::{NodeInfo, NodeMetrics};
use crate::slurm::{SlurmClusterStats, SlurmJobInfo};

/// Aggregated state across all nodes in the cluster.
#[derive(Debug, Clone)]
pub struct ClusterState {
    pub nodes: Vec<NodeMetrics>,
    pub node_infos: HashMap<String, NodeInfo>,
    pub healthy_count: usize,
    pub unhealthy_count: usize,
    pub total_requests_running: f64,
    pub total_requests_waiting: f64,
    pub avg_kv_cache_usage: f64,
    pub total_generation_tps: f64,
    pub total_prompt_tps: f64,
    /// SLURM jobs whose nodelists overlap the scrape targets.
    /// Multiple entries appear when targets span more than one running job.
    pub slurm_jobs: Vec<SlurmJobInfo>,
    /// Cluster-wide node counts from `sinfo` (total / idle across the whole
    /// SLURM cluster, not just the scrape targets). `None` when sinfo is
    /// unavailable or hasn't been polled yet.
    pub slurm_cluster: Option<SlurmClusterStats>,
    /// Mooncake Store leader addresses when configured (`--mooncake`):
    /// a single explicit address, or one auto-derived target per tracked
    /// SLURM job. Set even when a scrape fails so the UI can render an
    /// "unreachable" indicator instead of silently omitting the section.
    pub mooncake_addrs: Vec<String>,
    /// Latest Mooncake metrics, in `mooncake_addrs` order. A target whose
    /// scrape failed for the current tick has no entry here.
    pub mooncakes: Vec<MooncakeMetrics>,
}

impl ClusterState {
    /// Find the SLURM job that owns the given bare hostname, if any.
    pub fn slurm_job_for_host(&self, host: &str) -> Option<&SlurmJobInfo> {
        self.slurm_jobs.iter().find(|j| j.owns_host(host))
    }

    /// Resolve ambiguous "backend" Dynamo roles using deployment-wide context.
    ///
    /// Dynamo names the aggregated worker component "backend", but PD
    /// deployments reuse the same name for decode workers — only prefill
    /// workers report a distinct "prefill" label. A single node can't tell
    /// the two apart, so this runs after aggregation: within each scope
    /// (nodes owned by the same SLURM job; job-less nodes share one scope),
    /// "backend" becomes "D" when the scope also has a prefill badge, and
    /// is dropped otherwise (aggregated worker, no badge). Call after
    /// `slurm_jobs` is set.
    pub fn resolve_backend_roles(&mut self) {
        // Scope id: index of the owning job, or jobs.len() for job-less hosts.
        let scopes: Vec<usize> = self
            .nodes
            .iter()
            .map(|n| {
                let host = n.addr.split(':').next().unwrap_or(&n.addr);
                self.slurm_jobs
                    .iter()
                    .position(|j| j.owns_host(host))
                    .unwrap_or(self.slurm_jobs.len())
            })
            .collect();

        let mut scope_has_prefill = vec![false; self.slurm_jobs.len() + 1];
        for (n, &s) in self.nodes.iter().zip(&scopes) {
            if role_has_prefill(n.dynamo_role.as_deref()) {
                scope_has_prefill[s] = true;
            }
        }

        for (n, &s) in self.nodes.iter_mut().zip(&scopes) {
            let backend_is_decode = scope_has_prefill[s];
            n.dynamo_role = rewrite_backend(n.dynamo_role.take(), backend_is_decode);
            if let Some(engines) = n.engine_metrics.as_mut() {
                for e in engines {
                    e.dynamo_role = rewrite_backend(e.dynamo_role.take(), backend_is_decode);
                }
            }
        }
    }

    pub fn aggregate(nodes: Vec<NodeMetrics>, node_infos: HashMap<String, NodeInfo>) -> Self {
        let healthy_count = nodes.iter().filter(|n| n.is_healthy).count();
        let unhealthy_count = nodes.len() - healthy_count;

        let healthy_nodes: Vec<&NodeMetrics> = nodes.iter().filter(|n| n.is_healthy).collect();

        let total_requests_running: f64 = healthy_nodes.iter().map(|n| n.requests_running).sum();
        let total_requests_waiting: f64 = healthy_nodes.iter().map(|n| n.requests_waiting).sum();
        let total_generation_tps: f64 = healthy_nodes.iter().map(|n| n.generation_tps).sum();
        let total_prompt_tps: f64 = healthy_nodes.iter().map(|n| n.prompt_tps).sum();

        let avg_kv_cache_usage = if healthy_nodes.is_empty() {
            0.0
        } else {
            healthy_nodes.iter().map(|n| n.kv_cache_usage).sum::<f64>() / healthy_nodes.len() as f64
        };

        ClusterState {
            nodes,
            node_infos,
            healthy_count,
            unhealthy_count,
            total_requests_running,
            total_requests_waiting,
            avg_kv_cache_usage,
            total_generation_tps,
            total_prompt_tps,
            slurm_jobs: Vec::new(),
            slurm_cluster: None,
            mooncake_addrs: Vec::new(),
            mooncakes: Vec::new(),
        }
    }
}

fn role_has_prefill(role: Option<&str>) -> bool {
    role.is_some_and(|r| r.split('+').any(|b| b == "P" || b == "prefill"))
}

/// Map "backend" badge parts to "D" when the scope is PD-disaggregated, drop
/// them otherwise; non-backend parts pass through. None when nothing remains.
fn rewrite_backend(role: Option<String>, backend_is_decode: bool) -> Option<String> {
    let role = role?;
    if !role.split('+').any(|b| b == "backend") {
        return Some(role);
    }
    let parts: Vec<&str> = role
        .split('+')
        .filter_map(|b| {
            if b == "backend" {
                backend_is_decode.then_some("D")
            } else {
                Some(b)
            }
        })
        .collect();
    if parts.is_empty() {
        None
    } else {
        Some(parts.join("+"))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::SystemTime;

    fn node(addr: &str, role: Option<&str>) -> NodeMetrics {
        let mut n = NodeMetrics::offline(addr.to_string());
        n.dynamo_role = role.map(|s| s.to_string());
        n
    }

    fn job(id: &str, hosts: &[&str]) -> SlurmJobInfo {
        SlurmJobInfo {
            job_id: id.into(),
            job_name: format!("job-{id}"),
            start_time: SystemTime::now(),
            end_time: None,
            nodelist_compact: hosts.join(","),
            nodes: hosts.iter().map(|h| h.to_string()).collect(),
        }
    }

    /// PD deployment: prefill workers present, so "backend" hosts are the
    /// decode side and earn a "D" badge. Engine sub-rows resolve too.
    #[test]
    fn resolve_marks_backend_as_decode_when_prefill_present() {
        let mut decode = node("node02", Some("backend"));
        decode.engine_metrics = Some(vec![
            node("node02:8081", Some("backend")),
            node("node02:8082", Some("backend")),
        ]);
        let mut cluster =
            ClusterState::aggregate(vec![node("node01", Some("P")), decode], Default::default());
        cluster.resolve_backend_roles();
        assert_eq!(cluster.nodes[0].dynamo_role.as_deref(), Some("P"));
        assert_eq!(cluster.nodes[1].dynamo_role.as_deref(), Some("D"));
        let engines = cluster.nodes[1].engine_metrics.as_ref().unwrap();
        assert!(engines.iter().all(|e| e.dynamo_role.as_deref() == Some("D")));
    }

    /// Raw "prefill" (unfolded single-rank path) also marks the scope as PD.
    #[test]
    fn resolve_accepts_raw_prefill_label() {
        let mut cluster = ClusterState::aggregate(
            vec![
                node("node01:8000", Some("prefill")),
                node("node02:8000", Some("backend")),
            ],
            Default::default(),
        );
        cluster.resolve_backend_roles();
        assert_eq!(cluster.nodes[1].dynamo_role.as_deref(), Some("D"));
    }

    /// Aggregated deployment: no prefill anywhere, "backend" is the combined
    /// worker — a "[D]" badge would read as decode-only, so it's dropped.
    #[test]
    fn resolve_drops_backend_badge_without_prefill() {
        let mut cluster = ClusterState::aggregate(
            vec![
                node("node01", Some("backend")),
                node("node02", Some("backend")),
            ],
            Default::default(),
        );
        cluster.resolve_backend_roles();
        assert!(cluster.nodes.iter().all(|n| n.dynamo_role.is_none()));
    }

    /// Multi-job monitoring: each SLURM job is its own scope. Job 100 is PD
    /// (its backends become "D"); job 200 is aggregated (badge dropped) and
    /// must not be contaminated by job 100's prefill workers.
    #[test]
    fn resolve_scopes_backend_per_slurm_job() {
        let mut cluster = ClusterState::aggregate(
            vec![
                node("node01", Some("P")),
                node("node02", Some("backend")),
                node("node09", Some("backend")),
            ],
            Default::default(),
        );
        cluster.slurm_jobs = vec![job("100", &["node01", "node02"]), job("200", &["node09"])];
        cluster.resolve_backend_roles();
        assert_eq!(cluster.nodes[1].dynamo_role.as_deref(), Some("D"));
        assert_eq!(cluster.nodes[2].dynamo_role.as_deref(), None);
    }

    /// Already-resolved badges ("P", "D", "P+D") and unknown components pass
    /// through untouched.
    #[test]
    fn resolve_leaves_resolved_and_unknown_roles_alone() {
        let mut cluster = ClusterState::aggregate(
            vec![
                node("node01", Some("P+D")),
                node("node02", Some("router")),
                node("node03", None),
            ],
            Default::default(),
        );
        cluster.resolve_backend_roles();
        assert_eq!(cluster.nodes[0].dynamo_role.as_deref(), Some("P+D"));
        assert_eq!(cluster.nodes[1].dynamo_role.as_deref(), Some("router"));
        assert_eq!(cluster.nodes[2].dynamo_role.as_deref(), None);
    }
}
