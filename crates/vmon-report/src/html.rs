// SPDX-License-Identifier: Apache-2.0

use minijinja::{Environment, Value, context};

use crate::collector::TimeSeriesCollector;

const HTML_TEMPLATE: &str = r##"<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>vmon Report</title>
<style>
body { font-family: -apple-system, BlinkMacSystemFont, sans-serif; margin: 20px; background: #1a1a2e; color: #e0e0e0; }
h1 { color: #00d4ff; }
h2 { color: #e0e0e0; font-size: 16px; margin: 0 0 12px; }
.meta { color: #888; margin-bottom: 20px; overflow-wrap: anywhere; }
.charts { display: grid; grid-template-columns: 1fr 1fr; gap: 20px; }
.chart { min-width: 0; background: #16213e; border-radius: 8px; padding: 16px; }
.chart svg { display: block; width: 100%; }
.chart svg text { fill: #a0a0a0; font-size: 12px; }
.legend { display: flex; flex-wrap: wrap; gap: 6px; }
.legend button { background: transparent; border: 1px solid #475569; border-radius: 4px; padding: 4px 8px; cursor: pointer; overflow-wrap: anywhere; max-width: 100%; }
.legend button[aria-pressed="false"] { opacity: 0.5; text-decoration: line-through; }
.chart-values { color: #a0a0a0; font-size: 12px; white-space: pre-wrap; overflow-wrap: anywhere; max-height: 160px; overflow: auto; }
:focus-visible { outline: 2px solid #00d4ff; outline-offset: 2px; }
@media (max-width: 800px) { .charts { grid-template-columns: 1fr; } }
</style>
<script>{{ charts_src }}</script>
</head>
<body>
<h1>vmon Report</h1>
<div class="meta">
  Duration: {{ duration }} &middot; Samples: {{ sample_count }} &middot; Nodes: {{ nodes | join(", ") }}
</div>
<noscript>Enable JavaScript to view the report charts.</noscript>
<div class="charts">
  <section class="chart" id="throughput"></section>
  <section class="chart" id="kv_cache"></section>
  <section class="chart" id="ttft"></section>
  <section class="chart" id="itl_tpot"></section>
  <section class="chart" id="e2e"></section>
  <section class="chart" id="phases"></section>
  <section class="chart" id="queue"></section>
  <section class="chart" id="cache_hit"></section>
  <section class="chart" id="req_stats"></section>
  <section class="chart" id="preemptions"></section>
</div>
<script>
const DATA = {{ data_json }};
const NODES = {{ nodes_json }};
const COLORS = ['#00d4ff','#ff6b6b','#ffd93d','#6bcb77','#c084fc','#fb923c'];

function nodeDatasets(metric, label_suffix) {
  return NODES.map((node, i) => ({
    label: node + (label_suffix || ''),
    data: DATA.map(s => s.nodes[node] ? s.nodes[node][metric] : null),
    color: COLORS[i % COLORS.length],
  }));
}

function pctDatasets(metric, label_suffix) {
  return nodeDatasets(metric, label_suffix).map(d => ({...d, data: d.data.map(v => Number.isFinite(v) ? v * 100 : null)}));
}

// Throughput
makeChart('throughput', 'Throughput (tokens/s)', [
  ...nodeDatasets('generation_tps', ' gen'),
  ...nodeDatasets('prompt_tps', ' pf'),
]);

// KV Cache
makeChart('kv_cache', 'KV Cache Usage (%)', pctDatasets('kv_cache'));

// TTFT
makeChart('ttft', 'Time to First Token (ms)', [
  ...nodeDatasets('ttft_p50_ms', ' p50'),
  ...nodeDatasets('ttft_p99_ms', ' p99'),
]);

// ITL + TPOT
makeChart('itl_tpot', 'Inter-Token Latency & TPOT (ms)', [
  ...nodeDatasets('itl_p50_ms', ' ITL p50'),
  ...nodeDatasets('itl_p99_ms', ' ITL p99'),
  ...nodeDatasets('tpot_p50_ms', ' TPOT p50'),
  ...nodeDatasets('tpot_p99_ms', ' TPOT p99'),
]);

// E2E
makeChart('e2e', 'End-to-End Latency (ms)', [
  ...nodeDatasets('e2e_p50_ms', ' p50'),
  ...nodeDatasets('e2e_p99_ms', ' p99'),
]);

// Phases: prefill, decode, inference, queue
makeChart('phases', 'Phase Latency p99 (ms)', [
  ...nodeDatasets('prefill_p99_ms', ' prefill'),
  ...nodeDatasets('decode_p99_ms', ' decode'),
  ...nodeDatasets('inference_p99_ms', ' inference'),
  ...nodeDatasets('queue_p99_ms', ' queue'),
]);

// Queue depth
makeChart('queue', 'Queue Depth', [
  ...nodeDatasets('running', ' running'),
  ...nodeDatasets('waiting', ' waiting'),
]);

// Cache hit rate
makeChart('cache_hit', 'Cache Hit Rate (%)', [
  ...pctDatasets('prefix_cache_hit_rate', ' prefix'),
  ...pctDatasets('external_cache_hit_rate', ' external'),
]);

// Request stats
makeChart('req_stats', 'Avg Tokens per Request', [
  ...nodeDatasets('avg_prompt_tokens', ' prompt'),
  ...nodeDatasets('avg_generation_tokens', ' gen'),
  ...nodeDatasets('iteration_tokens_mean', ' iter'),
]);

// Preemptions
makeChart('preemptions', 'Preemptions/s', nodeDatasets('preemptions_per_sec'));
</script>
</body>
</html>
"##;

// First-party SVG rendering code, embedded for offline use.
const CHARTS_JS: &str = include_str!("charts.js");

/// Escape HTML parser delimiters as JSON escapes, including script-like comments.
fn escape_json_for_script(s: String) -> String {
    s.replace('&', "\\u0026")
        .replace('<', "\\u003c")
        .replace('>', "\\u003e")
        .replace('\u{2028}', "\\u2028")
        .replace('\u{2029}', "\\u2029")
}

pub fn render(collector: &TimeSeriesCollector) -> String {
    let duration = collector
        .samples
        .last()
        .map(|s| format!("{:.0}s", s.elapsed_secs))
        .unwrap_or_else(|| "0s".to_string());

    let data_json =
        escape_json_for_script(serde_json::to_string(&collector.samples).unwrap_or_default());
    let nodes_json =
        escape_json_for_script(serde_json::to_string(&collector.node_addrs).unwrap_or_default());

    let mut env = Environment::new();
    env.add_template("report.html", HTML_TEMPLATE).unwrap();
    let tmpl = env.get_template("report.html").unwrap();

    tmpl.render(context! {
        charts_src => Value::from_safe_string(CHARTS_JS.to_owned()),
        duration => duration,
        sample_count => collector.samples.len(),
        nodes => &collector.node_addrs,
        data_json => Value::from_safe_string(data_json),
        nodes_json => Value::from_safe_string(nodes_json),
    })
    .unwrap_or_else(|e| format!("Template error: {e}"))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn report_escapes_markup_but_preserves_json_data() {
        let node = "<img src=x onerror=alert(1)></script><!--<script>&\u{2028}:8000";
        let report = render(&TimeSeriesCollector::new(vec![node.into()]));
        assert!(!report.contains(node));
        assert!(report.contains("&lt;img"));
        assert_eq!(report.matches("<script>").count(), 2);
        assert_eq!(report.matches("</script>").count(), 2);
        let json = report.split("const NODES = ").nth(1).unwrap().split(';').next().unwrap();
        let nodes: Vec<String> = serde_json::from_str(json).unwrap();
        assert_eq!(nodes, vec![node]);
        assert!(!report.contains("<script src="));
        assert!(report.contains("createElementNS"));
    }
}
