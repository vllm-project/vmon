// SPDX-License-Identifier: Apache-2.0

use std::fmt;

#[derive(Debug, Clone, PartialEq)]
pub enum MetricType {
    Counter,
    Gauge,
    Histogram,
    Summary,
    Untyped,
}

impl fmt::Display for MetricType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            MetricType::Counter => write!(f, "counter"),
            MetricType::Gauge => write!(f, "gauge"),
            MetricType::Histogram => write!(f, "histogram"),
            MetricType::Summary => write!(f, "summary"),
            MetricType::Untyped => write!(f, "untyped"),
        }
    }
}

#[derive(Debug, Clone)]
pub struct Sample {
    pub name: String,
    pub labels: Vec<(String, String)>,
    pub value: f64,
}

impl Sample {
    pub fn label(&self, key: &str) -> Option<&str> {
        self.labels.iter().find(|(k, _)| k == key).map(|(_, v)| v.as_str())
    }
}

#[derive(Debug, Clone)]
pub struct MetricFamily {
    pub name: String,
    pub help: String,
    pub metric_type: MetricType,
    pub samples: Vec<Sample>,
}

#[derive(Debug, thiserror::Error)]
pub enum ParseError {
    #[error("invalid metric line: {0}")]
    InvalidLine(String),
    #[error("invalid float value: {0}")]
    InvalidFloat(String),
}

/// Parse Prometheus text exposition format into metric families.
pub fn parse_prometheus_text(input: &str) -> Result<Vec<MetricFamily>, ParseError> {
    let mut families: Vec<MetricFamily> = Vec::new();
    let mut current_name = String::new();
    let mut current_help = String::new();
    let mut current_type = MetricType::Untyped;
    let mut current_samples: Vec<Sample> = Vec::new();

    for line in input.lines() {
        let line = line.trim();
        if line.is_empty() {
            continue;
        }

        if let Some(rest) = line.strip_prefix("# HELP ") {
            if let Some(idx) = rest.find(' ') {
                let name = &rest[..idx];
                let help = &rest[idx + 1..];
                // If we were building a family, flush it
                if !current_name.is_empty() && !current_samples.is_empty() {
                    families.push(MetricFamily {
                        name: current_name.clone(),
                        help: current_help.clone(),
                        metric_type: current_type.clone(),
                        samples: std::mem::take(&mut current_samples),
                    });
                }
                current_name = name.to_string();
                current_help = help.to_string();
                current_type = MetricType::Untyped;
            }
            continue;
        }

        if let Some(rest) = line.strip_prefix("# TYPE ") {
            if let Some(idx) = rest.find(' ') {
                let name = &rest[..idx];
                let type_str = &rest[idx + 1..];
                let metric_type = match type_str {
                    "counter" => MetricType::Counter,
                    "gauge" => MetricType::Gauge,
                    "histogram" => MetricType::Histogram,
                    "summary" => MetricType::Summary,
                    _ => MetricType::Untyped,
                };
                // Prometheus client appends _total to counter TYPE/HELP lines;
                // normalize by stripping it so the family name matches the base.
                let normalized = if metric_type == MetricType::Counter {
                    name.strip_suffix("_total").unwrap_or(name)
                } else {
                    name
                };
                // Flush previous family if name changed
                if !current_name.is_empty()
                    && current_name != normalized
                    && !current_samples.is_empty()
                {
                    families.push(MetricFamily {
                        name: current_name.clone(),
                        help: current_help.clone(),
                        metric_type: current_type.clone(),
                        samples: std::mem::take(&mut current_samples),
                    });
                    current_help = String::new();
                }
                current_name = normalized.to_string();
                current_type = metric_type;
            }
            continue;
        }

        if line.starts_with('#') {
            continue;
        }

        // Parse sample line: name{labels} value [timestamp]
        let sample = parse_sample_line(line)?;

        // Check if this sample belongs to a new unnamed family
        let sample_base = sample
            .name
            .strip_suffix("_bucket")
            .or_else(|| sample.name.strip_suffix("_count"))
            .or_else(|| sample.name.strip_suffix("_sum"))
            .or_else(|| sample.name.strip_suffix("_total"))
            .or_else(|| sample.name.strip_suffix("_created"))
            .unwrap_or(&sample.name);

        if current_name.is_empty() || (sample_base != current_name && sample.name != current_name) {
            if !current_name.is_empty() && !current_samples.is_empty() {
                families.push(MetricFamily {
                    name: current_name.clone(),
                    help: current_help.clone(),
                    metric_type: current_type.clone(),
                    samples: std::mem::take(&mut current_samples),
                });
            }
            if sample_base != current_name && sample.name != current_name {
                current_name = sample_base.to_string();
                current_help = String::new();
                current_type = MetricType::Untyped;
            }
        }

        current_samples.push(sample);
    }

    // Flush last family
    if !current_name.is_empty() && !current_samples.is_empty() {
        families.push(MetricFamily {
            name: current_name,
            help: current_help,
            metric_type: current_type,
            samples: current_samples,
        });
    }

    Ok(families)
}

fn parse_sample_line(line: &str) -> Result<Sample, ParseError> {
    let (name, rest) = if let Some(brace) = line.find('{') {
        (&line[..brace], &line[brace..])
    } else {
        // No labels
        let mut parts = line.splitn(2, |c: char| c.is_whitespace());
        let name = parts.next().unwrap_or("");
        let value_str = parts.next().unwrap_or("").split_whitespace().next().unwrap_or("");
        let value = parse_f64(value_str)?;
        return Ok(Sample {
            name: name.to_string(),
            labels: Vec::new(),
            value,
        });
    };

    // Parse labels between { and }
    let close_brace = rest.find('}').ok_or_else(|| ParseError::InvalidLine(line.to_string()))?;
    let labels_str = &rest[1..close_brace];
    let after_brace = &rest[close_brace + 1..];

    let labels = parse_labels(labels_str);

    let value_str = after_brace.split_whitespace().next().unwrap_or("");
    let value = parse_f64(value_str)?;

    Ok(Sample {
        name: name.to_string(),
        labels,
        value,
    })
}

fn parse_labels(s: &str) -> Vec<(String, String)> {
    let mut labels = Vec::new();
    if s.is_empty() {
        return labels;
    }

    let mut remaining = s;
    while !remaining.is_empty() {
        // Find key
        let eq = match remaining.find('=') {
            Some(i) => i,
            None => break,
        };
        let key = remaining[..eq].trim().trim_start_matches(',').trim();
        remaining = &remaining[eq + 1..];

        // Value is quoted
        if !remaining.starts_with('"') {
            break;
        }
        remaining = &remaining[1..]; // skip opening quote

        // Find closing quote (handle escaped quotes)
        let mut value = String::new();
        let mut chars = remaining.chars();
        loop {
            match chars.next() {
                Some('\\') => {
                    if let Some(c) = chars.next() {
                        match c {
                            'n' => value.push('\n'),
                            '\\' => value.push('\\'),
                            '"' => value.push('"'),
                            _ => {
                                value.push('\\');
                                value.push(c);
                            }
                        }
                    }
                }
                Some('"') => break,
                Some(c) => value.push(c),
                None => break,
            }
        }

        labels.push((key.to_string(), value));
        remaining = chars.as_str().trim_start_matches(',').trim_start();
    }

    labels
}

fn parse_f64(s: &str) -> Result<f64, ParseError> {
    match s {
        "+Inf" | "Inf" => Ok(f64::INFINITY),
        "-Inf" => Ok(f64::NEG_INFINITY),
        "NaN" => Ok(f64::NAN),
        _ => s.parse::<f64>().map_err(|_| ParseError::InvalidFloat(s.to_string())),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_gauge() {
        let input = r#"# HELP vllm:num_requests_running Number of requests running
# TYPE vllm:num_requests_running gauge
vllm:num_requests_running{model_name="example/Example-Model"} 15.0
"#;
        let families = parse_prometheus_text(input).unwrap();
        assert_eq!(families.len(), 1);
        assert_eq!(families[0].name, "vllm:num_requests_running");
        assert_eq!(families[0].metric_type, MetricType::Gauge);
        assert_eq!(families[0].samples.len(), 1);
        assert_eq!(families[0].samples[0].value, 15.0);
        assert_eq!(
            families[0].samples[0].label("model_name"),
            Some("example/Example-Model")
        );
    }

    #[test]
    fn test_parse_counter() {
        // Prometheus client appends _total to counter TYPE/HELP lines;
        // parser should normalize the family name back to the base.
        let input = r#"# HELP vllm:prompt_tokens_total Number of prefill tokens
# TYPE vllm:prompt_tokens_total counter
vllm:prompt_tokens_total{model_name="example-model"} 123456.0
"#;
        let families = parse_prometheus_text(input).unwrap();
        assert_eq!(families.len(), 1);
        assert_eq!(families[0].name, "vllm:prompt_tokens");
        assert_eq!(families[0].metric_type, MetricType::Counter);
        assert_eq!(families[0].samples[0].value, 123456.0);
    }

    #[test]
    fn test_parse_counter_total_normalization() {
        // Full counter output from Prometheus client: _total samples + _created as separate gauge
        let input = r#"# HELP vllm:request_success_total Count of successful requests
# TYPE vllm:request_success_total counter
vllm:request_success_total{finished_reason="stop"} 980.0
vllm:request_success_total{finished_reason="length"} 20.0
# HELP vllm:request_success_created Count of successful requests
# TYPE vllm:request_success_created gauge
vllm:request_success_created{finished_reason="stop"} 1.77e+09
vllm:request_success_created{finished_reason="length"} 1.77e+09
"#;
        let families = parse_prometheus_text(input).unwrap();
        assert_eq!(families.len(), 2);
        // Counter family normalized to base name
        assert_eq!(families[0].name, "vllm:request_success");
        assert_eq!(families[0].metric_type, MetricType::Counter);
        assert_eq!(families[0].samples.len(), 2);
        // Created gauge kept as-is
        assert_eq!(families[1].name, "vllm:request_success_created");
        assert_eq!(families[1].metric_type, MetricType::Gauge);
    }

    #[test]
    fn test_parse_histogram() {
        let input = r#"# HELP vllm:e2e_request_latency_seconds Histogram of e2e latency
# TYPE vllm:e2e_request_latency_seconds histogram
vllm:e2e_request_latency_seconds_bucket{model_name="example-model",le="0.1"} 10.0
vllm:e2e_request_latency_seconds_bucket{model_name="example-model",le="0.5"} 30.0
vllm:e2e_request_latency_seconds_bucket{model_name="example-model",le="1.0"} 45.0
vllm:e2e_request_latency_seconds_bucket{model_name="example-model",le="+Inf"} 50.0
vllm:e2e_request_latency_seconds_sum{model_name="example-model"} 25.5
vllm:e2e_request_latency_seconds_count{model_name="example-model"} 50.0
"#;
        let families = parse_prometheus_text(input).unwrap();
        assert_eq!(families.len(), 1);
        assert_eq!(families[0].metric_type, MetricType::Histogram);
        assert_eq!(families[0].samples.len(), 6);
        // Check +Inf bucket
        let inf_sample = &families[0].samples[3];
        assert_eq!(inf_sample.label("le"), Some("+Inf"));
        assert_eq!(inf_sample.value, 50.0);
    }

    #[test]
    fn test_parse_no_labels() {
        let input = "process_cpu_seconds_total 42.5\n";
        let families = parse_prometheus_text(input).unwrap();
        assert_eq!(families.len(), 1);
        assert_eq!(families[0].samples[0].value, 42.5);
        assert!(families[0].samples[0].labels.is_empty());
    }

    #[test]
    fn test_parse_escaped_label() {
        let input = r#"metric{label="foo\"bar"} 1.0"#;
        let families = parse_prometheus_text(input).unwrap();
        assert_eq!(families[0].samples[0].label("label"), Some("foo\"bar"));
    }

    #[test]
    fn test_parse_multiple_families() {
        let input = r#"# HELP a_gauge A gauge
# TYPE a_gauge gauge
a_gauge 1.0
# HELP b_counter A counter
# TYPE b_counter counter
b_counter_total 2.0
"#;
        let families = parse_prometheus_text(input).unwrap();
        assert_eq!(families.len(), 2);
        assert_eq!(families[0].name, "a_gauge");
        assert_eq!(families[1].name, "b_counter");
    }

    #[test]
    fn test_parse_http_instrumentator() {
        let input = r#"# HELP http_requests_total Total HTTP requests
# TYPE http_requests_total counter
http_requests_total{handler="/v1/chat/completions",method="POST",status="2xx"} 201.0
http_requests_total{handler="/v1/chat/completions",method="POST",status="4xx"} 5.0
"#;
        let families = parse_prometheus_text(input).unwrap();
        assert_eq!(families.len(), 1);
        assert_eq!(families[0].name, "http_requests"); // _total stripped
        assert_eq!(families[0].samples.len(), 2);
        assert_eq!(families[0].samples[0].label("status"), Some("2xx"));
    }
}
