// SPDX-License-Identifier: Apache-2.0

/// A snapshot of a Prometheus histogram at a point in time.
#[derive(Debug, Clone, Default)]
pub struct HistogramSnapshot {
    /// Cumulative buckets: (upper_bound, cumulative_count), sorted by bound ascending.
    pub buckets: Vec<(f64, u64)>,
    pub sum: f64,
    pub count: u64,
}

impl HistogramSnapshot {
    /// Estimate a percentile (0.0..1.0) using linear interpolation between bucket boundaries.
    /// Equivalent to Prometheus `histogram_quantile()`.
    pub fn percentile(&self, q: f64) -> f64 {
        if self.count == 0 || self.buckets.is_empty() {
            return 0.0;
        }

        let rank = q * self.count as f64;

        let mut prev_bound: f64 = 0.0;
        let mut prev_count: u64 = 0;

        for &(bound, cum_count) in &self.buckets {
            if bound.is_infinite() {
                // If we still haven't found the bucket, return the last finite bound
                break;
            }
            if cum_count as f64 >= rank {
                // Linear interpolation within this bucket
                let bucket_count = cum_count - prev_count;
                if bucket_count == 0 {
                    return bound;
                }
                let fraction = (rank - prev_count as f64) / bucket_count as f64;
                return prev_bound + fraction * (bound - prev_bound);
            }
            prev_bound = bound;
            prev_count = cum_count;
        }

        // Fell through — return last finite bound
        prev_bound
    }

    /// Mean value of the histogram.
    pub fn mean(&self) -> f64 {
        if self.count == 0 {
            0.0
        } else {
            self.sum / self.count as f64
        }
    }

    /// Compute the delta histogram between self (current) and prev (previous snapshot).
    /// Used to get percentiles for a recent time window rather than all-time.
    pub fn delta(&self, prev: &Self) -> Self {
        if self.count < prev.count {
            // Counter reset detected — use current as-is
            return self.clone();
        }

        // Bucket schema mismatch (process restarted with different histogram
        // config, or one side parsed a partial response). Index-wise zipping
        // against mismatched bounds would produce nonsense deltas; treat as a
        // reset.
        if self.buckets.len() != prev.buckets.len()
            || self.buckets.iter().zip(prev.buckets.iter()).any(|((a, _), (b, _))| a != b)
        {
            return self.clone();
        }

        let delta_count = self.count - prev.count;
        // Guard against sum going backwards (float precision or partial reset)
        let delta_sum = if self.sum >= prev.sum {
            self.sum - prev.sum
        } else {
            self.sum
        };

        let delta_buckets: Vec<(f64, u64)> = self
            .buckets
            .iter()
            .zip(prev.buckets.iter())
            .map(|(&(bound, cur), &(_, prev_cum))| {
                let d = if cur >= prev_cum { cur - prev_cum } else { cur };
                (bound, d)
            })
            .collect();

        Self {
            buckets: delta_buckets,
            sum: delta_sum,
            count: delta_count,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_histogram(buckets: &[(f64, u64)], sum: f64, count: u64) -> HistogramSnapshot {
        HistogramSnapshot {
            buckets: buckets.to_vec(),
            sum,
            count,
        }
    }

    #[test]
    fn test_percentile_basic() {
        // 10 requests in [0, 0.1], 20 more in (0.1, 0.5], 15 more in (0.5, 1.0], 5 in (1.0, +Inf]
        let h = make_histogram(
            &[(0.1, 10), (0.5, 30), (1.0, 45), (f64::INFINITY, 50)],
            25.5,
            50,
        );

        // p50 = 50th percentile: rank = 25, falls in the (0.1, 0.5] bucket
        let p50 = h.percentile(0.5);
        // rank=25, prev_count=10, bucket_count=20, fraction=(25-10)/20=0.75
        // value = 0.1 + 0.75 * 0.4 = 0.4
        assert!((p50 - 0.4).abs() < 0.001, "p50={p50}");

        // p90: rank = 45, falls in (0.5, 1.0] bucket
        let p90 = h.percentile(0.9);
        // rank=45, prev_count=30, bucket_count=15, fraction=(45-30)/15=1.0
        // value = 0.5 + 1.0 * 0.5 = 1.0
        assert!((p90 - 1.0).abs() < 0.001, "p90={p90}");

        // p99: rank = 49.5, falls in (+Inf) — should return last finite bound 1.0
        let p99 = h.percentile(0.99);
        assert!((p99 - 1.0).abs() < 0.001, "p99={p99}");
    }

    #[test]
    fn test_percentile_empty() {
        let h = HistogramSnapshot::default();
        assert_eq!(h.percentile(0.5), 0.0);
    }

    #[test]
    fn test_mean() {
        let h = make_histogram(&[(1.0, 10), (f64::INFINITY, 10)], 5.0, 10);
        assert!((h.mean() - 0.5).abs() < 0.001);
    }

    #[test]
    fn test_delta() {
        let prev = make_histogram(
            &[(0.1, 5), (0.5, 15), (1.0, 20), (f64::INFINITY, 20)],
            10.0,
            20,
        );
        let curr = make_histogram(
            &[(0.1, 10), (0.5, 30), (1.0, 45), (f64::INFINITY, 50)],
            25.5,
            50,
        );

        let delta = curr.delta(&prev);
        assert_eq!(delta.count, 30);
        assert!((delta.sum - 15.5).abs() < 0.001);
        assert_eq!(delta.buckets[0], (0.1, 5));
        assert_eq!(delta.buckets[1], (0.5, 15));
        assert_eq!(delta.buckets[2], (1.0, 25));
        assert_eq!(delta.buckets[3], (f64::INFINITY, 30));
    }

    #[test]
    fn test_delta_counter_reset() {
        let prev = make_histogram(&[(1.0, 100), (f64::INFINITY, 100)], 50.0, 100);
        let curr = make_histogram(&[(1.0, 5), (f64::INFINITY, 5)], 2.0, 5);
        let delta = curr.delta(&prev);
        // Should return curr as-is on counter reset
        assert_eq!(delta.count, 5);
    }

    #[test]
    fn test_delta_bucket_schema_change_falls_back_to_current() {
        // Process restart with a different bucket config: same/higher count
        // but different bounds. Index-wise zipping would produce garbage,
        // so we must treat it as a reset and return current as-is.
        let prev = make_histogram(&[(0.1, 10), (0.5, 20), (f64::INFINITY, 20)], 5.0, 20);
        let curr = make_histogram(&[(0.2, 30), (1.0, 50), (f64::INFINITY, 50)], 20.0, 50);
        let delta = curr.delta(&prev);
        assert_eq!(delta.count, 50);
        assert_eq!(delta.buckets[0], (0.2, 30));
    }

    #[test]
    fn test_delta_bucket_count_change_falls_back_to_current() {
        let prev = make_histogram(&[(1.0, 5), (f64::INFINITY, 5)], 2.0, 5);
        let curr = make_histogram(
            &[(0.1, 1), (1.0, 8), (10.0, 10), (f64::INFINITY, 10)],
            7.0,
            10,
        );
        let delta = curr.delta(&prev);
        assert_eq!(delta.count, 10);
        assert_eq!(delta.buckets.len(), 4);
    }
}
