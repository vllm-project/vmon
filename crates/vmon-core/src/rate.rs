// SPDX-License-Identifier: Apache-2.0

use tokio::time::Instant;

/// Computes the rate (per second) of a monotonic counter, handling counter resets.
#[derive(Debug, Default)]
pub struct RateCalc {
    prev_value: Option<u64>,
    prev_time: Option<Instant>,
}

impl RateCalc {
    pub fn new() -> Self {
        Self::default()
    }

    /// Feed a new counter value and timestamp. Returns the rate (per second)
    /// if a previous sample exists.
    pub fn update(&mut self, value: u64, now: Instant) -> Option<f64> {
        let result = match (self.prev_value, self.prev_time) {
            (Some(prev_val), Some(prev_time)) => {
                let dt = now.duration_since(prev_time).as_secs_f64();
                if dt < 0.01 {
                    // Too small an interval — skip to avoid rate spikes
                    return None;
                }
                // Handle counter reset: if value < prev, assume reset and use value as delta
                let delta = if value >= prev_val {
                    value - prev_val
                } else {
                    value
                };
                Some(delta as f64 / dt)
            }
            _ => None,
        };

        self.prev_value = Some(value);
        self.prev_time = Some(now);
        result
    }
}

/// Computes a ratio from two counters (e.g., cache hit rate = hits / queries).
#[derive(Debug, Default)]
pub struct RatioCalc {
    prev_numerator: Option<u64>,
    prev_denominator: Option<u64>,
    prev_ratio: Option<f64>,
}

impl RatioCalc {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn update(&mut self, numerator: u64, denominator: u64) -> Option<f64> {
        let result = match (self.prev_numerator, self.prev_denominator) {
            (Some(prev_n), Some(prev_d)) => {
                let dn = if numerator >= prev_n {
                    numerator - prev_n
                } else {
                    numerator
                };
                let dd = if denominator >= prev_d {
                    denominator - prev_d
                } else {
                    denominator
                };
                if dd == 0 {
                    // No new data — keep the previous ratio
                    self.prev_ratio
                } else {
                    let r = dn as f64 / dd as f64;
                    self.prev_ratio = Some(r);
                    Some(r)
                }
            }
            _ => None,
        };

        self.prev_numerator = Some(numerator);
        self.prev_denominator = Some(denominator);
        result
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::Duration;

    #[test]
    fn test_rate_basic() {
        let mut calc = RateCalc::new();
        let t0 = Instant::now();
        assert!(calc.update(100, t0).is_none());
        let t1 = t0 + Duration::from_secs(2);
        let rate = calc.update(200, t1).unwrap();
        assert!((rate - 50.0).abs() < 0.001); // (200-100)/2 = 50
    }

    #[test]
    fn test_rate_counter_reset() {
        let mut calc = RateCalc::new();
        let t0 = Instant::now();
        calc.update(1000, t0);
        let t1 = t0 + Duration::from_secs(1);
        // Counter reset: new value < old
        let rate = calc.update(50, t1).unwrap();
        assert!((rate - 50.0).abs() < 0.001); // uses 50 as delta
    }

    #[test]
    fn test_ratio() {
        let mut calc = RatioCalc::new();
        assert!(calc.update(10, 100).is_none());
        let ratio = calc.update(20, 200).unwrap();
        assert!((ratio - 0.1).abs() < 0.001); // (20-10)/(200-100) = 0.1
    }

    #[test]
    fn test_ratio_zero_denominator() {
        let mut calc = RatioCalc::new();
        calc.update(10, 100);
        assert!(calc.update(10, 100).is_none()); // delta denom = 0, no prior ratio
    }

    #[test]
    fn test_ratio_holds_when_idle() {
        let mut calc = RatioCalc::new();
        calc.update(0, 0);
        let r = calc.update(10, 100).unwrap();
        assert!((r - 0.1).abs() < 0.001);
        // No new data — should hold the previous ratio
        let r2 = calc.update(10, 100).unwrap();
        assert!((r2 - 0.1).abs() < 0.001);
    }
}
