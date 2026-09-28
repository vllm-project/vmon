// SPDX-License-Identifier: Apache-2.0

use ratatui::style::{Color, Modifier, Style};

// ── Base Palette ──

pub const HEADER_BG: Color = Color::Rgb(30, 34, 42);
pub const ACCENT: Color = Color::Rgb(97, 175, 239);
pub const HEALTHY: Color = Color::Rgb(152, 195, 121);
pub const WARNING: Color = Color::Rgb(229, 192, 123);
pub const DANGER: Color = Color::Rgb(224, 108, 117);
pub const MUTED: Color = Color::Rgb(92, 99, 112);
pub const TEXT: Color = Color::Rgb(171, 178, 191);
pub const BRIGHT: Color = Color::Rgb(220, 223, 228);
pub const BAR_EMPTY: Color = Color::Rgb(52, 58, 70);
pub const SELECTED_BG: Color = Color::Rgb(44, 49, 58);
pub const PURPLE: Color = Color::Rgb(198, 120, 221);

/// Palette for distinguishing SLURM jobs in the header and JOB column.
/// Excludes red (reserved for danger/error states) and gray-ish tones
/// (reserved for muted/inactive/ended jobs).
pub const JOB_COLORS: [Color; 6] = [
    Color::Rgb(97, 175, 239),  // blue
    Color::Rgb(152, 195, 121), // green
    Color::Rgb(229, 192, 123), // yellow
    Color::Rgb(198, 120, 221), // purple
    Color::Rgb(86, 182, 194),  // cyan
    Color::Rgb(232, 152, 90),  // orange
];

/// Stable color assignment for a SLURM job_id, picked from `JOB_COLORS`.
/// Uses a small FNV-ish hash so the same id keeps the same color across
/// renders, vmon restarts, and `vmon-tui` versions.
pub fn job_color(job_id: &str) -> Color {
    let mut h: u32 = 0;
    for b in job_id.bytes() {
        h = h.wrapping_mul(31).wrapping_add(b as u32);
    }
    JOB_COLORS[(h as usize) % JOB_COLORS.len()]
}

// ── Semantic Styles ──

pub fn bold() -> Style {
    Style::default().fg(BRIGHT).add_modifier(Modifier::BOLD)
}

pub fn muted() -> Style {
    Style::default().fg(MUTED)
}

pub fn accent() -> Style {
    Style::default().fg(ACCENT).add_modifier(Modifier::BOLD)
}

pub fn label() -> Style {
    Style::default().fg(ACCENT)
}

// ── Load-Based Colors ──

pub fn load_color(pct: f64) -> Color {
    if pct > 90.0 {
        DANGER
    } else if pct > 70.0 {
        WARNING
    } else {
        HEALTHY
    }
}

pub fn latency_color(ms: f64) -> Color {
    if ms > 500.0 {
        DANGER
    } else if ms > 100.0 {
        WARNING
    } else {
        HEALTHY
    }
}

pub fn wait_color(count: f64) -> Color {
    if count > 50.0 {
        DANGER
    } else if count > 20.0 {
        WARNING
    } else {
        TEXT
    }
}

/// Color an HTTP error rate (0.0–1.0). 5%+ is red, 0.5%+ yellow, else neutral.
/// Pure-zero stays neutral so a healthy steady state doesn't paint green noise.
pub fn error_rate_color(ratio: f64) -> Color {
    let pct = ratio * 100.0;
    if pct >= 5.0 {
        DANGER
    } else if pct >= 0.5 {
        WARNING
    } else {
        TEXT
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn job_color_is_stable_for_same_id() {
        assert_eq!(job_color("1001"), job_color("1001"));
        assert_eq!(job_color("nova-7"), job_color("nova-7"));
    }

    #[test]
    fn job_color_picks_from_palette() {
        let c = job_color("1002");
        assert!(JOB_COLORS.contains(&c));
    }

    #[test]
    fn job_color_distinguishes_at_least_two_close_ids() {
        // Sequential job ids should not all collide; at least one pair differs.
        let colors: Vec<Color> = (1000..1008).map(|i| job_color(&i.to_string())).collect();
        let unique = colors.iter().collect::<std::collections::HashSet<_>>();
        assert!(
            unique.len() >= 2,
            "expected ≥2 distinct colors, got {colors:?}"
        );
    }
}
