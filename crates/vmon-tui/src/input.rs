// SPDX-License-Identifier: Apache-2.0

use crossterm::event::{KeyCode, KeyEvent, KeyModifiers};

pub enum Action {
    Quit,
    Up,
    Down,
    ToggleDetail,
    ToggleEngines,
    TabNext,
    TabPrev,
    ScrollDetailUp,
    ScrollDetailDown,
    EnterSearch,
    ExitSearch,
    SearchInput(char),
    SearchBackspace,
    /// Toggle sleep/wake on selected node (dev mode).
    ToggleSleep,
    /// Reset prefix cache on selected node (dev mode).
    ResetPrefixCache,
    /// Toggle real-time throughput graph view.
    ToggleGraph,
    /// Confirm a pending action.
    Confirm,
    /// Cancel a pending action.
    Cancel,
    /// Replay: toggle pause/resume.
    ReplayPause,
    /// Replay: increase playback speed.
    ReplayFaster,
    /// Replay: decrease playback speed.
    ReplaySlower,
    /// Replay: step forward one sample (when paused).
    ReplayStepForward,
    /// Replay: step backward one sample (when paused).
    ReplayStepBackward,
    /// Enter config diff mode (select second node to compare).
    DiffMode,
    /// Start a collect session, or toggle pause/resume on an active one.
    CollectToggle,
    /// Stop the active collect session and write the JSON file.
    CollectStop,
    None,
}

/// When in search mode, route keys to search input; otherwise normal mode.
/// `confirm_pending` is true when a confirmation prompt is shown.
/// `replay_mode` is true when replaying a collected report.
pub fn handle_key(
    key: KeyEvent,
    search_active: bool,
    confirm_pending: bool,
    replay_mode: bool,
) -> Action {
    // Confirmation dialog takes priority
    if confirm_pending {
        return match key.code {
            KeyCode::Char('y') | KeyCode::Char('Y') => Action::Confirm,
            _ => Action::Cancel,
        };
    }

    if search_active {
        return match key.code {
            KeyCode::Esc => Action::ExitSearch,
            KeyCode::Enter => Action::ExitSearch,
            KeyCode::Backspace => Action::SearchBackspace,
            KeyCode::Char(c) => Action::SearchInput(c),
            KeyCode::Up => Action::ScrollDetailUp,
            KeyCode::Down => Action::ScrollDetailDown,
            _ => Action::None,
        };
    }

    match key.code {
        KeyCode::Char('q') => Action::Quit,
        KeyCode::Char('c') if key.modifiers.contains(KeyModifiers::CONTROL) => Action::Quit,
        KeyCode::Up | KeyCode::Char('k') => Action::Up,
        KeyCode::Down | KeyCode::Char('j') => Action::Down,
        KeyCode::Enter => Action::ToggleDetail,
        // Space folds engines in both modes (muscle memory from live mode).
        // Pause moved to `p` in replay so it doesn't shadow the fold key.
        KeyCode::Char(' ') => Action::ToggleEngines,
        KeyCode::Char('p') if replay_mode => Action::ReplayPause,
        KeyCode::Tab | KeyCode::Char('l') => Action::TabNext,
        KeyCode::BackTab | KeyCode::Char('h') => Action::TabPrev,
        KeyCode::Char(']') => Action::ScrollDetailDown,
        KeyCode::Char('[') => Action::ScrollDetailUp,
        KeyCode::Char('/') => Action::EnterSearch,
        KeyCode::Esc => Action::ExitSearch,
        KeyCode::Char('>') | KeyCode::Char('.') if replay_mode => Action::ReplayFaster,
        KeyCode::Char('<') | KeyCode::Char(',') if replay_mode => Action::ReplaySlower,
        KeyCode::Char('n') if replay_mode => Action::ReplayStepForward,
        KeyCode::Char('N') if replay_mode => Action::ReplayStepBackward,
        KeyCode::Char('d') => Action::DiffMode,
        KeyCode::Char('s') if !replay_mode => Action::ToggleSleep,
        KeyCode::Char('R') if !replay_mode => Action::ResetPrefixCache,
        KeyCode::Char('c') if !replay_mode => Action::CollectToggle,
        KeyCode::Char('C') if !replay_mode => Action::CollectStop,
        KeyCode::Char('g') => Action::ToggleGraph,
        _ => Action::None,
    }
}
