//! Persisted configuration for automatic meeting detection.

use serde::{Deserialize, Serialize};
use sqlx::SqlitePool;
use tauri::{AppHandle, Manager, Runtime};

use crate::database::repositories::setting::{SettingsRepository, KEY_MEETING_DETECTION};
use crate::state::AppState;

/// User-facing knobs for the detector.
///
/// Everything is off until the user opts in: watching what holds the
/// microphone and starting a recorder on your behalf is not something to
/// enable behind somebody's back.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct MeetingDetectionSettings {
    /// Master switch. When false the monitor loop doesn't run at all.
    pub enabled: bool,
    /// Start recording when a meeting is detected.
    pub auto_start_recording: bool,
    /// Stop recording when the meeting ends. Only ever stops a recording
    /// the detector itself started.
    pub auto_stop_recording: bool,
    /// Show a desktop notification when a meeting is detected.
    pub notify_on_detection: bool,
    /// Keys from [`super::apps::KNOWN_MEETING_APPS`] to ignore.
    ///
    /// A deny list rather than an allow list, so apps added to the table in
    /// later versions work without the user revisiting this screen. Seeded
    /// from the table's `default_ignored` entries — see the note there on
    /// why a voice channel isn't a meeting.
    pub ignored_apps: Vec<String>,
    /// Treat *any* application holding the mic as a meeting, not just the
    /// known ones. Off by default — this also fires for voice memos, games,
    /// and dictation.
    pub detect_unknown_apps: bool,
    /// How often to look at the audio graph.
    pub poll_interval_secs: u64,
    /// How long an app must hold the mic before it counts as a meeting.
    /// Filters out the momentary grabs apps make while testing devices.
    pub start_delay_secs: u64,
    /// How long the mic must stay free before the meeting counts as over.
    /// Covers the gap when someone leaves one call and joins another.
    pub stop_grace_secs: u64,
}

impl Default for MeetingDetectionSettings {
    fn default() -> Self {
        Self {
            enabled: false,
            auto_start_recording: true,
            auto_stop_recording: true,
            notify_on_detection: true,
            ignored_apps: super::apps::default_ignored_keys(),
            detect_unknown_apps: false,
            poll_interval_secs: 5,
            start_delay_secs: 10,
            stop_grace_secs: 30,
        }
    }
}

impl MeetingDetectionSettings {
    /// Clamp anything the frontend could send that would make the monitor
    /// loop misbehave (a zero interval spins the CPU; a huge one makes the
    /// feature look broken).
    pub fn sanitized(mut self) -> Self {
        self.poll_interval_secs = self.poll_interval_secs.clamp(2, 60);
        self.start_delay_secs = self.start_delay_secs.min(300);
        self.stop_grace_secs = self.stop_grace_secs.min(600);
        self.ignored_apps.sort();
        self.ignored_apps.dedup();
        self
    }

    /// Whether `app_key` (or an unrecognized app, passed as `None`) should
    /// be treated as a meeting.
    pub fn accepts(&self, app_key: Option<&str>) -> bool {
        match app_key {
            Some(key) => !self.ignored_apps.iter().any(|k| k == key),
            None => self.detect_unknown_apps,
        }
    }
}

/// Fetch the SQLite pool, if `AppState` has been managed yet. `None` early
/// in startup, before the database has been initialized.
pub fn db_pool<R: Runtime>(app: &AppHandle<R>) -> Option<SqlitePool> {
    app.try_state::<AppState>()
        .map(|s| s.db_manager.pool().clone())
}

/// Load the saved settings, falling back to defaults when nothing has been
/// saved yet or the database isn't up.
pub async fn load<R: Runtime>(app: &AppHandle<R>) -> MeetingDetectionSettings {
    let Some(pool) = db_pool(app) else {
        return MeetingDetectionSettings::default();
    };
    match SettingsRepository::get_setting::<MeetingDetectionSettings>(&pool, KEY_MEETING_DETECTION)
        .await
    {
        Ok(Some(settings)) => settings.sanitized(),
        Ok(None) => MeetingDetectionSettings::default(),
        Err(e) => {
            log::warn!("Failed to load meeting detection settings: {}", e);
            MeetingDetectionSettings::default()
        }
    }
}

/// Persist settings.
pub async fn save<R: Runtime>(
    app: &AppHandle<R>,
    settings: &MeetingDetectionSettings,
) -> Result<(), String> {
    let pool = db_pool(app).ok_or_else(|| "Database is not ready yet".to_string())?;
    SettingsRepository::set_setting(&pool, KEY_MEETING_DETECTION, settings)
        .await
        .map_err(|e| format!("Failed to save meeting detection settings: {}", e))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn defaults_are_opt_in() {
        let settings = MeetingDetectionSettings::default();
        assert!(!settings.enabled);
        assert!(!settings.detect_unknown_apps);
    }

    #[test]
    fn sanitize_keeps_the_poll_interval_usable() {
        let mut settings = MeetingDetectionSettings::default();
        settings.poll_interval_secs = 0;
        assert_eq!(settings.sanitized().poll_interval_secs, 2);

        let mut settings = MeetingDetectionSettings::default();
        settings.poll_interval_secs = 10_000;
        assert_eq!(settings.sanitized().poll_interval_secs, 60);
    }

    #[test]
    fn ignored_apps_are_rejected_and_everything_else_accepted() {
        let mut settings = MeetingDetectionSettings::default();
        settings.ignored_apps = vec!["discord".to_string()];

        assert!(!settings.accepts(Some("discord")));
        assert!(settings.accepts(Some("zoom")));
    }

    #[test]
    fn unknown_apps_follow_their_own_toggle() {
        let mut settings = MeetingDetectionSettings::default();
        assert!(!settings.accepts(None));

        settings.detect_unknown_apps = true;
        assert!(settings.accepts(None));
    }

    #[test]
    fn apps_that_hold_the_mic_all_day_are_ignored_out_of_the_box() {
        assert!(!MeetingDetectionSettings::default().accepts(Some("discord")));
    }

    #[test]
    fn an_explicitly_saved_empty_ignore_list_is_honored() {
        // Un-ignoring Discord has to survive a reload — the seeded default
        // only applies when nothing has been saved.
        let settings: MeetingDetectionSettings =
            serde_json::from_str(r#"{"ignored_apps":[]}"#).expect("should deserialize");
        assert!(settings.ignored_apps.is_empty());
        assert!(settings.accepts(Some("discord")));
    }

    #[test]
    fn missing_fields_deserialize_to_defaults() {
        // Settings saved by an older build won't have newer fields.
        let settings: MeetingDetectionSettings =
            serde_json::from_str(r#"{"enabled":true}"#).expect("should deserialize");
        assert!(settings.enabled);
        assert_eq!(settings.poll_interval_secs, 5);
    }
}
