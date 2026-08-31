//! Tauri commands backing the meeting detection settings screen.

use tauri::{AppHandle, Runtime, State};

use super::apps::{self, MeetingAppInfo};
use super::detector::{CaptureStreamInfo, MeetingDetectionStatus, MeetingDetector};
use super::settings::MeetingDetectionSettings;

/// Current settings, as persisted.
#[tauri::command]
pub async fn api_get_meeting_detection_settings(
    detector: State<'_, MeetingDetector>,
) -> Result<MeetingDetectionSettings, String> {
    Ok(detector.settings().await)
}

/// Persist new settings and restart the monitor loop to match. Returns the
/// settings as actually stored, which may differ from what was sent — out
/// of range intervals are clamped.
#[tauri::command]
pub async fn api_set_meeting_detection_settings<R: Runtime>(
    app: AppHandle<R>,
    detector: State<'_, MeetingDetector>,
    settings: MeetingDetectionSettings,
) -> Result<MeetingDetectionSettings, String> {
    detector.apply(&app, settings).await
}

/// Settings plus live state: whether the loop is running, what meeting it
/// thinks is happening, and the last probe error if there was one.
#[tauri::command]
pub async fn api_get_meeting_detection_status(
    detector: State<'_, MeetingDetector>,
) -> Result<MeetingDetectionStatus, String> {
    Ok(detector.status().await)
}

/// The known-app table, for rendering the per-app toggles.
#[tauri::command]
pub async fn api_list_meeting_apps() -> Result<Vec<MeetingAppInfo>, String> {
    Ok(apps::app_catalog())
}

/// What's holding the microphone right now, classified under current
/// settings. Lets the user check the detector's view of their system
/// without having to start a call to find out.
#[tauri::command]
pub async fn api_probe_capture_streams(
    detector: State<'_, MeetingDetector>,
) -> Result<Vec<CaptureStreamInfo>, String> {
    detector.probe().await
}
