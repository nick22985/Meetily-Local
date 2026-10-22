//! What happens when the window manager asks the main window to close
//! (title-bar X, Super+Q, Alt+F4, ...).
//!
//! By default a close request closes the window and exits the app. With
//! `hide_on_close` on, it's trapped and the window is hidden to the tray
//! instead — the tray's Open item brings it back, and its Quit item exits
//! directly without raising a close request.
//!
//! Hidden rather than minimized: Hyprland (and some other tiling
//! compositors) ignore minimize requests, so minimizing would leave the
//! window where it is.

use std::sync::atomic::{AtomicBool, Ordering};

use serde::{Deserialize, Serialize};
use tauri::{AppHandle, Runtime, Window, WindowEvent};

use crate::database::repositories::setting::{SettingsRepository, KEY_WINDOW_BEHAVIOR};
use crate::meeting_detector::settings::db_pool;

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
#[serde(default)]
pub struct WindowBehaviorSettings {
    /// Hide to the tray instead of quitting when the window is closed.
    pub hide_on_close: bool,
}

/// Mirror of `hide_on_close`, read from the window event handler, which
/// is synchronous and can't wait on the database.
static HIDE_ON_CLOSE: AtomicBool = AtomicBool::new(false);

/// Load the saved settings and mirror them into the close-request flag.
/// Called once during app setup, after the database is ready.
pub async fn initialize<R: Runtime>(app: &AppHandle<R>) {
    let settings = load(app).await;
    HIDE_ON_CLOSE.store(settings.hide_on_close, Ordering::Relaxed);
}

async fn load<R: Runtime>(app: &AppHandle<R>) -> WindowBehaviorSettings {
    let Some(pool) = db_pool(app) else {
        return WindowBehaviorSettings::default();
    };
    match SettingsRepository::get_setting::<WindowBehaviorSettings>(&pool, KEY_WINDOW_BEHAVIOR)
        .await
    {
        Ok(settings) => settings.unwrap_or_default(),
        Err(e) => {
            log::warn!("Failed to load window behavior settings: {}", e);
            WindowBehaviorSettings::default()
        }
    }
}

/// Window event hook: turns a close request into a hide when enabled.
pub fn on_window_event<R: Runtime>(window: &Window<R>, event: &WindowEvent) {
    if let WindowEvent::CloseRequested { api, .. } = event {
        if HIDE_ON_CLOSE.load(Ordering::Relaxed) {
            api.prevent_close();
            if let Err(e) = window.hide() {
                log::warn!("Failed to hide window on close: {}", e);
            }
        }
    }
}

#[tauri::command]
pub async fn api_get_window_behavior_settings<R: Runtime>(
    app: AppHandle<R>,
) -> Result<WindowBehaviorSettings, String> {
    Ok(load(&app).await)
}

#[tauri::command]
pub async fn api_set_window_behavior_settings<R: Runtime>(
    app: AppHandle<R>,
    settings: WindowBehaviorSettings,
) -> Result<(), String> {
    let pool = db_pool(&app).ok_or_else(|| "Database is not ready yet".to_string())?;
    SettingsRepository::set_setting(&pool, KEY_WINDOW_BEHAVIOR, &settings)
        .await
        .map_err(|e| format!("Failed to save window behavior settings: {}", e))?;
    HIDE_ON_CLOSE.store(settings.hide_on_close, Ordering::Relaxed);
    Ok(())
}
