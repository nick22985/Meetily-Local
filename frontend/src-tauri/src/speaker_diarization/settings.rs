//! User-facing speaker diarization settings: which embedding model to use and
//! whether mic and system voices may share a speaker.
//!
//! Both are read when a diarizer is built (see `commands::build_diarizer`),
//! so a change applies from the next recording or batch job. The model choice
//! is also mirrored into [`model::set_active_model`] so path lookups, the
//! download command and enrollment all follow it.

use std::sync::atomic::{AtomicBool, Ordering};

use serde::{Deserialize, Serialize};
use tauri::{AppHandle, Runtime};

use super::model::{self, EMBEDDING_MODELS};
use crate::database::repositories::setting::{SettingsRepository, KEY_SPEAKER_DIARIZATION};
use crate::meeting_detector::settings::db_pool;

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct SpeakerDiarizationSettings {
    /// Filename of the active embedding model (one of [`EMBEDDING_MODELS`]).
    pub embedding_model: String,
    /// Never put mic and system segments under the same speaker. Right with
    /// headphones; wrong when the mic also hears the room's speakers.
    pub separate_sources: bool,
}

impl Default for SpeakerDiarizationSettings {
    fn default() -> Self {
        Self {
            embedding_model: model::DEFAULT_MODEL_FILENAME.to_string(),
            separate_sources: false,
        }
    }
}

static SEPARATE_SOURCES: AtomicBool = AtomicBool::new(false);

pub fn separate_sources() -> bool {
    SEPARATE_SOURCES.load(Ordering::Relaxed)
}

/// Load the saved settings and apply them. Called once during app setup,
/// after the database and the models directory are ready.
pub async fn initialize<R: Runtime>(app: &AppHandle<R>) {
    apply(&load(app).await);
}

fn apply(settings: &SpeakerDiarizationSettings) {
    model::set_active_model(&settings.embedding_model);
    SEPARATE_SOURCES.store(settings.separate_sources, Ordering::Relaxed);
}

async fn load<R: Runtime>(app: &AppHandle<R>) -> SpeakerDiarizationSettings {
    let Some(pool) = db_pool(app) else {
        return SpeakerDiarizationSettings::default();
    };
    match SettingsRepository::get_setting::<SpeakerDiarizationSettings>(
        &pool,
        KEY_SPEAKER_DIARIZATION,
    )
    .await
    {
        Ok(settings) => settings.unwrap_or_default(),
        Err(e) => {
            log::warn!("Failed to load speaker diarization settings: {}", e);
            SpeakerDiarizationSettings::default()
        }
    }
}

#[tauri::command]
pub async fn api_get_speaker_diarization_settings<R: Runtime>(
    app: AppHandle<R>,
) -> Result<SpeakerDiarizationSettings, String> {
    Ok(load(&app).await)
}

#[tauri::command]
pub async fn api_set_speaker_diarization_settings<R: Runtime>(
    app: AppHandle<R>,
    settings: SpeakerDiarizationSettings,
) -> Result<(), String> {
    if model::embedding_model_info(&settings.embedding_model).is_none() {
        return Err(format!("Unknown embedding model: {}", settings.embedding_model));
    }
    let pool = db_pool(&app).ok_or_else(|| "Database is not ready yet".to_string())?;
    SettingsRepository::set_setting(&pool, KEY_SPEAKER_DIARIZATION, &settings)
        .await
        .map_err(|e| format!("Failed to save speaker diarization settings: {}", e))?;
    apply(&settings);
    Ok(())
}

#[derive(Debug, Serialize)]
pub struct EmbeddingModelDto {
    pub filename: String,
    pub display_name: String,
    pub size_mb: u32,
    pub description: String,
    pub downloaded: bool,
    pub active: bool,
}

#[tauri::command]
pub async fn api_list_speaker_embedding_models() -> Result<Vec<EmbeddingModelDto>, String> {
    let active = model::model_filename();
    Ok(EMBEDDING_MODELS
        .iter()
        .map(|m| EmbeddingModelDto {
            filename: m.filename.to_string(),
            display_name: m.display_name.to_string(),
            size_mb: m.size_mb,
            description: m.description.to_string(),
            downloaded: model::embedding_model_path(m.filename)
                .map_or(false, |p| model::model_is_ready(&p)),
            active: m.filename == active,
        })
        .collect())
}
