// audio/transcription/engine.rs
//
// TranscriptionEngine enum and model initialization/validation logic.
//
// Two backends are supported, selected by the saved transcript provider:
//   * `localWhisper` — whisper-rs in this process (`whisper_engine`).
//   * `remoteWhisper` — an OpenAI-compatible endpoint on another machine
//     (`super::remote`), for hosts whose GPU can't keep up with realtime.
//
// Everything downstream of `TranscriptionEngine::transcribe_audio_with_confidence`
// is backend-agnostic: capture, mixing, VAD, diarization, and persistence all
// stay local either way — only the decode moves.

use anyhow::anyhow;
use log::{info, warn};
use std::sync::Arc;
use tauri::{AppHandle, Manager, Runtime};

use super::remote::{
    active_remote, set_active_remote, RemoteWhisperClient, RemoteWhisperConfig,
    REMOTE_WHISPER_PROVIDER, REMOTE_WHISPER_SETTING_KEY,
};

// Transcription engine abstraction.
pub enum TranscriptionEngine {
    Whisper(Arc<crate::whisper_engine::WhisperEngine>), // Local Whisper (direct access)
    Remote(Arc<RemoteWhisperClient>),                   // Offloaded to another machine over HTTP
}

impl TranscriptionEngine {
    pub async fn is_model_loaded(&self) -> bool {
        match self {
            Self::Whisper(engine) => engine.is_model_loaded().await,
            // The remote server owns its own model lifecycle; readiness is
            // established by `health_check` before recording starts rather
            // than polled per chunk.
            Self::Remote(_) => true,
        }
    }

    pub async fn get_current_model(&self) -> Option<String> {
        match self {
            Self::Whisper(engine) => engine.get_current_model().await,
            Self::Remote(client) => Some(client.model().to_string()),
        }
    }

    pub fn provider_name(&self) -> &str {
        match self {
            Self::Whisper(_) => "Whisper (direct)",
            Self::Remote(_) => "Whisper (remote)",
        }
    }

    /// Transcribe one chunk of f32 mono 16 kHz audio.
    ///
    /// Returns `(text, confidence, is_partial)`. Both backends produce the
    /// same confidence semantics — see
    /// [`RemoteWhisperClient::transcribe_audio_with_confidence`] for how the
    /// remote path reconstructs them from the wire response.
    pub async fn transcribe_audio_with_confidence(
        &self,
        audio_data: Vec<f32>,
        language: Option<String>,
        context_prompt: Option<String>,
    ) -> anyhow::Result<(String, f32, bool)> {
        match self {
            Self::Whisper(engine) => {
                engine
                    .transcribe_audio_with_confidence(audio_data, language, context_prompt)
                    .await
            }
            Self::Remote(client) => {
                client
                    .transcribe_audio_with_confidence(audio_data, language, context_prompt)
                    .await
            }
        }
    }
}

impl Clone for TranscriptionEngine {
    fn clone(&self) -> Self {
        match self {
            Self::Whisper(engine) => Self::Whisper(engine.clone()),
            Self::Remote(client) => Self::Remote(client.clone()),
        }
    }
}

/// Read the saved transcript provider, defaulting to local Whisper.
pub async fn configured_provider<R: Runtime>(app: &AppHandle<R>) -> String {
    match crate::api::api::api_get_transcript_config(app.clone(), app.clone().state()).await {
        Ok(Some(config)) => config.provider,
        Ok(None) => "localWhisper".to_string(),
        Err(e) => {
            warn!(
                "⚠️ Failed to read transcript config ({}); assuming local Whisper",
                e
            );
            "localWhisper".to_string()
        }
    }
}

/// Load the saved remote-endpoint settings.
pub async fn load_remote_config<R: Runtime>(
    app: &AppHandle<R>,
) -> Result<RemoteWhisperConfig, String> {
    let state = app
        .try_state::<crate::state::AppState>()
        .ok_or_else(|| "App state not available".to_string())?;

    crate::database::repositories::setting::SettingsRepository::get_setting::<RemoteWhisperConfig>(
        &state.db_manager.pool(),
        REMOTE_WHISPER_SETTING_KEY,
    )
    .await
    .map_err(|e| format!("Failed to read remote transcription settings: {}", e))?
    .ok_or_else(|| {
        "No remote transcription endpoint is configured. Add one in Settings → Transcription."
            .to_string()
    })
}

/// Build a remote client from the saved settings.
pub async fn build_remote_client<R: Runtime>(
    app: &AppHandle<R>,
) -> Result<Arc<RemoteWhisperClient>, String> {
    let config = load_remote_config(app).await?;
    RemoteWhisperClient::new(config)
        .map(Arc::new)
        .map_err(|e| e.to_string())
}

/// The backend the current session is transcribing with, without touching
/// the database.
///
/// Used by code that can't be handed an engine directly — notably the
/// streaming partial-decode task, which runs alongside the worker and would
/// otherwise hit the settings table on every decode. Returns `None` before
/// [`get_or_init_transcription_engine`] has run for the session.
pub fn current_transcription_engine() -> Option<TranscriptionEngine> {
    if let Some(client) = active_remote() {
        return Some(TranscriptionEngine::Remote(client));
    }
    let engine = {
        let guard = crate::whisper_engine::commands::WHISPER_ENGINE
            .lock()
            .unwrap_or_else(|e| e.into_inner());
        guard.as_ref().cloned()
    };
    engine.map(TranscriptionEngine::Whisper)
}

/// Validate that the configured transcription backend is ready before
/// recording starts.
///
/// For local Whisper that means the model file is present and loadable; for
/// a remote endpoint it means the server answers a real transcription
/// request. Both failures surface to the user before any audio is captured,
/// rather than as a silently empty transcript afterwards.
pub async fn validate_transcription_model_ready<R: Runtime>(
    app: &AppHandle<R>,
) -> Result<(), String> {
    if configured_provider(app).await == REMOTE_WHISPER_PROVIDER {
        info!("🔍 Validating remote transcription endpoint...");
        let client = build_remote_client(app).await?;
        return match client.health_check().await {
            Ok(()) => {
                info!(
                    "✅ Remote transcription endpoint reachable: {}",
                    client.base_url()
                );
                Ok(())
            }
            Err(e) => {
                warn!("❌ Remote transcription validation failed: {}", e);
                Err(e.to_string())
            }
        };
    }

    info!("🔍 Validating Whisper model...");

    if let Err(init_error) = crate::whisper_engine::commands::whisper_init().await {
        warn!("❌ Failed to initialize Whisper engine: {}", init_error);
        return Err(format!(
            "Failed to initialize speech recognition: {}",
            init_error
        ));
    }

    match crate::whisper_engine::commands::whisper_validate_model_ready_with_config(app).await {
        Ok(model_name) => {
            info!(
                "✅ Whisper model validation successful: {} is ready",
                model_name
            );
            Ok(())
        }
        Err(e) => {
            warn!("❌ Whisper model validation failed: {}", e);
            Err(e)
        }
    }
}

/// Get or initialize the transcription engine for live recording, honouring
/// the saved provider.
///
/// Also installs (or clears) the session's active remote backend, so the
/// partial-decode task picks up the same choice — see
/// [`current_transcription_engine`].
pub async fn get_or_init_transcription_engine<R: Runtime>(
    app: &AppHandle<R>,
) -> Result<TranscriptionEngine, String> {
    if configured_provider(app).await == REMOTE_WHISPER_PROVIDER {
        let client = build_remote_client(app).await?;
        info!(
            "🌐 Using remote transcription engine at {} (model '{}')",
            client.base_url(),
            client.model()
        );
        set_active_remote(Some(client.clone()));
        return Ok(TranscriptionEngine::Remote(client));
    }

    set_active_remote(None);
    info!("🎤 Initializing Whisper transcription engine");
    let whisper_engine = get_or_init_whisper(app).await?;
    Ok(TranscriptionEngine::Whisper(whisper_engine))
}

/// Get or initialize a transcription engine for a batch job (import or
/// retranscription).
///
/// `requested_provider`/`requested_model` come from the dialog's model
/// picker; when they're absent the saved config decides. Unlike the live
/// path this does not install a session-wide remote backend — batch jobs are
/// one-shot and must not disturb an in-progress recording's choice.
pub async fn get_or_init_batch_engine<R: Runtime>(
    app: &AppHandle<R>,
    requested_provider: Option<&str>,
    requested_model: Option<&str>,
) -> anyhow::Result<TranscriptionEngine> {
    let use_remote = match requested_provider {
        Some(provider) => provider == REMOTE_WHISPER_PROVIDER || provider == "remote",
        None => configured_provider(app).await == REMOTE_WHISPER_PROVIDER,
    };

    if use_remote {
        let mut config = load_remote_config(app).await.map_err(|e| anyhow!(e))?;
        // The picker offers the configured remote model by name; honour an
        // explicit override so a batch job can target a bigger model than
        // the one used for realtime.
        if let Some(model) = requested_model.filter(|m| !m.is_empty()) {
            config.model = model.to_string();
        }
        let client = Arc::new(RemoteWhisperClient::new(config)?);
        info!(
            "🌐 Batch transcription using remote engine at {} (model '{}')",
            client.base_url(),
            client.model()
        );
        return Ok(TranscriptionEngine::Remote(client));
    }

    Ok(TranscriptionEngine::Whisper(
        load_local_model_for_batch(app, requested_model).await?,
    ))
}

/// Ensure the local Whisper engine has `requested_model` (or the configured
/// one) loaded, and return it.
///
/// Shared by audio import and retranscription, which previously kept
/// byte-identical private copies of this. Unlike the live path's
/// [`get_or_init_whisper`], a batch caller may name a model explicitly —
/// the dialogs let the user pick one per job.
async fn load_local_model_for_batch<R: Runtime>(
    app: &AppHandle<R>,
    requested_model: Option<&str>,
) -> anyhow::Result<Arc<crate::whisper_engine::WhisperEngine>> {
    let engine = {
        let guard = crate::whisper_engine::commands::WHISPER_ENGINE
            .lock()
            .unwrap_or_else(|e| e.into_inner());
        guard.as_ref().cloned()
    };
    let engine = engine.ok_or_else(|| anyhow!("Whisper engine not initialized"))?;

    let target_model = match requested_model.filter(|m| !m.is_empty()) {
        Some(model) => model.to_string(),
        None => configured_local_model(app).await,
    };

    let current_model = engine.get_current_model().await;
    if current_model.as_deref() == Some(target_model.as_str()) {
        info!("Whisper model '{}' already loaded", target_model);
        return Ok(engine);
    }

    info!(
        "Loading Whisper model '{}' (current: {:?})",
        target_model, current_model
    );

    // Populates the internal model cache; a failure here isn't fatal on its
    // own — load_model reports the real problem.
    if let Err(e) = engine.discover_models().await {
        warn!("Model discovery error (continuing): {}", e);
    }

    engine
        .load_model(&target_model)
        .await
        .map_err(|e| anyhow!("Failed to load Whisper model '{}': {}", target_model, e))?;

    Ok(engine)
}

/// The saved local Whisper model name, or [`DEFAULT_WHISPER_MODEL`] when
/// nothing is saved or the saved provider isn't a local one.
///
/// [`DEFAULT_WHISPER_MODEL`]: crate::config::DEFAULT_WHISPER_MODEL
pub(crate) async fn configured_local_model<R: Runtime>(app: &AppHandle<R>) -> String {
    let default = crate::config::DEFAULT_WHISPER_MODEL.to_string();

    let Some(state) = app.try_state::<crate::state::AppState>() else {
        warn!("App state not available; using default Whisper model");
        return default;
    };

    let result: Result<Option<(String, String)>, _> =
        sqlx::query_as("SELECT provider, model FROM transcript_settings WHERE id = '1'")
            .fetch_optional(state.db_manager.pool())
            .await;

    match result {
        Ok(Some((provider, model)))
            if (provider == "localWhisper" || provider == "whisper") && !model.is_empty() =>
        {
            model
        }
        Ok(_) => default,
        Err(e) => {
            warn!(
                "Failed to query transcript config ({}); using default model",
                e
            );
            default
        }
    }
}

/// Get or initialize the Whisper engine, loading the model from saved config.
pub async fn get_or_init_whisper<R: Runtime>(
    app: &AppHandle<R>,
) -> Result<Arc<crate::whisper_engine::WhisperEngine>, String> {
    // Check if engine already exists and has a model loaded
    let existing_engine = {
        let engine_guard = crate::whisper_engine::commands::WHISPER_ENGINE
            .lock()
            .unwrap();
        engine_guard.as_ref().cloned()
    };

    if let Some(engine) = existing_engine {
        if engine.is_model_loaded().await {
            let current_model = engine
                .get_current_model()
                .await
                .unwrap_or_else(|| "unknown".to_string());

            // Check if loaded model matches saved config; reload if not.
            let configured_model =
                match crate::api::api::api_get_transcript_config(app.clone(), app.clone().state())
                    .await
                {
                    Ok(Some(config)) => {
                        info!(
                            "📝 Saved transcript config - provider: {}, model: {}",
                            config.provider, config.model
                        );
                        if config.provider == "localWhisper" && !config.model.is_empty() {
                            Some(config.model)
                        } else {
                            None
                        }
                    }
                    Ok(None) => None,
                    Err(e) => {
                        warn!("⚠️ Failed to get transcript config: {}", e);
                        None
                    }
                };

            if let Some(ref expected_model) = configured_model {
                if current_model == *expected_model {
                    info!(
                        "✅ Loaded model '{}' matches saved config, reusing",
                        current_model
                    );
                    return Ok(engine);
                } else {
                    info!(
                        "🔄 Loaded model '{}' doesn't match saved config '{}', reloading...",
                        current_model, expected_model
                    );
                    engine.unload_model().await;
                }
            } else {
                info!(
                    "✅ No specific model configured, using currently loaded model: '{}'",
                    current_model
                );
                return Ok(engine);
            }
        } else {
            info!("🔄 Whisper engine exists but no model loaded, will load model from config");
        }
    }

    info!("Initializing Whisper engine");
    if let Err(e) = crate::whisper_engine::commands::whisper_init().await {
        return Err(format!("Failed to initialize Whisper engine: {}", e));
    }

    let engine = {
        let engine_guard = crate::whisper_engine::commands::WHISPER_ENGINE
            .lock()
            .unwrap();
        engine_guard
            .as_ref()
            .cloned()
            .ok_or("Failed to get initialized engine")?
    };

    let model_to_load = match crate::api::api::api_get_transcript_config(
        app.clone(),
        app.clone().state(),
    )
    .await
    {
        Ok(Some(config)) => {
            info!(
                "Got transcript config from API - provider: {}, model: {}",
                config.provider, config.model
            );
            if config.provider == "localWhisper" {
                config.model
            } else {
                return Err(format!(
                    "Cannot initialize Whisper engine: config uses provider '{}'. Local recording requires 'localWhisper'.",
                    config.provider
                ));
            }
        }
        Ok(None) => {
            info!("No transcript config found in API, falling back to 'small'");
            "small".to_string()
        }
        Err(e) => {
            warn!(
                "Failed to get transcript config from API: {}, falling back to 'small'",
                e
            );
            "small".to_string()
        }
    };

    info!("Selected model to load: {}", model_to_load);
    let models = engine
        .discover_models()
        .await
        .map_err(|e| format!("Failed to discover models: {}", e))?;

    let model_info = models.iter().find(|model| model.name == model_to_load);

    match model_info {
        Some(model) => match model.status {
            crate::whisper_engine::ModelStatus::Available => {
                info!("Loading model: {}", model_to_load);
                engine
                    .load_model(&model_to_load)
                    .await
                    .map_err(|e| format!("Failed to load model '{}': {}", model_to_load, e))?;
                info!("✅ Model '{}' loaded successfully", model_to_load);
            }
            crate::whisper_engine::ModelStatus::Missing => {
                return Err(format!(
                    "Model '{}' is not downloaded. Please download it from settings first.",
                    model_to_load
                ));
            }
            crate::whisper_engine::ModelStatus::Downloading { progress } => {
                return Err(format!(
                    "Model '{}' is currently downloading ({}%). Please wait.",
                    model_to_load, progress
                ));
            }
            crate::whisper_engine::ModelStatus::Error(ref err) => {
                return Err(format!(
                    "Model '{}' has an error: {}. Please re-download.",
                    model_to_load, err
                ));
            }
            crate::whisper_engine::ModelStatus::Corrupted { .. } => {
                return Err(format!(
                    "Model '{}' is corrupted. Please delete and re-download.",
                    model_to_load
                ));
            }
        },
        None => {
            // Fall back to any available model
            let available_models: Vec<_> = models
                .iter()
                .filter(|m| matches!(m.status, crate::whisper_engine::ModelStatus::Available))
                .collect();

            if let Some(fallback_model) = available_models.first() {
                warn!(
                    "Model '{}' not found, falling back to available model: '{}'",
                    model_to_load, fallback_model.name
                );
                engine.load_model(&fallback_model.name).await.map_err(|e| {
                    format!(
                        "Failed to load fallback model '{}': {}",
                        fallback_model.name, e
                    )
                })?;
                info!(
                    "✅ Fallback model '{}' loaded successfully",
                    fallback_model.name
                );
            } else {
                return Err(format!(
                    "Model '{}' is not supported and no other models are available. Please download a model from settings.",
                    model_to_load
                ));
            }
        }
    }

    Ok(engine)
}
