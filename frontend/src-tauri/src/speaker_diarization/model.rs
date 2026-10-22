//! Speaker model file resolution and download coordination.
//!
//! Models live under `<app_data_dir>/speaker_models/`. The default embedding
//! model ([`DEFAULT_MODEL_FILENAME`]) is the 3D-Speaker CAM++ English
//! checkpoint — 28MB, 192-dim output, runs in <50ms on CPU per segment. The
//! user can pick another from [`EMBEDDING_MODELS`] (see
//! [`super::settings`]); the embedding dimension is read from the loaded
//! ONNX at runtime, and voice profiles record which model produced them.
//!
//! The pyannote segmentation model ([`SEGMENTATION_MODEL_FILENAME`]) is used
//! only by the post-recording offline diarization pass ([`super::offline`]).

use once_cell::sync::Lazy;
use std::path::{Path, PathBuf};
use std::sync::Mutex;
use tauri::{AppHandle, Manager, Runtime};

/// Default speaker-embedding model. English-trained (VoxCeleb), 192-dim,
/// fast on CPU. The 3D-Speaker family ships its preprocessing inside the
/// ONNX graph, so we feed raw 16 kHz mono audio directly.
pub const DEFAULT_MODEL_FILENAME: &str =
    "3dspeaker_speech_campplus_sv_en_voxceleb_16k.onnx";

/// A selectable speaker-embedding model. All are served from the same
/// sherpa-onnx release, and sherpa's extractor reads each model's
/// preprocessing from its ONNX metadata, so they're drop-in replacements.
#[derive(Debug, Clone, Copy)]
pub struct EmbeddingModelInfo {
    pub filename: &'static str,
    pub display_name: &'static str,
    pub size_mb: u32,
    pub description: &'static str,
}

/// Models offered in settings, default first. Use the `diarization_eval`
/// binary to compare them on your own recordings before switching.
pub const EMBEDDING_MODELS: &[EmbeddingModelInfo] = &[
    EmbeddingModelInfo {
        filename: DEFAULT_MODEL_FILENAME,
        display_name: "3D-Speaker CAM++ (default)",
        size_mb: 28,
        description: "Fast and small. Good default for most meetings.",
    },
    EmbeddingModelInfo {
        filename: "3dspeaker_speech_eres2net_sv_en_voxceleb_16k.onnx",
        display_name: "3D-Speaker ERes2Net",
        size_mb: 25,
        description: "Similar size to CAM++, sometimes better on noisy audio.",
    },
    EmbeddingModelInfo {
        filename: "wespeaker_en_voxceleb_resnet34_LM.onnx",
        display_name: "WeSpeaker ResNet34",
        size_mb: 25,
        description: "Strong mid-size model, 256-dim embeddings.",
    },
    EmbeddingModelInfo {
        filename: "nemo_en_titanet_large.onnx",
        display_name: "NeMo TitaNet Large",
        size_mb: 97,
        description: "Larger and slower; better separation of similar voices.",
    },
    EmbeddingModelInfo {
        filename: "wespeaker_en_voxceleb_resnet293_LM.onnx",
        display_name: "WeSpeaker ResNet293",
        size_mb: 109,
        description: "Most accurate, slowest. Best with a GPU-class CPU.",
    },
];

pub fn embedding_model_info(filename: &str) -> Option<&'static EmbeddingModelInfo> {
    EMBEDDING_MODELS.iter().find(|m| m.filename == filename)
}

/// pyannote segmentation-3.0 (sherpa-onnx export, ~6 MB). Finds speech and
/// speaker-change points, including overlapped speech, for the offline pass.
pub const SEGMENTATION_MODEL_FILENAME: &str = "pyannote_segmentation_3_0.onnx";

const SEGMENTATION_MODEL_URL: &str =
    "https://huggingface.co/csukuangfj/sherpa-onnx-pyannote-segmentation-3-0/resolve/main/model.onnx";

/// Silero VAD ONNX model used by sherpa-onnx's `VoiceActivityDetector`.
/// Lives alongside the speaker embedding model since both are sherpa-onnx
/// runtime artefacts. Tiny (~2.3 MB).
pub const SILERO_VAD_FILENAME: &str = "silero_vad.onnx";

const MODEL_BASE_URL: &str =
    "https://github.com/k2-fsa/sherpa-onnx/releases/download/speaker-recongition-models";

/// Sherpa's silero-vad model lives in a different release tag than the
/// speaker recognition models.
const SILERO_VAD_URL: &str =
    "https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/silero_vad.onnx";

static MODELS_DIR: Lazy<Mutex<Option<PathBuf>>> = Lazy::new(|| Mutex::new(None));

static ACTIVE_MODEL: Lazy<Mutex<String>> =
    Lazy::new(|| Mutex::new(DEFAULT_MODEL_FILENAME.to_string()));

/// Called once at app startup (from `lib.rs::setup`). Resolves
/// `<app_data_dir>/speaker_models/`, creates it if missing, and caches the
/// path so command handlers can look it up without an `AppHandle`.
pub fn set_models_dir<R: Runtime>(app: &AppHandle<R>) {
    let Ok(app_data_dir) = app.path().app_data_dir() else {
        log::warn!("Failed to resolve app_data_dir for speaker models");
        return;
    };
    let dir = app_data_dir.join("speaker_models");
    if let Err(e) = std::fs::create_dir_all(&dir) {
        log::error!("Failed to create speaker_models dir: {}", e);
        return;
    }
    log::info!("Speaker models directory: {}", dir.display());
    set_models_dir_path(dir);
}

/// Point model lookups at `dir` directly. For tools running outside the app
/// (the `diarization_eval` binary).
pub fn set_models_dir_path(dir: PathBuf) {
    if let Ok(mut guard) = MODELS_DIR.lock() {
        *guard = Some(dir);
    }
}

/// Configured models directory, or `None` if `set_models_dir` hasn't run yet.
pub fn models_dir() -> Option<PathBuf> {
    MODELS_DIR.lock().ok().and_then(|g| g.clone())
}

/// Select the active embedding model. Unknown filenames are rejected (and
/// leave the current selection alone) so a stale setting can't point the
/// app at a file it doesn't know how to download.
pub fn set_active_model(filename: &str) -> bool {
    if embedding_model_info(filename).is_none() {
        log::warn!("Ignoring unknown speaker embedding model '{}'", filename);
        return false;
    }
    if let Ok(mut guard) = ACTIVE_MODEL.lock() {
        *guard = filename.to_string();
    }
    true
}

/// Filename of the active speaker embedding model.
pub fn model_filename() -> String {
    ACTIVE_MODEL
        .lock()
        .map(|g| g.clone())
        .unwrap_or_else(|_| DEFAULT_MODEL_FILENAME.to_string())
}

/// Full path to the active speaker embedding model. `None` if
/// `set_models_dir` hasn't run.
pub fn default_model_path() -> Option<PathBuf> {
    models_dir().map(|d| d.join(model_filename()))
}

/// Path to a specific embedding model file.
pub fn embedding_model_path(filename: &str) -> Option<PathBuf> {
    models_dir().map(|d| d.join(filename))
}

/// Download URL for the active speaker embedding model.
pub fn model_download_url() -> String {
    embedding_model_url(&model_filename())
}

pub fn embedding_model_url(filename: &str) -> String {
    format!("{}/{}", MODEL_BASE_URL, filename)
}

/// Path to the pyannote segmentation model.
pub fn segmentation_model_path() -> Option<PathBuf> {
    models_dir().map(|d| d.join(SEGMENTATION_MODEL_FILENAME))
}

pub fn segmentation_model_url() -> &'static str {
    SEGMENTATION_MODEL_URL
}

/// Path to the silero-vad ONNX model used by sherpa-onnx's VAD.
pub fn silero_vad_path() -> Option<PathBuf> {
    models_dir().map(|d| d.join(SILERO_VAD_FILENAME))
}

/// Download URL for the silero-vad model.
pub fn silero_vad_download_url() -> &'static str {
    SILERO_VAD_URL
}

/// True when the model file exists on disk and is non-empty.
pub fn model_is_ready(path: &Path) -> bool {
    path.metadata().map(|m| m.len() > 0).unwrap_or(false)
}
