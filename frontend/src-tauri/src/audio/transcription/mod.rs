// audio/transcription/mod.rs
//
// Transcription module: engine management and worker pool. Two backends are
// available behind `TranscriptionEngine` — local whisper-rs, and a remote
// OpenAI-compatible endpoint (`remote`) for offloading decode to another
// machine. See worker.rs for TranscriptionError.

pub mod echo_dedup;
pub mod engine;
pub mod partial_worker;
pub mod remote;
pub mod worker;

// Re-export commonly used types
pub use engine::{
    get_or_init_batch_engine, get_or_init_transcription_engine, get_or_init_whisper,
    validate_transcription_model_ready, TranscriptionEngine,
};
pub use partial_worker::start_partial_decode_task;
pub use remote::{
    RemoteModel, RemoteModelCatalog, RemoteWhisperClient, RemoteWhisperConfig,
    REMOTE_WHISPER_PROVIDER,
};
pub use worker::{
    reset_speech_detected_flag, start_transcription_task, TranscriptUpdate, TranscriptionError,
};
