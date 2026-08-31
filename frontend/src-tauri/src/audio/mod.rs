// src/audio/mod.rs
pub mod audio_processing;
pub mod aec;
pub mod clip;
pub mod decoder;
pub mod playback;
pub mod encode;
pub mod ffmpeg;
pub mod vad;

// Device model + PipeWire-backed discovery
pub mod devices;

// Native PipeWire capture layer (Linux)
#[cfg(target_os = "linux")]
pub mod pw;

// Capturing one application's playback instead of the whole sink mix
pub mod app_audio;

// Device classification for adaptive buffering (Bluetooth vs wired)
pub mod device_detection;

// How a device selection (a node name, or a "default"/"communication"
// sentinel) resolves to the node we actually capture from
pub mod device_follower;
pub mod device_mode;
pub mod ffmpeg_mixer;

// Recording system
pub mod batch_processor;
pub mod buffer_pool;
pub mod hardware_detector;
pub mod incremental_saver;
pub mod pipeline;
pub mod recording_commands;
pub mod recording_manager;
pub mod recording_preferences;
pub mod recording_saver;
pub mod recording_state;
pub mod simple_level_monitor;
pub mod stream;

// Transcription module (provider abstraction, engine management, worker pool)
pub mod transcription;

// Shared utilities for import and retranscription
pub(crate) mod common;

// Shared constants
pub mod constants;

// Retranscription module (re-process stored audio with different settings)
pub mod retranscription;

// Import module (import external audio files as new meetings)
pub mod import;

pub use devices::{list_audio_devices, trigger_audio_permission, AudioDevice, DeviceType};

pub use buffer_pool::{AudioBufferPool, PooledBuffer};
pub use encode::{encode_single_audio, AudioInput};
pub use hardware_detector::{AdaptiveWhisperConfig, GpuType, HardwareProfile, PerformanceTier};
pub use pipeline::AudioPipelineManager;
pub use recording_commands::{
    get_transcription_status, is_recording, start_recording, start_recording_with_devices,
    stop_recording, RecordingArgs, TranscriptUpdate, TranscriptionStatus,
};
pub use recording_manager::RecordingManager;
pub use recording_preferences::{get_default_recordings_folder, RecordingPreferences};
pub use recording_saver::RecordingSaver;
pub use recording_state::{
    AudioChunk, AudioError, DeviceType as RecordingDeviceType, ProcessedAudioChunk, RecordingState,
};
pub use app_audio::{list_app_audio_sources, AppAudioCatalog, AppAudioSource};
pub use device_mode::{DeviceMode, DEVICE_COMMUNICATION, DEVICE_SYSTEM_DEFAULT};
pub use stream::AudioStreamManager;

pub use device_detection::{calculate_buffer_timeout, InputDeviceKind};
pub use ffmpeg_mixer::{BufferStats, FFmpegAudioMixer, RNNOISE_APPLY_ENABLED};

pub use vad::extract_speech_16k;

// Export decoder for retranscription
pub use decoder::{decode_audio_file, DecodedAudio};

// Export audio constants
pub use constants::AUDIO_EXTENSIONS;
