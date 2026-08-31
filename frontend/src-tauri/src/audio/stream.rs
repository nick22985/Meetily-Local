//! Capture stream management on top of the native PipeWire layer.
//!
//! Every stream delivers interleaved f32 @ 48 kHz stereo (the PipeWire
//! graph negotiates/resamples), feeding `AudioCapture` → pipeline
//! unchanged.
//!
//! A selection is either a stable PipeWire `node.name` or one of the
//! sentinels in [`super::device_mode`], which are resolved against the
//! graph at open time. Resolving up front rather than leaving the stream
//! untargeted means we know — and can report — which node we are on, and
//! it is what lets [`super::device_follower`] notice when that answer
//! changes mid-recording.

use anyhow::Result;
use log::{error, info, warn};
use std::sync::{Arc, Mutex};

use super::app_audio;
use super::device_follower::{ActiveRoute, DeviceFollower, FollowedSide};
use super::device_mode::DeviceMode;
use super::devices::AudioDevice;
use super::pipeline::AudioCapture;
use super::pw::{
    self, CaptureTarget, PwCaptureStream, PwDeviceKind, CAPTURE_CHANNELS, CAPTURE_RATE,
};
use super::recording_state::{DeviceType, RecordingState};

/// Which kind of PipeWire node a capture side reads from.
pub fn device_kind_for(device_type: DeviceType) -> PwDeviceKind {
    match device_type {
        DeviceType::Microphone => PwDeviceKind::Microphone,
        DeviceType::System => PwDeviceKind::System,
    }
}

/// Resolve a device selection to the `node.name` it currently points at.
///
/// `None` means we could not name one — an unset default, or a graph we
/// failed to read. Callers treat that as "let PipeWire pick", which is
/// the behaviour an untargeted stream has anyway.
pub fn resolve_selection(device_id: &str, device_type: DeviceType) -> Option<String> {
    let mode = DeviceMode::parse(device_id);
    if let DeviceMode::Explicit(node) = &mode {
        return Some(node.clone());
    }
    match pw::snapshot_graph() {
        Ok(snapshot) => mode.resolve(device_kind_for(device_type), &snapshot),
        Err(e) => {
            warn!(
                "Could not read the audio graph to resolve '{}': {} — letting PipeWire pick",
                device_id, e
            );
            None
        }
    }
}

/// Translate a selected device into a PipeWire capture target.
pub fn capture_target_for(device_id: &str, device_type: DeviceType) -> CaptureTarget {
    capture_target_for_node(
        resolve_selection(device_id, device_type).as_deref(),
        device_type,
    )
}

/// Build a capture target for an already-resolved node.
///
/// An `app:` selection on the system side resolves to an application
/// rather than a node; everything else is a `node.name` to point at.
pub fn capture_target_for_node(node: Option<&str>, device_type: DeviceType) -> CaptureTarget {
    if let Some(key) = app_capture_key(node, device_type) {
        return CaptureTarget::AppAudio {
            label: app_audio::label_for(key),
        };
    }
    // An app selection we are not honouring — the feature is switched
    // off, or it was left on the microphone side — is not a node name.
    // Drop it and let PipeWire's own default stand in.
    let node = node.filter(|id| !id.starts_with(app_audio::APP_PREFIX));
    match (node, device_type) {
        (Some(id), _) => CaptureTarget::Node {
            id: id.to_string(),
            kind: device_kind_for(device_type),
        },
        (None, DeviceType::Microphone) => CaptureTarget::DefaultMicrophone,
        (None, DeviceType::System) => CaptureTarget::DefaultSystem,
    }
}

/// The application a selection asks for, if it asks for one at all.
///
/// `None` for a device selection, for an app selection made while the
/// feature is switched off, and for one on the microphone side — a
/// microphone is a device, and there is nothing to tap.
fn app_capture_key(node: Option<&str>, device_type: DeviceType) -> Option<&str> {
    if device_type != DeviceType::System {
        return None;
    }
    app_audio::selected_app(node?)
}

/// A running capture stream bound to the recording pipeline.
pub struct AudioStream {
    device: Arc<AudioDevice>,
    /// The node this stream was opened on. `None` when we could not name
    /// one and left PipeWire to pick.
    resolved: Option<String>,
    stream: PwCaptureStream,
}

impl AudioStream {
    /// Open a stream, resolving the selection against the graph.
    pub async fn create(
        device: Arc<AudioDevice>,
        state: Arc<RecordingState>,
        device_type: DeviceType,
    ) -> Result<Self> {
        let resolved = resolve_selection(&device.name, device_type);
        Self::create_on(device, resolved, state, device_type).await
    }

    /// Open a stream on an already-resolved node.
    pub async fn create_on(
        device: Arc<AudioDevice>,
        resolved: Option<String>,
        state: Arc<RecordingState>,
        device_type: DeviceType,
    ) -> Result<Self> {
        info!(
            "🎵 Stream: opening PipeWire capture for '{}' ({:?}) on {}",
            device.name,
            device_type,
            resolved.as_deref().unwrap_or("<PipeWire's pick>")
        );

        let capture = AudioCapture::new(
            device.clone(),
            state,
            CAPTURE_RATE,
            CAPTURE_CHANNELS as u16,
            device_type,
        );

        let on_samples = Box::new(move |samples: &[f32]| capture.process_audio_data(samples));
        let stream = match app_capture_key(resolved.as_deref(), device_type) {
            // An app is tapped wherever it happens to be playing, so the
            // stream steers itself rather than being pointed at a node.
            Some(key) => PwCaptureStream::open_app(
                app_audio::label_for(key),
                app_audio::node_resolver(key.to_string()),
                on_samples,
            )?,
            None => PwCaptureStream::open(
                capture_target_for_node(resolved.as_deref(), device_type),
                on_samples,
            )?,
        };

        Ok(Self {
            device,
            resolved,
            stream,
        })
    }

    pub fn device(&self) -> &AudioDevice {
        &self.device
    }

    /// The node this stream is reading from, when we named one.
    pub fn resolved(&self) -> Option<&str> {
        self.resolved.as_deref()
    }

    pub fn stop(self) -> Result<()> {
        info!("Stopping audio stream for device: {}", self.device.name);
        self.stream.stop();
        Ok(())
    }
}

/// Manages the microphone + system capture stream pair.
///
/// The streams live behind mutexes because [`DeviceFollower`] reopens
/// them from its own task when a selection that follows the graph starts
/// pointing somewhere else. Nothing holds a lock across an await: a
/// replacement stream is built first, then swapped in.
pub struct AudioStreamManager {
    microphone_stream: Arc<Mutex<Option<AudioStream>>>,
    system_stream: Arc<Mutex<Option<AudioStream>>>,
    follower: Option<DeviceFollower>,
    route: Arc<Mutex<ActiveRoute>>,
    state: Arc<RecordingState>,
}

impl AudioStreamManager {
    pub fn new(state: Arc<RecordingState>) -> Self {
        Self {
            microphone_stream: Arc::new(Mutex::new(None)),
            system_stream: Arc::new(Mutex::new(None)),
            follower: None,
            route: Arc::new(Mutex::new(ActiveRoute::default())),
            state,
        }
    }

    /// Start audio streams for the given devices.
    pub async fn start_streams(
        &mut self,
        microphone_device: Option<Arc<AudioDevice>>,
        system_device: Option<Arc<AudioDevice>>,
    ) -> Result<()> {
        let mut sides = Vec::new();

        if let Some(mic_device) = microphone_device {
            info!("🎤 Creating microphone stream: {}", mic_device.name);
            match AudioStream::create(
                mic_device.clone(),
                self.state.clone(),
                DeviceType::Microphone,
            )
            .await
            {
                Ok(stream) => {
                    self.route.lock().unwrap().microphone = stream.resolved().map(str::to_string);
                    self.state.set_microphone_device(mic_device.clone());
                    *self.microphone_stream.lock().unwrap() = Some(stream);
                    sides.push(FollowedSide::new(
                        mic_device,
                        DeviceType::Microphone,
                        self.microphone_stream.clone(),
                    ));
                    info!("✅ Microphone stream created successfully");
                }
                Err(e) => {
                    error!("❌ Failed to create microphone stream: {}", e);
                    return Err(e);
                }
            }
        } else {
            info!("ℹ️ No microphone device specified, skipping microphone stream");
        }

        if let Some(sys_device) = system_device {
            info!("🔊 Creating system audio stream: {}", sys_device.name);
            match AudioStream::create(sys_device.clone(), self.state.clone(), DeviceType::System)
                .await
            {
                Ok(stream) => {
                    self.route.lock().unwrap().system = stream.resolved().map(str::to_string);
                    self.state.set_system_device(sys_device.clone());
                    *self.system_stream.lock().unwrap() = Some(stream);
                    sides.push(FollowedSide::new(
                        sys_device,
                        DeviceType::System,
                        self.system_stream.clone(),
                    ));
                    info!("✅ System audio stream created successfully");
                }
                Err(e) => {
                    // Don't fail the whole recording if only system audio fails.
                    error!("⚠️ Failed to create system audio stream: {}", e);
                }
            }
        } else {
            info!("ℹ️ No system device specified, skipping system audio stream");
        }

        if !self.has_active_streams() {
            return Err(anyhow::anyhow!("No audio streams could be created"));
        }

        // Only selections that can move are worth watching; an explicitly
        // named node stays where the user put it.
        sides.retain(|side| side.follows_graph());
        if !sides.is_empty() {
            self.follower = Some(DeviceFollower::spawn(
                sides,
                self.state.clone(),
                self.route.clone(),
            ));
        }

        Ok(())
    }

    /// Stop all audio streams.
    pub fn stop_streams(&mut self) -> Result<()> {
        info!("Stopping all audio streams");

        // Before the streams, so nothing gets reopened behind us.
        if let Some(follower) = self.follower.take() {
            follower.stop();
        }

        let mic = self.microphone_stream.lock().unwrap().take();
        if let Some(mic_stream) = mic {
            mic_stream.stop()?;
        }
        let sys = self.system_stream.lock().unwrap().take();
        if let Some(sys_stream) = sys {
            sys_stream.stop()?;
        }
        *self.route.lock().unwrap() = ActiveRoute::default();

        info!("All audio streams stopped");
        Ok(())
    }

    /// Which nodes the capture is currently reading from.
    pub fn active_route(&self) -> ActiveRoute {
        self.route.lock().unwrap().clone()
    }

    pub fn active_stream_count(&self) -> usize {
        self.microphone_stream.lock().unwrap().is_some() as usize
            + self.system_stream.lock().unwrap().is_some() as usize
    }

    pub fn has_active_streams(&self) -> bool {
        self.microphone_stream.lock().unwrap().is_some()
            || self.system_stream.lock().unwrap().is_some()
    }
}

impl Drop for AudioStreamManager {
    fn drop(&mut self) {
        if let Err(e) = self.stop_streams() {
            error!("Error stopping streams during drop: {}", e);
        }
    }
}
