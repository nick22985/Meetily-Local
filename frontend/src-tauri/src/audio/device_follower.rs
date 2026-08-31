//! Keeps capture streams pointed at the right node while recording.
//!
//! A selection of [`DeviceMode::SystemDefault`] or
//! [`DeviceMode::Communication`] is a question about the graph, not a
//! fixed answer, and the answer can change under us: the user hits a
//! profile-switcher keybind, unplugs an interface, or joins a call on a
//! headset. This watcher re-asks once a second and, when the answer has
//! moved, reopens that side's stream on the new node.
//!
//! Retargeting is a stop-then-open rather than a live `target.object`
//! rewrite. That costs a sub-second gap in the recording, which is
//! acceptable for a switch the user deliberately triggered, and it keeps
//! the stream lifecycle identical to the one that already works.
//!
//! Polling beats a long-lived registry watcher here: one combined
//! roundtrip per tick reuses [`crate::audio::pw::snapshot_graph`] as-is,
//! with no extra proxies to keep alive, and a second of latency on a
//! device switch is well under what the user notices.

use log::{error, info, warn};
use serde::{Deserialize, Serialize};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};
use std::time::Duration;
use tokio::task::JoinHandle;

use super::device_mode::DeviceMode;
use super::devices::AudioDevice;
use super::pw;
use super::recording_state::{DeviceType, RecordingState};
use super::stream::{device_kind_for, AudioStream};

/// How often the graph is re-checked.
const POLL_INTERVAL: Duration = Duration::from_secs(1);

/// The nodes the capture is currently reading from, for display.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct ActiveRoute {
    pub microphone: Option<String>,
    pub system: Option<String>,
}

/// One capture side the follower is responsible for.
pub struct FollowedSide {
    device: Arc<AudioDevice>,
    mode: DeviceMode,
    device_type: DeviceType,
    slot: Arc<Mutex<Option<AudioStream>>>,
    /// The node this side is meant to be on. `None` after a failed
    /// reopen, which makes the next tick try again.
    desired: Option<String>,
}

impl FollowedSide {
    pub fn new(
        device: Arc<AudioDevice>,
        device_type: DeviceType,
        slot: Arc<Mutex<Option<AudioStream>>>,
    ) -> Self {
        let mode = DeviceMode::parse(&device.name);
        let desired = slot
            .lock()
            .unwrap()
            .as_ref()
            .and_then(|stream| stream.resolved().map(str::to_string));
        Self {
            device,
            mode,
            device_type,
            slot,
            desired,
        }
    }

    /// Whether this side's selection can move at all.
    pub fn follows_graph(&self) -> bool {
        self.mode.follows_graph()
    }

    fn label(&self) -> &'static str {
        match self.device_type {
            DeviceType::Microphone => "microphone",
            DeviceType::System => "system audio",
        }
    }

    /// Re-open this side on `want` if that is not where it already is.
    ///
    /// Returns `true` when the stream was replaced.
    async fn retarget(
        &mut self,
        want: Option<String>,
        state: Arc<RecordingState>,
        stop: &AtomicBool,
    ) -> bool {
        if want == self.desired {
            return false;
        }
        // Nothing to move to and nothing currently claimed: wait for the
        // graph to name something rather than churning the stream.
        if want.is_none() && self.desired.is_none() {
            return false;
        }

        info!(
            "🔀 {} moved: {} → {} (following '{}')",
            self.label(),
            self.desired.as_deref().unwrap_or("<none>"),
            want.as_deref().unwrap_or("<none>"),
            self.device.name
        );

        // Stop before opening: two live streams would both feed the
        // pipeline and interleave into the mixer's ring buffer.
        let old = self.slot.lock().unwrap().take();
        if let Some(stream) = old {
            if let Err(e) = stream.stop() {
                warn!("Error stopping the old {} stream: {}", self.label(), e);
            }
        }

        if stop.load(Ordering::Relaxed) {
            return false;
        }

        match AudioStream::create_on(self.device.clone(), want.clone(), state, self.device_type)
            .await
        {
            Ok(stream) => {
                // The recording may have ended while we were opening.
                if stop.load(Ordering::Relaxed) {
                    let _ = stream.stop();
                    return false;
                }
                *self.slot.lock().unwrap() = Some(stream);
                self.desired = want;
                true
            }
            Err(e) => {
                error!(
                    "❌ Failed to reopen the {} stream on {}: {}",
                    self.label(),
                    want.as_deref().unwrap_or("<PipeWire's pick>"),
                    e
                );
                // Leave the slot empty and forget where we were, so the
                // next tick retries instead of deciding nothing changed.
                self.desired = None;
                false
            }
        }
    }
}

/// Watches the graph and retargets streams for the lifetime of a recording.
pub struct DeviceFollower {
    stop: Arc<AtomicBool>,
    task: JoinHandle<()>,
}

impl DeviceFollower {
    pub fn spawn(
        mut sides: Vec<FollowedSide>,
        state: Arc<RecordingState>,
        route: Arc<Mutex<ActiveRoute>>,
    ) -> Self {
        let stop = Arc::new(AtomicBool::new(false));
        let task_stop = stop.clone();

        let task = tokio::spawn(async move {
            info!(
                "👀 Following the audio graph for {} capture side(s)",
                sides.len()
            );

            loop {
                tokio::time::sleep(POLL_INTERVAL).await;
                if task_stop.load(Ordering::Relaxed) {
                    break;
                }

                let snapshot = match tokio::task::spawn_blocking(pw::snapshot_graph).await {
                    Ok(Ok(snapshot)) => snapshot,
                    Ok(Err(e)) => {
                        warn!("Could not read the audio graph: {} — retrying", e);
                        continue;
                    }
                    // The runtime is shutting down.
                    Err(_) => break,
                };

                for side in sides.iter_mut() {
                    if task_stop.load(Ordering::Relaxed) {
                        break;
                    }
                    let want = side
                        .mode
                        .resolve(device_kind_for(side.device_type), &snapshot);
                    if side.retarget(want, state.clone(), &task_stop).await {
                        let mut route = route.lock().unwrap();
                        match side.device_type {
                            DeviceType::Microphone => route.microphone = side.desired.clone(),
                            DeviceType::System => route.system = side.desired.clone(),
                        }
                    }
                }
            }

            info!("👋 Stopped following the audio graph");
        });

        Self { stop, task }
    }

    /// Stop watching. Safe to call from a synchronous context.
    pub fn stop(self) {
        self.stop.store(true, Ordering::Relaxed);
        self.task.abort();
    }
}
