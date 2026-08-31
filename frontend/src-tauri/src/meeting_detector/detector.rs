//! The monitor loop: watch what holds the microphone, and turn that into
//! meeting start/end events.

use std::collections::HashMap;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::Arc;
use std::time::Duration;

use serde::{Deserialize, Serialize};
use tauri::{AppHandle, Emitter, Manager, Runtime};
use tokio::sync::RwLock;

use crate::audio::pw::{self, PwCaptureClient};
use crate::audio::recording_commands;
use crate::notifications::commands::NotificationManagerState;
use crate::notifications::types::{Notification, NotificationType};

use super::apps::{self, MeetingApp};
use super::settings::MeetingDetectionSettings;
use super::voice_socket::{self, CallEvidence, CallProbe};

/// A meeting the detector is currently tracking.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DetectedMeeting {
    /// Key from the known-app table, or `None` for an unrecognized app
    /// picked up because `detect_unknown_apps` is on.
    pub app_key: Option<String>,
    /// What to call it in the UI.
    pub app_label: String,
    /// The PipeWire client that gave it away, for the "why did this fire?"
    /// question.
    pub source: String,
    /// RFC 3339, when the meeting was confirmed.
    pub detected_at: String,
    /// Whether the detector started the current recording itself.
    pub auto_recording: bool,
}

/// A live capture stream, classified. Surfaced by the settings UI so the
/// user can see what the detector sees.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CaptureStreamInfo {
    pub app_key: Option<String>,
    pub label: String,
    pub node_name: String,
    /// Everything we matched on, joined for display. The panel exists to
    /// answer "why was this classified that way", which it can't do
    /// without showing the actual inputs.
    pub matched_on: String,
    /// Whether this stream would count as a meeting under current settings.
    pub counts_as_meeting: bool,
    /// Whether the stream is actually running — i.e. whether the app is
    /// listening, as opposed to sitting on a capture stream it opened at
    /// launch and never closed.
    pub live: bool,
}

/// Everything the settings screen needs to render the detector's state.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MeetingDetectionStatus {
    pub monitoring: bool,
    pub settings: MeetingDetectionSettings,
    pub active_meeting: Option<DetectedMeeting>,
    /// Last error from the audio graph, if the most recent poll failed.
    pub last_error: Option<String>,
}

/// Shared handle to the detector. Cheap to clone; every clone talks to the
/// same monitor loop.
#[derive(Clone)]
pub struct MeetingDetector {
    settings: Arc<RwLock<MeetingDetectionSettings>>,
    active: Arc<RwLock<Option<DetectedMeeting>>>,
    last_error: Arc<RwLock<Option<String>>>,
    monitoring: Arc<AtomicBool>,
    /// Bumped every time the loop should restart. A running loop exits as
    /// soon as it notices its own generation is stale, which is how a
    /// settings change takes effect without waiting out the old interval.
    generation: Arc<AtomicU64>,
    /// True while a recording the *detector* started is running. Guards
    /// auto-stop so we never cut short a recording the user started by hand.
    auto_recording: Arc<AtomicBool>,
}

impl Default for MeetingDetector {
    fn default() -> Self {
        Self::new()
    }
}

impl MeetingDetector {
    pub fn new() -> Self {
        Self {
            settings: Arc::new(RwLock::new(MeetingDetectionSettings::default())),
            active: Arc::new(RwLock::new(None)),
            last_error: Arc::new(RwLock::new(None)),
            monitoring: Arc::new(AtomicBool::new(false)),
            generation: Arc::new(AtomicU64::new(0)),
            auto_recording: Arc::new(AtomicBool::new(false)),
        }
    }

    pub async fn settings(&self) -> MeetingDetectionSettings {
        self.settings.read().await.clone()
    }

    pub async fn status(&self) -> MeetingDetectionStatus {
        MeetingDetectionStatus {
            monitoring: self.monitoring.load(Ordering::Relaxed),
            settings: self.settings.read().await.clone(),
            active_meeting: self.active.read().await.clone(),
            last_error: self.last_error.read().await.clone(),
        }
    }

    /// Load persisted settings and start monitoring if they're enabled.
    /// Called once during app setup, after the database is ready.
    pub async fn initialize<R: Runtime>(&self, app: &AppHandle<R>) {
        let settings = super::settings::load(app).await;
        let enabled = settings.enabled;
        *self.settings.write().await = settings;
        if enabled {
            self.restart(app);
        }
    }

    /// Replace the settings, persist them, and restart the monitor loop so
    /// the change takes effect immediately.
    pub async fn apply<R: Runtime>(
        &self,
        app: &AppHandle<R>,
        settings: MeetingDetectionSettings,
    ) -> Result<MeetingDetectionSettings, String> {
        let settings = settings.sanitized();
        super::settings::save(app, &settings).await?;
        let enabled = settings.enabled;
        *self.settings.write().await = settings.clone();

        if enabled {
            self.restart(app);
        } else {
            self.stop().await;
        }
        Ok(settings)
    }

    /// Retire any running loop and spawn a fresh one.
    fn restart<R: Runtime>(&self, app: &AppHandle<R>) {
        let generation = self.generation.fetch_add(1, Ordering::SeqCst) + 1;
        let detector = self.clone();
        let app = app.clone();
        self.monitoring.store(true, Ordering::Relaxed);
        tauri::async_runtime::spawn(async move {
            monitor_loop(detector, app, generation).await;
        });
    }

    /// Retire any running loop without touching the current recording — a
    /// recording in progress is the user's to stop.
    async fn stop(&self) {
        self.generation.fetch_add(1, Ordering::SeqCst);
        self.monitoring.store(false, Ordering::Relaxed);
        *self.active.write().await = None;
        *self.last_error.write().await = None;
    }

    /// One-shot look at the audio graph, classified under current settings.
    /// Backs the "what's using your microphone right now" panel.
    pub async fn probe(&self) -> Result<Vec<CaptureStreamInfo>, String> {
        let settings = self.settings().await;
        let clients = capture_clients().await.map_err(|e| e.to_string())?;
        let calls = probe_calls(&clients);
        Ok(clients
            .iter()
            .filter(|client| client.capture)
            .map(|client| {
                let app = apps::classify(&client.match_keys());
                let key = app.map(|a| a.key);
                CaptureStreamInfo {
                    app_key: key.map(str::to_string),
                    label: display_label(client, app),
                    node_name: client.node_name.clone(),
                    matched_on: client.match_keys().join(", "),
                    counts_as_meeting: settings.accepts(key) && is_live(client, &calls),
                    live: is_live(client, &calls),
                }
            })
            .collect())
    }
}

/// Enumerate capture streams off the async runtime — the PipeWire roundtrip
/// blocks its thread for up to a few seconds.
async fn capture_clients() -> anyhow::Result<Vec<PwCaptureClient>> {
    tokio::task::spawn_blocking(pw::enumerate_capture_streams)
        .await
        .map_err(|e| anyhow::anyhow!("capture stream probe panicked: {}", e))?
}

/// Best available human-readable name for a capture client.
fn display_label(client: &PwCaptureClient, app: Option<&'static MeetingApp>) -> String {
    if let Some(app) = app {
        return app.label.to_string();
    }
    client
        .app_name
        .clone()
        .or_else(|| client.binary.clone())
        .unwrap_or_else(|| client.node_name.clone())
}

/// Ask, for every stream whose app holds the microphone outside of calls,
/// whether that app currently has a media connection open.
///
/// Only those apps are asked, because only for them is the audio graph
/// inconclusive — and the probe costs a directory scan per process.
///
/// A stream missing from the result was never asked about; one answering
/// [`CallProbe::Unreadable`] was asked and couldn't be told. Both mean
/// "don't know", never "no call" — they are kept apart only so the log
/// says which.
fn probe_calls(clients: &[PwCaptureClient]) -> HashMap<u32, CallEvidence> {
    let asking: Vec<&PwCaptureClient> = clients
        .iter()
        .filter(|client| {
            client.capture
                && !client.is_meetily()
                && apps::classify(&client.match_keys()).is_some_and(|app| app.idles_with_mic)
        })
        .collect();
    if asking.is_empty() {
        return HashMap::new();
    }

    // No socket table at all is a different failure from a process we
    // can't read, but it lands in the same place: every stream we meant to
    // ask about goes down as unanswerable.
    let sockets = voice_socket::udp_sockets();

    let mut answers = HashMap::new();
    for client in asking {
        let evidence = match &sockets {
            Some(sockets) => voice_socket::probe_app(client.binary.as_deref(), client.pid, sockets),
            None => CallEvidence {
                probe: CallProbe::Unreadable,
                connected: 0,
                open: 0,
            },
        };
        answers.insert(client.node_id, evidence);
    }
    answers
}

/// The probe's verdict for a stream, if it was asked about at all.
fn call_verdict(calls: &HashMap<u32, CallEvidence>, node_id: u32) -> Option<CallProbe> {
    calls.get(&node_id).map(|evidence| evidence.probe)
}

/// The probe result as it appears in the detection log.
///
/// The socket counts are printed for every answer, not just the positive
/// ones, because the two kinds carry very different weight: a connected
/// socket names a media server, an unconnected one only says the app has a
/// UDP port open. Seeing which one fired — or that neither did while a
/// call was plainly running — is what makes a wrong verdict diagnosable.
fn describe_call(evidence: Option<&CallEvidence>) -> String {
    let Some(evidence) = evidence else {
        return "not asked".to_string();
    };
    let verdict = match evidence.probe {
        CallProbe::Connected => "connected",
        CallProbe::Idle => "none",
        CallProbe::Unreadable => "unreadable",
    };
    format!(
        "{} ({} peered, {} bound)",
        verdict, evidence.connected, evidence.open
    )
}

/// Whether a stream could be part of a meeting at this instant, before any
/// question of which app it belongs to or whether the user watches it.
///
/// The socket probe gets the deciding vote where it has one. Where it
/// doesn't — an unreadable pid, or an app that was never asked — this is
/// just "is the node running", and [`mute_stale_streams`] stays the
/// fallback for apps that sit on the microphone.
fn is_live(client: &PwCaptureClient, calls: &HashMap<u32, CallEvidence>) -> bool {
    client.running && call_verdict(calls, client.node_id) != Some(CallProbe::Idle)
}

/// Whether the app behind a meeting is still listening.
///
/// A muted stream still counts here. Muting says "this isn't the *start* of
/// a meeting"; it says nothing about one already under way. A stream that
/// has gone idle does not count: that is precisely how a call ends in an
/// app that keeps its capture stream open afterwards, which is most of them.
fn still_capturing(
    clients: &[PwCaptureClient],
    app_key: Option<&str>,
    calls: &HashMap<u32, CallEvidence>,
) -> bool {
    clients.iter().any(|client| {
        !client.is_meetily()
            && client.capture
            && is_live(client, calls)
            && apps::classify(&client.match_keys()).map(|app| app.key) == app_key
    })
}

/// Why a capture stream isn't counted as the start of a meeting.
///
/// The two reasons look the same from the outside and lift on opposite
/// signals, which is the whole point of telling them apart.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum MuteReason {
    /// It was already capturing before we started watching, and nothing
    /// about it says a call. Joining one is news, so that lifts it.
    Stale,
    /// A recording of this call was stopped by hand. Being in a call is
    /// exactly the state it was stopped in, so only leaving lifts it —
    /// otherwise stopping a recording mid-call starts another one
    /// seconds later, over and over.
    Stopped,
}

/// Streams that don't count as the start of a meeting, and why.
type Muted = HashMap<u32, MuteReason>;

/// Drop the mutes whose reason has stopped being true.
///
/// A stream that has closed or gone idle is unmuted whatever the reason:
/// both mean the app let go of the microphone, and what it does next is
/// news. The call probe is what separates the two — see [`MuteReason`].
fn lift_mutes(clients: &[PwCaptureClient], calls: &HashMap<u32, CallEvidence>, muted: &mut Muted) {
    muted.retain(|id, reason| {
        let live = clients
            .iter()
            .any(|client| client.node_id == *id && client.running);
        if !live {
            return false;
        }
        match reason {
            MuteReason::Stale => call_verdict(calls, *id) != Some(CallProbe::Connected),
            // Held while we cannot tell: "we couldn't read the sockets" is
            // not the same answer as "the call is over", and only the
            // second one should start a recording the user just stopped.
            MuteReason::Stopped => call_verdict(calls, *id) != Some(CallProbe::Idle),
        }
    });
}

/// Stop counting the streams that were already open at a moment when we
/// started looking for a *new* meeting.
///
/// Only the apps that hold the microphone outside of calls - see
/// [`MeetingApp::idles_with_mic`]. Discord opens a capture stream when it
/// launches and keeps it `Running` all day; a browser holds one for
/// whichever page asked, as likely a transcription demo in a background
/// tab as a call. Treating either as a meeting means a recording that
/// starts by itself and then never stops, because the stream never goes
/// away. Everything else gets the benefit of the doubt: if Zoom has the
/// microphone you are in a call, and launching Meetily mid-call should
/// still record it.
///
/// The mute is per stream, not per app, which is what makes it recover: the
/// same browser joining a real call opens a second stream with its own
/// object id, and that one is heard.
fn mute_stale_streams(
    clients: &[PwCaptureClient],
    calls: &HashMap<u32, CallEvidence>,
    muted: &mut Muted,
    why: &str,
) {
    for client in clients {
        if client.is_meetily() || !client.capture {
            continue;
        }
        if !apps::classify(&client.match_keys()).is_some_and(|app| app.idles_with_mic) {
            continue;
        }
        // Unless we can see it is genuinely in a call, in which case there
        // is nothing stale about it and launching Meetily mid-call should
        // record it like any other.
        if call_verdict(calls, client.node_id) == Some(CallProbe::Connected) {
            continue;
        }
        // Never downgrades a stream muted by a manual stop: that mute
        // outlasts this one, and lifts on the opposite signal.
        if !muted.contains_key(&client.node_id) {
            muted.insert(client.node_id, MuteReason::Stale);
            log::info!(
                "Not counting {} (stream {}) as a meeting: it {}",
                client.node_name,
                client.node_id,
                why
            );
        }
    }
}

/// Stop counting every stream behind a meeting, ambiguous or not.
fn mute_meeting_streams(clients: &[PwCaptureClient], app_key: Option<&str>, muted: &mut Muted) {
    for client in clients {
        if client.is_meetily() || !client.capture {
            continue;
        }
        if apps::classify(&client.match_keys()).map(|app| app.key) == app_key {
            muted.insert(client.node_id, MuteReason::Stopped);
        }
    }
}

/// A one-line account of every capture stream and the verdict on it, for
/// the log. Deliberately verbose about the match keys: when an app isn't
/// recognised, the keys are the only thing that says what pattern it
/// *should* have been matched by.
fn describe_clients(
    clients: &[PwCaptureClient],
    settings: &MeetingDetectionSettings,
    calls: &HashMap<u32, CallEvidence>,
    muted: &Muted,
    candidate: Option<&DetectedMeeting>,
) -> String {
    let picked = match candidate {
        Some(meeting) => format!("picked {}", meeting.app_label),
        None => "picked nothing".to_string(),
    };

    if clients.iter().all(|client| !client.capture) {
        return format!("nothing is capturing audio, {}", picked);
    }

    let streams: Vec<String> = clients
        .iter()
        .filter(|client| client.capture)
        .map(|client| {
            if client.is_meetily() {
                return format!("[{}: us, skipped]", client.node_name);
            }
            let app = apps::classify(&client.match_keys());
            let verdict = if !client.running {
                "open but idle, nothing flowing".to_string()
            } else if call_verdict(calls, client.node_id) == Some(CallProbe::Idle) {
                "capturing, but not connected to a call".to_string()
            } else if let Some(reason) = muted.get(&client.node_id) {
                match reason {
                    MuteReason::Stale => "muted until it reopens".to_string(),
                    MuteReason::Stopped => {
                        "you stopped this one; muted until the call ends".to_string()
                    }
                }
            } else {
                match app {
                    Some(app) if settings.accepts(Some(app.key)) => format!("{}, watched", app.key),
                    Some(app) => format!("{}, in your ignore list", app.key),
                    None if settings.detect_unknown_apps => "unrecognised, watched".to_string(),
                    None => "unrecognised, not watched".to_string(),
                }
            };
            format!(
                "[{} #{}: {} (keys: {}; media.name: {}; media.role: {}; pid: {}; call: {})]",
                client.node_name,
                client.node_id,
                verdict,
                client.match_keys().join(" / "),
                client.media_name.as_deref().unwrap_or("-"),
                client.media_role.as_deref().unwrap_or("-"),
                client
                    .pid
                    .map_or_else(|| "-".to_string(), |pid| pid.to_string()),
                describe_call(calls.get(&client.node_id)),
            )
        })
        .collect();

    // Playback is listed but not judged. A call is two-way, and an app
    // that holds the microphone forever is only distinguishable from one
    // in a call by whether it is also playing voice back — so the log
    // carries the other half of the picture even though nothing acts on
    // it yet.
    let playback: Vec<String> = clients
        .iter()
        .filter(|client| !client.capture && client.running)
        .map(|client| {
            format!(
                "[{} #{}: {} (media.name: {})]",
                client.node_name,
                client.node_id,
                apps::classify(&client.match_keys()).map_or("unrecognised", |app| app.key),
                client.media_name.as_deref().unwrap_or("-"),
            )
        })
        .collect();

    format!(
        "{} capturing {} — {} — playing back {}",
        streams.len(),
        streams.join(" "),
        picked,
        if playback.is_empty() {
            "nothing".to_string()
        } else {
            playback.join(" ")
        }
    )
}

/// The meeting-worthy stream to latch onto, if any. Known apps win over
/// unknown ones so a browser in the background doesn't shadow Zoom.
fn pick_meeting(
    clients: &[PwCaptureClient],
    settings: &MeetingDetectionSettings,
    calls: &HashMap<u32, CallEvidence>,
    muted: &Muted,
) -> Option<DetectedMeeting> {
    let mut best: Option<(u8, DetectedMeeting)> = None;

    for client in clients {
        if client.is_meetily() || !client.capture {
            continue;
        }
        // Open but idle, or open with no call on the other end of it: the
        // app is holding a stream, not using one. Discord keeps one for as
        // long as it is running, links and all.
        if !is_live(client, calls) {
            continue;
        }
        let app = apps::classify(&client.match_keys());
        let key = app.map(|a| a.key);
        if !settings.accepts(key) {
            continue;
        }
        if muted.contains_key(&client.node_id) {
            continue;
        }

        let rank = apps::rank(app);
        if best
            .as_ref()
            .is_some_and(|(best_rank, _)| *best_rank <= rank)
        {
            continue;
        }

        best = Some((
            rank,
            DetectedMeeting {
                app_key: key.map(str::to_string),
                app_label: display_label(client, app),
                source: client
                    .binary
                    .clone()
                    .or_else(|| client.app_name.clone())
                    .unwrap_or_else(|| client.node_name.clone()),
                detected_at: chrono::Utc::now().to_rfc3339(),
                auto_recording: false,
            },
        ));
    }

    best.map(|(_, meeting)| meeting)
}

/// Watch the audio graph until this loop's generation is retired.
///
/// Debouncing is deliberate on both ends. Apps grab the mic briefly when
/// you open their audio settings or run a device test, and they drop it for
/// a moment when you move between calls — reacting to either would mean
/// spurious recordings and truncated ones.
async fn monitor_loop<R: Runtime>(detector: MeetingDetector, app: AppHandle<R>, generation: u64) {
    log::info!(
        "Meeting detection monitor started (generation {})",
        generation
    );

    let mut present_secs: u64 = 0;
    let mut absent_secs: u64 = 0;
    let mut switch_secs: u64 = 0;
    let mut last_snapshot: Option<String> = None;
    // Streams that don't count as the start of a meeting until they close
    // and reopen. See `mute_stale_streams` and `mute_meeting_streams`.
    let mut muted: Muted = Muted::new();
    let mut primed = false;
    let mut was_recording = false;

    loop {
        if detector.generation.load(Ordering::SeqCst) != generation {
            log::info!(
                "Meeting detection monitor generation {} retired",
                generation
            );
            return;
        }

        let settings = detector.settings().await;
        if !settings.enabled {
            detector.monitoring.store(false, Ordering::Relaxed);
            return;
        }

        let interval = settings.poll_interval_secs;

        match capture_clients().await {
            Ok(clients) => {
                *detector.last_error.write().await = None;

                let calls = probe_calls(&clients);

                // A stream that has closed or gone idle can no longer be
                // muted, and depending on why it was muted, joining or
                // leaving a call lifts it too. See `lift_mutes`.
                lift_mutes(&clients, &calls, &mut muted);

                // A recording that was running and now isn't was stopped by
                // hand. Starting it again on the next poll would be
                // maddening, so every stream behind that meeting goes quiet
                // until it reopens - otherwise an app that holds the
                // microphone indefinitely (Discord idling in a voice
                // channel is the usual one) is re-detected seconds later,
                // forever.
                let recording = recording_commands::is_recording().await;
                if was_recording && !recording {
                    let ended = detector.active.write().await.take();
                    if let Some(ended) = ended {
                        log::info!(
                            "Recording stopped by hand during {}; not watching it again until it reopens",
                            ended.app_label
                        );
                        mute_meeting_streams(&clients, ended.app_key.as_deref(), &mut muted);
                        detector.auto_recording.store(false, Ordering::SeqCst);
                        present_secs = 0;
                        absent_secs = 0;
                        switch_secs = 0;
                    }
                }
                was_recording = recording;

                // The first look at the graph is the same situation as the
                // moment just after a meeting ends: whatever already holds
                // the microphone has held it since before we were watching,
                // so it isn't news.
                if !primed {
                    primed = true;
                    mute_stale_streams(
                        &clients,
                        &calls,
                        &mut muted,
                        "was already capturing when monitoring started",
                    );
                }

                let candidate = pick_meeting(&clients, &settings, &calls, &muted);
                let active = detector.active.read().await.clone();

                // Log the audio graph whenever it changes. Diagnosing "why
                // wasn't my call detected" from the outside is guesswork;
                // this puts the whole decision in the normal log, without
                // needing RUST_LOG or the settings panel, and without
                // spamming a line every poll interval.
                let snapshot =
                    describe_clients(&clients, &settings, &calls, &muted, candidate.as_ref());
                if last_snapshot.as_deref() != Some(snapshot.as_str()) {
                    log::info!("Meeting detection: {}", snapshot);
                    last_snapshot = Some(snapshot);
                }

                match (active, candidate) {
                    // A meeting is running. The only question that decides
                    // whether it is over is whether the app behind it still
                    // has the microphone - not whether *something* does.
                    // Asking the looser question is what let a background
                    // browser tab inherit a Discord call the moment it
                    // ended and keep the recording running indefinitely.
                    (Some(current), candidate) => {
                        present_secs = 0;

                        if !still_capturing(&clients, current.app_key.as_deref(), &calls) {
                            switch_secs = 0;
                            absent_secs = absent_secs.saturating_add(interval);
                            if absent_secs >= settings.stop_grace_secs {
                                absent_secs = 0;
                                end_meeting(&detector, &app, &settings).await;
                                // Anything still holding the microphone now
                                // has held it right through someone else's
                                // meeting. That is not a new one starting.
                                mute_stale_streams(
                                    &clients,
                                    &calls,
                                    &mut muted,
                                    "was already capturing when the last meeting ended",
                                );
                            } else {
                                log::info!(
                                    "{} stopped capturing; ending the meeting in {}s unless it comes back",
                                    current.app_label,
                                    settings.stop_grace_secs - absent_secs
                                );
                            }
                        } else {
                            absent_secs = 0;
                            // Stronger evidence than what we latched onto -
                            // you were on a browser call and then joined
                            // Zoom. Escalating like that is worth a switch;
                            // the reverse never is, because the weaker
                            // signal is usually just a tab left open.
                            let promoted = candidate.filter(|next| {
                                next.app_key != current.app_key
                                    && apps::rank(apps::by_key(next.app_key.as_deref()))
                                        < apps::rank(apps::by_key(current.app_key.as_deref()))
                            });
                            match promoted {
                                Some(next) => {
                                    switch_secs = switch_secs.saturating_add(interval);
                                    if switch_secs >= settings.start_delay_secs {
                                        switch_secs = 0;
                                        end_meeting(&detector, &app, &settings).await;
                                        begin_meeting(&detector, &app, next, &settings).await;
                                    }
                                }
                                None => switch_secs = 0,
                            }
                        }
                    }
                    // Nothing yet - count how long a candidate has been
                    // holding the microphone.
                    (None, Some(candidate)) => {
                        absent_secs = 0;
                        switch_secs = 0;
                        present_secs = present_secs.saturating_add(interval);
                        if present_secs >= settings.start_delay_secs {
                            present_secs = 0;
                            begin_meeting(&detector, &app, candidate, &settings).await;
                        }
                    }
                    (None, None) => {
                        present_secs = 0;
                        absent_secs = 0;
                        switch_secs = 0;
                    }
                }
            }
            Err(e) => {
                log::warn!("Meeting detection probe failed: {}", e);
                *detector.last_error.write().await = Some(e.to_string());
            }
        }

        tokio::time::sleep(Duration::from_secs(interval)).await;
    }
}

/// Confirm a meeting: announce it, and start recording if asked to.
async fn begin_meeting<R: Runtime>(
    detector: &MeetingDetector,
    app: &AppHandle<R>,
    mut meeting: DetectedMeeting,
    settings: &MeetingDetectionSettings,
) {
    log::info!(
        "Meeting detected: {} (via {})",
        meeting.app_label,
        meeting.source
    );

    if settings.auto_start_recording {
        if recording_commands::is_recording().await {
            // Already recording — nothing to start, and nothing of ours to
            // stop later either.
            log::info!("Meeting detected while already recording; leaving it alone");
        } else {
            let name = meeting_name(&meeting.app_label);
            match recording_commands::start_recording_with_meeting_name(
                app.clone(),
                Some(name.clone()),
            )
            .await
            {
                Ok(()) => {
                    log::info!("Auto-started recording for detected meeting: {}", name);
                    meeting.auto_recording = true;
                    detector.auto_recording.store(true, Ordering::SeqCst);
                }
                Err(e) => log::error!("Failed to auto-start recording: {}", e),
            }
        }
    }

    if settings.notify_on_detection {
        notify(app, &meeting).await;
    }

    *detector.active.write().await = Some(meeting.clone());
    if let Err(e) = app.emit("meeting-detected", &meeting) {
        log::error!("Failed to emit meeting-detected: {}", e);
    }
}

/// The meeting is over: announce it, and stop the recording if it's ours.
async fn end_meeting<R: Runtime>(
    detector: &MeetingDetector,
    app: &AppHandle<R>,
    settings: &MeetingDetectionSettings,
) {
    let meeting = detector.active.write().await.take();
    let Some(meeting) = meeting else {
        return;
    };
    log::info!("Meeting ended: {}", meeting.app_label);

    let ours = detector.auto_recording.swap(false, Ordering::SeqCst);
    if settings.auto_stop_recording && ours && recording_commands::is_recording().await {
        stop_recording(app).await;
    }

    if let Err(e) = app.emit("meeting-ended", &meeting) {
        log::error!("Failed to emit meeting-ended: {}", e);
    }
}

/// Stop the recording the same way the tray does — same save path shape,
/// same `recording-stop-complete` event the frontend listens for to persist
/// the transcript — minus the focus grab, since nobody asked us to jump in
/// front of what they were doing.
async fn stop_recording<R: Runtime>(app: &AppHandle<R>) {
    let Ok(data_dir) = app.path().app_data_dir() else {
        log::error!("Failed to resolve app data dir; cannot auto-stop recording");
        return;
    };
    let timestamp = chrono::Local::now().format("%Y-%m-%dT%H-%M-%S").to_string();
    let save_path = data_dir.join(format!("recording-{}.wav", timestamp));

    match recording_commands::stop_recording(
        app.clone(),
        recording_commands::RecordingArgs {
            save_path: save_path.to_string_lossy().to_string(),
        },
    )
    .await
    {
        Ok(()) => {
            log::info!("Auto-stopped recording after meeting ended");
            if let Err(e) = app.emit("recording-stop-complete", true) {
                log::error!("Failed to emit recording-stop-complete: {}", e);
            }
        }
        Err(e) => log::error!("Failed to auto-stop recording: {}", e),
    }
}

/// Name auto-started meetings after the app and the time, so a day of them
/// is still tellable apart in the sidebar.
fn meeting_name(app_label: &str) -> String {
    format!(
        "{} - {}",
        app_label,
        chrono::Local::now().format("%b %-d, %-I:%M %p")
    )
}

/// Desktop notification for a detected meeting. Best-effort: the manager
/// isn't available until the notification system has initialized, and a
/// missed notification is not worth failing the detection over.
async fn notify<R: Runtime>(app: &AppHandle<R>, meeting: &DetectedMeeting) {
    let Some(state) = app.try_state::<NotificationManagerState<R>>() else {
        return;
    };
    let manager_lock = state.read().await;
    let Some(manager) = manager_lock.as_ref() else {
        return;
    };

    let body = if meeting.auto_recording {
        format!("{} started - recording now.", meeting.app_label)
    } else {
        format!("{} started.", meeting.app_label)
    };
    let notification =
        Notification::new("Meeting detected", body, NotificationType::RecordingStarted);
    if let Err(e) = manager.show_notification(notification).await {
        log::warn!("Failed to show meeting detected notification: {}", e);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A probe result standing in for one the tests didn't run. The
    /// counts only reach the log, so they are left at zero.
    fn answer(probe: CallProbe) -> CallEvidence {
        CallEvidence {
            probe,
            connected: 0,
            open: 0,
        }
    }

    fn nothing_asked() -> HashMap<u32, CallEvidence> {
        HashMap::new()
    }

    fn nothing_muted() -> Muted {
        Muted::new()
    }

    fn client(node_name: &str, binary: Option<&str>) -> PwCaptureClient {
        stream(0, node_name, binary)
    }

    fn stream(node_id: u32, node_name: &str, binary: Option<&str>) -> PwCaptureClient {
        PwCaptureClient {
            node_id,
            linked: true,
            running: true,
            capture: true,
            node_name: node_name.to_string(),
            app_name: None,
            binary: binary.map(str::to_string),
            app_id: None,
            client_name: None,
            pid: None,
            media_role: None,
            media_name: None,
        }
    }

    #[test]
    fn our_own_capture_stream_is_never_a_meeting() {
        // Otherwise recording would keep the detector latched forever.
        let clients = vec![client(pw::MEETILY_CAPTURE_NODE, Some("meetily"))];
        assert!(pick_meeting(
            &clients,
            &MeetingDetectionSettings::default(),
            &nothing_asked(),
            &nothing_muted(),
        )
        .is_none());
    }

    #[test]
    fn a_known_app_holding_the_mic_is_a_meeting() {
        let clients = vec![client("zoom", Some("zoom"))];
        let meeting = pick_meeting(
            &clients,
            &MeetingDetectionSettings::default(),
            &nothing_asked(),
            &nothing_muted(),
        )
        .unwrap();
        assert_eq!(meeting.app_key.as_deref(), Some("zoom"));
    }

    #[test]
    fn unknown_apps_are_ignored_unless_asked_for() {
        let clients = vec![client("audacity", Some("audacity"))];
        let mut settings = MeetingDetectionSettings::default();
        assert!(pick_meeting(&clients, &settings, &nothing_asked(), &nothing_muted()).is_none());

        settings.detect_unknown_apps = true;
        let meeting =
            pick_meeting(&clients, &settings, &nothing_asked(), &nothing_muted()).unwrap();
        assert!(meeting.app_key.is_none());
        assert_eq!(meeting.app_label, "audacity");
    }

    #[test]
    fn a_flatpak_client_with_only_an_app_id_is_classified() {
        // The Flatpak sandbox restricts /proc, so `application.process.binary`
        // is frequently absent and the app id is all we get.
        let mut settings = MeetingDetectionSettings::default();
        settings.ignored_apps.clear();
        let clients = vec![PwCaptureClient {
            node_id: 0,
            linked: true,
            running: true,
            capture: true,
            node_name: "node-57".to_string(),
            app_name: None,
            binary: None,
            app_id: Some("com.discordapp.Discord".to_string()),
            client_name: None,
            pid: None,
            media_role: None,
            media_name: None,
        }];
        let meeting =
            pick_meeting(&clients, &settings, &nothing_asked(), &nothing_muted()).unwrap();
        assert_eq!(meeting.app_key.as_deref(), Some("discord"));
    }

    #[test]
    fn an_electron_client_is_named_by_its_pipewire_client_not_its_node() {
        // Observed shape: Flatpak Discord's stream node and its own
        // `application.name` are both "WEBRTC VoiceEngine", with no binary
        // and no app id anywhere on the node. Only the owning client object
        // knows it is Discord.
        let mut settings = MeetingDetectionSettings::default();
        settings.ignored_apps.clear();
        let clients = vec![PwCaptureClient {
            node_id: 0,
            linked: true,
            running: true,
            capture: true,
            node_name: "WEBRTC VoiceEngine".to_string(),
            app_name: Some("WEBRTC VoiceEngine".to_string()),
            binary: None,
            app_id: None,
            client_name: Some("Discord".to_string()),
            pid: None,
            media_role: None,
            media_name: None,
        }];
        let meeting =
            pick_meeting(&clients, &settings, &nothing_asked(), &nothing_muted()).unwrap();
        assert_eq!(meeting.app_key.as_deref(), Some("discord"));
    }

    #[test]
    fn an_unidentifiable_webrtc_stream_still_counts_as_a_call() {
        // Same stream with the client join coming up empty. We can no
        // longer say which app it is, but "something is running the WebRTC
        // voice engine" is still a call.
        let mut settings = MeetingDetectionSettings::default();
        settings.ignored_apps.clear();
        let clients = vec![client("WEBRTC VoiceEngine", None)];
        let meeting =
            pick_meeting(&clients, &settings, &nothing_asked(), &nothing_muted()).unwrap();
        assert_eq!(meeting.app_key.as_deref(), Some("webrtc"));
    }

    #[test]
    fn a_muted_stream_steps_aside_for_the_next_one() {
        // Discord idling in a voice channel holds the microphone forever.
        // Once its recording has been stopped by hand its stream is muted,
        // and the browser call behind it becomes the meeting - even though
        // Discord would otherwise outrank it.
        let mut settings = MeetingDetectionSettings::default();
        settings.ignored_apps.clear();
        let clients = vec![
            stream(11, "discord", Some("Discord")),
            stream(12, "brave", Some("brave")),
        ];

        let mut muted = Muted::new();
        assert_eq!(
            pick_meeting(&clients, &settings, &nothing_asked(), &muted)
                .unwrap()
                .app_key
                .as_deref(),
            Some("discord")
        );

        mute_meeting_streams(&clients, Some("discord"), &mut muted);
        assert_eq!(muted, Muted::from([(11, MuteReason::Stopped)]));
        assert_eq!(
            pick_meeting(&clients, &settings, &nothing_asked(), &muted)
                .unwrap()
                .app_key
                .as_deref(),
            Some("browser")
        );
    }

    #[test]
    fn stopping_a_recording_mid_call_does_not_start_another_one() {
        // The reported shape: stop the recording while still in the Discord
        // call. The stream is still there, still running, still connected —
        // every signal that would otherwise say "a meeting is happening".
        let mut settings = MeetingDetectionSettings::default();
        settings.ignored_apps.clear();
        let clients = vec![stream(232, "WEBRTC VoiceEngine", Some("Discord"))];
        let in_a_call = HashMap::from([(232, answer(CallProbe::Connected))]);

        let mut muted = Muted::new();
        mute_meeting_streams(&clients, Some("discord"), &mut muted);

        // Being in the call is what it was stopped in, so it cannot lift
        // the mute — however many polls go by.
        for _ in 0..3 {
            lift_mutes(&clients, &in_a_call, &mut muted);
            assert!(pick_meeting(&clients, &settings, &in_a_call, &muted).is_none());
        }

        // Leaving the call does lift it, and the next call is heard.
        let left = HashMap::from([(232, answer(CallProbe::Idle))]);
        lift_mutes(&clients, &left, &mut muted);
        assert!(muted.is_empty());
        assert!(pick_meeting(&clients, &settings, &in_a_call, &muted).is_some());
    }

    #[test]
    fn a_stale_mute_still_lifts_when_the_call_starts() {
        // The opposite reason, unchanged: Discord idling at startup is
        // written off, and joining a call is what makes it news again.
        let clients = vec![stream(232, "WEBRTC VoiceEngine", Some("Discord"))];

        let mut muted = Muted::new();
        mute_stale_streams(&clients, &nothing_asked(), &mut muted, "test");
        assert_eq!(muted, Muted::from([(232, MuteReason::Stale)]));

        let in_a_call = HashMap::from([(232, answer(CallProbe::Connected))]);
        lift_mutes(&clients, &in_a_call, &mut muted);
        assert!(muted.is_empty());
    }

    #[test]
    fn a_stopped_stream_that_closes_is_heard_again() {
        // Quitting the app entirely takes the stream with it, which ends
        // the call as surely as leaving it does.
        let clients = vec![stream(232, "WEBRTC VoiceEngine", Some("Discord"))];
        let mut muted = Muted::new();
        mute_meeting_streams(&clients, Some("discord"), &mut muted);

        lift_mutes(&[], &nothing_asked(), &mut muted);
        assert!(muted.is_empty());
    }

    #[test]
    fn a_tab_that_already_had_the_mic_is_not_a_new_meeting() {
        // The shape that made this necessary: a transcription page left
        // open in a background tab. It holds the microphone from before the
        // app launched and never lets go, so as "a meeting" it would start
        // a recording that could never end.
        let mut settings = MeetingDetectionSettings::default();
        settings.ignored_apps.clear();
        let clients = vec![stream(31, "brave input", Some("brave"))];

        let mut muted = Muted::new();
        mute_stale_streams(&clients, &nothing_asked(), &mut muted, "test");
        assert!(pick_meeting(&clients, &settings, &nothing_asked(), &muted).is_none());

        // The same browser joining a real call opens a second stream, and
        // that one has never been written off.
        let joined = vec![
            stream(31, "brave input", Some("brave")),
            stream(32, "brave input", Some("brave")),
        ];
        assert_eq!(
            pick_meeting(&joined, &settings, &nothing_asked(), &muted)
                .unwrap()
                .app_key
                .as_deref(),
            Some("browser")
        );
    }

    #[test]
    fn a_voice_engine_already_open_at_startup_is_not_a_meeting() {
        // The shape that kept auto-starting a recording at launch: Discord
        // running in the background with node #232 open, named only
        // "WEBRTC VoiceEngine". Nothing there says a call is happening.
        let mut settings = MeetingDetectionSettings::default();
        settings.ignored_apps.clear();
        let clients = vec![stream(232, "WEBRTC VoiceEngine", None)];

        let mut muted = Muted::new();
        mute_stale_streams(&clients, &nothing_asked(), &mut muted, "test");
        assert!(pick_meeting(&clients, &settings, &nothing_asked(), &muted).is_none());

        // Actually joining a call opens a new stream, which is heard.
        let joined = vec![
            stream(232, "WEBRTC VoiceEngine", None),
            stream(240, "WEBRTC VoiceEngine", None),
        ];
        assert!(pick_meeting(&joined, &settings, &nothing_asked(), &muted).is_some());
    }

    #[test]
    fn a_connected_call_outweighs_the_stale_stream_rule() {
        // Discord's node is indistinguishable in the audio graph whether or
        // not you are in a voice channel, so the socket probe is the only
        // thing that can tell them apart - and when it can, it decides.
        let mut settings = MeetingDetectionSettings::default();
        settings.ignored_apps.clear();
        let clients = vec![stream(232, "WEBRTC VoiceEngine", Some("Discord"))];

        let in_a_call = HashMap::from([(232, answer(CallProbe::Connected))]);
        let mut muted = Muted::new();
        mute_stale_streams(&clients, &in_a_call, &mut muted, "test");
        assert!(muted.is_empty(), "a live call is not a stale stream");
        assert_eq!(
            pick_meeting(&clients, &settings, &in_a_call, &muted)
                .unwrap()
                .app_key
                .as_deref(),
            Some("discord")
        );

        // And the same stream with nothing on the other end of it is not a
        // meeting even before any muting happens.
        let idle = HashMap::from([(232, answer(CallProbe::Idle))]);
        assert!(pick_meeting(&clients, &settings, &idle, &nothing_muted()).is_none());
        assert!(!still_capturing(&clients, Some("discord"), &idle));
    }

    #[test]
    fn a_call_hanging_up_ends_the_meeting() {
        // The stream stays open and `Running` afterwards - Discord never
        // closes it - so the socket going away is the whole signal.
        let clients = vec![stream(232, "WEBRTC VoiceEngine", Some("Discord"))];
        assert!(still_capturing(
            &clients,
            Some("discord"),
            &HashMap::from([(232, answer(CallProbe::Connected))])
        ));
        assert!(!still_capturing(
            &clients,
            Some("discord"),
            &HashMap::from([(232, answer(CallProbe::Idle))])
        ));
    }

    #[test]
    fn an_unanswerable_probe_changes_nothing() {
        // A sandboxed client whose pid we cannot read must fall back to the
        // old behaviour, not be written off as "not in a call".
        let mut settings = MeetingDetectionSettings::default();
        settings.ignored_apps.clear();
        let clients = vec![stream(41, "zoom", Some("zoom"))];
        assert!(pick_meeting(&clients, &settings, &nothing_asked(), &nothing_muted()).is_some());
        assert!(still_capturing(&clients, Some("zoom"), &nothing_asked()));

        // Asked and unreadable has to behave the same as never asked. It
        // is a separate value only so the log can say which happened.
        let unreadable = HashMap::from([(41, answer(CallProbe::Unreadable))]);
        assert!(pick_meeting(&clients, &settings, &unreadable, &nothing_muted()).is_some());
        assert!(still_capturing(&clients, Some("zoom"), &unreadable));
    }

    #[test]
    fn a_conferencing_client_already_in_a_call_is_not_written_off() {
        // Only ambiguous apps get the doubt. Launching Meetily while
        // already in a Zoom call should still record it.
        let mut settings = MeetingDetectionSettings::default();
        settings.ignored_apps.clear();
        let clients = vec![stream(41, "zoom", Some("zoom"))];

        let mut muted = Muted::new();
        mute_stale_streams(&clients, &nothing_asked(), &mut muted, "test");
        assert!(muted.is_empty());
        assert!(pick_meeting(&clients, &settings, &nothing_asked(), &muted).is_some());
    }

    #[test]
    fn an_idle_stream_is_not_a_meeting() {
        // The shape this was written for: node #232, "WEBRTC VoiceEngine",
        // open continuously for as long as Discord is running and outliving
        // whole restarts of Meetily. Holding a capture stream is not being
        // in a call; having audio actually flowing through it is. The
        // node stays linked to the source the whole time, which is why
        // link state was no help here.
        let mut settings = MeetingDetectionSettings::default();
        settings.ignored_apps.clear();

        let mut idle = stream(232, "WEBRTC VoiceEngine", None);
        idle.running = false;
        assert!(pick_meeting(
            &[idle.clone()],
            &settings,
            &nothing_asked(),
            &nothing_muted()
        )
        .is_none());
        assert!(!still_capturing(&[idle], Some("webrtc"), &nothing_asked()));

        let live = stream(232, "WEBRTC VoiceEngine", None);
        assert!(pick_meeting(&[live], &settings, &nothing_asked(), &nothing_muted()).is_some());
    }

    #[test]
    fn a_call_ending_ends_the_meeting_even_if_the_stream_stays_open() {
        // Which is the normal case: the app drops back to an idle stream
        // rather than closing it, so waiting for the node to disappear
        // means waiting forever.
        let mut after = stream(232, "WEBRTC VoiceEngine", None);
        after.running = false;
        assert!(!still_capturing(&[after], Some("webrtc"), &nothing_asked()));
    }

    #[test]
    fn a_meeting_runs_until_its_own_app_lets_go() {
        // A browser tab capturing alongside a call does not keep the call
        // alive, and does not inherit it either.
        let during = vec![
            stream(51, "discord", Some("Discord")),
            stream(52, "brave input", Some("brave")),
        ];
        assert!(still_capturing(&during, Some("discord"), &nothing_asked()));

        let after = vec![stream(52, "brave input", Some("brave"))];
        assert!(!still_capturing(&after, Some("discord"), &nothing_asked()));
    }

    #[test]
    fn only_stronger_evidence_can_take_over_a_running_meeting() {
        // Zoom starting during a browser call is worth a switch; a browser
        // tab opening during a Zoom call is not.
        let zoom = apps::rank(apps::by_key(Some("zoom")));
        let browser = apps::rank(apps::by_key(Some("browser")));
        let unknown = apps::rank(apps::by_key(None));
        assert!(zoom < browser);
        assert!(browser < unknown);
    }

    #[test]
    fn a_dedicated_client_beats_a_browser_holding_the_mic() {
        // The exact shape that broke this: a browser tab sitting on a
        // transcription page while a real call is happening elsewhere.
        let mut settings = MeetingDetectionSettings::default();
        settings.ignored_apps.clear();
        let clients = vec![
            client("firefox", Some("firefox")),
            client("discord", Some("discord")),
        ];
        let meeting =
            pick_meeting(&clients, &settings, &nothing_asked(), &nothing_muted()).unwrap();
        assert_eq!(meeting.app_key.as_deref(), Some("discord"));
    }

    #[test]
    fn enumeration_order_does_not_decide_the_winner() {
        let mut settings = MeetingDetectionSettings::default();
        settings.ignored_apps.clear();
        let forward = vec![
            client("firefox", Some("firefox")),
            client("zoom", Some("zoom")),
        ];
        let reversed = vec![
            client("zoom", Some("zoom")),
            client("firefox", Some("firefox")),
        ];
        assert_eq!(
            pick_meeting(&forward, &settings, &nothing_asked(), &nothing_muted())
                .unwrap()
                .app_key,
            pick_meeting(&reversed, &settings, &nothing_asked(), &nothing_muted())
                .unwrap()
                .app_key,
        );
    }

    #[test]
    fn ignored_apps_do_not_trigger() {
        let clients = vec![client("discord", Some("discord"))];
        let mut settings = MeetingDetectionSettings::default();
        settings.ignored_apps = vec!["discord".to_string()];
        assert!(pick_meeting(&clients, &settings, &nothing_asked(), &nothing_muted()).is_none());
    }

    #[test]
    fn a_known_app_wins_over_an_unknown_one() {
        let mut settings = MeetingDetectionSettings::default();
        settings.detect_unknown_apps = true;
        let clients = vec![
            client("audacity", Some("audacity")),
            client("zoom", Some("zoom")),
        ];
        let meeting =
            pick_meeting(&clients, &settings, &nothing_asked(), &nothing_muted()).unwrap();
        assert_eq!(meeting.app_key.as_deref(), Some("zoom"));
    }
}
