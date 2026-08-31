//! Recording one application instead of everything the speakers play.
//!
//! The system-audio side normally captures a sink's monitor, which is the
//! whole mix — the call, the video in the other window, the notification
//! that arrived mid-sentence. A selection of `app:<key>` says to capture
//! one application's playback instead, by tapping that app's own output
//! ports (see [`crate::audio::pw`]'s app capture).
//!
//! An app is named by a key rather than a node id because node ids do not
//! survive anything: Discord opens a fresh stream node per call, and a
//! new one again for a notification sound. Known conferencing apps use
//! the same keys as the meeting detector's table, so "Discord" means the
//! same thing on both screens; anything else playing audio is offered
//! under an identifier taken from the stream itself.
//!
//! The whole feature is behind [`enabled`], which mirrors the
//! `app_audio_capture` recording preference. Tapping ports directly is a
//! good deal more moving parts than reading a monitor, and if it
//! misbehaves the switch has to put things back to the ordinary path
//! without the user having to work out which selection caused it.

use anyhow::Result;
use log::{info, warn};
use serde::{Deserialize, Serialize};
use std::sync::atomic::{AtomicBool, Ordering};

use crate::audio::pw::{self, PwCaptureClient, PwGraphSnapshot};
use crate::meeting_detector::apps;
use crate::meeting_detector::voice_socket::{self, CallProbe};

/// Prefix marking a system-audio selection as an application rather than
/// a device. Shared with the frontend — see `lib/audioDeviceModes.ts`.
pub const APP_PREFIX: &str = "app:";

/// Whether per-app capture is switched on. Mirrors the recording
/// preference; consulted where a stream is opened, which has no route to
/// the database.
static ENABLED: AtomicBool = AtomicBool::new(false);

pub fn set_enabled(on: bool) {
    if ENABLED.swap(on, Ordering::Relaxed) != on {
        info!(
            "Per-app system audio capture is now {}",
            if on { "on" } else { "off" }
        );
    }
}

pub fn enabled() -> bool {
    ENABLED.load(Ordering::Relaxed)
}

/// The app key in a selection, if it is one and the feature is on.
///
/// Returns `None` for a device selection, and also for an app selection
/// made while the feature was on and left behind when it was switched
/// off — the caller falls back to the ordinary sink capture, which is the
/// point of the switch.
pub fn selected_app(selection: &str) -> Option<&str> {
    let key = selection.strip_prefix(APP_PREFIX)?;
    if !enabled() {
        warn!(
            "'{}' asks for per-app capture, which is switched off — \
             recording the default system audio instead",
            selection
        );
        return None;
    }
    Some(key)
}

/// One application the system-audio picker can point at.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AppAudioSource {
    /// Stable identifier, stored in the selection as `app:<key>`.
    pub key: String,
    pub label: String,
    /// Extra context for the picker, where the label isn't obvious.
    pub note: Option<String>,
    /// Whether the app has a playback stream open right now. Not a
    /// requirement for picking it — you set this up before the call, not
    /// during — but worth showing.
    pub playing: bool,
}

/// What the picker needs to render, in one call.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AppAudioCatalog {
    /// Mirrors the `app_audio_capture` preference. `sources` is empty
    /// when this is false, so a picker can render the list unconditionally.
    pub enabled: bool,
    pub sources: Vec<AppAudioSource>,
}

/// Reserved key for "whichever application is in a call".
///
/// Not an app: it is answered against the graph on every tick, the same
/// way [`crate::audio::device_mode::DEVICE_COMMUNICATION`] is, so joining
/// a Discord call or a Meet tab part-way through a recording moves the
/// capture onto it and leaving moves it back off.
pub const CALL_APP_KEY: &str = "call";

/// What [`CALL_APP_KEY`] is called on screen and in logs.
pub const CALL_APP_LABEL: &str = "Whichever app is in a call";

/// Whether this capture stream means its app is in a call right now.
///
/// The stream alone is not enough for the apps this exists for: Discord
/// holds a running capture stream from launch to quit, in a call or not.
/// The socket probe is the same in-call check the meeting detector uses,
/// and it is read the same way — anything it cannot answer counts as a
/// call rather than being dropped, because a missed call is a silent
/// recording and a spurious one is merely a bit more audio.
fn capture_is_in_call(
    stream: &PwCaptureClient,
    sockets: Option<&voice_socket::UdpSockets>,
) -> bool {
    if !stream.capture || !stream.running || stream.is_meetily() {
        return false;
    }
    match sockets {
        Some(sockets) => {
            voice_socket::probe_app(stream.binary.as_deref(), stream.pid, sockets).probe
                != CallProbe::Idle
        }
        None => true,
    }
}

/// The conferencing apps that are in a call right now.
///
/// Only apps in the detector's table: "in a call" is a question about
/// conferencing software, and an app we don't recognize holding the
/// microphone is as likely a voice memo as a meeting.
fn apps_in_call(snapshot: &PwGraphSnapshot) -> Vec<&'static str> {
    // Which app a stream belongs to is free to work out; whether it is in
    // a call costs a walk of /proc. Classify first so an idle machine
    // doesn't pay for the probe once a second.
    let listening: Vec<(&PwCaptureClient, &'static str)> = snapshot
        .streams
        .iter()
        .filter(|s| s.capture && s.running && !s.is_meetily())
        .filter_map(|s| apps::classify(&s.match_keys()).map(|app| (s, app.key)))
        .collect();
    if listening.is_empty() {
        return Vec::new();
    }

    let sockets = voice_socket::udp_sockets();
    let mut keys: Vec<&'static str> = listening
        .into_iter()
        .filter(|(stream, _)| capture_is_in_call(stream, sockets.as_ref()))
        .map(|(_, key)| key)
        .collect();
    keys.sort_unstable();
    keys.dedup();
    keys
}

/// The playback nodes of every app currently in a call.
///
/// The tap is on the app, not the call, so a browser in a Meet call
/// brings its other tabs with it — which tab holds the call is not
/// knowable from the audio graph. Everything outside that app still
/// stays out, which is the point.
pub fn call_app_nodes(snapshot: &PwGraphSnapshot) -> Vec<u32> {
    playback_nodes_for(snapshot, &apps_in_call(snapshot))
}

/// The playback nodes belonging to any of `keys`.
fn playback_nodes_for(snapshot: &PwGraphSnapshot, keys: &[&str]) -> Vec<u32> {
    snapshot
        .streams
        .iter()
        .filter(|s| !s.capture && !s.is_meetily())
        .filter(|s| keys.iter().any(|key| matches(s, key)))
        .map(|s| s.node_id)
        .collect()
}

/// Whether a playback stream belongs to `key`.
///
/// Known apps go through the detector's table, so every name Discord
/// ships under — `discord`, `com.discordapp.Discord`, a Flatpak wrapper —
/// resolves to the same key. Anything else is matched as a substring of
/// the identifiers the stream exposes, which is what the key was taken
/// from in the first place.
pub fn matches(stream: &PwCaptureClient, key: &str) -> bool {
    let identifiers = stream.match_keys();
    if apps::by_key(Some(key)).is_some() {
        return apps::classify(&identifiers).is_some_and(|app| app.key == key);
    }
    identifiers.iter().any(|id| id.contains(key))
}

/// The nodes `key` is playing on right now.
///
/// Idle streams are included: an app that has opened its playback stream
/// but isn't making noise yet is exactly the state a call starts from,
/// and linking it in advance means the first word isn't the one that gets
/// lost.
pub fn playback_nodes(snapshot: &PwGraphSnapshot, key: &str) -> Vec<u32> {
    if key == CALL_APP_KEY {
        return call_app_nodes(snapshot);
    }
    snapshot
        .streams
        .iter()
        .filter(|s| !s.capture && !s.is_meetily() && matches(s, key))
        .map(|s| s.node_id)
        .collect()
}

/// The identifier and display name to offer an app under, when it isn't
/// one the detector's table knows.
fn derive_source(stream: &PwCaptureClient) -> Option<AppAudioSource> {
    let label = stream
        .client_name
        .as_deref()
        .or(stream.app_name.as_deref())
        .or(stream.binary.as_deref())
        .unwrap_or(&stream.node_name)
        .trim()
        .to_string();
    let key = stream
        .binary
        .as_deref()
        .or(stream.app_id.as_deref())
        .unwrap_or(&label)
        .trim()
        .to_lowercase();

    (!key.is_empty() && !label.is_empty()).then_some(AppAudioSource {
        key,
        label,
        note: None,
        playing: true,
    })
}

/// Everything the system-audio picker can offer: the conferencing apps we
/// know by name, plus whatever else is currently playing.
///
/// The known ones are listed whether or not they are running, because the
/// selection is made before the meeting and has to survive the app being
/// restarted.
pub fn list_sources() -> Result<Vec<AppAudioSource>> {
    let snapshot = pw::snapshot_graph()?;
    let playing: Vec<&PwCaptureClient> = snapshot
        .streams
        .iter()
        .filter(|s| !s.capture && !s.is_meetily())
        .collect();

    let in_call = call_app_nodes(&snapshot);

    let mut sources: Vec<AppAudioSource> = apps::app_catalog()
        .into_iter()
        .map(|app| AppAudioSource {
            playing: playing.iter().any(|s| matches(s, &app.key)),
            key: app.key,
            label: app.label,
            note: app.note,
        })
        .collect();

    for stream in playing {
        if apps::classify(&stream.match_keys()).is_some() {
            continue;
        }
        let Some(source) = derive_source(stream) else {
            continue;
        };
        // The reserved key is not up for grabs by an app that happens to
        // ship under that name.
        if source.key == CALL_APP_KEY || sources.iter().any(|s| s.key == source.key) {
            continue;
        }
        sources.push(source);
    }

    // Whatever is making noise now is the likeliest pick.
    sources.sort_by(|a, b| {
        (!a.playing, a.label.to_lowercase()).cmp(&(!b.playing, b.label.to_lowercase()))
    });

    // Ahead of the named apps: it is the answer for "record my meetings"
    // without having to know in advance which app the meeting will be in.
    sources.insert(
        0,
        AppAudioSource {
            key: CALL_APP_KEY.to_string(),
            label: CALL_APP_LABEL.to_string(),
            note: Some(
                "Records the app you are in a call with, and nothing else. \
                 Switches by itself when a call starts or ends."
                    .to_string(),
            ),
            playing: !in_call.is_empty(),
        },
    );
    Ok(sources)
}

/// The target resolver an app capture polls: which nodes to be linked to,
/// asked again every second for the life of the recording.
///
/// `None` reports "couldn't tell", which holds the existing links; an
/// empty list is the real answer that the app has stopped playing.
pub fn node_resolver(key: String) -> Box<dyn Fn() -> Option<Vec<u32>> + Send> {
    Box::new(move || match pw::snapshot_graph() {
        Ok(snapshot) => Some(playback_nodes(&snapshot, &key)),
        Err(e) => {
            warn!("Could not read the audio graph to find {}: {}", key, e);
            None
        }
    })
}

/// How an app key should be named in logs and on screen.
///
/// Known apps get the detector's label; anything else is offered under a
/// key derived from its own identifiers, which is already the closest
/// thing to a name we have.
pub fn label_for(key: &str) -> String {
    if key == CALL_APP_KEY {
        return CALL_APP_LABEL.to_string();
    }
    apps::by_key(Some(key))
        .map(|app| app.label.to_string())
        .unwrap_or_else(|| key.to_string())
}

/// The picker's list of applications.
#[tauri::command]
pub async fn list_app_audio_sources() -> Result<AppAudioCatalog, String> {
    if !enabled() {
        return Ok(AppAudioCatalog {
            enabled: false,
            sources: Vec::new(),
        });
    }
    let sources = tokio::task::spawn_blocking(list_sources)
        .await
        .map_err(|e| format!("Application lookup did not finish: {}", e))?
        .map_err(|e| format!("Could not read the audio graph: {}", e))?;
    Ok(AppAudioCatalog {
        enabled: true,
        sources,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn stream(node_id: u32, binary: Option<&str>, client: Option<&str>) -> PwCaptureClient {
        PwCaptureClient {
            node_id,
            node_name: "playback".to_string(),
            app_name: None,
            binary: binary.map(str::to_string),
            app_id: None,
            client_name: client.map(str::to_string),
            pid: None,
            media_role: None,
            media_name: None,
            linked: true,
            running: true,
            capture: false,
        }
    }

    #[test]
    fn a_known_app_matches_every_name_it_ships_under() {
        let native = stream(1, Some("Discord"), Some("Discord"));
        let flatpak = stream(2, None, Some("com.discordapp.Discord"));
        assert!(matches(&native, "discord"));
        assert!(matches(&flatpak, "discord"));
    }

    #[test]
    fn a_known_key_does_not_match_a_different_app() {
        let firefox = stream(1, Some("firefox"), Some("Firefox"));
        assert!(!matches(&firefox, "discord"));
        assert!(matches(&firefox, "browser"));
    }

    #[test]
    fn an_unknown_app_matches_on_its_own_identifiers() {
        let spotify = stream(1, Some("spotify"), Some("Spotify"));
        assert!(matches(&spotify, "spotify"));
        assert!(!matches(&spotify, "discord"));
    }

    fn snapshot(streams: Vec<PwCaptureClient>) -> PwGraphSnapshot {
        PwGraphSnapshot {
            streams,
            devices: Default::default(),
            links: Vec::new(),
            defaults: Default::default(),
        }
    }

    #[test]
    fn a_call_taps_that_app_and_nothing_else() {
        let discord_playback = stream(1, Some("Discord"), Some("Discord"));
        let music = stream(2, Some("spotify"), Some("Spotify"));
        let graph = snapshot(vec![discord_playback, music]);
        assert_eq!(playback_nodes_for(&graph, &["discord"]), vec![1]);
    }

    #[test]
    fn nothing_is_tapped_while_no_app_is_listening() {
        // No capture stream at all, so the answer is settled without the
        // socket probe — and so is this test.
        let graph = snapshot(vec![stream(1, Some("Discord"), Some("Discord"))]);
        assert!(apps_in_call(&graph).is_empty());
        assert!(call_app_nodes(&graph).is_empty());
    }

    #[test]
    fn an_app_selection_is_ignored_while_the_feature_is_off() {
        set_enabled(false);
        assert_eq!(selected_app("app:discord"), None);
        set_enabled(true);
        assert_eq!(selected_app("app:discord"), Some("discord"));
        assert_eq!(
            selected_app("alsa_output.pci-0000_00_1f.3.analog-stereo"),
            None
        );
        set_enabled(false);
    }
}
