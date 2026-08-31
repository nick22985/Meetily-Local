//! Native PipeWire capture layer (Linux).
//!
//! This replaces the previous cpal-ALSA + pactl + `PIPEWIRE_NODE` env-var
//! stack. Everything talks to PipeWire directly:
//!
//! - **Enumeration** reads Audio/Source and Audio/Sink nodes from the
//!   registry. Devices are identified by their stable `node.name` and
//!   displayed by `node.description` — no string-suffix conventions.
//! - **Capture** opens a PipeWire capture stream with `target.object`
//!   set to the chosen node. For sinks, `stream.capture.sink` makes
//!   PipeWire deliver the sink's monitor — no `.monitor` pseudo-sources,
//!   no env vars, no default-device indirection.
//! - **Format** is negotiated to interleaved f32 / 48 kHz / stereo; the
//!   PipeWire graph performs any resampling/channel mixing, so consumers
//!   always receive the same shape.
//!
//! Each stream runs its own PipeWire main loop on a dedicated thread
//! (PipeWire objects are not `Send`); samples are handed to a caller
//! provided callback from the loop thread.

use anyhow::{anyhow, Context as _, Result};
use log::{debug, info, warn};
use serde::{Deserialize, Serialize};
use std::cell::{Cell, RefCell};
use std::collections::{HashMap, HashSet};
use std::rc::Rc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::mpsc as std_mpsc;
use std::sync::Arc;
use std::thread::JoinHandle;
use std::time::Duration;

use pipewire as pw;
use pw::spa;

mod app_capture;

/// Sample shape every capture stream delivers, regardless of device.
pub const CAPTURE_RATE: u32 = 48_000;
pub const CAPTURE_CHANNELS: u32 = 2;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum PwDeviceKind {
    /// A real capture node (Audio/Source) — microphones, capture cards.
    Microphone,
    /// A playback node (Audio/Sink) captured via its monitor ports.
    System,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PwDevice {
    /// Stable PipeWire `node.name` (e.g.
    /// `alsa_input.usb-RODE_Microphones_RODE_NT-USB-00.pro-input-0`).
    /// This is what selection, preferences, and capture use.
    pub id: String,
    /// Human-readable `node.description` for the picker.
    pub label: String,
    pub kind: PwDeviceKind,
}

/// Enumerate audio devices from the PipeWire registry.
///
/// Runs a short-lived main loop on the calling thread and returns once
/// the registry roundtrip completes (or errors after `timeout`).
pub fn enumerate_devices() -> Result<Vec<PwDevice>> {
    enumerate_devices_with_timeout(Duration::from_secs(3))
}

fn enumerate_devices_with_timeout(timeout: Duration) -> Result<Vec<PwDevice>> {
    let (tx, rx) = std_mpsc::channel::<Result<Vec<PwDevice>>>();

    // PipeWire objects are !Send; run the whole roundtrip on one thread.
    let worker = std::thread::Builder::new()
        .name("pw-enumerate".into())
        .spawn(move || {
            let _ = tx.send(enumerate_on_current_thread());
        })
        .context("failed to spawn pw-enumerate thread")?;

    let result = rx
        .recv_timeout(timeout)
        .map_err(|_| anyhow!("PipeWire device enumeration timed out"))?;
    let _ = worker.join();
    result
}

fn enumerate_on_current_thread() -> Result<Vec<PwDevice>> {
    pw::init();

    let mainloop = pw::main_loop::MainLoopRc::new(None)?;
    let context = pw::context::ContextRc::new(&mainloop, None)?;
    let core = context.connect_rc(None)?;
    let registry = core.get_registry()?;

    let devices: Rc<RefCell<Vec<PwDevice>>> = Rc::new(RefCell::new(Vec::new()));
    let devices_cb = devices.clone();

    let _reg_listener = registry
        .add_listener_local()
        .global(move |global| {
            if global.type_ != pw::types::ObjectType::Node {
                return;
            }
            let Some(props) = global.props else { return };
            let Some(class) = props.get("media.class") else {
                return;
            };
            let kind = match class {
                "Audio/Source" => PwDeviceKind::Microphone,
                "Audio/Sink" => PwDeviceKind::System,
                _ => return,
            };
            let Some(name) = props.get("node.name") else {
                return;
            };
            let label = props
                .get("node.description")
                .or_else(|| props.get("node.nick"))
                .unwrap_or(name)
                .to_string();
            devices_cb.borrow_mut().push(PwDevice {
                id: name.to_string(),
                label,
                kind,
            });
        })
        .register();

    // Roundtrip: when the server answers our sync, every global that
    // existed before it has been announced.
    let done = Rc::new(RefCell::new(false));
    let done_cb = done.clone();
    let mainloop_cb = mainloop.clone();
    let pending = core.sync(0)?;
    let _core_listener = core
        .add_listener_local()
        .done(move |id, seq| {
            if id == pw::core::PW_ID_CORE && seq == pending {
                *done_cb.borrow_mut() = true;
                mainloop_cb.quit();
            }
        })
        .register();

    mainloop.run();

    if !*done.borrow() {
        return Err(anyhow!("PipeWire registry roundtrip did not complete"));
    }

    let mut out = devices.borrow().clone();
    // Stable ordering: microphones first, then sinks, alphabetical labels.
    out.sort_by(|a, b| {
        (a.kind == PwDeviceKind::System, &a.label).cmp(&(b.kind == PwDeviceKind::System, &b.label))
    });
    debug!("pw: enumerated {} audio nodes", out.len());
    Ok(out)
}

/// `node.name` our own capture streams register under.
///
/// Meaningful to [`enumerate_capture_streams`] callers: we hold the
/// microphone open too while recording, and a meeting detector that
/// counted our own stream would latch on permanently.
pub const MEETILY_CAPTURE_NODE: &str = "meetily-capture";

/// An application stream in the audio graph — a microphone it holds open,
/// or audio it is playing back.
///
/// These are the graph's `Stream/Input/Audio` and `Stream/Output/Audio`
/// nodes. On Linux those, not a process name, are what "someone is in a
/// call" actually looks like: Zoom's macOS-only meeting helper process has
/// no equivalent here, and a browser's tab title (the usual way to spot
/// Google Meet) is unreadable under Wayland.
///
/// Playback is collected because a call is two-way and a held microphone
/// alone is not evidence of one. Discord keeps a capture stream open and
/// `Running` for as long as it is running, in a call or not; the voice
/// playback stream beside it is the part that only exists during one.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PwCaptureClient {
    /// PipeWire object id of the stream node.
    ///
    /// Identifies this particular stream rather than the app behind it,
    /// which is the distinction the meeting detector needs: a browser
    /// holding the mic for a background tab and then joining a call in
    /// another tab opens a second stream with its own id.
    pub node_id: u32,
    /// `node.name` of the stream node.
    pub node_name: String,
    /// `application.name`, when the client set one.
    pub app_name: Option<String>,
    /// `application.process.binary` — the executable's basename. Often
    /// absent for Flatpak clients, whose sandbox restricts `/proc`.
    pub binary: Option<String>,
    /// `application.id` — reverse-DNS app id, e.g. `com.discordapp.Discord`.
    /// Always set by Flatpak, and the most reliable identifier we get from
    /// a sandboxed client.
    pub app_id: Option<String>,
    /// `application.name` of the PipeWire *client* that owns this node.
    ///
    /// Worth having separately because it is frequently the only place the
    /// real application name appears. Electron apps name the node after
    /// their audio engine — Discord's stream node and its
    /// `application.name` are both literally "WEBRTC VoiceEngine" — while
    /// the client object is named after the app.
    pub client_name: Option<String>,
    /// `application.process.id`, when the client exposed it.
    ///
    /// Treat it as a hint, never an answer. A sandboxed client reports it
    /// from inside its own pid namespace — Discord arrives claiming pid
    /// 157, which on this side of the sandbox is an early-boot kernel
    /// thread. `pipewire.sec.pid` is no better: everything speaking the
    /// PulseAudio protocol connects through `pipewire-pulse`, so the pid
    /// the daemon reads off the socket is the bridge's, identical for
    /// every app. Anything looking a stream up in `/proc` has to confirm
    /// the pid really is the binary it claims to be.
    pub pid: Option<u32>,
    /// `media.role`. WebRTC clients conventionally set "Communication",
    /// which distinguishes a call from, say, a voice recorder.
    pub media_role: Option<String>,
    /// `media.name` — the client's own name for this stream. Diagnostic
    /// only: it is not matched against, because clients put wildly
    /// inconsistent things here.
    pub media_name: Option<String>,
    /// Whether PipeWire currently has this node linked to anything.
    ///
    /// Diagnostic only. It reads like it should mean "in use", but the
    /// session manager leaves idle capture streams linked to the source
    /// they asked for, so it does not separate a call from an app sitting
    /// on a stream. Kept because it costs nothing to collect and it is
    /// worth having in the log when a stream misbehaves.
    pub linked: bool,
    /// Whether the node is in PipeWire's `Running` state.
    ///
    /// This is the difference between *holding* a capture stream and
    /// *using* one, and it is the only thing that reliably tells them
    /// apart. Apps routinely open a stream and sit on it for as long as
    /// they are running — Discord's "WEBRTC VoiceEngine" node outlives
    /// whole restarts of this app — and such a node stays present, stays
    /// linked, and reports `Idle`.
    ///
    /// Defaults to `true` when the node's info never arrived, so a
    /// failure to read state degrades to the old presence-only behaviour
    /// rather than to detecting nothing at all.
    pub running: bool,
    /// Whether this is a capture stream (`Stream/Input/Audio`) rather than
    /// a playback one.
    pub capture: bool,
}

impl PwCaptureClient {
    /// Lowercased identifiers to match app rules against, most specific
    /// first. `application.name` is a display string ("Google Chrome")
    /// while the binary is stable ("chrome"), so both are worth checking —
    /// as is the Flatpak app id, which is often all a sandboxed client
    /// exposes.
    pub fn match_keys(&self) -> Vec<String> {
        [
            self.binary.as_deref(),
            // Ahead of `application.name` deliberately: Electron apps often
            // report a generic name there ("Chromium"), while the app id is
            // specific and is the only identifier a Flatpak client reliably
            // provides.
            self.app_id.as_deref(),
            // Also ahead of it: the node's own `application.name` can be an
            // audio-engine name rather than an app name, and the client's
            // is the app's.
            self.client_name.as_deref(),
            self.app_name.as_deref(),
            Some(self.node_name.as_str()),
        ]
        .into_iter()
        .flatten()
        .map(|s| s.to_lowercase())
        .collect()
    }

    /// Whether this stream is one of ours.
    pub fn is_meetily(&self) -> bool {
        self.node_name == MEETILY_CAPTURE_NODE || self.pid == Some(std::process::id())
    }
}

/// A device node in the graph, as [`snapshot_graph`] saw it.
#[derive(Debug, Clone)]
pub struct PwGraphNode {
    /// Stable `node.name`.
    pub name: String,
    pub kind: PwDeviceKind,
}

/// `metadata.name` of the object holding the default source/sink.
const METADATA_DEFAULT: &str = "default";

/// The system's default source and sink.
///
/// PipeWire keeps these in the `default` metadata object rather than on
/// the nodes, under two pairs of keys:
///
/// - `default.audio.source` / `default.audio.sink` — what the session
///   manager currently resolves the default to. This is what `pactl
///   get-default-source` / `get-default-sink` report.
/// - `default.configured.audio.source` / `.sink` — the node the user
///   explicitly picked. `pactl set-default-source` / `set-default-sink`
///   and any `pw-metadata` caller write *here*; the session manager
///   resolves those into the pair above, falling back to a
///   priority-ordered pick when the configured node is gone.
///
/// There is no second, "communication" default. The Windows
/// `eCommunications` role has no Linux counterpart, and both pairs of
/// keys above are global — so a script that writes
/// `default.configured.audio.sink` to set a "communication" output is
/// writing the very key `pactl set-default-sink` writes, and whichever
/// runs last silently wins. Following the device a *call* is on has to
/// be done by asking the graph what the call is linked to; see
/// [`PwGraphSnapshot::call_route`].
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct PwDefaults {
    /// `default.audio.source`.
    pub source: Option<String>,
    /// `default.audio.sink`.
    pub sink: Option<String>,
    /// `default.configured.audio.source`.
    pub configured_source: Option<String>,
    /// `default.configured.audio.sink`.
    pub configured_sink: Option<String>,
}

impl PwDefaults {
    /// The default source, preferring the resolved key: the configured
    /// one can still name a device that has since been unplugged.
    pub fn source(&self) -> Option<&str> {
        self.source.as_deref().or(self.configured_source.as_deref())
    }

    /// The default sink, on the same preference as [`source`](Self::source).
    pub fn sink(&self) -> Option<&str> {
        self.sink.as_deref().or(self.configured_sink.as_deref())
    }

    /// Fold in one `property` event from the `default` metadata.
    fn apply(&mut self, key: Option<&str>, value: Option<&str>) {
        // A null key is the metadata's "every property was cleared".
        let Some(key) = key else {
            *self = Self::default();
            return;
        };
        let slot = match key {
            "default.audio.source" => &mut self.source,
            "default.audio.sink" => &mut self.sink,
            "default.configured.audio.source" => &mut self.configured_source,
            "default.configured.audio.sink" => &mut self.configured_sink,
            _ => return,
        };
        // A null value is a removal of that key.
        *slot = value.and_then(parse_default_node_name);
    }
}

/// Pull the node name out of a `default.*` metadata value.
///
/// The value is SPA JSON — `{ "name": "alsa_input.…" }` in every current
/// PipeWire. A bare string is accepted too, rather than losing the
/// default over a shape we didn't expect.
fn parse_default_node_name(value: &str) -> Option<String> {
    if let Ok(parsed) = serde_json::from_str::<serde_json::Value>(value) {
        if let Some(name) = parsed.get("name").and_then(|n| n.as_str()) {
            return Some(name.to_string());
        }
        if let Some(name) = parsed.as_str() {
            return Some(name.to_string());
        }
    }
    let trimmed = value.trim();
    (!trimmed.is_empty() && !trimmed.starts_with('{')).then(|| trimmed.to_string())
}

/// The devices an in-progress call is using.
///
/// Either side is `None` when there is nothing to point at — no call
/// running, or a call whose stream the session manager has not linked to
/// a device.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct CallRoute {
    /// `node.name` of the source the call is recording from.
    pub source: Option<String>,
    /// `node.name` of the sink the call is playing out of.
    pub sink: Option<String>,
}

/// One pass over the audio graph: every device node, every application
/// stream, how they are linked, and what the defaults point at.
///
/// Collected together because between them they answer a single
/// question — "which node should we be capturing from right now" — and
/// splitting them up would cost a registry roundtrip each.
#[derive(Debug, Clone)]
pub struct PwGraphSnapshot {
    /// Application streams. Ours are included; callers filter them out
    /// with [`PwCaptureClient::is_meetily`].
    pub streams: Vec<PwCaptureClient>,
    /// Audio/Source and Audio/Sink nodes, by PipeWire object id.
    pub devices: HashMap<u32, PwGraphNode>,
    /// `(producer node id, consumer node id)` for every link.
    pub links: Vec<(u32, u32)>,
    pub defaults: PwDefaults,
}

/// Whether a stream's `media.role` says it belongs to a call.
///
/// Two vocabularies are in play and clients pick one at random. Native
/// PipeWire clients use the `PW_KEY_MEDIA_ROLE` set, whose voice value is
/// `Communication`; anything speaking the PulseAudio protocol — which is
/// most conferencing apps, Discord included — uses Pulse's older set,
/// whose voice value is `phone`. Matching only the first would miss the
/// apps this is for. Case is not load-bearing either: the PipeWire set is
/// capitalised, the Pulse one isn't, and clients are inconsistent.
///
/// A role is a *hint*, not proof — plenty of call apps set nothing at
/// all. See [`PwGraphSnapshot::call_route_matching`] for pairing it with
/// a stronger signal.
pub fn has_call_role(stream: &PwCaptureClient) -> bool {
    const CALL_ROLES: [&str; 2] = ["communication", "phone"];
    stream
        .media_role
        .as_deref()
        .map(str::to_ascii_lowercase)
        .is_some_and(|role| CALL_ROLES.contains(&role.as_str()))
}

impl PwGraphSnapshot {
    /// Which devices the active call is on, going by `media.role` alone.
    ///
    /// Callers with a better idea of what counts as a call app should use
    /// [`call_route_matching`](Self::call_route_matching) instead; role
    /// tags are widely omitted.
    pub fn call_route(&self) -> CallRoute {
        self.call_route_matching(has_call_role)
    }

    /// Which devices the streams `is_call` accepts are on.
    ///
    /// Our own capture stream is always excluded — it carries a call role
    /// itself, and following it would only pin us to the device we
    /// already picked.
    ///
    /// Only `Running` streams count, for the reason the meeting detector
    /// gives: apps sit on an open, linked, `Idle` capture stream for as
    /// long as they are running.
    pub fn call_route_matching(&self, is_call: impl Fn(&PwCaptureClient) -> bool) -> CallRoute {
        let mut route = CallRoute::default();

        // Sorted so that with two calls up we keep answering with the
        // same one instead of flapping between their devices.
        let mut candidates: Vec<&PwCaptureClient> = self
            .streams
            .iter()
            .filter(|s| !s.is_meetily() && s.running && is_call(s))
            .collect();
        candidates.sort_by_key(|s| s.node_id);

        for stream in candidates {
            if stream.capture {
                if route.source.is_none() {
                    route.source = self.device_feeding(stream.node_id, PwDeviceKind::Microphone);
                }
            } else if route.sink.is_none() {
                route.sink = self.device_fed_by(stream.node_id, PwDeviceKind::System);
            }
        }

        route
    }

    /// The device node producing into `consumer`.
    fn device_feeding(&self, consumer: u32, kind: PwDeviceKind) -> Option<String> {
        self.links
            .iter()
            .filter(|(_, to)| *to == consumer)
            .find_map(|(from, _)| self.device_named(*from, kind))
    }

    /// The device node that `producer` feeds.
    fn device_fed_by(&self, producer: u32, kind: PwDeviceKind) -> Option<String> {
        self.links
            .iter()
            .filter(|(from, _)| *from == producer)
            .find_map(|(_, to)| self.device_named(*to, kind))
    }

    fn device_named(&self, id: u32, kind: PwDeviceKind) -> Option<String> {
        self.devices
            .get(&id)
            .filter(|node| node.kind == kind)
            .map(|node| node.name.clone())
    }
}

/// Take one snapshot of the audio graph.
pub fn snapshot_graph() -> Result<PwGraphSnapshot> {
    let (tx, rx) = std_mpsc::channel::<Result<PwGraphSnapshot>>();

    // PipeWire objects are !Send; run the whole roundtrip on one thread.
    let worker = std::thread::Builder::new()
        .name("pw-graph".into())
        .spawn(move || {
            let _ = tx.send(snapshot_graph_on_current_thread());
        })
        .context("failed to spawn pw-graph thread")?;

    let result = rx
        .recv_timeout(Duration::from_secs(3))
        .map_err(|_| anyhow!("PipeWire graph snapshot timed out"))?;
    let _ = worker.join();
    result
}

/// Enumerate the applications currently capturing audio.
///
/// The same registry roundtrip as [`enumerate_devices`], collecting the
/// client-side stream nodes instead of the device nodes. Our own streams
/// are included; callers filter them with [`PwCaptureClient::is_meetily`].
pub fn enumerate_capture_streams() -> Result<Vec<PwCaptureClient>> {
    Ok(snapshot_graph()?.streams)
}

/// The props a PipeWire *client* object carries about the process behind it.
///
/// Stream nodes are matched back to these by their `client.id`. The
/// registry hands out only a subset of each object's properties, and for
/// stream nodes that subset routinely omits everything identifying —
/// `application.process.binary` and `application.id` in particular. The
/// client object is where they actually live.
/// What binding a node's proxy adds to what the registry already said.
///
/// The registry announces a subset of each object's properties, and for
/// stream nodes that subset is thin: Discord's capture node arrives
/// carrying nothing but "WEBRTC VoiceEngine", while the node's own info
/// carries `application.process.binary = Discord`. State isn't in the
/// registry's subset at all.
#[derive(Debug, Clone, Default)]
struct PwNodeDetails {
    running: bool,
    app_name: Option<String>,
    pid: Option<u32>,
    binary: Option<String>,
    app_id: Option<String>,
    media_role: Option<String>,
    media_name: Option<String>,
}

#[derive(Debug, Clone, Default)]
struct PwClientInfo {
    app_name: Option<String>,
    binary: Option<String>,
    app_id: Option<String>,
    pid: Option<u32>,
}

fn snapshot_graph_on_current_thread() -> Result<PwGraphSnapshot> {
    pw::init();

    let mainloop = pw::main_loop::MainLoopRc::new(None)?;
    let context = pw::context::ContextRc::new(&mainloop, None)?;
    let core = context.connect_rc(None)?;
    // The reference-counted flavour, so the listener below can bind the
    // objects it is being told about.
    let registry = core.get_registry_rc()?;

    // Nodes are collected alongside the id of the client that owns them;
    // the two are joined after the roundtrip, since the registry emits
    // them in no particular order.
    let nodes: Rc<RefCell<Vec<(Option<u32>, PwCaptureClient)>>> = Rc::new(RefCell::new(Vec::new()));
    let owners: Rc<RefCell<HashMap<u32, PwClientInfo>>> = Rc::new(RefCell::new(HashMap::new()));
    // Audio/Source and Audio/Sink nodes, so the far end of a link can be
    // named rather than left as a bare object id.
    let devices: Rc<RefCell<HashMap<u32, PwGraphNode>>> = Rc::new(RefCell::new(HashMap::new()));
    // Every link, as (producer, consumer). Collected in the same pass
    // because the registry hands links out as globals like everything
    // else, in no guaranteed order relative to the nodes.
    let links: Rc<RefCell<Vec<(u32, u32)>>> = Rc::new(RefCell::new(Vec::new()));
    // Node id -> everything only the node itself will tell us.
    let details: Rc<RefCell<HashMap<u32, PwNodeDetails>>> = Rc::new(RefCell::new(HashMap::new()));
    // Filled in by the `default` metadata's property events.
    let defaults: Rc<RefCell<PwDefaults>> = Rc::new(RefCell::new(PwDefaults::default()));
    // The proxies and their listeners both have to outlive the roundtrip
    // that delivers the info events; dropping either one unsubscribes.
    let bound: Rc<RefCell<Vec<(pw::node::Node, pw::node::NodeListener)>>> =
        Rc::new(RefCell::new(Vec::new()));
    let bound_meta: Rc<RefCell<Vec<(pw::metadata::Metadata, pw::metadata::MetadataListener)>>> =
        Rc::new(RefCell::new(Vec::new()));
    let nodes_cb = nodes.clone();
    let owners_cb = owners.clone();
    let devices_cb = devices.clone();
    let links_cb = links.clone();
    let details_cb = details.clone();
    let defaults_cb = defaults.clone();
    let bound_cb = bound.clone();
    let bound_meta_cb = bound_meta.clone();
    let registry_cb = registry.clone();

    let _reg_listener = registry
        .add_listener_local()
        .global(move |global| {
            let Some(props) = global.props else { return };

            if global.type_ == pw::types::ObjectType::Client {
                owners_cb.borrow_mut().insert(
                    global.id,
                    PwClientInfo {
                        app_name: props.get("application.name").map(str::to_string),
                        binary: props.get("application.process.binary").map(str::to_string),
                        app_id: props.get("application.id").map(str::to_string),
                        pid: props
                            .get("application.process.id")
                            .and_then(|p| p.parse().ok()),
                    },
                );
                return;
            }

            if global.type_ == pw::types::ObjectType::Metadata {
                // Only the `default` object carries the default
                // source/sink; `settings` and `route-settings` are
                // unrelated and would waste a proxy.
                if props.get("metadata.name") != Some(METADATA_DEFAULT) {
                    return;
                }
                match registry_cb.bind::<pw::metadata::Metadata, _>(global) {
                    Ok(meta) => {
                        let defaults = defaults_cb.clone();
                        let listener = meta
                            .add_listener_local()
                            .property(move |_subject, key, _type_, value| {
                                defaults.borrow_mut().apply(key, value);
                                0
                            })
                            .register();
                        bound_meta_cb.borrow_mut().push((meta, listener));
                    }
                    Err(e) => debug!("pw: could not bind the default-device metadata: {}", e),
                }
                return;
            }

            if global.type_ == pw::types::ObjectType::Link {
                let from = props.get("link.output.node").and_then(|id| id.parse().ok());
                let to = props.get("link.input.node").and_then(|id| id.parse().ok());
                if let (Some(from), Some(to)) = (from, to) {
                    links_cb.borrow_mut().push((from, to));
                }
                return;
            }

            if global.type_ != pw::types::ObjectType::Node {
                return;
            }

            let media_class = props.get("media.class");

            // Device nodes. Not streams, but a link's far end is one of
            // these, and naming it is the whole point of following a
            // call onto the device it is using.
            let device_kind = match media_class {
                Some("Audio/Source") => Some(PwDeviceKind::Microphone),
                Some("Audio/Sink") => Some(PwDeviceKind::System),
                _ => None,
            };
            if let Some(kind) = device_kind {
                if let Some(name) = props.get("node.name") {
                    devices_cb.borrow_mut().insert(
                        global.id,
                        PwGraphNode {
                            name: name.to_string(),
                            kind,
                        },
                    );
                }
                return;
            }

            // An app reading from the graph (recording) or writing to it
            // (playback). The session manager's own internal streams are
            // uninteresting here.
            let capture = match media_class {
                Some("Stream/Input/Audio") => true,
                Some("Stream/Output/Audio") => false,
                _ => return,
            };
            // Registry globals carry only a subset of a node's properties,
            // and `node.name` is not guaranteed to be in it for stream
            // nodes the way it is for the device nodes above. Falling back
            // to the object id keeps the stream visible — dropping it would
            // make the app invisible to the detector.
            let node_name = props
                .get("node.name")
                .map(str::to_string)
                .unwrap_or_else(|| format!("node-{}", global.id));
            let client_id = props.get("client.id").and_then(|id| id.parse().ok());

            // Ask the node itself whether anything is flowing through it.
            // The answer comes back as an info event on a later roundtrip,
            // which is why there are two of them below.
            match registry_cb.bind::<pw::node::Node, _>(global) {
                Ok(node) => {
                    let id = global.id;
                    let details = details_cb.clone();
                    let listener = node
                        .add_listener_local()
                        .info(move |info| {
                            let mut found = PwNodeDetails {
                                running: matches!(info.state(), pw::node::NodeState::Running),
                                ..Default::default()
                            };
                            if let Some(props) = info.props() {
                                found.app_name = props.get("application.name").map(str::to_string);
                                found.binary =
                                    props.get("application.process.binary").map(str::to_string);
                                found.app_id = props.get("application.id").map(str::to_string);
                                found.pid = props
                                    .get("application.process.id")
                                    .and_then(|p| p.parse().ok());
                                found.media_role = props.get("media.role").map(str::to_string);
                                found.media_name = props.get("media.name").map(str::to_string);
                            }
                            details.borrow_mut().insert(id, found);
                        })
                        .register();
                    bound_cb.borrow_mut().push((node, listener));
                }
                Err(e) => debug!(
                    "pw: could not bind node {} to read its state: {}",
                    global.id, e
                ),
            }

            nodes_cb.borrow_mut().push((
                client_id,
                PwCaptureClient {
                    node_id: global.id,
                    node_name,
                    app_name: props.get("application.name").map(str::to_string),
                    binary: props.get("application.process.binary").map(str::to_string),
                    app_id: props.get("application.id").map(str::to_string),
                    client_name: None,
                    pid: props
                        .get("application.process.id")
                        .and_then(|p| p.parse().ok()),
                    media_role: props.get("media.role").map(str::to_string),
                    media_name: props.get("media.name").map(str::to_string),
                    capture,
                    // Both filled in after the roundtrips below.
                    linked: false,
                    running: true,
                },
            ));
        })
        .register();

    // Which sync to quit on. It changes between the two roundtrips, so the
    // listener reads it from a cell rather than capturing one sequence.
    let done = Rc::new(Cell::new(false));
    let expected = Rc::new(Cell::new(core.sync(0)?));
    let done_cb = done.clone();
    let expected_cb = expected.clone();
    let mainloop_cb = mainloop.clone();
    let _core_listener = core
        .add_listener_local()
        .done(move |id, seq| {
            if id == pw::core::PW_ID_CORE && seq == expected_cb.get() {
                done_cb.set(true);
                mainloop_cb.quit();
            }
        })
        .register();

    mainloop.run();

    if !done.get() {
        return Err(anyhow!("PipeWire registry roundtrip did not complete"));
    }

    // The node and metadata proxies were bound while that first roundtrip
    // was already in flight, so their events only land on a second one.
    let bound_any = !bound.borrow().is_empty() || !bound_meta.borrow().is_empty();
    if bound_any {
        done.set(false);
        expected.set(core.sync(0)?);
        mainloop.run();
        if !done.get() {
            warn!("pw: node state roundtrip did not complete; treating open streams as running");
        }
    }

    // Fold the owning client's identity into each stream. Anything the
    // node already stated wins — it is the more specific of the two — so
    // this only ever fills gaps.
    let owners = owners.borrow();
    let links = links.borrow().clone();
    let linked: HashSet<u32> = links.iter().flat_map(|(from, to)| [*from, *to]).collect();
    let details = details.borrow();
    let streams: Vec<PwCaptureClient> = nodes
        .borrow()
        .iter()
        .cloned()
        .map(|(client_id, mut node)| {
            // The node's own info goes on first: it is more specific than
            // anything the owning client can say, and it is where the
            // registry's thin property subset gets filled back in.
            if let Some(found) = details.get(&node.node_id) {
                node.running = found.running;
                node.app_name = node.app_name.or_else(|| found.app_name.clone());
                node.binary = node.binary.or_else(|| found.binary.clone());
                node.app_id = node.app_id.or_else(|| found.app_id.clone());
                node.pid = node.pid.or(found.pid);
                node.media_role = node.media_role.or_else(|| found.media_role.clone());
                node.media_name = node.media_name.or_else(|| found.media_name.clone());
            }
            if let Some(owner) = client_id.and_then(|id| owners.get(&id)) {
                node.client_name = owner.app_name.clone();
                node.binary = node.binary.or_else(|| owner.binary.clone());
                node.app_id = node.app_id.or_else(|| owner.app_id.clone());
                node.pid = node.pid.or(owner.pid);
            }
            node.linked = linked.contains(&node.node_id);
            node
        })
        .collect();

    debug!("pw: {} app(s) currently capturing audio", streams.len());
    for c in &streams {
        debug!(
            "pw:   capture node={} app={:?} client={:?} id={:?} binary={:?} linked={} running={} capture={}",
            c.node_name,
            c.app_name,
            c.client_name,
            c.app_id,
            c.binary,
            c.linked,
            c.running,
            c.capture
        );
    }

    let defaults = defaults.borrow().clone();
    debug!(
        "pw: defaults source={:?} sink={:?} (configured {:?} / {:?})",
        defaults.source, defaults.sink, defaults.configured_source, defaults.configured_sink
    );

    let devices = devices.borrow().clone();

    Ok(PwGraphSnapshot {
        streams,
        devices,
        links,
        defaults,
    })
}

/// Target for a capture stream.
#[derive(Debug, Clone)]
pub enum CaptureTarget {
    /// Default source (microphone) as chosen in the system.
    DefaultMicrophone,
    /// Default sink's monitor.
    DefaultSystem,
    /// A specific node by `node.name`.
    Node { id: String, kind: PwDeviceKind },
    /// One application's playback, tapped from that app's own output
    /// ports rather than the sink every app shares. `label` is for logs
    /// and the UI; which nodes it means is answered separately, and
    /// repeatedly, by the resolver passed to
    /// [`PwCaptureStream::open_app`].
    AppAudio { label: String },
}

impl CaptureTarget {
    fn is_sink_capture(&self) -> bool {
        matches!(
            self,
            CaptureTarget::DefaultSystem
                | CaptureTarget::Node {
                    kind: PwDeviceKind::System,
                    ..
                }
        )
    }

    fn node_name(&self) -> Option<&str> {
        match self {
            CaptureTarget::Node { id, .. } => Some(id),
            _ => None,
        }
    }
}

enum LoopCommand {
    Terminate,
    /// Replace the set of application nodes an app capture is tapping.
    /// Ignored by the device capture loop, which has no links of its own.
    SetTargets(Vec<u32>),
}

/// A running PipeWire capture stream on its own loop thread.
///
/// `on_samples` is invoked from the loop thread with interleaved f32
/// frames at [`CAPTURE_RATE`] Hz / [`CAPTURE_CHANNELS`] channels.
/// Dropping (or calling [`stop`]) shuts the loop down and joins the
/// thread.
pub struct PwCaptureStream {
    quit_tx: Option<pw::channel::Sender<LoopCommand>>,
    handle: Option<JoinHandle<()>>,
    /// Set only for an app capture: the poller that keeps asking which
    /// nodes the app is playing on, and the flag that stops it.
    resolver_stop: Option<Arc<AtomicBool>>,
    resolver: Option<JoinHandle<()>>,
    pub target_desc: String,
}

impl PwCaptureStream {
    pub fn open(target: CaptureTarget, on_samples: Box<dyn FnMut(&[f32]) + Send>) -> Result<Self> {
        let (quit_tx, quit_rx) = pw::channel::channel::<LoopCommand>();
        // Reports either the successful connect or the first error back
        // to the caller so open() is synchronous and fallible.
        let (ready_tx, ready_rx) = std_mpsc::channel::<Result<()>>();

        let target_desc = format!("{:?}", target);
        let thread_target = target.clone();

        let handle = std::thread::Builder::new()
            .name("pw-capture".into())
            .spawn(move || {
                if let Err(e) = run_capture_loop(thread_target, on_samples, quit_rx, &ready_tx) {
                    // If setup already succeeded this is a runtime error;
                    // otherwise ready_tx already carried it to open().
                    let _ = ready_tx.send(Err(e));
                }
            })
            .context("failed to spawn pw-capture thread")?;

        match ready_rx.recv_timeout(Duration::from_secs(5)) {
            Ok(Ok(())) => {
                info!("pw: capture stream running ({})", target_desc);
                Ok(Self {
                    quit_tx: Some(quit_tx),
                    handle: Some(handle),
                    resolver_stop: None,
                    resolver: None,
                    target_desc,
                })
            }
            Ok(Err(e)) => {
                let _ = handle.join();
                Err(e)
            }
            Err(_) => {
                // Loop thread never reported: ask it to quit and bail.
                let _ = quit_tx.send(LoopCommand::Terminate);
                let _ = handle.join();
                Err(anyhow!("PipeWire capture stream setup timed out"))
            }
        }
    }

    /// Open a capture that taps one application's playback.
    ///
    /// `resolve_targets` answers "which nodes is that app playing on
    /// right now" and is asked once a second for as long as the stream
    /// lives, because the answer moves: an app opens a second stream for
    /// a notification sound, a call ends and its stream goes away, the
    /// app restarts. Everything it names is linked into this one capture
    /// stream, which mixes them.
    ///
    /// The stream starts with no links at all and stays silent until the
    /// app plays something. That is deliberate — silence is the correct
    /// recording of an app that is not making noise — and the mixer
    /// already treats a system side that stops delivering as silence
    /// rather than stalling the microphone (see `pipeline.rs`).
    pub fn open_app(
        label: String,
        resolve_targets: Box<dyn Fn() -> Option<Vec<u32>> + Send>,
        on_samples: Box<dyn FnMut(&[f32]) + Send>,
    ) -> Result<Self> {
        let (quit_tx, quit_rx) = pw::channel::channel::<LoopCommand>();
        let (ready_tx, ready_rx) = std_mpsc::channel::<Result<()>>();

        let target_desc = format!("AppAudio {{ label: {:?} }}", label);
        let loop_label = label.clone();

        let handle = std::thread::Builder::new()
            .name("pw-app-capture".into())
            .spawn(move || {
                if let Err(e) =
                    app_capture::run_app_capture_loop(loop_label, on_samples, quit_rx, &ready_tx)
                {
                    let _ = ready_tx.send(Err(e));
                }
            })
            .context("failed to spawn pw-app-capture thread")?;

        match ready_rx.recv_timeout(Duration::from_secs(5)) {
            Ok(Ok(())) => {}
            Ok(Err(e)) => {
                let _ = handle.join();
                return Err(e);
            }
            Err(_) => {
                let _ = quit_tx.send(LoopCommand::Terminate);
                let _ = handle.join();
                return Err(anyhow!("PipeWire app capture setup timed out"));
            }
        }

        let stop = Arc::new(AtomicBool::new(false));
        let resolver = spawn_target_poller(
            label.clone(),
            resolve_targets,
            quit_tx.clone(),
            stop.clone(),
        )
        .inspect_err(|_| {
            let _ = quit_tx.send(LoopCommand::Terminate);
        })?;

        info!("pw: app capture running ({})", target_desc);
        Ok(Self {
            quit_tx: Some(quit_tx),
            handle: Some(handle),
            resolver_stop: Some(stop),
            resolver: Some(resolver),
            target_desc,
        })
    }

    pub fn stop(mut self) {
        self.shutdown();
    }

    fn shutdown(&mut self) {
        // The poller holds a sender into the loop, so it goes first.
        if let Some(stop) = self.resolver_stop.take() {
            stop.store(true, Ordering::Relaxed);
        }
        if let Some(handle) = self.resolver.take() {
            let _ = handle.join();
        }
        if let Some(tx) = self.quit_tx.take() {
            let _ = tx.send(LoopCommand::Terminate);
        }
        if let Some(handle) = self.handle.take() {
            let _ = handle.join();
        }
    }
}

impl Drop for PwCaptureStream {
    fn drop(&mut self) {
        self.shutdown();
    }
}

/// How often an app capture re-asks which nodes the app is playing on.
const APP_TARGET_POLL: Duration = Duration::from_secs(1);

/// Slice the poll interval so stopping doesn't wait a whole one out.
const APP_TARGET_POLL_SLICE: Duration = Duration::from_millis(100);

/// Run `resolve` on a loop, pushing every change into the capture loop.
///
/// Polling rather than watching the registry: resolving "is this Discord"
/// needs the client object behind a stream node as well as the node
/// itself, which is exactly the join [`snapshot_graph`] already does, and
/// a second of latency on a stream appearing is inaudible next to the
/// app's own startup.
fn spawn_target_poller(
    label: String,
    resolve: Box<dyn Fn() -> Option<Vec<u32>> + Send>,
    tx: pw::channel::Sender<LoopCommand>,
    stop: Arc<AtomicBool>,
) -> Result<JoinHandle<()>> {
    std::thread::Builder::new()
        .name("pw-app-targets".into())
        .spawn(move || {
            let mut last: Option<Vec<u32>> = None;
            while !stop.load(Ordering::Relaxed) {
                // `None` means the graph could not be read this tick;
                // hold the links we have rather than cutting the app off
                // over one failed roundtrip.
                let Some(mut ids) = resolve() else {
                    wait_a_tick(&stop);
                    continue;
                };
                ids.sort_unstable();
                ids.dedup();

                if last.as_ref() != Some(&ids) {
                    debug!("pw: {} is playing on node(s) {:?}", label, ids);
                    if tx.send(LoopCommand::SetTargets(ids.clone())).is_err() {
                        // The loop is gone; nothing left to steer.
                        break;
                    }
                    last = Some(ids);
                }

                wait_a_tick(&stop);
            }
        })
        .context("failed to spawn pw-app-targets thread")
}

/// Sleep one poll interval, in slices, so a stop doesn't wait it out.
fn wait_a_tick(stop: &AtomicBool) {
    let mut waited = Duration::ZERO;
    while waited < APP_TARGET_POLL && !stop.load(Ordering::Relaxed) {
        std::thread::sleep(APP_TARGET_POLL_SLICE);
        waited += APP_TARGET_POLL_SLICE;
    }
}

/// The one format every capture stream negotiates: interleaved f32 at
/// [`CAPTURE_RATE`] Hz, [`CAPTURE_CHANNELS`] channels. The PipeWire graph
/// does whatever resampling and channel mixing that takes, so consumers
/// downstream never see anything else.
fn capture_format_pod() -> Result<Vec<u8>> {
    let mut audio_info = spa::param::audio::AudioInfoRaw::new();
    audio_info.set_format(spa::param::audio::AudioFormat::F32LE);
    audio_info.set_rate(CAPTURE_RATE);
    audio_info.set_channels(CAPTURE_CHANNELS);
    let pod_object = spa::pod::Object {
        type_: spa::utils::SpaTypes::ObjectParamFormat.as_raw(),
        id: spa::param::ParamType::EnumFormat.as_raw(),
        properties: audio_info.into(),
    };
    Ok(spa::pod::serialize::PodSerializer::serialize(
        std::io::Cursor::new(Vec::new()),
        &spa::pod::Value::Object(pod_object),
    )
    .map_err(|e| anyhow!("failed to serialize format pod: {:?}", e))?
    .0
    .into_inner())
}

/// Hand one dequeued buffer to the caller's callback.
fn deliver_samples(stream: &pw::stream::Stream, on_samples: &mut dyn FnMut(&[f32])) {
    let Some(mut buffer) = stream.dequeue_buffer() else {
        return;
    };
    let datas = buffer.datas_mut();
    if datas.is_empty() {
        return;
    }
    let data = &mut datas[0];
    let n_bytes = data.chunk().size() as usize;
    if n_bytes == 0 {
        return;
    }
    if let Some(bytes) = data.data() {
        let n_samples = n_bytes / std::mem::size_of::<f32>();
        let (_, samples, _) = unsafe { bytes[..n_bytes].align_to::<f32>() };
        // MAP_BUFFERS memory is f32-aligned in practice; the align_to
        // guard keeps us safe if it ever isn't.
        if samples.len() == n_samples {
            on_samples(samples);
        }
    }
}

fn run_capture_loop(
    target: CaptureTarget,
    mut on_samples: Box<dyn FnMut(&[f32]) + Send>,
    quit_rx: pw::channel::Receiver<LoopCommand>,
    ready_tx: &std_mpsc::Sender<Result<()>>,
) -> Result<()> {
    pw::init();

    let mainloop = pw::main_loop::MainLoopRc::new(None)?;
    let context = pw::context::ContextRc::new(&mainloop, None)?;
    let core = context.connect_rc(None)?;

    let mainloop_quit = mainloop.clone();
    let _quit_attach = quit_rx.attach(mainloop.loop_(), move |cmd| {
        if matches!(cmd, LoopCommand::Terminate) {
            mainloop_quit.quit();
        }
    });

    let mut props = pw::properties::properties! {
        *pw::keys::MEDIA_TYPE => "Audio",
        *pw::keys::MEDIA_CATEGORY => "Capture",
        *pw::keys::MEDIA_ROLE => "Communication",
        *pw::keys::APP_NAME => "Meetily",
        *pw::keys::NODE_NAME => MEETILY_CAPTURE_NODE,
    };
    if target.is_sink_capture() {
        props.insert(*pw::keys::STREAM_CAPTURE_SINK, "true");
    }
    if let Some(node) = target.node_name() {
        props.insert(*pw::keys::TARGET_OBJECT, node);
    }

    let stream = pw::stream::StreamBox::new(&core, "meetily-capture", props)?;

    let mainloop_err = mainloop.clone();
    let _listener = stream
        .add_local_listener_with_user_data(())
        .state_changed(move |_stream, _ud, _old, new| {
            if let pw::stream::StreamState::Error(e) = new {
                warn!("pw: capture stream error state: {}", e);
                mainloop_err.quit();
            }
        })
        .process(move |stream, _ud| {
            deliver_samples(stream, &mut *on_samples);
        })
        .register()?;

    let pod_bytes = capture_format_pod()?;
    let mut params = [spa::pod::Pod::from_bytes(&pod_bytes)
        .ok_or_else(|| anyhow!("failed to build format pod"))?];

    stream.connect(
        spa::utils::Direction::Input,
        None,
        pw::stream::StreamFlags::AUTOCONNECT
            | pw::stream::StreamFlags::MAP_BUFFERS
            | pw::stream::StreamFlags::RT_PROCESS,
        &mut params,
    )?;

    let _ = ready_tx.send(Ok(()));
    mainloop.run();
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::{AtomicUsize, Ordering};
    use std::sync::Arc;

    /// Requires a running PipeWire session — run explicitly:
    /// `cargo test -p meetily --lib pw::tests -- --ignored --nocapture`
    #[test]
    #[ignore]
    fn enumerate_smoke() {
        let devices = enumerate_devices().expect("enumeration failed");
        println!("enumerated {} devices:", devices.len());
        for d in &devices {
            println!("  [{:?}] {} — {}", d.kind, d.label, d.id);
        }
        assert!(
            devices.iter().any(|d| d.kind == PwDeviceKind::Microphone),
            "expected at least one microphone node"
        );
    }

    /// Open a call (or anything that takes the mic) before running this.
    #[test]
    #[ignore]
    fn enumerate_capture_streams_smoke() {
        let clients = enumerate_capture_streams().expect("enumeration failed");
        println!("{} app(s) capturing audio:", clients.len());
        for c in &clients {
            println!(
                "  {} — app={:?} id={:?} binary={:?} pid={:?} role={:?} media={:?}",
                c.node_name, c.app_name, c.app_id, c.binary, c.pid, c.media_role, c.media_name
            );
        }
    }

    #[test]
    #[ignore]
    fn capture_default_system_smoke() {
        let count = Arc::new(AtomicUsize::new(0));
        let count_cb = count.clone();
        let stream = PwCaptureStream::open(
            CaptureTarget::DefaultSystem,
            Box::new(move |samples| {
                count_cb.fetch_add(samples.len(), Ordering::Relaxed);
            }),
        )
        .expect("failed to open default system capture");
        std::thread::sleep(Duration::from_secs(2));
        stream.stop();
        let total = count.load(Ordering::Relaxed);
        println!("captured {} samples in 2s", total);
        // 2s of 48kHz stereo ≈ 192_000 samples; accept a generous lower
        // bound to absorb startup latency.
        assert!(total > 48_000, "expected ~192k samples, got {}", total);
    }
}
