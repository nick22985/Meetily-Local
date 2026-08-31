//! Capturing one application's audio instead of everything the speakers
//! are playing.
//!
//! The ordinary system-audio path captures a sink's monitor, which is the
//! whole mix: the call, the video playing in another window, and the
//! notification that arrived halfway through. Tapping a single app means
//! not going near the sink at all.
//!
//! PipeWire makes that possible because an application's playback stream
//! is itself a node with its own output ports, and a port can feed more
//! than one link. So this opens a capture stream that the session manager
//! is told to leave alone (`node.autoconnect = false`, no
//! `target.object`), and then links the app's output ports straight into
//! it. The app keeps playing to its sink exactly as before — we are a
//! second consumer of the same ports, not a redirect — so nothing about
//! what the user hears changes, and nothing has to be moved to a null
//! sink and moved back afterwards.
//!
//! Which nodes belong to the app is decided elsewhere (see
//! `audio::app_audio`) and arrives as [`LoopCommand::SetTargets`]; this
//! module only knows node ids. Both ends move on their own — the app
//! opens a stream for a notification sound, a call ends, a port
//! disappears — so links are reconciled from the registry rather than set
//! up once at open time.

use anyhow::{anyhow, Context as _, Result};
use log::{debug, info, warn};
use std::cell::RefCell;
use std::collections::HashMap;
use std::rc::Rc;
use std::sync::mpsc as std_mpsc;

use pipewire as pw;
use pw::spa;

use super::{capture_format_pod, deliver_samples, LoopCommand, MEETILY_CAPTURE_NODE};

/// One audio port, as the registry announced it.
#[derive(Debug, Clone)]
struct PortRef {
    /// Global object id — what a link refers to.
    id: u32,
    /// `port.id`, the port's index within its node. Only used to keep an
    /// order that matches the node's channel order.
    index: u32,
    /// `audio.channel` — "FL", "FR", "MONO", or absent on nodes that
    /// don't bother to say.
    channel: Option<String>,
}

/// Pairs an app's output ports with our capture stream's input ports.
///
/// Both sides arrive piecemeal from the registry and change while a
/// recording runs, so every event just updates the picture and calls
/// [`reconcile`](Self::reconcile); nothing here assumes an order.
struct AppLinker {
    core: pw::core::CoreRc,
    /// The app this capture is for, for logging.
    label: String,
    /// Our own capture node, once the server has created it.
    sink_node: Option<u32>,
    /// Input ports, by node id. Only ours matter, but a port can be
    /// announced before the stream has told us which node is ours, so
    /// they are kept until that question is answered.
    inputs: HashMap<u32, Vec<PortRef>>,
    /// Output ports of every node in the graph, by node id. Kept for all
    /// of them because a node that isn't a target now may become one.
    outputs: HashMap<u32, Vec<PortRef>>,
    /// Which node each known port belongs to, so a `global_remove` —
    /// which carries nothing but an id — can be attributed.
    port_owner: HashMap<u32, u32>,
    /// The nodes we have been told to tap.
    targets: Vec<u32>,
    /// Links we created, by the node they tap. Dropping a proxy destroys
    /// the link, which is how unlinking happens.
    links: HashMap<u32, Vec<pw::link::Link>>,
}

impl AppLinker {
    fn new(core: pw::core::CoreRc, label: String) -> Self {
        Self {
            core,
            label,
            sink_node: None,
            inputs: HashMap::new(),
            outputs: HashMap::new(),
            port_owner: HashMap::new(),
            targets: Vec::new(),
            links: HashMap::new(),
        }
    }

    /// Our capture node's id, learned from the stream once the server has
    /// created the node.
    fn set_sink_node(&mut self, node_id: u32) {
        if node_id == pw::constants::ID_ANY || self.sink_node == Some(node_id) {
            return;
        }
        debug!("pw: app capture node is {}", node_id);
        self.sink_node = Some(node_id);
        // Every other node's inputs were only being held in case they
        // turned out to be ours.
        self.inputs.retain(|node, _| *node == node_id);
        self.reconcile();
    }

    /// Our capture node's input ports, in channel order.
    fn sink_ports(&self) -> &[PortRef] {
        self.sink_node
            .and_then(|node| self.inputs.get(&node))
            .map(Vec::as_slice)
            .unwrap_or_default()
    }

    /// Record a port the registry has announced.
    fn add_port(&mut self, node_id: u32, direction: Option<&str>, port: PortRef) {
        let slot = match direction {
            // Ours, or not yet known to be anyone's.
            Some("in") if self.sink_node.is_none_or(|ours| ours == node_id) => {
                self.inputs.entry(node_id).or_default()
            }
            Some("in") => return,
            Some("out") => self.outputs.entry(node_id).or_default(),
            // A port that won't say which way it faces is no use to a link.
            _ => return,
        };

        if slot.iter().any(|p| p.id == port.id) {
            return;
        }
        slot.push(port.clone());
        slot.sort_by_key(|p| p.index);
        self.port_owner.insert(port.id, node_id);

        // A new output port on a node we are already tapping is a channel
        // we aren't carrying yet; relink it from scratch.
        if direction == Some("out") {
            self.links.remove(&node_id);
        }
        self.reconcile();
    }

    /// Forget an object that has left the graph — a port, or one of the
    /// nodes we were tapping.
    fn remove_global(&mut self, id: u32) {
        if let Some(node_id) = self.port_owner.remove(&id) {
            if let Some(ports) = self.inputs.get_mut(&node_id) {
                ports.retain(|p| p.id != id);
            }
            if let Some(ports) = self.outputs.get_mut(&node_id) {
                ports.retain(|p| p.id != id);
            }
            // The server destroyed every link on that port along with it,
            // so our proxies for this node are stale: drop them and
            // relink from whatever ports are left.
            self.links.remove(&node_id);
            self.reconcile();
            return;
        }

        if self.outputs.remove(&id).is_some() {
            debug!("pw: {} stream {} went away", self.label, id);
        }
        self.links.remove(&id);
        self.inputs.remove(&id);
        if self.sink_node == Some(id) {
            self.sink_node = None;
        }
    }

    /// Replace the set of nodes we should be tapping.
    fn set_targets(&mut self, targets: Vec<u32>) {
        self.targets = targets;
        // Dropping the proxies of a node we no longer want unlinks it.
        self.links.retain(|node, _| self.targets.contains(node));
        self.reconcile();
    }

    /// Link every target that isn't linked yet.
    fn reconcile(&mut self) {
        let Some(sink_node) = self.sink_node else {
            return;
        };
        let sinks = self.sink_ports().to_vec();
        if sinks.is_empty() {
            return;
        }

        for node in self.targets.clone() {
            if self.links.contains_key(&node) {
                continue;
            }
            // Its ports haven't been announced yet; the announcement will
            // bring us back here.
            let Some(ports) = self.outputs.get(&node) else {
                continue;
            };
            let pairs = channel_map(ports, &sinks);
            if pairs.is_empty() {
                continue;
            }

            let mut links = Vec::with_capacity(pairs.len());
            for (out_port, in_port) in pairs {
                match self.link(node, out_port, sink_node, in_port) {
                    Ok(link) => links.push(link),
                    Err(e) => warn!(
                        "pw: could not tap {} (node {} port {}): {}",
                        self.label, node, out_port, e
                    ),
                }
            }
            if links.is_empty() {
                continue;
            }
            info!(
                "pw: tapping {} — node {} on {} link(s)",
                self.label,
                node,
                links.len()
            );
            self.links.insert(node, links);
        }
    }

    fn link(
        &self,
        out_node: u32,
        out_port: u32,
        in_node: u32,
        in_port: u32,
    ) -> Result<pw::link::Link> {
        let mut props = pw::properties::PropertiesBox::new();
        props.insert(*pw::keys::LINK_OUTPUT_NODE, out_node.to_string());
        props.insert(*pw::keys::LINK_OUTPUT_PORT, out_port.to_string());
        props.insert(*pw::keys::LINK_INPUT_NODE, in_node.to_string());
        props.insert(*pw::keys::LINK_INPUT_PORT, in_port.to_string());
        // Ours to clean up: a link that outlived the recording would keep
        // feeding a node that no longer exists.
        props.insert(*pw::keys::OBJECT_LINGER, "false");

        self.core
            .create_object::<pw::link::Link>("link-factory", &props)
            .map_err(|e| anyhow!("link-factory refused the link: {}", e))
    }
}

/// Pair an app's output ports with our input ports, by channel.
///
/// Matching on `audio.channel` rather than position, because an app that
/// plays mono and one that plays 5.1 both have to end up in the same
/// stereo capture. A mono source feeds both our channels — halving it
/// into one ear instead would be worse than the duplication. Anything we
/// have no channel for (a centre or LFE channel) is dropped rather than
/// downmixed; what this exists to record is speech, which is in front.
fn channel_map(sources: &[PortRef], sinks: &[PortRef]) -> Vec<(u32, u32)> {
    if sources.is_empty() || sinks.is_empty() {
        return Vec::new();
    }

    sinks
        .iter()
        .enumerate()
        .filter_map(|(i, sink)| {
            let src = sources
                .iter()
                .find(|s| match (&s.channel, &sink.channel) {
                    (Some(a), Some(b)) => a == b,
                    _ => false,
                })
                // A single source port is the whole signal, whatever it
                // calls itself.
                .or_else(|| (sources.len() == 1).then(|| &sources[0]))
                // Otherwise fall back on position, which is right for the
                // stereo pair that didn't label itself.
                .or_else(|| sources.get(i))?;
            Some((src.id, sink.id))
        })
        .collect()
}

/// Run an app capture until told to stop.
///
/// Mirrors `super::run_capture_loop`, minus the device targeting and plus
/// the linking; see the module docs for why the two can't be one loop.
pub(super) fn run_app_capture_loop(
    label: String,
    mut on_samples: Box<dyn FnMut(&[f32]) + Send>,
    quit_rx: pw::channel::Receiver<LoopCommand>,
    ready_tx: &std_mpsc::Sender<Result<()>>,
) -> Result<()> {
    pw::init();

    let mainloop = pw::main_loop::MainLoopRc::new(None)?;
    let context = pw::context::ContextRc::new(&mainloop, None)?;
    let core = context.connect_rc(None)?;
    let registry = core.get_registry_rc()?;

    let linker = Rc::new(RefCell::new(AppLinker::new(core.clone(), label.clone())));

    let mainloop_quit = mainloop.clone();
    let linker_cmd = linker.clone();
    let _quit_attach = quit_rx.attach(mainloop.loop_(), move |cmd| match cmd {
        LoopCommand::Terminate => mainloop_quit.quit(),
        LoopCommand::SetTargets(ids) => linker_cmd.borrow_mut().set_targets(ids),
    });

    let linker_add = linker.clone();
    let linker_remove = linker.clone();
    let _reg_listener = registry
        .add_listener_local()
        .global(move |global| {
            if global.type_ != pw::types::ObjectType::Port {
                return;
            }
            let Some(props) = global.props else { return };
            // MIDI and video ports share the registry with audio ones and
            // cannot be linked to an audio stream.
            if !props
                .get("format.dsp")
                .is_some_and(|f| f.contains("audio"))
            {
                return;
            }
            let Some(node_id) = props.get("node.id").and_then(|v| v.parse::<u32>().ok()) else {
                return;
            };
            let port = PortRef {
                id: global.id,
                index: props
                    .get("port.id")
                    .and_then(|v| v.parse::<u32>().ok())
                    .unwrap_or(global.id),
                channel: props.get("audio.channel").map(str::to_string),
            };
            linker_add
                .borrow_mut()
                .add_port(node_id, props.get("port.direction"), port);
        })
        .global_remove(move |id| {
            linker_remove.borrow_mut().remove_global(id);
        })
        .register();

    let mut props = pw::properties::properties! {
        *pw::keys::MEDIA_TYPE => "Audio",
        *pw::keys::MEDIA_CATEGORY => "Capture",
        *pw::keys::MEDIA_ROLE => "Communication",
        *pw::keys::APP_NAME => "Meetily",
        *pw::keys::NODE_NAME => MEETILY_CAPTURE_NODE,
    };
    // Nothing must connect this node for us: the session manager's idea
    // of a capture stream is "put it on the default source", which is the
    // microphone, and it would happily do that alongside our own links.
    props.insert(*pw::keys::NODE_AUTOCONNECT, "false");
    // Linked to a playback stream rather than a device, the node still
    // needs the graph's driver to run it.
    props.insert(*pw::keys::NODE_WANT_DRIVER, "true");

    let stream = pw::stream::StreamBox::new(&core, "meetily-app-capture", props)?;

    let mainloop_err = mainloop.clone();
    let linker_state = linker.clone();
    let _listener = stream
        .add_local_listener_with_user_data(())
        .state_changed(move |stream, _ud, _old, new| {
            if let pw::stream::StreamState::Error(e) = new {
                warn!("pw: app capture stream error state: {}", e);
                mainloop_err.quit();
                return;
            }
            // The node id only exists once the server has created the
            // node, which is what this transition is telling us.
            linker_state.borrow_mut().set_sink_node(stream.node_id());
        })
        .process(move |stream, _ud| {
            deliver_samples(stream, &mut *on_samples);
        })
        .register()?;

    let pod_bytes = capture_format_pod()?;
    let mut params = [spa::pod::Pod::from_bytes(&pod_bytes)
        .ok_or_else(|| anyhow!("failed to build format pod"))?];

    stream
        .connect(
            spa::utils::Direction::Input,
            None,
            pw::stream::StreamFlags::MAP_BUFFERS | pw::stream::StreamFlags::RT_PROCESS,
            &mut params,
        )
        .context("failed to connect the app capture stream")?;

    // Usually already known here; the state_changed above covers the case
    // where the server hasn't answered yet.
    linker.borrow_mut().set_sink_node(stream.node_id());

    let _ = ready_tx.send(Ok(()));
    mainloop.run();
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn port(id: u32, index: u32, channel: Option<&str>) -> PortRef {
        PortRef {
            id,
            index,
            channel: channel.map(str::to_string),
        }
    }

    fn stereo_sink() -> Vec<PortRef> {
        vec![port(10, 0, Some("FL")), port(11, 1, Some("FR"))]
    }

    #[test]
    fn stereo_pairs_by_channel_not_position() {
        // Announced in the opposite order to ours, which is exactly the
        // case position-matching would get wrong.
        let sources = vec![port(1, 0, Some("FR")), port(2, 1, Some("FL"))];
        let mut map = channel_map(&sources, &stereo_sink());
        map.sort();
        assert_eq!(map, vec![(1, 11), (2, 10)]);
    }

    #[test]
    fn a_mono_source_feeds_both_channels() {
        let sources = vec![port(1, 0, Some("MONO"))];
        let map = channel_map(&sources, &stereo_sink());
        assert_eq!(map, vec![(1, 10), (1, 11)]);
    }

    #[test]
    fn unlabelled_ports_fall_back_to_position() {
        let sources = vec![port(1, 0, None), port(2, 1, None)];
        let map = channel_map(&sources, &stereo_sink());
        assert_eq!(map, vec![(1, 10), (2, 11)]);
    }

    #[test]
    fn surround_keeps_the_front_pair() {
        let sources = vec![
            port(1, 0, Some("FL")),
            port(2, 1, Some("FR")),
            port(3, 2, Some("FC")),
            port(4, 3, Some("LFE")),
        ];
        let map = channel_map(&sources, &stereo_sink());
        assert_eq!(map, vec![(1, 10), (2, 11)]);
    }

    #[test]
    fn nothing_to_link_is_not_an_error() {
        assert!(channel_map(&[], &stereo_sink()).is_empty());
        assert!(channel_map(&stereo_sink(), &[]).is_empty());
    }
}
