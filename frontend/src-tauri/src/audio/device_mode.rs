//! How a device *selection* becomes an actual PipeWire node.
//!
//! The picker stores either a stable `node.name` or one of two
//! sentinels, and the difference matters while a recording is running:
//! a named node is pinned, whereas the sentinels are questions that have
//! to be re-asked as the graph changes.
//!
//! - [`DEVICE_SYSTEM_DEFAULT`] — the system default source/sink. Tracked
//!   live, so switching the default mid-recording (a profile-switcher
//!   keybind, unplugging an interface) moves the capture with it.
//! - [`DEVICE_COMMUNICATION`] — whatever device the call in progress is
//!   on, falling back to the system default when nothing is in a call.
//!   Linux has no separate "communication default" for this to read (see
//!   [`crate::audio::pw::PwDefaults`]), so it is answered by looking at
//!   what the call's streams are actually linked to.

use crate::audio::app_audio::APP_PREFIX;
use crate::audio::pw::{self, CallRoute, PwDeviceKind, PwGraphSnapshot};
use crate::meeting_detector::apps;

/// Sentinel for "the system default source/sink".
pub const DEVICE_SYSTEM_DEFAULT: &str = "default";

/// Sentinel for "whatever device the active call is using".
pub const DEVICE_COMMUNICATION: &str = "communication";

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum DeviceMode {
    /// Follow the system default.
    SystemDefault,
    /// Follow the active call, falling back to the system default.
    Communication,
    /// One specific `node.name`, never moved.
    Explicit(String),
    /// One application's audio, wherever it happens to be playing.
    /// System side only.
    App(String),
}

impl DeviceMode {
    pub fn parse(device_id: &str) -> Self {
        match device_id {
            DEVICE_SYSTEM_DEFAULT => Self::SystemDefault,
            DEVICE_COMMUNICATION => Self::Communication,
            other => match other.strip_prefix(APP_PREFIX) {
                Some(app) => Self::App(app.to_string()),
                None => Self::Explicit(other.to_string()),
            },
        }
    }

    /// Whether this selection can move while a recording is running, and
    /// so needs the graph re-checked.
    ///
    /// An app selection does not: it moves with the app, but the capture
    /// stream follows that itself by relinking, with no gap and nothing
    /// for [`crate::audio::device_follower`] to reopen.
    pub fn follows_graph(&self) -> bool {
        matches!(self, Self::SystemDefault | Self::Communication)
    }

    /// The `node.name` this selection currently points at.
    ///
    /// `None` means "we could not name a node" — for the sentinels that
    /// happens when the graph has no default set at all. Callers fall
    /// back to letting PipeWire pick, which is what an untargeted stream
    /// does anyway.
    pub fn resolve(&self, kind: PwDeviceKind, snapshot: &PwGraphSnapshot) -> Option<String> {
        match self {
            Self::Explicit(node) => Some(node.clone()),
            // Its own answer, and deliberately not a node name: the app
            // is the route, and reporting it as one keeps the follower
            // from deciding the stream has moved. There is nothing to tap
            // on the microphone side, so there it stands for the default.
            Self::App(app) => match kind {
                PwDeviceKind::System => Some(format!("{}{}", APP_PREFIX, app)),
                PwDeviceKind::Microphone => default_for(kind, snapshot),
            },
            Self::SystemDefault => default_for(kind, snapshot),
            Self::Communication => {
                let route = call_route(snapshot);
                let on_call = match kind {
                    PwDeviceKind::Microphone => route.source,
                    PwDeviceKind::System => route.sink,
                };
                on_call.or_else(|| default_for(kind, snapshot))
            }
        }
    }
}

fn default_for(kind: PwDeviceKind, snapshot: &PwGraphSnapshot) -> Option<String> {
    match kind {
        PwDeviceKind::Microphone => snapshot.defaults.source(),
        PwDeviceKind::System => snapshot.defaults.sink(),
    }
    .map(str::to_string)
}

/// Which devices the call in progress is on.
///
/// A stream counts as a call if either signal fires:
///
/// - its `media.role` names one of the call roles, or
/// - the app behind it is in the meeting detector's table.
///
/// The role alone is not enough. It is optional, and the apps this
/// feature exists for routinely omit it — Discord's playback stream in
/// particular usually carries no role at all, which would leave the
/// system side falling back to the default sink and silently recording
/// the wrong device. The app table is the same one that decides whether
/// something is a meeting worth auto-recording, so the two features
/// agree on what a call is.
pub fn call_route(snapshot: &PwGraphSnapshot) -> CallRoute {
    snapshot.call_route_matching(|stream| {
        pw::has_call_role(stream) || apps::classify(&stream.match_keys()).is_some()
    })
}
