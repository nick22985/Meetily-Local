//! Automatic meeting detection.
//!
//! Watches the PipeWire graph for applications that hold a capture stream —
//! that is, for whoever currently has the microphone open — and turns a
//! sustained grab by a conferencing app into a meeting start, with optional
//! auto-record.
//!
//! # Why the audio graph
//!
//! The obvious approach, and the one the upstream `meeting-minutes-autodetect`
//! fork takes, is to look for known processes and window titles. That doesn't
//! survive the trip to Linux: Zoom's telltale `CptHost` helper is a macOS
//! implementation detail, browser tab titles (the usual way to spot a Google
//! Meet tab) aren't readable under Wayland, and matching on "is Teams
//! running" fires all day for an app most people never close.
//!
//! Capture streams sidestep all of it. An app opens one when it starts
//! listening and closes it when it stops, which is exactly the signal we
//! want, and it's the same signal regardless of which app or which meeting
//! service is involved. The cost is that we can't tell *which* call it is —
//! only that one is happening — which is a trade worth making.
//!
//! # Layout
//!
//! - [`apps`] — the table of conferencing apps and how their capture
//!   clients are recognized
//! - [`settings`] — persisted user preferences, stored in `app_settings`
//! - [`detector`] — the monitor loop, debouncing, and recording control
//! - [`commands`] — the Tauri command surface
//! - [`voice_socket`] — the in-call check for apps that hold the
//!   microphone whether or not a call is happening

pub mod apps;
pub mod commands;
pub mod detector;
pub mod settings;
pub mod voice_socket;

pub use detector::{CaptureStreamInfo, DetectedMeeting, MeetingDetectionStatus, MeetingDetector};
pub use settings::MeetingDetectionSettings;
