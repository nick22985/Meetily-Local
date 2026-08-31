//! The table of applications whose microphone use counts as a meeting.
//!
//! Matching is done against every identifier PipeWire exposes for a capture
//! stream — the process binary, the Flatpak app id, the owning client's name,
//! the node's own `application.name`, and the node name — all lowercased. See
//! [`crate::audio::pw::PwCaptureClient::match_keys`], which orders them most
//! specific first. Substring matching is deliberate: distributions ship the
//! same app under names like `zoom`, `ZoomLauncher`, and `zoom.real`, and
//! Flatpak wrappers prefix or suffix almost everything.

use serde::{Deserialize, Serialize};

/// One entry in the known-app table.
pub struct MeetingApp {
    /// Stable identifier. Persisted in the user's ignore list, so renaming
    /// one silently re-enables it for everybody — don't.
    pub key: &'static str,
    /// What the settings UI calls it.
    pub label: &'static str,
    /// Extra context for the settings UI, where the label isn't obvious.
    pub note: Option<&'static str>,
    /// Lowercased substrings to match against a capture client.
    pub patterns: &'static [&'static str],
    /// Whether holding the microphone is weak evidence for this app.
    ///
    /// True where the entry doesn't identify a *program* so much as a
    /// situation: a browser tab that took the mic might be a call or might
    /// be a transcription demo, and a bare WebRTC engine is whatever app
    /// declined to say who it was. When something less ambiguous is also
    /// capturing, that wins — and a stream this weak that was already open
    /// before we started watching is written off rather than recorded.
    pub ambiguous: bool,
    /// Whether this app holds the microphone open outside of calls.
    ///
    /// Discord opens a capture stream when it launches and keeps it, in a
    /// call or not, `Running` throughout; browsers hold one for whichever
    /// page asked; Mumble sits in a channel all afternoon. For these, a
    /// stream that was already open when we started watching says nothing
    /// at all, so it is written off rather than recorded. Apps without
    /// this get the benefit of the doubt: if Zoom has the microphone you
    /// are in a call, and launching Meetily mid-call should record it.
    ///
    /// Distinct from [`ambiguous`](Self::ambiguous), which is about how
    /// much the app's *identity* tells us. Discord is unambiguous — we
    /// know exactly what it is — and still idles with the mic open.
    pub idles_with_mic: bool,
    /// Start out in the user's ignore list.
    ///
    /// For apps where holding the microphone doesn't mean a meeting is
    /// happening — a voice channel you idle in all afternoon opens exactly
    /// the same capture stream a call does, and auto-recording that is
    /// hours of disk for nothing. Still togglable; just not the default.
    pub default_ignored: bool,
}

/// The serializable view of [`MeetingApp`] handed to the frontend.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MeetingAppInfo {
    pub key: String,
    pub label: String,
    pub note: Option<String>,
}

/// Applications treated as conferencing tools.
///
/// Browsers are one entry rather than several: which *site* holds the
/// microphone is not knowable from the audio graph, and it doesn't need to
/// be — a browser only opens a capture stream when a page asked for the
/// mic, and on a work machine that page is a call.
pub const KNOWN_MEETING_APPS: &[MeetingApp] = &[
    MeetingApp {
        key: "zoom",
        label: "Zoom",
        note: None,
        ambiguous: false,
        idles_with_mic: false,
        default_ignored: false,
        patterns: &["zoom"],
    },
    MeetingApp {
        key: "teams",
        label: "Microsoft Teams",
        note: None,
        ambiguous: false,
        idles_with_mic: false,
        default_ignored: false,
        patterns: &["teams", "msteams", "ms-teams"],
    },
    MeetingApp {
        key: "browser",
        label: "Browser calls",
        note: Some(
            "Google Meet, Jitsi, Whereby, Teams for web - any tab that takes the mic. \
             A tab already holding it when Meetily starts is skipped, so a \
             transcription page left open doesn't read as a call",
        ),
        ambiguous: true,
        idles_with_mic: true,
        default_ignored: false,
        patterns: &[
            "chrome",
            "chromium",
            "brave",
            "vivaldi",
            "msedge",
            "microsoft-edge",
            "microsoft.edge",
            "firefox",
            "librewolf",
            "zen",
            "epiphany",
            "gnome-web",
        ],
    },
    MeetingApp {
        key: "slack",
        label: "Slack",
        note: Some("Huddles"),
        ambiguous: false,
        idles_with_mic: false,
        default_ignored: false,
        patterns: &["slack"],
    },
    MeetingApp {
        key: "discord",
        label: "Discord",
        note: Some(
            "Idling in a voice channel looks exactly like a call, so this can \
             record a whole afternoon - raise the stop grace period if that bites",
        ),
        ambiguous: false,
        idles_with_mic: true,
        default_ignored: false,
        patterns: &["discord", "vesktop", "webcord", "armcord", "legcord"],
    },
    MeetingApp {
        key: "webex",
        label: "Webex",
        note: None,
        ambiguous: false,
        idles_with_mic: false,
        default_ignored: false,
        patterns: &["webex", "ciscospark"],
    },
    MeetingApp {
        key: "skype",
        label: "Skype",
        note: None,
        ambiguous: false,
        idles_with_mic: false,
        default_ignored: false,
        patterns: &["skype", "skypeforlinux"],
    },
    MeetingApp {
        key: "element",
        label: "Element",
        note: Some("Matrix calls"),
        ambiguous: false,
        idles_with_mic: false,
        default_ignored: false,
        patterns: &["element", "schildichat", "riot"],
    },
    MeetingApp {
        key: "signal",
        label: "Signal",
        note: None,
        ambiguous: false,
        idles_with_mic: false,
        default_ignored: false,
        patterns: &["signal-desktop", "signal"],
    },
    MeetingApp {
        key: "telegram",
        label: "Telegram",
        note: None,
        ambiguous: false,
        idles_with_mic: false,
        default_ignored: false,
        patterns: &["telegram"],
    },
    MeetingApp {
        key: "jami",
        label: "Jami",
        note: None,
        ambiguous: false,
        idles_with_mic: false,
        default_ignored: false,
        patterns: &["jami"],
    },
    MeetingApp {
        key: "webrtc",
        label: "Voice call",
        note: Some(
            "An app running the WebRTC voice engine without telling PipeWire who \
             it is - some Flatpak and Electron clients look like this. Weak \
             evidence on its own: Discord opens one of these at launch and \
             holds it whether or not you are in a call, so one already open \
             when Meetily starts is skipped",
        ),
        ambiguous: true,
        idles_with_mic: true,
        default_ignored: false,
        patterns: &["webrtc voiceengine"],
    },
    MeetingApp {
        key: "mumble",
        label: "Mumble",
        note: Some("Off by default - same persistent voice channels as Discord"),
        ambiguous: false,
        idles_with_mic: true,
        default_ignored: true,
        patterns: &["mumble"],
    },
];

/// The known-app table in the shape the settings UI wants.
pub fn app_catalog() -> Vec<MeetingAppInfo> {
    KNOWN_MEETING_APPS
        .iter()
        .map(|app| MeetingAppInfo {
            key: app.key.to_string(),
            label: app.label.to_string(),
            note: app.note.map(str::to_string),
        })
        .collect()
}

/// Look an entry up by its stable key. `None` for an unrecognized app,
/// which has no entry by definition.
pub fn by_key(key: Option<&str>) -> Option<&'static MeetingApp> {
    let key = key?;
    KNOWN_MEETING_APPS.iter().find(|app| app.key == key)
}

/// How strong a signal this app is, lowest first. Used to break ties when
/// several apps hold the microphone at once.
pub fn rank(app: Option<&MeetingApp>) -> u8 {
    match app {
        Some(app) if !app.ambiguous => 0,
        Some(_) => 1,
        None => 2,
    }
}

/// Keys that start out ignored. Seeds
/// [`crate::meeting_detector::MeetingDetectionSettings::default`].
pub fn default_ignored_keys() -> Vec<String> {
    KNOWN_MEETING_APPS
        .iter()
        .filter(|app| app.default_ignored)
        .map(|app| app.key.to_string())
        .collect()
}

/// Find the app a set of lowercased identifiers belongs to.
///
/// `keys` comes from [`crate::audio::pw::PwCaptureClient::match_keys`] and is
/// ordered most-specific first, so the binary name wins over a vaguer
/// display name.
pub fn classify(keys: &[String]) -> Option<&'static MeetingApp> {
    for key in keys {
        for app in KNOWN_MEETING_APPS {
            if app.patterns.iter().any(|p| key.contains(p)) {
                return Some(app);
            }
        }
    }
    None
}

#[cfg(test)]
mod tests {
    use super::*;

    fn keys(items: &[&str]) -> Vec<String> {
        items.iter().map(|s| s.to_lowercase()).collect()
    }

    #[test]
    fn binary_names_classify_to_their_app() {
        assert_eq!(classify(&keys(&["zoom"])).unwrap().key, "zoom");
        assert_eq!(classify(&keys(&["firefox"])).unwrap().key, "browser");
        assert_eq!(classify(&keys(&["ms-teams"])).unwrap().key, "teams");
    }

    #[test]
    fn flatpak_and_launcher_variants_still_match() {
        // Flatpak wrappers and distro launchers decorate the binary name.
        assert_eq!(classify(&keys(&["ZoomLauncher"])).unwrap().key, "zoom");
        assert_eq!(
            classify(&keys(&["org.chromium.Chromium"])).unwrap().key,
            "browser"
        );
        assert_eq!(classify(&keys(&["signal-desktop"])).unwrap().key, "signal");
    }

    #[test]
    fn the_binary_wins_over_the_display_name() {
        // `application.name` is whatever the app felt like calling itself;
        // the binary is the thing we can trust, so it's checked first.
        let app = classify(&keys(&["firefox", "Zoom Meeting"])).unwrap();
        assert_eq!(app.key, "browser");
    }

    #[test]
    fn discord_forks_classify_as_discord() {
        // The third-party clients don't all have "discord" in the name.
        for binary in ["Discord", "com.discordapp.Discord", "vesktop", "legcord"] {
            assert_eq!(
                classify(&keys(&[binary])).unwrap().key,
                "discord",
                "{} should classify as discord",
                binary
            );
        }
    }

    #[test]
    fn flatpak_app_ids_classify() {
        // Sandboxed clients often expose `application.id` and nothing else.
        for (id, expected) in [
            ("com.discordapp.Discord", "discord"),
            ("us.zoom.Zoom", "zoom"),
            ("com.microsoft.Teams", "teams"),
            ("com.microsoft.Edge", "browser"),
            ("org.chromium.Chromium", "browser"),
            ("im.riot.Riot", "element"),
            ("info.mumble.Mumble", "mumble"),
        ] {
            assert_eq!(
                classify(&keys(&[id])).map(|a| a.key),
                Some(expected),
                "{} should classify as {}",
                id,
                expected
            );
        }
    }

    #[test]
    fn unrelated_apps_are_not_meetings() {
        assert!(classify(&keys(&["audacity"])).is_none());
        assert!(classify(&keys(&["obs"])).is_none());
        assert!(classify(&keys(&["meetily-capture"])).is_none());
    }

    #[test]
    fn persistent_voice_channel_apps_start_out_ignored() {
        // Recognized, but not watched until asked for.
        assert_eq!(classify(&keys(&["mumble"])).unwrap().key, "mumble");
        let ignored = default_ignored_keys();
        assert!(ignored.contains(&"mumble".to_string()));
        assert!(!ignored.contains(&"zoom".to_string()));
        // Discord idles in voice channels the same way, but it is far and
        // away the most common thing people actually want recorded, and
        // shipping it ignored just reads as the detector being broken.
        assert!(!ignored.contains(&"discord".to_string()));
    }

    #[test]
    fn a_dedicated_client_outranks_a_browser() {
        let discord = classify(&keys(&["discord"]));
        let browser = classify(&keys(&["firefox"]));
        assert!(rank(discord) < rank(browser));
        assert!(rank(browser) < rank(None));
    }

    #[test]
    fn apps_that_sit_on_the_mic_are_flagged_as_such() {
        // The ones that open a capture stream and keep it whether or not
        // anything is happening. Not the same question as `ambiguous`:
        // Discord is perfectly identifiable and still does this.
        for binary in ["discord", "firefox", "mumble", "WEBRTC VoiceEngine"] {
            assert!(
                classify(&keys(&[binary])).unwrap().idles_with_mic,
                "{} holds the mic outside calls",
                binary
            );
        }
        for binary in ["zoom", "ms-teams", "webex"] {
            assert!(
                !classify(&keys(&[binary])).unwrap().idles_with_mic,
                "{} takes the mic only for a call",
                binary
            );
        }
    }

    #[test]
    fn an_anonymous_voice_engine_is_weak_evidence() {
        // It says "some app is doing voice", which is a situation, not an
        // app. A named client saying the same thing is worth more.
        let webrtc = classify(&keys(&["WEBRTC VoiceEngine"]));
        assert!(webrtc.unwrap().ambiguous);
        assert!(rank(classify(&keys(&["zoom"]))) < rank(webrtc));
    }

    #[test]
    fn the_webrtc_engine_is_only_a_fallback() {
        // A named app always wins, because `classify` walks the keys in
        // most-specific-first order and the engine name is last.
        assert_eq!(
            classify(&keys(&["Discord", "WEBRTC VoiceEngine"]))
                .unwrap()
                .key,
            "discord"
        );
        assert_eq!(
            classify(&keys(&["Chromium", "WEBRTC VoiceEngine"]))
                .unwrap()
                .key,
            "browser"
        );
        // Nothing else to go on: still a call.
        assert_eq!(
            classify(&keys(&["WEBRTC VoiceEngine"])).unwrap().key,
            "webrtc"
        );
    }

    #[test]
    fn app_keys_are_unique() {
        let mut seen = std::collections::HashSet::new();
        for app in KNOWN_MEETING_APPS {
            assert!(seen.insert(app.key), "duplicate app key: {}", app.key);
        }
    }
}
