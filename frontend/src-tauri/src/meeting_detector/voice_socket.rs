//! Whether an app is actually in a voice call, read from its sockets.
//!
//! PipeWire cannot answer this for the apps that need it most. Discord
//! opens a capture stream when it launches and keeps it `Running` whether
//! or not you are in a voice channel — same node id, same links, same
//! state, with a playback stream open beside it the whole time. Every
//! signal the audio graph has to offer says "in a call" permanently, which
//! leaves the detector choosing between recording all day and never
//! recording Discord at all.
//!
//! What does change is the network. A voice call means a UDP socket to a
//! media server, opened on joining and closed on leaving, so reading
//! `/proc` for the processes behind a capture stream and looking for one
//! is the only reliable in-call signal Linux offers.
//!
//! Two things about that socket are not obvious. It is often *unconnected*
//! — WebRTC stacks bind a port and `sendto` each peer rather than calling
//! `connect`, so `/proc/net/udp` shows an all-zero remote address even
//! mid-call, and looking only for connected sockets finds nothing. And it
//! rarely lives in the process that opened the audio stream: an Electron
//! app talks to PipeWire from its audio service and carries voice in its
//! network service. Both are why the search here is by binary name across
//! every matching process, rather than by the pid the audio graph named.
//!
//! Deliberately conservative. Anything unreadable answers
//! [`CallProbe::Unreadable`] — "can't tell" — rather than guessing, and
//! callers fall back to whatever they did before.

use std::collections::HashSet;
use std::fs;

/// What the probe could work out about one app.
///
/// The third variant is the point of having an enum rather than a bool:
/// "we looked and there is no call" and "we could not look" have to be
/// told apart, because the first should end a meeting and the second
/// must not.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CallProbe {
    /// The app holds a media socket — it is in a call.
    Connected,
    /// The app was read, and holds no such socket.
    Idle,
    /// Nothing about the app could be read: no process of that name, or
    /// none whose descriptors we may list.
    Unreadable,
}

/// A verdict with the counts it was reached from.
///
/// The counts exist for the detection log. The two kinds of socket are
/// evidence of very different strength — a connected one names a media
/// server, an unconnected one is only "this app has a UDP port open" —
/// and keeping them apart in the log is what makes a wrong answer
/// diagnosable instead of just wrong.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct CallEvidence {
    pub probe: CallProbe,
    /// Sockets connected to a peer, on a port that isn't web or DNS.
    pub connected: usize,
    /// Sockets bound to an ephemeral port with no peer at all.
    pub open: usize,
}

/// Every UDP socket on the machine that could be carrying a call, split by
/// how strong the evidence is.
///
/// Read once per poll and shared across the apps being asked about — this
/// is a single file read, while the per-process part is a directory scan
/// each.
#[derive(Debug, Default)]
pub struct UdpSockets {
    connected: HashSet<u64>,
    open: HashSet<u64>,
}

/// Remote ports that mean something other than a call.
///
/// Browsers hold QUIC connections to :443 open constantly, everything
/// resolves names over :53, and desktops chatter over mDNS and NTP. Media
/// servers are on ephemeral ports, so excluding the well-known ones costs
/// nothing and removes the false positives that show up in practice.
const NOT_MEDIA_PORTS: &[u16] = &[53, 80, 123, 443, 5353];

/// Local ports an unconnected socket can be bound to and still count.
///
/// Below 1024 is a service, not a client. mDNS is the one ephemeral-ish
/// port every desktop app has open regardless.
const NOT_MEDIA_LOCAL_PORTS: &[u16] = &[5353];

/// Read `/proc/net/udp` and its v6 twin.
///
/// `None` if neither could be read, which means the question can't be
/// answered at all rather than that nobody is in a call.
pub fn udp_sockets() -> Option<UdpSockets> {
    let mut sockets = UdpSockets::default();
    let mut read_any = false;

    for path in ["/proc/net/udp", "/proc/net/udp6"] {
        let Ok(table) = fs::read_to_string(path) else {
            continue;
        };
        read_any = true;
        for line in table.lines().skip(1) {
            match classify_row(line) {
                Some((true, inode)) => sockets.connected.insert(inode),
                Some((false, inode)) => sockets.open.insert(inode),
                None => continue,
            };
        }
    }

    read_any.then_some(sockets)
}

/// One row of `/proc/net/udp`: whether it is connected, and its inode.
///
/// Columns are `sl local rem st tx:rx tr:when retrnsmt uid timeout inode`,
/// with addresses as `HEXADDR:HEXPORT`. Both v4 and v6 use the same
/// layout; only the address width differs, which doesn't matter here
/// because the port is always after the colon.
///
/// `None` for a row that can't be carrying a call either way.
fn classify_row(line: &str) -> Option<(bool, u64)> {
    let mut fields = line.split_whitespace();
    let _sl = fields.next()?;
    let local = fields.next()?;
    let remote = fields.next()?;
    // uid and timeout sit between the flags and the inode.
    let inode: u64 = fields.nth(6)?.parse().ok()?;

    let (peer, remote_port) = remote.rsplit_once(':')?;
    let remote_port = u16::from_str_radix(remote_port, 16).ok()?;
    // An all-zero peer is a socket nobody called `connect` on. That is the
    // usual state for a listener, and also for a WebRTC stack mid-call.
    if !peer.bytes().all(|b| b == b'0') {
        if remote_port == 0 || NOT_MEDIA_PORTS.contains(&remote_port) {
            return None;
        }
        return Some((true, inode));
    }

    let (_, local_port) = local.rsplit_once(':')?;
    let local_port = u16::from_str_radix(local_port, 16).ok()?;
    if local_port < 1024 || NOT_MEDIA_LOCAL_PORTS.contains(&local_port) {
        return None;
    }
    Some((false, inode))
}

/// The command name behind `pid`, as `/proc` sees it.
pub fn process_name(pid: u32) -> Option<String> {
    let comm = fs::read_to_string(format!("/proc/{}/comm", pid)).ok()?;
    Some(comm.trim().to_string())
}

/// Every pid whose command name is `name`.
pub fn pids_named(name: &str) -> Vec<u32> {
    let Ok(entries) = fs::read_dir("/proc") else {
        return Vec::new();
    };

    let mut pids = Vec::new();
    for entry in entries.flatten() {
        let Some(pid) = entry.file_name().to_str().and_then(|n| n.parse().ok()) else {
            continue;
        };
        if process_name(pid).is_some_and(|comm| same_process(&comm, name)) {
            pids.push(pid);
        }
    }
    pids
}

/// Whether a `/proc` command name refers to the given binary.
///
/// `comm` is the basename and is cut off at 15 characters, so a long
/// binary name only ever matches as a prefix. Comparison is
/// case-insensitive because PipeWire reports `Discord` while some builds
/// exec themselves as `discord`.
fn same_process(comm: &str, name: &str) -> bool {
    let name = name.rsplit('/').next().unwrap_or(name).to_lowercase();
    let comm = comm.to_lowercase();
    comm == name || (comm.len() == 15 && name.starts_with(&comm))
}

/// Whether the app behind a capture stream is in a call.
///
/// Takes the binary name and the pid the audio graph reported, and treats
/// the pid as a hint rather than an answer: it is only used if the process
/// it points at really is that binary. Everything routed through
/// `pipewire-pulse` reports the bridge's pid, identical for every app, so
/// without that check the probe answers for the wrong process entirely.
pub fn probe_app(
    binary: Option<&str>,
    reported: Option<u32>,
    sockets: &UdpSockets,
) -> CallEvidence {
    let mut pids: Vec<u32> = Vec::new();

    match (reported, binary) {
        (Some(pid), Some(name)) => {
            if process_name(pid).is_some_and(|comm| same_process(&comm, name)) {
                pids.push(pid);
            }
        }
        // Nothing to check the pid against, so take it at face value.
        (Some(pid), None) => {
            if process_name(pid).is_some() {
                pids.push(pid);
            }
        }
        (None, _) => {}
    }

    if let Some(name) = binary {
        for pid in pids_named(name) {
            if !pids.contains(&pid) {
                pids.push(pid);
            }
        }
    }

    let mut evidence = CallEvidence {
        probe: CallProbe::Unreadable,
        connected: 0,
        open: 0,
    };
    let mut read_any = false;

    for pid in pids {
        // A process of the app we couldn't open says nothing; keep going
        // through the rest.
        let Some(held) = held_sockets(pid) else {
            continue;
        };
        read_any = true;
        evidence.connected += held
            .iter()
            .filter(|i| sockets.connected.contains(i))
            .count();
        evidence.open += held.iter().filter(|i| sockets.open.contains(i)).count();
    }

    if !read_any {
        return evidence;
    }
    evidence.probe = if evidence.connected > 0 || evidence.open > 0 {
        CallProbe::Connected
    } else {
        CallProbe::Idle
    };
    evidence
}

/// The socket inodes `pid` has open, or `None` if it can't be read.
fn held_sockets(pid: u32) -> Option<HashSet<u64>> {
    let entries = fs::read_dir(format!("/proc/{}/fd", pid)).ok()?;

    let mut inodes = HashSet::new();
    for entry in entries.flatten() {
        // Racing a process that is closing descriptors is normal; a
        // vanished one just isn't a socket we were looking for.
        let Ok(target) = fs::read_link(entry.path()) else {
            continue;
        };
        if let Some(inode) = socket_inode(&target.to_string_lossy()) {
            inodes.insert(inode);
        }
    }
    Some(inodes)
}

/// The inode out of a `socket:[12345]` symlink target.
fn socket_inode(target: &str) -> Option<u64> {
    target
        .strip_prefix("socket:[")?
        .strip_suffix(']')?
        .parse()
        .ok()
}

#[cfg(test)]
mod tests {
    use super::*;

    // Real rows, trimmed to the columns that matter.
    const MDNS: &str = "  123: 00000000:14E9 00000000:0000 07 00000000:00000000 00:00000000 00000000  1000        0 445566 2 0000000000000000 0";
    const VOICE: &str = "  124: 0100007F:A1B2 5DB8C43A:C350 01 00000000:00000000 00:00000000 00000000  1000        0 778899 2 0000000000000000 0";
    const QUIC: &str = "  125: 0100007F:A1B3 5DB8C43A:01BB 01 00000000:00000000 00:00000000 00000000  1000        0 998877 2 0000000000000000 0";
    const BOUND: &str = "  126: 0100007F:C351 00000000:0000 07 00000000:00000000 00:00000000 00000000  1000        0 112233 2 0000000000000000 0";
    const SERVICE: &str = "  127: 00000000:0035 00000000:0000 07 00000000:00000000 00:00000000 00000000     0        0 334455 2 0000000000000000 0";

    #[test]
    fn a_connected_socket_on_a_media_port_is_strong_evidence() {
        assert_eq!(classify_row(VOICE), Some((true, 778899)));
    }

    #[test]
    fn a_bound_socket_with_no_peer_is_weak_evidence() {
        // What a WebRTC stack that sends with `sendto` looks like — the
        // case that made the connected-only rule miss Discord entirely.
        assert_eq!(classify_row(BOUND), Some((false, 112233)));
    }

    #[test]
    fn quic_to_a_web_server_is_not_a_call() {
        // 0x01BB is 443. A browser has these open constantly.
        assert_eq!(classify_row(QUIC), None);
    }

    #[test]
    fn the_ports_every_desktop_has_open_are_not_calls() {
        // 0x14E9 is 5353, mDNS. 0x0035 is 53.
        assert_eq!(classify_row(MDNS), None);
        assert_eq!(classify_row(SERVICE), None);
    }

    #[test]
    fn junk_rows_are_skipped_rather_than_panicking() {
        for line in ["", "   ", "sl local_address rem_address", "1: x"] {
            assert_eq!(classify_row(line), None);
        }
    }

    #[test]
    fn socket_links_yield_their_inode() {
        assert_eq!(socket_inode("socket:[778899]"), Some(778899));
        assert_eq!(socket_inode("/dev/null"), None);
        assert_eq!(socket_inode("socket:[]"), None);
    }

    #[test]
    fn command_names_match_their_binary_loosely() {
        assert!(same_process("Discord", "discord"));
        assert!(same_process("discord", "/usr/bin/Discord"));
        // comm is cut at 15 characters, so the long name only prefixes.
        assert!(same_process("SomeVeryLongBin", "SomeVeryLongBinary"));
        assert!(!same_process("chrome", "discord"));
    }

    #[test]
    fn an_app_with_no_process_of_that_name_cannot_be_answered_for() {
        // Better an honest "don't know" than "not in a call", which would
        // silently end a meeting. pid 1 stands in for the pid a sandboxed
        // client reports: readable, but not the app it claims to be.
        let probe = probe_app(
            Some("no-such-binary-anywhere"),
            Some(1),
            &UdpSockets::default(),
        );
        assert_eq!(probe.probe, CallProbe::Unreadable);
    }

    #[test]
    fn an_app_found_by_name_is_answered_for() {
        // Our own test binary is a process we can definitely read, and one
        // whose pid the caller got wrong.
        let us = process_name(std::process::id()).expect("our own comm");
        let probe = probe_app(Some(&us), Some(u32::MAX), &UdpSockets::default());
        assert_eq!(probe.probe, CallProbe::Idle);
    }

    #[test]
    fn either_kind_of_socket_is_enough_to_be_in_a_call() {
        let us = process_name(std::process::id()).expect("our own comm");
        let ours = held_sockets(std::process::id()).expect("our own descriptors");

        for inode in ours {
            let weak = UdpSockets {
                connected: HashSet::new(),
                open: HashSet::from([inode]),
            };
            let probe = probe_app(Some(&us), None, &weak);
            assert_eq!(probe.probe, CallProbe::Connected);
            assert_eq!((probe.connected, probe.open), (0, 1));
            return;
        }
    }
}
