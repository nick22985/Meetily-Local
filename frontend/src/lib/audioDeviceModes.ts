/**
 * Device selections that are questions, not answers.
 *
 * A picker entry is normally a stable PipeWire `node.name`, but two
 * sentinels stand for "ask the audio graph": the system default, and
 * whichever device the call in progress is on. Both are re-checked once
 * a second while recording, so switching audio profiles or joining a
 * call mid-meeting moves the capture with it.
 *
 * These strings are the wire format shared with the Rust side — see
 * `audio/device_mode.rs`. Keep them in sync.
 */

/** The system default source/sink. */
export const DEVICE_SYSTEM_DEFAULT = "default";

/** Whatever device the active call uses, falling back to the default. */
export const DEVICE_COMMUNICATION = "communication";

/**
 * Prefix marking a system-audio selection as an application rather than
 * a device: `app:discord` records only what Discord plays, leaving the
 * video in the other window out of the recording.
 *
 * Behind the `app_audio_capture` preference on the Rust side, which is
 * why the picker asks for the list rather than assuming one — see
 * `useAppAudioSources`.
 */
export const APP_PREFIX = "app:";

/**
 * Reserved app key: whichever application is in a call.
 *
 * Not an app but a question, re-asked once a second while recording, so
 * joining a Discord call or a Meet tab part-way through moves the
 * capture onto it. See `audio/app_audio.rs`.
 */
export const CALL_APP_KEY = "call";

/** One application the system-audio side can be pointed at. */
export interface AppAudioSource {
  /** Stable key; the selection is `app:<key>`. */
  key: string;
  label: string;
  /** Extra context for the picker, where the label isn't obvious. */
  note: string | null;
  /** Whether the app is playing something right now. */
  playing: boolean;
}

/** Everything the app picker needs, from `list_app_audio_sources`. */
export interface AppAudioCatalog {
  /** Mirrors the `app_audio_capture` preference. */
  enabled: boolean;
  sources: AppAudioSource[];
}

/** What the capture is actually reading from, per side. */
export interface ActiveRoute {
  microphone: string | null;
  system: string | null;
}

export const MIC_MODE_OPTIONS = [
  { value: DEVICE_SYSTEM_DEFAULT, label: "Default microphone" },
  { value: DEVICE_COMMUNICATION, label: "Follow the active call" },
];

export const SYSTEM_MODE_OPTIONS = [
  { value: DEVICE_SYSTEM_DEFAULT, label: "Default system audio" },
  { value: DEVICE_COMMUNICATION, label: "Follow the active call" },
];

/** The app key in a selection, or `null` if it names a device. */
export function appKey(id: string | null | undefined): string | null {
  return id?.startsWith(APP_PREFIX) ? id.slice(APP_PREFIX.length) : null;
}

/** The selection value for an app. */
export function appSelection(key: string): string {
  return `${APP_PREFIX}${key}`;
}

/** Picker entries for a list of apps. */
export function appOptions(
  sources: AppAudioSource[],
): { value: string; label: string }[] {
  return sources.map((source) => ({
    value: appSelection(source.key),
    label: appOptionLabel(source),
  }));
}

function appOptionLabel(source: AppAudioSource): string {
  if (!source.playing) return source.label;
  // "playing" means something different for the reserved key: there is a
  // call in progress, not a stream making noise.
  return source.key === CALL_APP_KEY
    ? `${source.label} (in a call now)`
    : `${source.label} (playing)`;
}

/**
 * Whether this selection can move on its own while recording.
 *
 * An app selection does not: it moves with the app, but the capture
 * follows that itself, so there is no resolved node to display.
 */
export function followsGraph(id: string | null | undefined): boolean {
  return id == null || id === DEVICE_SYSTEM_DEFAULT || id === DEVICE_COMMUNICATION;
}

/**
 * The picker value for a stored selection.
 *
 * `null` is how the context has always stored "system default", so it
 * maps onto the sentinel rather than getting its own entry.
 */
export function selectionValue(id: string | null | undefined): string {
  return id ?? DEVICE_SYSTEM_DEFAULT;
}

/** The stored form of a picker value — the default collapses back to `null`. */
export function storedSelection(value: string): string | null {
  return value === DEVICE_SYSTEM_DEFAULT ? null : value;
}
