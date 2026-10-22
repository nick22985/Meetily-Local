import { Transcript } from "@/types";

/**
 * Renders stored transcript segments as the plain text an LLM is asked to
 * summarize.
 *
 * Two things matter here and neither is cosmetic:
 *
 * 1. **Speaker labels.** Segments carry a diarized `speaker` ("Me",
 *    "Speaker 2", or a stored profile name). Feeding the model bare text
 *    strips out who said what, so it cannot attribute a decision or an
 *    action item to anyone — the summary comes back as a shapeless
 *    "the team discussed…" blur. The action-item extractor already renders
 *    `[MM:SS] Speaker: text` (see `transcript_action_items.rs`); this
 *    matches that format so both paths see the same shape of transcript.
 *
 * 2. **Turn merging.** VAD cuts speech at every ~250-400 ms of silence, so
 *    one person's paragraph arrives as a dozen rows. Emitting a timestamp
 *    line per row buries the actual words under prefix noise (often more
 *    prefix characters than speech) and hands the model a transcript that
 *    reads as if everyone speaks in fragments. Consecutive rows from the
 *    same speaker are joined into one turn.
 */

/** Characters after which a long monologue is split into a fresh turn, so a
 * 10-minute stretch by one speaker still carries interior timestamps for
 * quoting and grounding. */
const MAX_TURN_CHARS = 1200;

/** Label used when a segment has no diarized speaker and no usable source. */
const UNKNOWN_SPEAKER = "Speaker";

function formatTime(
  seconds: number | undefined,
  fallbackTimestamp: string,
): string {
  if (seconds === undefined) {
    // Transcripts recorded before audio-relative timing existed only have
    // wall-clock time; better than no anchor at all.
    return `[${fallbackTimestamp}]`;
  }
  const totalSecs = Math.max(0, Math.floor(seconds));
  const mins = Math.floor(totalSecs / 60);
  const secs = totalSecs % 60;
  return `[${mins.toString().padStart(2, "0")}:${secs.toString().padStart(2, "0")}]`;
}

/** Speaker label for a segment, falling back to the audio stream's identity
 * when diarization produced nothing (no speaker model downloaded). */
function speakerLabel(t: Transcript): string {
  const diarized = t.speaker?.trim();
  if (diarized) return diarized;
  if (t.source === "mic") return "Me";
  return UNKNOWN_SPEAKER;
}

interface Turn {
  speaker: string;
  time: string;
  text: string;
}

/**
 * Format transcript segments as `[MM:SS] Speaker: text` turns, one per line,
 * merging consecutive segments from the same speaker.
 */
export function renderTranscriptForLlm(transcripts: Transcript[]): string {
  const turns: Turn[] = [];

  for (const t of transcripts) {
    const text = t.text?.trim();
    if (!text) continue;

    const speaker = speakerLabel(t);
    const last = turns[turns.length - 1];

    if (
      last &&
      last.speaker === speaker &&
      last.text.length + text.length + 1 <= MAX_TURN_CHARS
    ) {
      last.text = `${last.text} ${text}`;
      continue;
    }

    turns.push({
      speaker,
      time: formatTime(t.audio_start_time, t.timestamp),
      text,
    });
  }

  return turns.map((t) => `${t.time} ${t.speaker}: ${t.text}`).join("\n");
}
