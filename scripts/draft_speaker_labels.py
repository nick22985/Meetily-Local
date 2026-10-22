#!/usr/bin/env python3
"""Draft speaker reference labels for diarization_eval from saved meetings.

Exports each meeting's transcript speakers as an Audacity label track, so a
ground-truth reference is a correction pass rather than labelling from
scratch: open the meeting's audio.mp4 in Audacity, File > Import > Labels the
draft, fix the speaker names and boundaries, then File > Export > Labels over
the same file. Also writes a manifest.tsv for `diarization_eval --manifest`.

The drafts are the app's own output — score against them only after
correcting them, or the eval just measures agreement with the old pipeline.

Usage:
  scripts/draft_speaker_labels.py --list
  scripts/draft_speaker_labels.py OUT_DIR MEETING...   (id, or a substring of the title or folder name)
"""

import argparse
import os
import sqlite3
import sys

DB = os.path.expanduser("~/.local/share/com.meetily.ai/meeting_minutes.sqlite")
# Same-speaker segments closer than this are merged into one label.
MERGE_GAP_SECS = 1.0


def connect():
    return sqlite3.connect(f"file:{DB}?mode=ro", uri=True)


def list_meetings(db):
    print(f"{'id':46} {'span':>4} {'spk':>4}  title")
    for mid, title, n_spk, secs in db.execute(
        """select m.id, m.title, count(distinct t.speaker),
                  max(t.audio_end_time) - min(t.audio_start_time)
           from meetings m join transcripts t on t.meeting_id = m.id
           group by m.id having count(*) >= 50 order by m.created_at desc"""
    ):
        print(f"{mid:46} {(secs or 0) / 60:4.0f} {n_spk:4}  {title}")


def find_meetings(db, keys):
    found = []
    for key in keys:
        rows = db.execute(
            "select id, title, folder_path from meetings where id = ? or title like ? or folder_path like ?",
            (key, f"%{key}%", f"%{key}%"),
        ).fetchall()
        if not rows:
            sys.exit(f"no meeting matches {key!r}")
        found.extend(rows)
    return found


def labels_for(db, meeting_id):
    labels = []
    for start, end, speaker in db.execute(
        """select audio_start_time, audio_end_time, coalesce(speaker, '?')
           from transcripts
           where meeting_id = ? and audio_start_time is not null
           order by audio_start_time""",
        (meeting_id,),
    ):
        if labels and labels[-1][2] == speaker and start - labels[-1][1] <= MERGE_GAP_SECS:
            labels[-1][1] = max(labels[-1][1], end)
        else:
            labels.append([start, end, speaker])
    return labels


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--list", action="store_true", help="list meetings with transcripts")
    ap.add_argument("out_dir", nargs="?")
    ap.add_argument("meetings", nargs="*")
    args = ap.parse_args()

    db = connect()
    if args.list or not args.out_dir:
        list_meetings(db)
        return
    if not args.meetings:
        ap.error("give at least one meeting id or title substring")

    os.makedirs(args.out_dir, exist_ok=True)
    manifest = []
    for mid, title, folder in find_meetings(db, args.meetings):
        labels = labels_for(db, mid)
        path = os.path.join(args.out_dir, f"{mid}.txt")
        with open(path, "w") as f:
            for start, end, speaker in labels:
                f.write(f"{start:.3f}\t{end:.3f}\t{speaker}\n")
        audio = os.path.join(folder or "", "audio.mp4")
        if not os.path.exists(audio):
            print(f"warning: {audio} not found", file=sys.stderr)
        manifest.append(f"{audio}\t{os.path.abspath(path)}")
        print(f"{path}: {len(labels)} labels, {len({l[2] for l in labels})} speakers  ({title})")

    with open(os.path.join(args.out_dir, "manifest.tsv"), "w") as f:
        f.write("\n".join(manifest) + "\n")
    print(f"wrote {os.path.join(args.out_dir, 'manifest.tsv')}")


if __name__ == "__main__":
    main()
