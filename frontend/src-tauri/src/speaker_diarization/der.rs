//! Diarization error rate (DER) scoring, for the `diarization_eval` binary.
//!
//! DER is the fraction of reference speech time that is attributed wrongly:
//!
//! ```text
//! DER = (missed speech + false alarm + speaker confusion) / reference speech
//! ```
//!
//! Scoring runs on 10 ms frames. Each frame counts `N_ref` reference and
//! `N_hyp` hypothesis speakers (overlap allowed): missed is
//! `max(0, N_ref − N_hyp)`, false alarm `max(0, N_hyp − N_ref)`, and
//! confusion `min(N_ref, N_hyp) − N_correct`, where `N_correct` counts
//! reference speakers whose mapped hypothesis speaker is also active. The
//! reference→hypothesis speaker mapping is the one-to-one assignment that
//! maximizes total overlap (Hungarian algorithm), as in NIST md-eval.
//!
//! Frames within `collar` seconds of a reference segment boundary are not
//! scored, since annotators can't place boundaries more precisely than that.
//! 0.25 s is the customary collar; 0 is the strict setting.
//!
//! References load from RTTM (`SPEAKER <file> <chan> <start> <dur> ... <name>`)
//! or Audacity label tracks (`<start>\t<end>\t<name>`), so a meeting can be
//! annotated in Audacity and exported directly.

use anyhow::{anyhow, Context, Result};
use std::collections::HashMap;
use std::path::Path;

const FRAME_SECS: f64 = 0.01;

/// One speaker-labelled span, reference or hypothesis.
#[derive(Debug, Clone, PartialEq)]
pub struct LabeledSegment {
    pub start: f64,
    pub end: f64,
    pub speaker: String,
}

impl LabeledSegment {
    pub fn new(start: f64, end: f64, speaker: impl Into<String>) -> Self {
        Self {
            start,
            end,
            speaker: speaker.into(),
        }
    }
}

/// Parse RTTM `SPEAKER` lines. Other record types and comments are skipped.
pub fn parse_rttm(text: &str) -> Result<Vec<LabeledSegment>> {
    let mut segments = Vec::new();
    for (n, line) in text.lines().enumerate() {
        let line = line.trim();
        if line.is_empty() || line.starts_with('#') || line.starts_with(';') {
            continue;
        }
        let fields: Vec<&str> = line.split_whitespace().collect();
        if fields[0] != "SPEAKER" {
            continue;
        }
        if fields.len() < 8 {
            return Err(anyhow!("RTTM line {}: expected at least 8 fields", n + 1));
        }
        let start: f64 = fields[3]
            .parse()
            .with_context(|| format!("RTTM line {}: bad start '{}'", n + 1, fields[3]))?;
        let dur: f64 = fields[4]
            .parse()
            .with_context(|| format!("RTTM line {}: bad duration '{}'", n + 1, fields[4]))?;
        segments.push(LabeledSegment::new(start, start + dur, fields[7]));
    }
    Ok(segments)
}

/// Parse an Audacity label track export: `start<TAB>end<TAB>label` per line.
/// Lines starting with `\` (Audacity's spectral-selection rows) are skipped.
pub fn parse_audacity_labels(text: &str) -> Result<Vec<LabeledSegment>> {
    let mut segments = Vec::new();
    for (n, line) in text.lines().enumerate() {
        let line = line.trim_end();
        if line.trim().is_empty() || line.starts_with('\\') {
            continue;
        }
        let mut fields = line.splitn(3, '\t');
        let (Some(start), Some(end), Some(label)) = (fields.next(), fields.next(), fields.next())
        else {
            return Err(anyhow!("Label line {}: expected start<TAB>end<TAB>label", n + 1));
        };
        let start: f64 = start
            .trim()
            .parse()
            .with_context(|| format!("Label line {}: bad start '{}'", n + 1, start))?;
        let end: f64 = end
            .trim()
            .parse()
            .with_context(|| format!("Label line {}: bad end '{}'", n + 1, end))?;
        segments.push(LabeledSegment::new(start, end, label.trim()));
    }
    Ok(segments)
}

/// Load a reference: `.rttm` files as RTTM, anything else as Audacity labels.
pub fn load_reference(path: &Path) -> Result<Vec<LabeledSegment>> {
    let text = std::fs::read_to_string(path)
        .with_context(|| format!("Failed to read {}", path.display()))?;
    let is_rttm = path
        .extension()
        .map_or(false, |e| e.eq_ignore_ascii_case("rttm"));
    if is_rttm {
        parse_rttm(&text)
    } else {
        parse_audacity_labels(&text)
    }
}

/// Render segments as RTTM, e.g. to inspect a hypothesis in other tools.
pub fn to_rttm(file_id: &str, segments: &[LabeledSegment]) -> String {
    segments
        .iter()
        .map(|s| {
            format!(
                "SPEAKER {} 1 {:.3} {:.3} <NA> <NA> {} <NA> <NA>\n",
                file_id,
                s.start,
                s.end - s.start,
                s.speaker.replace(char::is_whitespace, "_")
            )
        })
        .collect()
}

#[derive(Debug, Clone, Default)]
pub struct DerReport {
    /// Scored reference speech, in speaker-seconds (overlap counts twice).
    pub reference_secs: f64,
    pub missed_secs: f64,
    pub false_alarm_secs: f64,
    pub confusion_secs: f64,
    pub reference_speakers: usize,
    pub hypothesis_speakers: usize,
    /// Reference speaker → mapped hypothesis speaker (unmapped ones absent).
    pub mapping: Vec<(String, String)>,
}

impl DerReport {
    pub fn der(&self) -> f64 {
        if self.reference_secs <= 0.0 {
            return 0.0;
        }
        (self.missed_secs + self.false_alarm_secs + self.confusion_secs) / self.reference_secs
    }

    /// Confusion as a fraction of reference speech — the part a better
    /// embedding model or clustering actually moves; missed speech and
    /// false alarms are mostly VAD.
    pub fn confusion_rate(&self) -> f64 {
        if self.reference_secs <= 0.0 {
            0.0
        } else {
            self.confusion_secs / self.reference_secs
        }
    }
}

/// Per-frame activity: for each frame, the indices of active speakers.
fn rasterize(segments: &[LabeledSegment], speakers: &[String], n_frames: usize) -> Vec<Vec<usize>> {
    let index: HashMap<&str, usize> = speakers
        .iter()
        .enumerate()
        .map(|(i, s)| (s.as_str(), i))
        .collect();
    let mut frames = vec![Vec::new(); n_frames];
    for s in segments {
        let spk = index[s.speaker.as_str()];
        let a = (s.start / FRAME_SECS).round().max(0.0) as usize;
        let b = ((s.end / FRAME_SECS).round() as usize).min(n_frames);
        for f in frames.iter_mut().take(b).skip(a) {
            if !f.contains(&spk) {
                f.push(spk);
            }
        }
    }
    frames
}

fn speaker_names(segments: &[LabeledSegment]) -> Vec<String> {
    let mut names: Vec<String> = Vec::new();
    for s in segments {
        if !names.contains(&s.speaker) {
            names.push(s.speaker.clone());
        }
    }
    names
}

/// Score `hypothesis` against `reference`; see the module docs.
pub fn compute_der(
    reference: &[LabeledSegment],
    hypothesis: &[LabeledSegment],
    collar: f64,
) -> DerReport {
    let end = reference
        .iter()
        .chain(hypothesis)
        .map(|s| s.end)
        .fold(0.0f64, f64::max);
    let n_frames = (end / FRAME_SECS).ceil() as usize + 1;

    let ref_names = speaker_names(reference);
    let hyp_names = speaker_names(hypothesis);
    let ref_frames = rasterize(reference, &ref_names, n_frames);
    let hyp_frames = rasterize(hypothesis, &hyp_names, n_frames);

    let mut scored = vec![true; n_frames];
    if collar > 0.0 {
        let c = (collar / FRAME_SECS).round() as isize;
        for s in reference {
            for boundary in [s.start, s.end] {
                let b = (boundary / FRAME_SECS).round() as isize;
                for f in (b - c).max(0)..(b + c).min(n_frames as isize) {
                    scored[f as usize] = false;
                }
            }
        }
    }

    // Overlap between each reference and hypothesis speaker, in frames.
    let mut overlap = vec![vec![0u64; hyp_names.len()]; ref_names.len()];
    for f in 0..n_frames {
        if !scored[f] {
            continue;
        }
        for &r in &ref_frames[f] {
            for &h in &hyp_frames[f] {
                overlap[r][h] += 1;
            }
        }
    }
    let assignment = max_weight_assignment(&overlap);

    let mut report = DerReport {
        reference_speakers: ref_names.len(),
        hypothesis_speakers: hyp_names.len(),
        ..Default::default()
    };
    for f in 0..n_frames {
        if !scored[f] {
            continue;
        }
        let (r, h) = (&ref_frames[f], &hyp_frames[f]);
        let correct = r
            .iter()
            .filter(|&&ri| assignment[ri].map_or(false, |hi| h.contains(&hi)))
            .count();
        let (nr, nh) = (r.len(), h.len());
        report.reference_secs += nr as f64 * FRAME_SECS;
        report.missed_secs += nr.saturating_sub(nh) as f64 * FRAME_SECS;
        report.false_alarm_secs += nh.saturating_sub(nr) as f64 * FRAME_SECS;
        report.confusion_secs += (nr.min(nh) - correct) as f64 * FRAME_SECS;
    }
    report.mapping = assignment
        .iter()
        .enumerate()
        .filter_map(|(r, h)| h.map(|h| (ref_names[r].clone(), hyp_names[h].clone())))
        .collect();
    report
}

/// One-to-one assignment of rows to columns maximizing the summed weight
/// (Hungarian algorithm on the padded square cost matrix). Returns each
/// row's column, or `None` when it's matched only to padding or to a column
/// it never overlaps.
fn max_weight_assignment(weights: &[Vec<u64>]) -> Vec<Option<usize>> {
    let rows = weights.len();
    let cols = weights.first().map_or(0, |r| r.len());
    let n = rows.max(cols);
    if n == 0 {
        return vec![None; rows];
    }
    let max = weights.iter().flatten().copied().max().unwrap_or(0) as i64;
    let cost = |i: usize, j: usize| -> i64 {
        if i < rows && j < cols {
            max - weights[i][j] as i64
        } else {
            max
        }
    };

    // Classic O(n³) potentials formulation, 1-indexed with a sentinel column 0.
    let inf = i64::MAX / 4;
    let mut u = vec![0i64; n + 1];
    let mut v = vec![0i64; n + 1];
    let mut p = vec![0usize; n + 1]; // p[j] = row matched to column j
    let mut way = vec![0usize; n + 1];
    for i in 1..=n {
        p[0] = i;
        let mut j0 = 0usize;
        let mut minv = vec![inf; n + 1];
        let mut used = vec![false; n + 1];
        loop {
            used[j0] = true;
            let i0 = p[j0];
            let mut delta = inf;
            let mut j1 = 0usize;
            for j in 1..=n {
                if used[j] {
                    continue;
                }
                let cur = cost(i0 - 1, j - 1) - u[i0] - v[j];
                if cur < minv[j] {
                    minv[j] = cur;
                    way[j] = j0;
                }
                if minv[j] < delta {
                    delta = minv[j];
                    j1 = j;
                }
            }
            for j in 0..=n {
                if used[j] {
                    u[p[j]] += delta;
                    v[j] -= delta;
                } else {
                    minv[j] -= delta;
                }
            }
            j0 = j1;
            if p[j0] == 0 {
                break;
            }
        }
        loop {
            let j1 = way[j0];
            p[j0] = p[j1];
            j0 = j1;
            if j0 == 0 {
                break;
            }
        }
    }

    let mut result = vec![None; rows];
    for j in 1..=n {
        let (i, c) = (p[j] - 1, j - 1);
        if i < rows && c < cols && weights[i][c] > 0 {
            result[i] = Some(c);
        }
    }
    result
}

#[cfg(test)]
mod tests {
    use super::*;

    fn seg(start: f64, end: f64, speaker: &str) -> LabeledSegment {
        LabeledSegment::new(start, end, speaker)
    }

    #[test]
    fn perfect_hypothesis_with_other_names_scores_zero() {
        let reference = vec![seg(0.0, 5.0, "alice"), seg(5.0, 10.0, "bob")];
        let hypothesis = vec![seg(0.0, 5.0, "Speaker 2"), seg(5.0, 10.0, "Speaker 1")];
        let r = compute_der(&reference, &hypothesis, 0.0);
        assert!(r.der() < 1e-9, "{:?}", r);
        assert!((r.reference_secs - 10.0).abs() < 0.02);
    }

    #[test]
    fn everything_one_speaker_confuses_the_minority() {
        let reference = vec![seg(0.0, 6.0, "a"), seg(6.0, 10.0, "b")];
        let hypothesis = vec![seg(0.0, 10.0, "x")];
        let r = compute_der(&reference, &hypothesis, 0.0);
        assert!((r.confusion_secs - 4.0).abs() < 0.02, "{:?}", r);
        assert!((r.der() - 0.4).abs() < 0.01);
    }

    #[test]
    fn missed_and_false_alarm_are_separated() {
        let reference = vec![seg(0.0, 4.0, "a")];
        let hypothesis = vec![seg(2.0, 6.0, "x")];
        let r = compute_der(&reference, &hypothesis, 0.0);
        assert!((r.missed_secs - 2.0).abs() < 0.02, "{:?}", r);
        assert!((r.false_alarm_secs - 2.0).abs() < 0.02, "{:?}", r);
        assert!(r.confusion_secs.abs() < 1e-9);
    }

    #[test]
    fn collar_forgives_boundary_slop() {
        let reference = vec![seg(0.0, 5.0, "a"), seg(5.0, 10.0, "b")];
        let hypothesis = vec![seg(0.0, 5.2, "x"), seg(5.2, 10.0, "y")];
        assert!(compute_der(&reference, &hypothesis, 0.0).der() > 0.0);
        assert!(compute_der(&reference, &hypothesis, 0.25).der() < 1e-9);
    }

    #[test]
    fn overlap_counts_each_speaker() {
        let reference = vec![seg(0.0, 4.0, "a"), seg(2.0, 4.0, "b")];
        let hypothesis = vec![seg(0.0, 4.0, "x")];
        let r = compute_der(&reference, &hypothesis, 0.0);
        assert!((r.reference_secs - 6.0).abs() < 0.02, "{:?}", r);
        assert!((r.missed_secs - 2.0).abs() < 0.02, "{:?}", r);
    }

    #[test]
    fn assignment_is_one_to_one() {
        // Both reference speakers overlap "x" most, but only one can have it.
        let reference = vec![seg(0.0, 6.0, "a"), seg(6.0, 10.0, "b")];
        let hypothesis = vec![seg(0.0, 9.0, "x"), seg(9.0, 10.0, "y")];
        let r = compute_der(&reference, &hypothesis, 0.0);
        assert_eq!(r.mapping.len(), 2);
        assert!(r.mapping.contains(&("a".to_string(), "x".to_string())));
        assert!(r.mapping.contains(&("b".to_string(), "y".to_string())));
    }

    #[test]
    fn parses_rttm_and_audacity_labels() {
        let rttm = "SPEAKER m1 1 0.50 2.00 <NA> <NA> alice <NA> <NA>\n\
                    # comment\nSPKR-INFO m1 1 <NA> <NA> <NA> unknown alice <NA>\n";
        assert_eq!(parse_rttm(rttm).unwrap(), vec![seg(0.5, 2.5, "alice")]);

        let labels = "0.500000\t2.500000\talice\n\\\t0\t100\n3.0\t4.0\tbob smith\n";
        assert_eq!(
            parse_audacity_labels(labels).unwrap(),
            vec![seg(0.5, 2.5, "alice"), seg(3.0, 4.0, "bob smith")]
        );
    }

    #[test]
    fn rttm_round_trips() {
        let segments = vec![seg(0.5, 2.5, "alice"), seg(3.0, 4.0, "bob")];
        assert_eq!(parse_rttm(&to_rttm("m", &segments)).unwrap(), segments);
    }
}
