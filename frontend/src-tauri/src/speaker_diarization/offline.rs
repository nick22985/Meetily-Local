//! Whole-recording offline diarization.
//!
//! The live pipeline only ever sees one VAD segment at a time, so it can't
//! find a speaker change inside a segment without a silence gap (beyond the
//! coarse 1s-window check in [`super::Diarizer::speaker_turns`]), and it never
//! sees overlapped speech as such. Once the audio is on disk we can do much
//! better: sherpa-onnx's `OfflineSpeakerDiarization` runs pyannote
//! segmentation-3.0 over the whole waveform (frame-level speech activity and
//! speaker-change detection, overlap-aware), embeds each local speaker with
//! the active embedding model, and clusters the embeddings globally.
//!
//! The result is a list of [`OfflineTurn`]s in seconds from the start of the
//! audio. Refinement maps the live segments onto them (see
//! [`super::refinement::refine_with_offline_turns`]); batch transcription also
//! splits its segments at turn changes ([`speaker_change_points`]).
//!
//! Everything here degrades to "no turns": a missing segmentation model, an
//! undecodable file or a failed run is logged and the caller falls back to
//! HAC over the live embeddings.

use anyhow::{anyhow, Result};
use sherpa_onnx::{
    FastClusteringConfig, OfflineSpeakerDiarization, OfflineSpeakerDiarizationConfig,
    OfflineSpeakerSegmentationModelConfig, OfflineSpeakerSegmentationPyannoteModelConfig,
    SpeakerEmbeddingExtractorConfig,
};
use std::path::Path;

use super::model;

/// sherpa's fast-clustering cutoff. It's a cosine *distance* on the
/// embeddings: larger merges more aggressively (fewer speakers). 0.5 is
/// sherpa's recommended default for the 3D-Speaker/WeSpeaker models; tune
/// with the `diarization_eval` binary.
pub const OFFLINE_CLUSTER_THRESHOLD: f32 = 0.5;

/// Turn runs shorter than this are absorbed by a neighbour when splitting
/// batch segments, so a stray 200ms blip of another speaker doesn't chop a
/// sentence in half.
pub const MIN_TURN_RUN_SECS: f64 = 1.0;

/// One speaker turn from the offline pass. `speaker` is an arbitrary index,
/// consistent within one run only.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct OfflineTurn {
    pub start: f64,
    pub end: f64,
    pub speaker: usize,
}

fn default_threads() -> i32 {
    let cores = std::thread::available_parallelism()
        .map(|n| n.get())
        .unwrap_or(2);
    (cores / 2).clamp(1, 4) as i32
}

/// True when both the segmentation model and the active embedding model are
/// on disk.
pub fn models_ready() -> bool {
    let seg = model::segmentation_model_path().map_or(false, |p| model::model_is_ready(&p));
    let emb = model::default_model_path().map_or(false, |p| model::model_is_ready(&p));
    seg && emb
}

/// Diarize 16 kHz mono `samples` with explicit model paths. Used directly by
/// the eval binary; app code goes through [`diarize_samples`].
pub fn diarize_samples_with(
    samples_16k: &[f32],
    segmentation_model: &Path,
    embedding_model: &Path,
    threshold: f32,
) -> Result<Vec<OfflineTurn>> {
    if samples_16k.is_empty() {
        return Ok(Vec::new());
    }
    for path in [segmentation_model, embedding_model] {
        if !model::model_is_ready(path) {
            return Err(anyhow!("Model not found: {}", path.display()));
        }
    }

    let threads = default_threads();
    let config = OfflineSpeakerDiarizationConfig {
        segmentation: OfflineSpeakerSegmentationModelConfig {
            pyannote: OfflineSpeakerSegmentationPyannoteModelConfig {
                model: Some(segmentation_model.to_string_lossy().into_owned()),
                ..Default::default()
            },
            num_threads: threads,
            ..Default::default()
        },
        embedding: SpeakerEmbeddingExtractorConfig {
            model: Some(embedding_model.to_string_lossy().into_owned()),
            num_threads: threads,
            ..Default::default()
        },
        clustering: FastClusteringConfig {
            num_clusters: -1,
            threshold,
        },
        ..Default::default()
    };

    let diarizer = OfflineSpeakerDiarization::create(&config)
        .ok_or_else(|| anyhow!("Failed to create OfflineSpeakerDiarization"))?;
    if diarizer.sample_rate() != 16_000 {
        return Err(anyhow!(
            "Segmentation model expects {} Hz audio, not 16000",
            diarizer.sample_rate()
        ));
    }

    let started = std::time::Instant::now();
    let result = diarizer
        .process(samples_16k)
        .ok_or_else(|| anyhow!("Offline diarization returned no result"))?;
    let turns: Vec<OfflineTurn> = result
        .sort_by_start_time()
        .into_iter()
        .filter(|s| s.speaker >= 0 && s.end > s.start)
        .map(|s| OfflineTurn {
            start: s.start as f64,
            end: s.end as f64,
            speaker: s.speaker as usize,
        })
        .collect();
    log::info!(
        "Offline diarization: {:.1}s of audio -> {} turns, {} speakers in {:.1}s",
        samples_16k.len() as f64 / 16_000.0,
        turns.len(),
        result.num_speakers(),
        started.elapsed().as_secs_f64()
    );
    Ok(turns)
}

/// Diarize 16 kHz mono `samples` with the active embedding model.
pub fn diarize_samples(samples_16k: &[f32]) -> Result<Vec<OfflineTurn>> {
    let segmentation = model::segmentation_model_path()
        .ok_or_else(|| anyhow!("Speaker models directory not configured"))?;
    let embedding = model::default_model_path()
        .ok_or_else(|| anyhow!("Speaker models directory not configured"))?;
    diarize_samples_with(samples_16k, &segmentation, &embedding, OFFLINE_CLUSTER_THRESHOLD)
}

/// [`diarize_samples`] when the models are present, logging and returning
/// `None` on any failure so callers can fall back to HAC.
pub fn diarize_if_available(samples_16k: &[f32]) -> Option<Vec<OfflineTurn>> {
    if !models_ready() {
        log::info!("Offline diarization skipped: segmentation or embedding model not downloaded");
        return None;
    }
    match diarize_samples(samples_16k) {
        Ok(turns) if !turns.is_empty() => Some(turns),
        Ok(_) => {
            log::info!("Offline diarization found no speech turns");
            None
        }
        Err(e) => {
            log::warn!("Offline diarization failed: {}", e);
            None
        }
    }
}

/// Decode an audio file, resample to 16 kHz mono and diarize it. Blocking.
pub fn diarize_file_if_available(path: &Path) -> Option<Vec<OfflineTurn>> {
    if !models_ready() {
        log::info!("Offline diarization skipped: segmentation or embedding model not downloaded");
        return None;
    }
    let decoded = match crate::audio::decoder::decode_audio_file(path) {
        Ok(d) => d,
        Err(e) => {
            log::warn!("Offline diarization: failed to decode {}: {}", path.display(), e);
            return None;
        }
    };
    diarize_if_available(&decoded.to_whisper_format())
}

/// Times (seconds, strictly inside `(start, end)`) where the dominant offline
/// speaker changes over `[start, end)`. Runs shorter than `min_run_secs` are
/// absorbed by a neighbour first, and gaps with no turn inherit the speaker
/// around them, so the result only marks sustained changes.
pub fn speaker_change_points(
    turns: &[OfflineTurn],
    start: f64,
    end: f64,
    min_run_secs: f64,
) -> Vec<f64> {
    const FRAME: f64 = 0.01;
    if end - start <= FRAME {
        return Vec::new();
    }
    let n = ((end - start) / FRAME).ceil() as usize;

    // Dominant speaker per frame: the covering turn that's longest (a long
    // turn is the more reliable one where pyannote marks overlap).
    let relevant: Vec<&OfflineTurn> = turns
        .iter()
        .filter(|t| t.end > start && t.start < end)
        .collect();
    let mut frames: Vec<Option<usize>> = (0..n)
        .map(|i| {
            let t = start + (i as f64 + 0.5) * FRAME;
            relevant
                .iter()
                .filter(|turn| turn.start <= t && t < turn.end)
                .max_by(|a, b| (a.end - a.start).total_cmp(&(b.end - b.start)))
                .map(|turn| turn.speaker)
        })
        .collect();

    // Gaps take the previous speaker, leading gaps the first one seen.
    let first = frames.iter().flatten().copied().next();
    let Some(first) = first else {
        return Vec::new();
    };
    let mut last = first;
    for f in frames.iter_mut() {
        match f {
            Some(s) => last = *s,
            None => *f = Some(last),
        }
    }

    // Run-length encode as (speaker, frame count).
    let mut runs: Vec<(usize, usize)> = Vec::new();
    for s in frames.into_iter().flatten() {
        match runs.last_mut() {
            Some((sp, len)) if *sp == s => *len += 1,
            _ => runs.push((s, 1)),
        }
    }

    // Absorb short runs, shortest first, into their longer neighbour.
    let min_frames = (min_run_secs / FRAME).round() as usize;
    loop {
        if runs.len() < 2 {
            break;
        }
        let (idx, &(_, len)) = runs
            .iter()
            .enumerate()
            .min_by_key(|(_, (_, len))| *len)
            .expect("runs is non-empty");
        if len >= min_frames {
            break;
        }
        let target = match (idx.checked_sub(1), runs.get(idx + 1)) {
            (Some(p), Some(next)) => {
                if runs[p].1 >= next.1 {
                    p
                } else {
                    idx + 1
                }
            }
            (Some(p), None) => p,
            (None, _) => idx + 1,
        };
        runs[target].1 += len;
        runs.remove(idx);
        // Neighbours of the removed run may now share a speaker.
        let mut merged: Vec<(usize, usize)> = Vec::with_capacity(runs.len());
        for r in runs.drain(..) {
            match merged.last_mut() {
                Some((sp, len)) if *sp == r.0 => *len += r.1,
                _ => merged.push(r),
            }
        }
        runs = merged;
    }

    let mut points = Vec::with_capacity(runs.len().saturating_sub(1));
    let mut frame = 0usize;
    for (_, len) in runs.iter().take(runs.len().saturating_sub(1)) {
        frame += len;
        points.push(start + frame as f64 * FRAME);
    }
    points
}

#[cfg(test)]
mod tests {
    use super::*;

    fn turn(start: f64, end: f64, speaker: usize) -> OfflineTurn {
        OfflineTurn { start, end, speaker }
    }

    #[test]
    fn single_speaker_has_no_change_points() {
        let turns = vec![turn(0.0, 10.0, 0)];
        assert!(speaker_change_points(&turns, 1.0, 9.0, 1.0).is_empty());
    }

    #[test]
    fn sustained_change_is_found() {
        let turns = vec![turn(0.0, 4.0, 0), turn(4.0, 10.0, 1)];
        let points = speaker_change_points(&turns, 0.0, 10.0, 1.0);
        assert_eq!(points.len(), 1);
        assert!((points[0] - 4.0).abs() < 0.02, "{:?}", points);
    }

    #[test]
    fn short_blip_is_absorbed() {
        let turns = vec![turn(0.0, 4.0, 0), turn(4.0, 4.3, 1), turn(4.3, 10.0, 0)];
        assert!(speaker_change_points(&turns, 0.0, 10.0, 1.0).is_empty());
    }

    #[test]
    fn gaps_inherit_the_surrounding_speaker() {
        let turns = vec![turn(0.0, 3.0, 0), turn(5.0, 10.0, 1)];
        let points = speaker_change_points(&turns, 0.0, 10.0, 1.0);
        assert_eq!(points.len(), 1);
        assert!((points[0] - 5.0).abs() < 0.02, "{:?}", points);
    }

    #[test]
    fn no_turns_means_no_points() {
        assert!(speaker_change_points(&[], 0.0, 10.0, 1.0).is_empty());
    }

    #[test]
    fn points_are_inside_the_range() {
        let turns = vec![turn(0.0, 2.0, 0), turn(2.0, 6.0, 1), turn(6.0, 12.0, 2)];
        let points = speaker_change_points(&turns, 1.0, 9.0, 0.5);
        assert_eq!(points.len(), 2);
        assert!(points.iter().all(|&p| p > 1.0 && p < 9.0));
    }
}
