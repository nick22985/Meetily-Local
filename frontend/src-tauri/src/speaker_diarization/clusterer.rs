//! Online speaker clustering.
//!
//! Greedy single-pass clusterer suitable for real-time use. Each incoming
//! embedding is compared by cosine similarity to every existing centroid;
//! if the best match is above `threshold` it joins that cluster and updates
//! the running centroid, otherwise it seeds a new cluster.
//!
//! Two refinements over plain greedy clustering:
//!
//! - **Duration weighting.** Centroids are averaged by segment duration, not
//!   segment count, and a segment shorter than [`MIN_SEED_SECS`] may not seed
//!   a new cluster: it attaches to the nearest existing one instead. Short
//!   embeddings are the noisiest in the system, and a "yeah" or "mm-hm" that
//!   happens to land under threshold used to spawn a spurious "Speaker N".
//! - **Source hint.** With `separate_sources` on, a cluster belongs to the
//!   capture source (mic or system) that seeded it and only accepts segments
//!   from that source. That's only true when mic and system audio can't hear
//!   the same voices (headphones), so it's opt-in.
//!
//! Embeddings are L2-normalized on insertion so cosine similarity reduces
//! to a dot product — important for hot-path performance.

use super::embedding_math::{dot, l2_normalize};
use crate::audio::recording_state::DeviceType;

/// Segments shorter than this (seconds) can't seed a new cluster while any
/// compatible cluster exists. CAM++-class embeddings degrade sharply below
/// ~1.5s of speech.
pub const MIN_SEED_SECS: f32 = 1.5;

/// Floor for a segment's weight so a zero/unknown duration still counts.
const MIN_WEIGHT_SECS: f32 = 0.1;

#[derive(Debug, Clone)]
struct Centroid {
    /// Stable id used to derive the user-facing label.
    id: usize,
    /// Duration-weighted average of L2-normalized embeddings (re-normalized
    /// each update).
    embedding: Vec<f32>,
    /// Total seconds of speech merged into this centroid; weights the
    /// running average.
    weight: f32,
    /// Capture source of the segment that seeded the cluster.
    source: Option<DeviceType>,
}

#[derive(Debug, Clone)]
pub struct OnlineSpeakerClusterer {
    centroids: Vec<Centroid>,
    threshold: f32,
    separate_sources: bool,
    next_id: usize,
}

impl OnlineSpeakerClusterer {
    pub fn new(threshold: f32, separate_sources: bool) -> Self {
        Self {
            centroids: Vec::new(),
            threshold,
            separate_sources,
            next_id: 0,
        }
    }

    pub fn num_speakers(&self) -> usize {
        self.centroids.len()
    }

    /// Assign `embedding` (a segment `duration_secs` long, captured from
    /// `source`) to the best matching cluster, or create a new one. Returns
    /// the assigned cluster id.
    pub fn assign(
        &mut self,
        mut embedding: Vec<f32>,
        duration_secs: f32,
        source: Option<DeviceType>,
    ) -> usize {
        l2_normalize(&mut embedding);
        let weight = duration_secs.max(MIN_WEIGHT_SECS);

        let mut best: Option<(usize, f32)> = None;
        for (idx, c) in self.centroids.iter().enumerate() {
            if !self.compatible(c, source) {
                continue;
            }
            let sim = dot(&c.embedding, &embedding);
            if best.map_or(true, |(_, s)| sim > s) {
                best = Some((idx, sim));
            }
        }

        if let Some((idx, sim)) = best {
            let centroid = &mut self.centroids[idx];
            if sim >= self.threshold {
                merge_into_centroid(&mut centroid.embedding, &embedding, centroid.weight, weight);
                centroid.weight += weight;
                return centroid.id;
            }
            // Too short to trust as a new voice: label it with the nearest
            // cluster, but don't let the noisy embedding move that centroid.
            if duration_secs < MIN_SEED_SECS {
                return centroid.id;
            }
        }

        let id = self.next_id;
        self.next_id += 1;
        self.centroids.push(Centroid {
            id,
            embedding,
            weight,
            source,
        });
        id
    }

    fn compatible(&self, centroid: &Centroid, source: Option<DeviceType>) -> bool {
        !self.separate_sources
            || centroid.source.is_none()
            || source.is_none()
            || centroid.source == source
    }

    /// User-facing label like "Speaker 1" (1-indexed).
    pub fn label_for(&self, cluster_id: usize) -> String {
        format!("Speaker {}", cluster_id + 1)
    }
}

/// Update a running centroid with a new (already-normalized) embedding.
/// Uses a duration-weighted mean, then re-normalizes.
fn merge_into_centroid(centroid: &mut [f32], new_emb: &[f32], prev_weight: f32, weight: f32) {
    debug_assert_eq!(centroid.len(), new_emb.len());
    let denom = prev_weight + weight;
    for (c, e) in centroid.iter_mut().zip(new_emb) {
        *c = (*c * prev_weight + *e * weight) / denom;
    }
    l2_normalize(centroid);
}

#[cfg(test)]
mod tests {
    use super::*;

    fn vec3(x: f32, y: f32, z: f32) -> Vec<f32> {
        vec![x, y, z]
    }

    #[test]
    fn similar_embeddings_merge() {
        let mut c = OnlineSpeakerClusterer::new(0.9, false);
        let id1 = c.assign(vec3(1.0, 0.0, 0.0), 3.0, None);
        let id2 = c.assign(vec3(0.99, 0.01, 0.0), 3.0, None);
        assert_eq!(id1, id2);
        assert_eq!(c.num_speakers(), 1);
    }

    #[test]
    fn dissimilar_embeddings_separate() {
        let mut c = OnlineSpeakerClusterer::new(0.5, false);
        let id1 = c.assign(vec3(1.0, 0.0, 0.0), 3.0, None);
        let id2 = c.assign(vec3(0.0, 1.0, 0.0), 3.0, None);
        assert_ne!(id1, id2);
        assert_eq!(c.num_speakers(), 2);
    }

    #[test]
    fn cluster_ids_are_stable_and_one_indexed_in_label() {
        let mut c = OnlineSpeakerClusterer::new(0.5, false);
        let id1 = c.assign(vec3(1.0, 0.0, 0.0), 3.0, None);
        let id2 = c.assign(vec3(0.0, 1.0, 0.0), 3.0, None);
        assert_eq!(c.label_for(id1), "Speaker 1");
        assert_eq!(c.label_for(id2), "Speaker 2");
    }

    #[test]
    fn empty_embedding_does_not_panic() {
        let mut c = OnlineSpeakerClusterer::new(0.5, false);
        // L2-normalizing a zero vector is a no-op; we accept it as a degenerate
        // first cluster rather than rejecting (caller is responsible for valid input).
        let id = c.assign(vec![0.0; 4], 3.0, None);
        assert_eq!(id, 0);
    }

    #[test]
    fn short_segment_attaches_instead_of_seeding() {
        let mut c = OnlineSpeakerClusterer::new(0.5, false);
        let id1 = c.assign(vec3(1.0, 0.0, 0.0), 3.0, None);
        // Orthogonal, but only 0.6s long: it may not open a new speaker.
        let id2 = c.assign(vec3(0.0, 1.0, 0.0), 0.6, None);
        assert_eq!(id1, id2);
        assert_eq!(c.num_speakers(), 1);
        // The same voice with enough audio does seed its own cluster...
        let id3 = c.assign(vec3(0.0, 1.0, 0.0), 2.0, None);
        assert_ne!(id1, id3);
        // ...and the attached short segment didn't drag the first centroid.
        let id4 = c.assign(vec3(1.0, 0.0, 0.0), 3.0, None);
        assert_eq!(id4, id1);
    }

    #[test]
    fn first_short_segment_still_seeds() {
        let mut c = OnlineSpeakerClusterer::new(0.5, false);
        let id = c.assign(vec3(1.0, 0.0, 0.0), 0.4, None);
        assert_eq!(id, 0);
        assert_eq!(c.num_speakers(), 1);
    }

    #[test]
    fn longer_segments_pull_the_centroid_harder() {
        let mut c = OnlineSpeakerClusterer::new(0.3, false);
        c.assign(vec3(1.0, 0.0, 0.0), 1.5, None);
        // Joins (sim ~0.71) with a much longer duration.
        c.assign(vec3(0.7071, 0.7071, 0.0), 10.0, None);
        let centroid = &c.centroids[0].embedding;
        assert!(centroid[1] > 0.6, "centroid {:?} should lean toward the long segment", centroid);
    }

    #[test]
    fn separate_sources_keeps_mic_and_system_apart() {
        let mut c = OnlineSpeakerClusterer::new(0.5, true);
        let mic = c.assign(vec3(1.0, 0.0, 0.0), 3.0, Some(DeviceType::Microphone));
        let sys = c.assign(vec3(1.0, 0.0, 0.0), 3.0, Some(DeviceType::System));
        assert_ne!(mic, sys);
        // A short system segment attaches to the system cluster, not the mic one.
        let short = c.assign(vec3(0.0, 1.0, 0.0), 0.5, Some(DeviceType::System));
        assert_eq!(short, sys);
    }

    #[test]
    fn sources_mix_when_separation_is_off() {
        let mut c = OnlineSpeakerClusterer::new(0.5, false);
        let mic = c.assign(vec3(1.0, 0.0, 0.0), 3.0, Some(DeviceType::Microphone));
        let sys = c.assign(vec3(1.0, 0.0, 0.0), 3.0, Some(DeviceType::System));
        assert_eq!(mic, sys);
    }
}
