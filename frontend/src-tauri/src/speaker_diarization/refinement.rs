//! Post-recording offline refinement.
//!
//! The live path clusters greedily: each embedding is compared against the
//! centroids that exist *at that moment* and joins the best one above
//! threshold. That decision is irreversible and order-dependent — an early
//! embedding can seed a cluster that later turns out to be two people, and
//! the same voice arriving before its cluster has a stable centroid can be
//! stranded under its own "Speaker N".
//!
//! Once recording stops we have every embedding, so we re-group the session.
//! There are two ways to get the new partition:
//!
//! - [`refine_with_offline_turns`]: the preferred path. A full offline
//!   diarization of the recorded audio (pyannote segmentation + embedding
//!   clustering, see [`super::offline`]) yields speaker turns; each history
//!   record takes the offline speaker it overlaps most. This sees the whole
//!   waveform, including overlapped speech and turn changes the live VAD
//!   never cut at.
//! - [`refine`]: the fallback when the segmentation model or the audio isn't
//!   available. **Average-linkage hierarchical agglomerative clustering**
//!   (HAC) over the live embeddings. Every pairwise distance is known up
//!   front, so the result depends only on the *set* of embeddings, not the
//!   order they arrived in — HAC can split a cluster the live pass wrongly
//!   merged and merge clusters it wrongly split. Speaker count is unknown,
//!   so the dendrogram is cut on a distance threshold
//!   (`1.0 - similarity_threshold`) rather than a fixed k.
//!
//! HAC is **duration-weighted**: cluster sizes in the average-linkage update
//! are seconds of speech, not segment counts, and records shorter than
//! [`MIN_SEED_SECS`] don't take part in the clustering at all — they're
//! attached afterwards to the nearest resulting cluster. With
//! `separate_sources` on, mic and system records are **cannot-link**: their
//! pairwise distance is infinite, so no cluster ever mixes the two sources.
//!
//! ## Profile handling
//!
//! Segments the live pass matched to a stored profile ("pinned" rows) are
//! authoritative: their label and profile id are never changed here. They do
//! participate in the clustering, though, and lend their identity outward —
//! a cluster containing pinned "John" rows labels its *unpinned* members
//! "John" too. This repairs the common live-pass artifact where half a
//! voice's segments cross the profile-match threshold and the rest sit just
//! under it, splitting one person into "John" plus a parallel "Speaker N".
//!
//! Clusters with no pinned members get a second chance against the stored
//! profiles: the cluster centroid (duration-weighted over the whole session,
//! so much less noisy than any single segment) is searched against the
//! matcher. Only clusters that still match nothing fall back to
//! "Speaker N", numbered by first appearance so the first unrecognized voice
//! heard is "Speaker 1".

use crate::audio::recording_state::DeviceType;
use crate::speaker_diarization::clusterer::MIN_SEED_SECS;
use crate::speaker_diarization::embedding_math::{
    cosine_distance, dot, l2_normalize, weighted_average_and_normalize,
};
use crate::speaker_diarization::offline::OfflineTurn;
use crate::speaker_diarization::{EmbeddingRecord, SpeakerProfileMatcher};
use serde::Serialize;
use std::collections::HashMap;

/// Weight given to a record whose duration is unknown, and the floor for
/// any record's weight (so a zero-length record can't zero out a cluster).
const DEFAULT_WEIGHT_SECS: f32 = 1.0;
const MIN_WEIGHT_SECS: f32 = 0.1;

#[derive(Debug, Serialize, Clone)]
pub struct RefinedAssignment {
    pub sequence_id: u64,
    pub speaker: String,
    /// Profile id after refinement. Always `Some` when the live pass matched
    /// a profile (we never unmatch); may also be newly `Some` for a segment
    /// whose cluster was folded into a stored profile.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub voice_profile_id: Option<String>,
    /// True when this segment's label or profile link changed vs the live
    /// pass — the frontend can use this to flash-update only the affected
    /// rows, and the DB writer uses it to skip no-op updates.
    pub changed: bool,
    /// The label the live pass assigned. The DB update matches on this so a
    /// row the user renamed underneath us is never clobbered.
    #[serde(skip)]
    pub previous_speaker: String,
}

/// Knobs shared by both refinement paths.
#[derive(Debug, Clone, Copy)]
pub struct RefineOptions {
    /// Cosine **similarity** threshold for the HAC path (see
    /// `REFINE_CLUSTER_THRESHOLD`); internally the distance cutoff is
    /// `1.0 - threshold`.
    pub threshold: f32,
    /// Never put mic and system records in the same cluster.
    pub separate_sources: bool,
}

/// Re-cluster `history` with duration-weighted HAC and return one
/// assignment per record. `matcher` provides the stored voice profiles used
/// to name clusters (see module docs); `None` disables the profile steps
/// and everything unpinned falls back to "Speaker N".
///
/// Output order matches `history` order.
pub fn refine(
    history: &[EmbeddingRecord],
    options: &RefineOptions,
    matcher: Option<&SpeakerProfileMatcher>,
) -> Vec<RefinedAssignment> {
    if history.is_empty() {
        return Vec::new();
    }
    let cluster_of = hac_partition(history, options);
    assignments_from_partition(history, &cluster_of, matcher)
}

/// Re-group `history` by the offline diarizer's speaker `turns` (see module
/// docs). Falls back to [`refine`] when no record overlaps any turn — e.g.
/// the history carries no timing, or the turns came from different audio.
pub fn refine_with_offline_turns(
    history: &[EmbeddingRecord],
    turns: &[OfflineTurn],
    options: &RefineOptions,
    matcher: Option<&SpeakerProfileMatcher>,
) -> Vec<RefinedAssignment> {
    if history.is_empty() {
        return Vec::new();
    }
    match partition_from_turns(history, turns, options.separate_sources) {
        Some(cluster_of) => assignments_from_partition(history, &cluster_of, matcher),
        None => {
            log::info!("No history record overlaps an offline speaker turn; falling back to HAC");
            refine(history, options, matcher)
        }
    }
}

fn record_weight(record: &EmbeddingRecord) -> f32 {
    record
        .duration_secs()
        .unwrap_or(DEFAULT_WEIGHT_SECS)
        .max(MIN_WEIGHT_SECS)
}

/// Records too short to shape the clustering. Unknown duration counts as
/// long enough — the caller had no timing to judge by.
fn is_short(record: &EmbeddingRecord) -> bool {
    record.duration_secs().map_or(false, |d| d < MIN_SEED_SECS)
}

fn record_source(record: &EmbeddingRecord, separate_sources: bool) -> Option<DeviceType> {
    if separate_sources {
        record.source
    } else {
        None
    }
}

fn sources_compatible(a: Option<DeviceType>, b: Option<DeviceType>) -> bool {
    match (a, b) {
        (Some(a), Some(b)) => a == b,
        _ => true,
    }
}

/// HAC over the long records, then attach the short ones to the nearest
/// compatible cluster. Returns a cluster index per record.
fn hac_partition(history: &[EmbeddingRecord], options: &RefineOptions) -> Vec<usize> {
    let mut seeds: Vec<usize> = (0..history.len()).filter(|&i| !is_short(&history[i])).collect();
    if seeds.is_empty() {
        seeds = (0..history.len()).collect();
    }
    let rest: Vec<usize> = {
        let is_seed: std::collections::HashSet<usize> = seeds.iter().copied().collect();
        (0..history.len()).filter(|i| !is_seed.contains(i)).collect()
    };

    let seed_clusters = cluster_subset(history, &seeds, options);
    let mut cluster_of = vec![usize::MAX; history.len()];
    for (k, &i) in seeds.iter().enumerate() {
        cluster_of[i] = seed_clusters[k];
    }
    let num_seed_clusters = seed_clusters.iter().max().map_or(0, |m| m + 1);

    let orphans = attach_to_nearest(history, &mut cluster_of, &rest, options.separate_sources);
    if !orphans.is_empty() {
        // No compatible cluster for these (only possible with source
        // separation) — let them form clusters among themselves.
        let orphan_clusters = cluster_subset(history, &orphans, options);
        for (k, &i) in orphans.iter().enumerate() {
            cluster_of[i] = num_seed_clusters + orphan_clusters[k];
        }
    }
    cluster_of
}

/// Run HAC over `history[indices]` and return a cluster index per entry of
/// `indices`.
fn cluster_subset(
    history: &[EmbeddingRecord],
    indices: &[usize],
    options: &RefineOptions,
) -> Vec<usize> {
    let embeddings: Vec<Vec<f32>> = indices.iter().map(|&i| history[i].embedding.clone()).collect();
    let weights: Vec<f32> = indices.iter().map(|&i| record_weight(&history[i])).collect();
    let sources: Vec<Option<DeviceType>> = indices
        .iter()
        .map(|&i| record_source(&history[i], options.separate_sources))
        .collect();
    agglomerative_cluster(&embeddings, &weights, &sources, 1.0 - options.threshold)
}

/// Assign each record in `pending` to the cluster (already present in
/// `cluster_of`) whose duration-weighted centroid is most similar and whose
/// source is compatible. Returns the records that had no compatible cluster.
fn attach_to_nearest(
    history: &[EmbeddingRecord],
    cluster_of: &mut [usize],
    pending: &[usize],
    separate_sources: bool,
) -> Vec<usize> {
    let mut members: HashMap<usize, Vec<usize>> = HashMap::new();
    for (i, &c) in cluster_of.iter().enumerate() {
        if c != usize::MAX {
            members.entry(c).or_default().push(i);
        }
    }
    let centroids: Vec<(usize, Vec<f32>, Option<DeviceType>)> = members
        .iter()
        .map(|(&c, idx)| {
            let source = idx
                .iter()
                .find_map(|&i| record_source(&history[i], separate_sources));
            (c, weighted_centroid(idx, history), source)
        })
        .collect();

    let mut orphans = Vec::new();
    for &i in pending {
        let record = &history[i];
        let source = record_source(record, separate_sources);
        let mut normed = record.embedding.clone();
        l2_normalize(&mut normed);
        let best = centroids
            .iter()
            .filter(|(_, _, s)| sources_compatible(*s, source))
            .map(|(c, centroid, _)| (*c, dot(centroid, &normed)))
            .max_by(|a, b| a.1.total_cmp(&b.1));
        match best {
            Some((c, _)) => cluster_of[i] = c,
            None => orphans.push(i),
        }
    }
    orphans
}

/// Map every record to the offline speaker it overlaps most (split by
/// source when `separate_sources`), then attach records with no overlap to
/// the nearest resulting cluster by embedding. `None` if no record overlaps
/// any turn.
fn partition_from_turns(
    history: &[EmbeddingRecord],
    turns: &[OfflineTurn],
    separate_sources: bool,
) -> Option<Vec<usize>> {
    let mut key_index: HashMap<(usize, Option<DeviceType>), usize> = HashMap::new();
    let mut cluster_of = vec![usize::MAX; history.len()];
    let mut unmatched = Vec::new();

    for (i, record) in history.iter().enumerate() {
        let speaker = match (record.start_secs, record.end_secs) {
            (Some(start), Some(end)) => dominant_speaker(turns, start, end),
            _ => None,
        };
        match speaker {
            Some(speaker) => {
                let key = (speaker, record_source(record, separate_sources));
                let next = key_index.len();
                cluster_of[i] = *key_index.entry(key).or_insert(next);
            }
            None => unmatched.push(i),
        }
    }
    if key_index.is_empty() {
        return None;
    }

    let num_clusters = key_index.len();
    let orphans = attach_to_nearest(history, &mut cluster_of, &unmatched, separate_sources);
    // Orphans only exist under source separation, when no cluster carries
    // the record's source; give each source its own cluster.
    let mut orphan_cluster: HashMap<Option<DeviceType>, usize> = HashMap::new();
    for i in orphans {
        let source = record_source(&history[i], separate_sources);
        let next = num_clusters + orphan_cluster.len();
        cluster_of[i] = *orphan_cluster.entry(source).or_insert(next);
    }
    Some(cluster_of)
}

/// The offline speaker with the most overlap with `[start, end)`, or `None`
/// when nothing overlaps. Ties resolve to the lower speaker index.
fn dominant_speaker(turns: &[OfflineTurn], start: f64, end: f64) -> Option<usize> {
    let mut overlap: HashMap<usize, f64> = HashMap::new();
    for turn in turns {
        let o = end.min(turn.end) - start.max(turn.start);
        if o > 0.0 {
            *overlap.entry(turn.speaker).or_insert(0.0) += o;
        }
    }
    overlap
        .into_iter()
        .max_by(|a, b| a.1.total_cmp(&b.1).then(b.0.cmp(&a.0)))
        .map(|(speaker, _)| speaker)
}

fn weighted_centroid(member_indices: &[usize], history: &[EmbeddingRecord]) -> Vec<f32> {
    let embeddings: Vec<Vec<f32>> = member_indices
        .iter()
        .map(|&i| history[i].embedding.clone())
        .collect();
    let weights: Vec<f32> = member_indices.iter().map(|&i| record_weight(&history[i])).collect();
    weighted_average_and_normalize(&embeddings, &weights)
}

/// Turn a partition (`cluster_of[i]` = cluster of `history[i]`) into
/// per-record assignments: name each cluster (pinned profile, centroid
/// match, or "Speaker N" by first appearance) and keep pinned rows as-is.
fn assignments_from_partition(
    history: &[EmbeddingRecord],
    cluster_of: &[usize],
    matcher: Option<&SpeakerProfileMatcher>,
) -> Vec<RefinedAssignment> {
    let mut members: HashMap<usize, Vec<usize>> = HashMap::new();
    for (i, &c) in cluster_of.iter().enumerate() {
        members.entry(c).or_default().push(i);
    }

    // Resolve each cluster's identity in history (first-appearance) order so
    // the "Speaker N" numbering is stable and human-friendly: the first
    // unrecognized voice to speak is "Speaker 1". Raw cluster indices carry
    // no meaningful order.
    let mut identity_of: HashMap<usize, (String, Option<String>)> = HashMap::new();
    let mut next_label = 1usize;
    for &cluster in cluster_of {
        if identity_of.contains_key(&cluster) {
            continue;
        }
        let identity = cluster_identity(&members[&cluster], history, matcher)
            .unwrap_or_else(|| {
                let label = format!("Speaker {}", next_label);
                next_label += 1;
                (label, None)
            });
        identity_of.insert(cluster, identity);
    }

    history
        .iter()
        .enumerate()
        .map(|(i, record)| {
            if let Some(profile_id) = &record.voice_profile_id {
                // Pinned: a live profile match is authoritative.
                return RefinedAssignment {
                    sequence_id: record.sequence_id,
                    speaker: record.label.clone(),
                    voice_profile_id: Some(profile_id.clone()),
                    changed: false,
                    previous_speaker: record.label.clone(),
                };
            }
            let (label, profile_id) = identity_of[&cluster_of[i]].clone();
            RefinedAssignment {
                sequence_id: record.sequence_id,
                speaker: label.clone(),
                changed: label != record.label || profile_id.is_some(),
                voice_profile_id: profile_id,
                previous_speaker: record.label.clone(),
            }
        })
        .collect()
}

/// A cluster's (label, profile id) when it can be tied to a stored profile:
/// the dominant pinned profile among its members, else a matcher hit on the
/// cluster centroid. `None` means "unrecognized voice" (caller assigns a
/// "Speaker N").
fn cluster_identity(
    member_indices: &[usize],
    history: &[EmbeddingRecord],
    matcher: Option<&SpeakerProfileMatcher>,
) -> Option<(String, Option<String>)> {
    // Dominant pinned profile. Iterating in history order with a
    // strictly-greater comparison makes ties resolve to the profile pinned
    // first, deterministically.
    let mut counts: HashMap<&str, usize> = HashMap::new();
    let mut best: Option<(&EmbeddingRecord, usize)> = None;
    for &i in member_indices {
        let record = &history[i];
        if let Some(profile_id) = &record.voice_profile_id {
            let count = counts.entry(profile_id.as_str()).or_insert(0);
            *count += 1;
            if best.map_or(true, |(_, c)| *count > c) {
                best = Some((record, *count));
            }
        }
    }
    if let Some((record, _)) = best {
        return Some((record.label.clone(), record.voice_profile_id.clone()));
    }

    // No pinned members — try the (low-noise) cluster centroid against the
    // stored profiles.
    let matcher = matcher?;
    let centroid = weighted_centroid(member_indices, history);
    matcher
        .search(&centroid)
        .map(|m| (m.name, Some(m.profile_id)))
}

/// Average-linkage HAC over `embeddings` using cosine distance, cutting the
/// dendrogram once the nearest pair exceeds `max_distance`.
///
/// `weights` (seconds of speech) are the initial cluster sizes in the
/// Lance-Williams update, so a long segment pulls a merged cluster's
/// distances toward itself. Points whose `sources` are both `Some` and
/// differ start infinitely far apart (cannot-link); average linkage
/// propagates the infinity, so such points never end up together.
///
/// Returns a cluster index per input (indices are arbitrary but consistent;
/// the caller relabels them).
///
/// Cost: O(n²) memory for the distance matrix. Merging keeps a cached
/// nearest-neighbour per active cluster, so the usual case is ~O(n²) total
/// rather than the O(n³) a naive full rescan per merge would cost.
fn agglomerative_cluster(
    embeddings: &[Vec<f32>],
    weights: &[f32],
    sources: &[Option<DeviceType>],
    max_distance: f32,
) -> Vec<usize> {
    debug_assert_eq!(embeddings.len(), weights.len());
    debug_assert_eq!(embeddings.len(), sources.len());
    let n = embeddings.len();
    if n == 0 {
        return Vec::new();
    }
    if n == 1 {
        return vec![0];
    }

    // Work on L2-normalized copies so cosine similarity is a plain dot product.
    let normed: Vec<Vec<f32>> = embeddings
        .iter()
        .map(|e| {
            let mut v = e.clone();
            l2_normalize(&mut v);
            v
        })
        .collect();

    // dist[i][j] — cosine distance. Only entries between *active* clusters
    // are meaningful; merged-away rows are ignored via `active`.
    let mut dist = vec![vec![0.0f32; n]; n];
    for i in 0..n {
        for j in (i + 1)..n {
            let d = if sources_compatible(sources[i], sources[j]) {
                cosine_distance(&normed[i], &normed[j])
            } else {
                f32::INFINITY
            };
            dist[i][j] = d;
            dist[j][i] = d;
        }
    }

    let mut active: Vec<bool> = vec![true; n];
    let mut size: Vec<f32> = weights.iter().map(|w| w.max(MIN_WEIGHT_SECS)).collect();
    // Which original points live in each active cluster.
    let mut members: Vec<Vec<usize>> = (0..n).map(|i| vec![i]).collect();

    // nn[i] = (nearest active j, distance) for active i. Recomputed lazily
    // when a merge invalidates it.
    let mut nn: Vec<Option<(usize, f32)>> = (0..n)
        .map(|i| nearest_neighbour(i, &dist, &active))
        .collect();

    loop {
        // Global closest pair, via the cached per-cluster nearest neighbours.
        let mut best: Option<(usize, usize, f32)> = None;
        for i in 0..n {
            if !active[i] {
                continue;
            }
            if let Some((j, d)) = nn[i] {
                if best.map_or(true, |(_, _, bd)| d < bd) {
                    best = Some((i, j, d));
                }
            }
        }

        let Some((i, j, d)) = best else { break };
        // Dendrogram cut: nothing left that's close enough to merge.
        if d >= max_distance {
            break;
        }

        // Merge j into i using the Lance-Williams update for average linkage:
        //   d(k, i∪j) = (|i|·d(k,i) + |j|·d(k,j)) / (|i| + |j|)
        // with |·| the cluster's total speech duration. An infinite
        // (cannot-link) input stays infinite.
        let (si, sj) = (size[i], size[j]);
        let total = si + sj;
        for k in 0..n {
            if !active[k] || k == i || k == j {
                continue;
            }
            let merged = (si * dist[i][k] + sj * dist[j][k]) / total;
            dist[i][k] = merged;
            dist[k][i] = merged;
        }
        size[i] = total;
        let moved = std::mem::take(&mut members[j]);
        members[i].extend(moved);
        active[j] = false;
        nn[j] = None;

        // i's distances all changed, and anyone pointing at i or j is stale.
        for k in 0..n {
            if !active[k] {
                continue;
            }
            if k == i || matches!(nn[k], Some((t, _)) if t == i || t == j) {
                nn[k] = nearest_neighbour(k, &dist, &active);
            }
        }
    }

    let mut out = vec![0usize; n];
    let mut cluster_index = 0usize;
    for i in 0..n {
        if !active[i] {
            continue;
        }
        for &m in &members[i] {
            out[m] = cluster_index;
        }
        cluster_index += 1;
    }
    out
}

/// Nearest active cluster to `i` (excluding itself), or `None` if `i` is the
/// only active cluster left.
fn nearest_neighbour(i: usize, dist: &[Vec<f32>], active: &[bool]) -> Option<(usize, f32)> {
    if !active[i] {
        return None;
    }
    let mut best: Option<(usize, f32)> = None;
    for (j, &is_active) in active.iter().enumerate() {
        if !is_active || j == i {
            continue;
        }
        let d = dist[i][j];
        if best.map_or(true, |(_, bd)| d < bd) {
            best = Some((j, d));
        }
    }
    best
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::speaker_diarization::EmbeddingRecord;

    const OPTS: RefineOptions = RefineOptions {
        threshold: 0.55,
        separate_sources: false,
    };

    fn rec(seq: u64, emb: Vec<f32>, label: &str) -> EmbeddingRecord {
        EmbeddingRecord {
            sequence_id: seq,
            embedding: emb,
            voice_profile_id: None,
            label: label.to_string(),
            start_secs: None,
            end_secs: None,
            source: None,
        }
    }

    fn pinned(seq: u64, emb: Vec<f32>, label: &str, profile_id: &str) -> EmbeddingRecord {
        EmbeddingRecord {
            voice_profile_id: Some(profile_id.to_string()),
            ..rec(seq, emb, label)
        }
    }

    fn timed(
        seq: u64,
        emb: Vec<f32>,
        start: f64,
        end: f64,
        source: Option<DeviceType>,
    ) -> EmbeddingRecord {
        EmbeddingRecord {
            start_secs: Some(start),
            end_secs: Some(end),
            source,
            ..rec(seq, emb, "Speaker 1")
        }
    }

    /// Number of distinct speaker labels in a refinement result.
    fn distinct_speakers(out: &[RefinedAssignment]) -> usize {
        out.iter()
            .map(|a| a.speaker.as_str())
            .collect::<std::collections::HashSet<_>>()
            .len()
    }

    #[test]
    fn empty_history_returns_empty() {
        let out = refine(&[], &OPTS, None);
        assert!(out.is_empty());
    }

    #[test]
    fn profile_matched_rows_are_pinned() {
        let r = pinned(1, vec![1.0, 0.0], "John", "profile-1");
        let out = refine(&[r], &OPTS, None);
        assert_eq!(out.len(), 1);
        assert_eq!(out[0].speaker, "John");
        assert_eq!(out[0].voice_profile_id.as_deref(), Some("profile-1"));
        assert!(!out[0].changed);
    }

    /// The profile-extension path: an unpinned row that clusters with pinned
    /// "John" rows inherits the profile instead of surfacing as a parallel
    /// "Speaker N" — while the pinned row itself never changes.
    #[test]
    fn cluster_mates_of_a_pinned_row_inherit_its_profile() {
        let history = vec![
            rec(1, vec![1.0, 0.0, 0.0], "Speaker 1"),
            pinned(2, vec![1.0, 0.0, 0.0], "John", "profile-1"),
            rec(3, vec![0.0, 1.0, 0.0], "Speaker 2"),
        ];
        let out = refine(&history, &OPTS, None);
        // The unpinned same-voice row folds into John, flagged as changed.
        assert_eq!(out[0].speaker, "John");
        assert_eq!(out[0].voice_profile_id.as_deref(), Some("profile-1"));
        assert!(out[0].changed);
        // The pinned row is untouched.
        assert_eq!(out[1].speaker, "John");
        assert!(!out[1].changed);
        // The remaining unrecognized voice is the first unnamed cluster.
        assert_eq!(out[2].speaker, "Speaker 1");
        assert!(out[2].voice_profile_id.is_none());
    }

    /// Two pinned rows of *different* profiles keep their own labels even if
    /// HAC puts them in one cluster — pins are always authoritative.
    #[test]
    fn pinned_rows_never_change_even_when_clustered_together() {
        let history = vec![
            pinned(1, vec![1.0, 0.0, 0.0], "John", "profile-1"),
            pinned(2, vec![0.98, 0.02, 0.0], "Mary", "profile-2"),
        ];
        let out = refine(&history, &OPTS, None);
        assert_eq!(out[0].speaker, "John");
        assert_eq!(out[1].speaker, "Mary");
        assert!(!out[0].changed);
        assert!(!out[1].changed);
    }

    /// Unpinned members of a mixed cluster follow the *dominant* pinned
    /// profile, with ties broken by first appearance.
    #[test]
    fn dominant_pinned_profile_wins_a_mixed_cluster() {
        let voice = vec![1.0, 0.0, 0.0];
        let history = vec![
            rec(1, voice.clone(), "Speaker 1"),
            pinned(2, voice.clone(), "Mary", "profile-2"),
            pinned(3, voice.clone(), "John", "profile-1"),
            pinned(4, voice.clone(), "John", "profile-1"),
        ];
        let out = refine(&history, &OPTS, None);
        assert_eq!(out[0].speaker, "John");
        assert_eq!(out[0].voice_profile_id.as_deref(), Some("profile-1"));
        // Pins keep their own labels regardless.
        assert_eq!(out[1].speaker, "Mary");
        assert_eq!(out[2].speaker, "John");
    }

    /// A cluster with no pinned members is searched against the stored
    /// profiles by centroid, so a voice whose individual segments all sat
    /// just under the live match threshold still gets named.
    #[test]
    fn unpinned_cluster_matches_profile_by_centroid() {
        let matcher = SpeakerProfileMatcher::new(
            3,
            vec![crate::speaker_diarization::StoredProfile {
                id: "profile-1".to_string(),
                name: "John".to_string(),
                centroid: vec![1.0, 0.0, 0.0],
                exemplars: Vec::new(),
            }],
            0.60,
        );
        let history = vec![
            rec(1, vec![0.99, 0.01, 0.0], "Speaker 1"),
            rec(2, vec![0.98, 0.02, 0.0], "Speaker 1"),
            rec(3, vec![0.0, 1.0, 0.0], "Speaker 2"),
        ];
        let out = refine(&history, &OPTS, Some(&matcher));
        assert_eq!(out[0].speaker, "John");
        assert_eq!(out[0].voice_profile_id.as_deref(), Some("profile-1"));
        assert!(out[0].changed);
        assert_eq!(out[1].speaker, "John");
        // The orthogonal voice matches nothing and stays unnamed.
        assert_eq!(out[2].speaker, "Speaker 1");
        assert!(out[2].voice_profile_id.is_none());
    }

    #[test]
    fn two_clear_groups_produce_two_clusters() {
        let history = vec![
            rec(1, vec![1.0, 0.0, 0.0], "Speaker 1"),
            rec(2, vec![0.98, 0.02, 0.0], "Speaker 1"),
            rec(3, vec![0.0, 1.0, 0.0], "Speaker 2"),
            rec(4, vec![0.02, 0.98, 0.0], "Speaker 2"),
        ];
        let out = refine(&history, &OPTS, None);
        assert_eq!(distinct_speakers(&out), 2);
        assert_eq!(out[0].speaker, out[1].speaker);
        assert_eq!(out[2].speaker, out[3].speaker);
        assert_ne!(out[0].speaker, out[2].speaker);
    }

    #[test]
    fn a_third_distinct_voice_produces_three_clusters() {
        let history = vec![
            rec(1, vec![1.0, 0.0, 0.0], "Speaker 1"),
            rec(2, vec![0.98, 0.02, 0.0], "Speaker 1"),
            rec(3, vec![0.0, 1.0, 0.0], "Speaker 2"),
            rec(4, vec![0.02, 0.98, 0.0], "Speaker 2"),
            rec(5, vec![0.0, 0.0, 1.0], "Speaker 3"),
            rec(6, vec![0.0, 0.02, 0.98], "Speaker 3"),
        ];
        let out = refine(&history, &OPTS, None);
        assert_eq!(distinct_speakers(&out), 3);
        assert_eq!(out[4].speaker, out[5].speaker);
        assert_ne!(out[4].speaker, out[0].speaker);
        assert_ne!(out[4].speaker, out[2].speaker);
    }

    /// The whole point of offline refinement: the grouping must depend on the
    /// *set* of embeddings, not the order they arrived in.
    #[test]
    fn clustering_is_order_independent() {
        let a1 = vec![1.0, 0.0, 0.0];
        let a2 = vec![0.97, 0.03, 0.0];
        let b1 = vec![0.0, 1.0, 0.0];
        let b2 = vec![0.03, 0.97, 0.0];

        // Same four embeddings, two different arrival orders.
        let forward = vec![
            rec(1, a1.clone(), "Speaker 1"),
            rec(2, a2.clone(), "Speaker 1"),
            rec(3, b1.clone(), "Speaker 2"),
            rec(4, b2.clone(), "Speaker 2"),
        ];
        let interleaved = vec![
            rec(1, b1.clone(), "Speaker 1"),
            rec(2, a1.clone(), "Speaker 2"),
            rec(3, b2.clone(), "Speaker 1"),
            rec(4, a2.clone(), "Speaker 2"),
        ];

        let out_f = refine(&forward, &OPTS, None);
        let out_i = refine(&interleaved, &OPTS, None);

        assert_eq!(distinct_speakers(&out_f), 2);
        assert_eq!(distinct_speakers(&out_i), 2);

        // In both runs the A's group together and the B's group together —
        // the *partition* is identical even though arrival order differs.
        assert_eq!(out_f[0].speaker, out_f[1].speaker); // a1 == a2
        assert_eq!(out_f[2].speaker, out_f[3].speaker); // b1 == b2
        assert_eq!(out_i[1].speaker, out_i[3].speaker); // a1 == a2
        assert_eq!(out_i[0].speaker, out_i[2].speaker); // b1 == b2
        assert_ne!(out_i[0].speaker, out_i[1].speaker); // A != B
    }

    /// A voice the live pass stranded under its own label gets folded back
    /// into the cluster it belongs to — something re-running the greedy pass
    /// in the same order could never fix.
    #[test]
    fn live_mis_split_is_repaired_and_marked_changed() {
        // All three embeddings are the same voice, but the live pass tagged
        // the middle one "Speaker 2" (warmup noise / unstable centroid).
        let history = vec![
            rec(1, vec![1.0, 0.0, 0.0], "Speaker 1"),
            rec(2, vec![0.99, 0.01, 0.0], "Speaker 2"),
            rec(3, vec![0.98, 0.02, 0.0], "Speaker 1"),
        ];
        let out = refine(&history, &OPTS, None);
        assert_eq!(distinct_speakers(&out), 1);
        assert_eq!(out[0].speaker, "Speaker 1");
        assert_eq!(out[1].speaker, "Speaker 1");
        // The repaired row is flagged, the already-correct ones are not.
        assert!(out[1].changed);
        assert!(!out[0].changed);
        assert_eq!(out[1].previous_speaker, "Speaker 2");
    }

    /// Labels are assigned by first appearance, so the first voice heard is
    /// always "Speaker 1" regardless of internal HAC cluster indices.
    #[test]
    fn labels_follow_first_appearance_order() {
        let history = vec![
            rec(1, vec![0.0, 1.0, 0.0], "Speaker 6"),
            rec(2, vec![1.0, 0.0, 0.0], "Speaker 10"),
            rec(3, vec![0.0, 0.98, 0.02], "Speaker 6"),
        ];
        let out = refine(&history, &OPTS, None);
        assert_eq!(out[0].speaker, "Speaker 1");
        assert_eq!(out[1].speaker, "Speaker 2");
        assert_eq!(out[2].speaker, "Speaker 1");
    }

    #[test]
    fn single_unpinned_record_is_speaker_one() {
        let out = refine(&[rec(1, vec![1.0, 0.0, 0.0], "Speaker 4")], &OPTS, None);
        assert_eq!(out.len(), 1);
        assert_eq!(out[0].speaker, "Speaker 1");
        assert!(out[0].changed);
    }

    #[test]
    fn every_assignment_preserves_its_sequence_id() {
        let history = vec![
            rec(41, vec![1.0, 0.0, 0.0], "Speaker 1"),
            rec(42, vec![0.0, 1.0, 0.0], "Speaker 2"),
        ];
        let out = refine(&history, &OPTS, None);
        assert_eq!(out[0].sequence_id, 41);
        assert_eq!(out[1].sequence_id, 42);
    }

    /// A short, ambiguous segment can't bridge two voices into one cluster:
    /// it sits out the HAC and is attached to its nearest cluster afterwards.
    #[test]
    fn short_segment_cannot_bridge_two_voices() {
        let history = vec![
            timed(1, vec![1.0, 0.0, 0.0], 0.0, 4.0, None),
            timed(2, vec![0.6, 0.6, 0.0], 4.0, 4.6, None),
            timed(3, vec![0.0, 1.0, 0.0], 5.0, 9.0, None),
        ];
        let opts = RefineOptions { threshold: 0.45, ..OPTS };
        let out = refine(&history, &opts, None);
        assert_ne!(out[0].speaker, out[2].speaker);
        // Still gets a label from one of the two real clusters.
        assert!(out[1].speaker == out[0].speaker || out[1].speaker == out[2].speaker);
    }

    #[test]
    fn all_short_history_still_clusters() {
        let history = vec![
            timed(1, vec![1.0, 0.0, 0.0], 0.0, 0.5, None),
            timed(2, vec![0.99, 0.01, 0.0], 1.0, 1.5, None),
            timed(3, vec![0.0, 1.0, 0.0], 2.0, 2.5, None),
        ];
        let out = refine(&history, &OPTS, None);
        assert_eq!(distinct_speakers(&out), 2);
        assert_eq!(out[0].speaker, out[1].speaker);
    }

    /// With source separation on, identical embeddings from mic and system
    /// still land in different clusters.
    #[test]
    fn separate_sources_are_cannot_link() {
        let v = vec![1.0, 0.0, 0.0];
        let history = vec![
            timed(1, v.clone(), 0.0, 3.0, Some(DeviceType::Microphone)),
            timed(2, v.clone(), 3.0, 6.0, Some(DeviceType::System)),
            timed(3, v.clone(), 6.0, 9.0, Some(DeviceType::Microphone)),
        ];
        let separated = refine(
            &history,
            &RefineOptions { separate_sources: true, ..OPTS },
            None,
        );
        assert_eq!(distinct_speakers(&separated), 2);
        assert_eq!(separated[0].speaker, separated[2].speaker);
        assert_ne!(separated[0].speaker, separated[1].speaker);

        let mixed = refine(&history, &OPTS, None);
        assert_eq!(distinct_speakers(&mixed), 1);
    }

    fn turn(start: f64, end: f64, speaker: usize) -> OfflineTurn {
        OfflineTurn { start, end, speaker }
    }

    /// Offline turns decide the grouping even when the live embeddings would
    /// have said otherwise.
    #[test]
    fn offline_turns_drive_the_partition() {
        let v = vec![1.0, 0.0, 0.0];
        let history = vec![
            timed(1, v.clone(), 0.0, 3.0, None),
            timed(2, v.clone(), 3.0, 6.0, None),
            timed(3, v.clone(), 6.0, 9.0, None),
        ];
        let turns = vec![turn(0.0, 3.2, 0), turn(3.2, 6.0, 1), turn(6.0, 9.0, 0)];
        let out = refine_with_offline_turns(&history, &turns, &OPTS, None);
        assert_eq!(out[0].speaker, "Speaker 1");
        assert_eq!(out[1].speaker, "Speaker 2");
        assert_eq!(out[2].speaker, "Speaker 1");
    }

    #[test]
    fn record_outside_every_turn_attaches_by_embedding() {
        let history = vec![
            timed(1, vec![1.0, 0.0, 0.0], 0.0, 3.0, None),
            timed(2, vec![0.0, 1.0, 0.0], 3.0, 6.0, None),
            timed(3, vec![0.02, 0.98, 0.0], 20.0, 22.0, None),
        ];
        let turns = vec![turn(0.0, 3.0, 0), turn(3.0, 6.0, 1)];
        let out = refine_with_offline_turns(&history, &turns, &OPTS, None);
        assert_eq!(out[2].speaker, out[1].speaker);
        assert_ne!(out[0].speaker, out[1].speaker);
    }

    #[test]
    fn offline_speaker_is_split_by_source_when_separating() {
        let v = vec![1.0, 0.0, 0.0];
        let history = vec![
            timed(1, v.clone(), 0.0, 3.0, Some(DeviceType::Microphone)),
            timed(2, v.clone(), 3.0, 6.0, Some(DeviceType::System)),
        ];
        let turns = vec![turn(0.0, 6.0, 0)];
        let opts = RefineOptions { separate_sources: true, ..OPTS };
        let out = refine_with_offline_turns(&history, &turns, &opts, None);
        assert_ne!(out[0].speaker, out[1].speaker);
    }

    #[test]
    fn untimed_history_falls_back_to_hac() {
        let history = vec![
            rec(1, vec![1.0, 0.0, 0.0], "Speaker 1"),
            rec(2, vec![0.0, 1.0, 0.0], "Speaker 1"),
        ];
        let out = refine_with_offline_turns(&history, &[turn(0.0, 10.0, 0)], &OPTS, None);
        assert_eq!(distinct_speakers(&out), 2);
    }
}
