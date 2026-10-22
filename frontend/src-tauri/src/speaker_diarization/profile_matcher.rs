//! Match a live embedding against stored voice profiles.
//!
//! Each profile is a centroid plus a few **exemplars** — embeddings captured
//! at separate naming / merge / enrollment events. A centroid averages a voice
//! across every mic and room it was heard in, which blurs it; an exemplar
//! keeps one of those conditions sharp. An embedding's score against a profile
//! is the better of its centroid similarity and its best exemplar similarity
//! minus [`EXEMPLAR_MARGIN`] (a single exemplar is a noisier estimate than the
//! centroid, so it has to clear the bar by a bit more).
//!
//! Loaded once per session (live recording or batch job) with the profiles
//! built by the active embedding model. Entries whose dimension doesn't match
//! the model are skipped (logged) so a stale row can't crash the matcher.

use crate::speaker_diarization::embedding_math::{dot, l2_normalize};

/// Penalty applied to exemplar similarity before comparing to the threshold.
pub const EXEMPLAR_MARGIN: f32 = 0.05;

/// Most exemplars kept per profile; see [`prune_exemplars`].
pub const MAX_EXEMPLARS: usize = 8;

#[derive(Debug, Clone)]
pub struct ProfileMatch {
    pub profile_id: String,
    pub name: String,
    pub score: f32,
}

/// One stored profile as loaded from the DB.
#[derive(Debug, Clone)]
pub struct StoredProfile {
    pub id: String,
    pub name: String,
    pub centroid: Vec<f32>,
    pub exemplars: Vec<Vec<f32>>,
}

pub struct SpeakerProfileMatcher {
    profiles: Vec<StoredProfile>,
    threshold: f32,
}

impl SpeakerProfileMatcher {
    /// Build a matcher for embeddings of dimension `dim`. Centroids and
    /// exemplars are L2-normalized here; mismatched-dim entries are dropped.
    pub fn new<I>(dim: usize, profiles: I, threshold: f32) -> Self
    where
        I: IntoIterator<Item = StoredProfile>,
    {
        let mut loaded = Vec::new();
        let mut skipped = 0usize;
        let mut exemplar_count = 0usize;
        for mut p in profiles {
            if p.centroid.len() != dim {
                log::warn!(
                    "Skipping voice profile '{}' (id={}): embedding dim {} != active model dim {}",
                    p.name,
                    p.id,
                    p.centroid.len(),
                    dim
                );
                skipped += 1;
                continue;
            }
            l2_normalize(&mut p.centroid);
            p.exemplars.retain(|e| e.len() == dim);
            for e in p.exemplars.iter_mut() {
                l2_normalize(e);
            }
            exemplar_count += p.exemplars.len();
            loaded.push(p);
        }
        log::info!(
            "Profile matcher loaded {} profiles with {} exemplars ({} skipped, dim={}, threshold={})",
            loaded.len(),
            exemplar_count,
            skipped,
            dim,
            threshold
        );
        Self {
            profiles: loaded,
            threshold,
        }
    }

    pub fn num_profiles(&self) -> usize {
        self.profiles.len()
    }

    /// Find the best matching stored profile for `embedding`, or `None` if no
    /// profile scores at or above threshold.
    pub fn search(&self, embedding: &[f32]) -> Option<ProfileMatch> {
        let mut query = embedding.to_vec();
        l2_normalize(&mut query);
        self.profiles
            .iter()
            .filter(|p| p.centroid.len() == query.len())
            .map(|p| (p, profile_score(p, &query)))
            .filter(|(_, score)| *score >= self.threshold)
            .max_by(|a, b| a.1.total_cmp(&b.1))
            .map(|(p, score)| ProfileMatch {
                profile_id: p.id.clone(),
                name: p.name.clone(),
                score,
            })
    }
}

fn profile_score(profile: &StoredProfile, query: &[f32]) -> f32 {
    let centroid = dot(&profile.centroid, query);
    let exemplar = profile
        .exemplars
        .iter()
        .map(|e| dot(e, query) - EXEMPLAR_MARGIN)
        .fold(f32::NEG_INFINITY, f32::max);
    centroid.max(exemplar)
}

/// Trim `exemplars` (`(embedding, sample_count)`) to at most `cap` by
/// repeatedly dropping the most redundant one — the exemplar most similar to
/// some other exemplar. That keeps the set spread across the conditions it
/// was captured in instead of keeping the newest or oldest.
pub fn prune_exemplars(exemplars: &mut Vec<(Vec<f32>, i64)>, cap: usize) {
    while exemplars.len() > cap.max(1) {
        let normed: Vec<Vec<f32>> = exemplars
            .iter()
            .map(|(e, _)| {
                let mut v = e.clone();
                l2_normalize(&mut v);
                v
            })
            .collect();
        let redundancy = |i: usize| {
            (0..normed.len())
                .filter(|&j| j != i && normed[j].len() == normed[i].len())
                .map(|j| dot(&normed[i], &normed[j]))
                .fold(f32::NEG_INFINITY, f32::max)
        };
        let victim = (0..normed.len())
            .max_by(|&a, &b| redundancy(a).total_cmp(&redundancy(b)))
            .expect("exemplars is non-empty");
        exemplars.remove(victim);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn profile(id: &str, centroid: Vec<f32>, exemplars: Vec<Vec<f32>>) -> StoredProfile {
        StoredProfile {
            id: id.to_string(),
            name: id.to_uppercase(),
            centroid,
            exemplars,
        }
    }

    #[test]
    fn centroid_match_above_threshold() {
        let m = SpeakerProfileMatcher::new(2, vec![profile("a", vec![1.0, 0.0], vec![])], 0.6);
        let hit = m.search(&[0.9, 0.1]).unwrap();
        assert_eq!(hit.profile_id, "a");
        assert_eq!(hit.name, "A");
        assert!(m.search(&[0.0, 1.0]).is_none());
    }

    /// An embedding far from the centroid but close to one exemplar (the
    /// voice under other conditions) still matches.
    #[test]
    fn exemplar_rescues_a_condition_the_centroid_blurs() {
        let p = profile("a", vec![1.0, 0.0, 0.0], vec![vec![0.0, 1.0, 0.0]]);
        let m = SpeakerProfileMatcher::new(3, vec![p], 0.6);
        let hit = m.search(&[0.1, 0.99, 0.0]).unwrap();
        assert_eq!(hit.profile_id, "a");
        assert!(hit.score < 0.99);
    }

    #[test]
    fn exemplar_margin_applies() {
        // Exemplar similarity exactly at threshold fails once the margin is
        // subtracted.
        let p = profile("a", vec![0.0, 1.0], vec![vec![1.0, 0.0]]);
        let m = SpeakerProfileMatcher::new(2, vec![p], 0.6);
        let q = [0.6f32, 0.8];
        assert!(m.search(&q).is_some()); // centroid sim 0.8
        let p = profile("b", vec![0.0, 1.0], vec![vec![1.0, 0.0]]);
        let m = SpeakerProfileMatcher::new(2, vec![p], 0.85);
        assert!(m.search(&[0.88, 0.47]).is_none()); // exemplar 0.88-0.05 < 0.85
    }

    #[test]
    fn best_profile_wins() {
        let m = SpeakerProfileMatcher::new(
            2,
            vec![
                profile("a", vec![1.0, 0.0], vec![]),
                profile("b", vec![0.8, 0.6], vec![]),
            ],
            0.5,
        );
        assert_eq!(m.search(&[0.7, 0.7]).unwrap().profile_id, "b");
    }

    #[test]
    fn wrong_dim_profiles_are_skipped() {
        let m = SpeakerProfileMatcher::new(
            2,
            vec![profile("a", vec![1.0, 0.0, 0.0], vec![]), profile("b", vec![1.0, 0.0], vec![])],
            0.5,
        );
        assert_eq!(m.num_profiles(), 1);
    }

    #[test]
    fn pruning_drops_the_most_redundant_exemplar() {
        let mut ex = vec![
            (vec![1.0, 0.0], 1),
            (vec![0.0, 1.0], 1),
            (vec![0.99, 0.01], 1),
        ];
        prune_exemplars(&mut ex, 2);
        assert_eq!(ex.len(), 2);
        // One of the two near-duplicates went; the distinct one stayed.
        assert!(ex.iter().any(|(e, _)| e[1] == 1.0));
    }

    #[test]
    fn pruning_under_cap_is_a_no_op() {
        let mut ex = vec![(vec![1.0, 0.0], 1)];
        prune_exemplars(&mut ex, MAX_EXEMPLARS);
        assert_eq!(ex.len(), 1);
    }
}
