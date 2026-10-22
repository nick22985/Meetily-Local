-- Selectable speaker-embedding models + multi-exemplar voice profiles
--
-- Embeddings from different models live in unrelated vector spaces, so a
-- profile has to remember which model produced it; the matcher only loads
-- profiles for the active model. Every profile so far was built with the
-- original default (3D-Speaker CAM++).
--
-- A single centroid blurs a voice heard through different mics and rooms.
-- `voice_profile_exemplars` keeps up to a handful of extra embeddings per
-- profile (one per naming/merge/enrollment event); a live embedding matches a
-- profile if it is close to the centroid *or* to any exemplar.

ALTER TABLE voice_profiles ADD COLUMN embedding_model TEXT NOT NULL
    DEFAULT '3dspeaker_speech_campplus_sv_en_voxceleb_16k.onnx';

CREATE TABLE IF NOT EXISTS voice_profile_exemplars (
    id TEXT PRIMARY KEY,
    profile_id TEXT NOT NULL REFERENCES voice_profiles(id) ON DELETE CASCADE,
    -- Packed little-endian f32, L2-normalized.
    embedding BLOB NOT NULL,
    embedding_dim INTEGER NOT NULL,
    embedding_model TEXT NOT NULL,
    -- Segments that went into this exemplar.
    sample_count INTEGER NOT NULL DEFAULT 1,
    created_at TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_voice_profile_exemplars_profile
    ON voice_profile_exemplars(profile_id);
