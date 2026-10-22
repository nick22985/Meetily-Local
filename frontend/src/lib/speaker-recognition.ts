// Frontend wrappers for the speaker-recognition settings and model commands
// (src-tauri/src/speaker_diarization/settings.rs and commands.rs).

import { invoke } from "@tauri-apps/api/core";

export interface SpeakerDiarizationSettings {
  /** Filename of the active speaker embedding model. */
  embedding_model: string;
  /** Never give mic and system voices the same speaker label. */
  separate_sources: boolean;
}

export interface SpeakerEmbeddingModel {
  filename: string;
  display_name: string;
  size_mb: number;
  description: string;
  downloaded: boolean;
  active: boolean;
}

export interface SpeakerModelStatus {
  model_filename: string;
  model_path: string | null;
  /** The active embedding model is on disk. */
  is_ready: boolean;
  download_url: string;
  /** The segmentation model used after each recording is on disk. */
  segmentation_ready: boolean;
}

/** `{ progress: 0..100 }` across every file being fetched. */
export const SPEAKER_MODEL_DOWNLOAD_PROGRESS = "speaker-model-download-progress";

export async function getSpeakerDiarizationSettings(): Promise<SpeakerDiarizationSettings> {
  return invoke<SpeakerDiarizationSettings>(
    "api_get_speaker_diarization_settings",
  );
}

export async function setSpeakerDiarizationSettings(
  settings: SpeakerDiarizationSettings,
): Promise<void> {
  return invoke("api_set_speaker_diarization_settings", { settings });
}

export async function listSpeakerEmbeddingModels(): Promise<
  SpeakerEmbeddingModel[]
> {
  return invoke<SpeakerEmbeddingModel[]>("api_list_speaker_embedding_models");
}

export async function getSpeakerModelStatus(): Promise<SpeakerModelStatus> {
  return invoke<SpeakerModelStatus>("speaker_model_status");
}

/** Downloads whichever of the active embedding model and the segmentation
 *  model are missing. Resolves once both are on disk. */
export async function downloadSpeakerModels(): Promise<void> {
  return invoke("speaker_model_download");
}
