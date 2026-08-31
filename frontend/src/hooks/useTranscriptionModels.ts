import { useState, useCallback, useRef } from "react";
import { invoke } from "@tauri-apps/api/core";

interface RawModelInfo {
  name: string;
  size_mb: number;
  status:
    | "Available"
    | "Missing"
    | { Downloading: { progress: number } }
    | { Error: string };
}

export interface ModelOption {
  /** Matches the provider string the Rust batch commands expect. */
  provider: "whisper" | "remoteWhisper";
  name: string;
  displayName: string;
  size_mb: number;
}

interface TranscriptModelConfig {
  provider?: string;
  model?: string;
}

/**
 * Custom hook for fetching and managing transcription models offered by the
 * import and retranscription dialogs.
 *
 * This hook centralizes the model fetching logic that was previously duplicated
 * in ImportAudioDialog and RetranscribeDialog components.
 *
 * Locally downloaded Whisper models are always listed. When the configured
 * provider is the remote endpoint, its model is listed too and selected by
 * default — but the local ones stay available, so a batch job can still be run
 * on this machine when the remote server is unreachable.
 *
 * @param transcriptModelConfig - User's saved model configuration from context
 * @returns Object containing available models, selected model key, loading state, and fetch function
 */
export function useTranscriptionModels(
  transcriptModelConfig: TranscriptModelConfig | undefined,
) {
  const [availableModels, setAvailableModels] = useState<ModelOption[]>([]);
  const [selectedModelKey, setSelectedModelKey] = useState<string>("");
  const [loadingModels, setLoadingModels] = useState(false);
  // Track whether the user has manually changed the model selection
  const userSelectedRef = useRef(false);

  // Wrap setSelectedModelKey to track user-initiated changes
  const setSelectedModelKeyWithTracking = useCallback((key: string) => {
    userSelectedRef.current = true;
    setSelectedModelKey(key);
  }, []);

  const fetchModels = useCallback(async () => {
    setLoadingModels(true);
    const allModels: ModelOption[] = [];

    // Fetch Whisper models
    try {
      const whisperModels = await invoke<RawModelInfo[]>(
        "whisper_get_available_models",
      );
      const availableWhisper = whisperModels
        .filter((m) => m.status === "Available")
        .map((m) => ({
          provider: "whisper" as const,
          name: m.name,
          displayName: `🏠 Whisper: ${m.name}`,
          size_mb: m.size_mb,
        }));
      allModels.push(...availableWhisper);
    } catch (err) {
      console.error("Failed to fetch Whisper models:", err);
    }

    const configuredProvider = transcriptModelConfig?.provider || "";
    const configuredModel = transcriptModelConfig?.model || "";

    if (configuredProvider === "remoteWhisper" && configuredModel) {
      allModels.unshift({
        provider: "remoteWhisper" as const,
        name: configuredModel,
        displayName: `🖧 Remote: ${configuredModel}`,
        size_mb: 0,
      });
    }

    setAvailableModels(allModels);

    // Try to match the configured model.
    // Note: 'localWhisper' in saved config maps to 'whisper' provider in model list.
    const configuredMatch = allModels.find(
      (m) =>
        m.name === configuredModel &&
        ((configuredProvider === "localWhisper" && m.provider === "whisper") ||
          (configuredProvider === "remoteWhisper" &&
            m.provider === "remoteWhisper")),
    );

    // Only set default model if user hasn't manually selected one
    if (!userSelectedRef.current) {
      if (configuredMatch) {
        // Use the configured model if available
        setSelectedModelKey(
          `${configuredMatch.provider}:${configuredMatch.name}`,
        );
      } else if (allModels.length > 0) {
        // Fall back to first available model
        setSelectedModelKey(`${allModels[0].provider}:${allModels[0].name}`);
      }
    }

    setLoadingModels(false);
  }, [transcriptModelConfig]);

  // Reset user selection tracking (call when dialog opens fresh)
  const resetSelection = useCallback(() => {
    userSelectedRef.current = false;
  }, []);

  return {
    availableModels,
    selectedModelKey,
    setSelectedModelKey: setSelectedModelKeyWithTracking,
    loadingModels,
    fetchModels,
    resetSelection,
  };
}
