"use client";

import { useCallback, useEffect, useRef, useState } from "react";
import { listen, type UnlistenFn } from "@tauri-apps/api/event";
import { Download } from "lucide-react";

import { Button } from "./ui/button";
import { Label } from "./ui/label";
import { Progress } from "./ui/progress";
import { Switch } from "./ui/switch";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "./ui/select";
import { Heading, Text } from "@/components/ui/typography";
import { Alert, AlertDescription } from "@/components/ui/alert";
import {
  downloadSpeakerModels,
  getSpeakerDiarizationSettings,
  getSpeakerModelStatus,
  listSpeakerEmbeddingModels,
  setSpeakerDiarizationSettings,
  SPEAKER_MODEL_DOWNLOAD_PROGRESS,
  type SpeakerDiarizationSettings,
  type SpeakerEmbeddingModel,
  type SpeakerModelStatus,
} from "@/lib/speaker-recognition";

interface SpeakerRecognitionSettingsProps {
  /** Fires with the active embedding model's filename once loaded and on
   *  every change, so the saved-speakers table can flag stale profiles. */
  onActiveModelChange?: (filename: string) => void;
}

export function SpeakerRecognitionSettings({
  onActiveModelChange,
}: SpeakerRecognitionSettingsProps) {
  const [settings, setSettings] = useState<SpeakerDiarizationSettings | null>(
    null,
  );
  const [models, setModels] = useState<SpeakerEmbeddingModel[]>([]);
  const [status, setStatus] = useState<SpeakerModelStatus | null>(null);
  const [downloadProgress, setDownloadProgress] = useState<number | null>(
    null,
  );
  const [error, setError] = useState<string | null>(null);
  const unlistenRef = useRef<UnlistenFn | null>(null);

  const refresh = useCallback(async () => {
    try {
      const [s, m, st] = await Promise.all([
        getSpeakerDiarizationSettings(),
        listSpeakerEmbeddingModels(),
        getSpeakerModelStatus(),
      ]);
      setSettings(s);
      setModels(m);
      setStatus(st);
      onActiveModelChange?.(s.embedding_model);
    } catch (err) {
      setError(err instanceof Error ? err.message : String(err));
    }
  }, [onActiveModelChange]);

  useEffect(() => {
    void refresh();
    return () => unlistenRef.current?.();
  }, [refresh]);

  const download = useCallback(async () => {
    setError(null);
    setDownloadProgress(0);
    unlistenRef.current?.();
    unlistenRef.current = await listen<{ progress: number }>(
      SPEAKER_MODEL_DOWNLOAD_PROGRESS,
      (e) => setDownloadProgress(e.payload.progress),
    );
    try {
      await downloadSpeakerModels();
    } catch (err) {
      setError(
        `Download failed: ${err instanceof Error ? err.message : String(err)}`,
      );
    } finally {
      unlistenRef.current?.();
      unlistenRef.current = null;
      setDownloadProgress(null);
      await refresh();
    }
  }, [refresh]);

  const save = useCallback(
    async (next: SpeakerDiarizationSettings) => {
      const previous = settings;
      setSettings(next);
      setError(null);
      try {
        await setSpeakerDiarizationSettings(next);
      } catch (err) {
        setSettings(previous);
        setError(err instanceof Error ? err.message : String(err));
        return false;
      }
      return true;
    },
    [settings],
  );

  const handleModelChange = async (filename: string) => {
    if (!settings || filename === settings.embedding_model) return;
    if (!(await save({ ...settings, embedding_model: filename }))) return;
    const model = models.find((m) => m.filename === filename);
    if (model && !model.downloaded) {
      await download();
    } else {
      await refresh();
    }
  };

  const active = models.find((m) => m.filename === settings?.embedding_model);
  const missing = status && (!status.is_ready || !status.segmentation_ready);
  const downloading = downloadProgress !== null;

  return (
    <div className="space-y-4">
      <div>
        <Heading level={3}>Recognition</Heading>
        <Text size="caption" tone="muted">
          How voices are told apart. Changes apply from the next recording.
        </Text>
      </div>

      {error && (
        <Alert variant="destructive">
          <AlertDescription>{error}</AlertDescription>
        </Alert>
      )}

      <div className="space-y-1">
        <Label htmlFor="speaker-embedding-model">Voice model</Label>
        <Select
          value={settings?.embedding_model ?? ""}
          onValueChange={(v) => void handleModelChange(v)}
          disabled={!settings || downloading}
        >
          <SelectTrigger id="speaker-embedding-model">
            <SelectValue placeholder="Loading…" />
          </SelectTrigger>
          <SelectContent>
            {models.map((m) => (
              <SelectItem key={m.filename} value={m.filename}>
                {m.display_name}
                <span className="text-muted-foreground">
                  {" "}
                  · {m.size_mb} MB{m.downloaded ? "" : " · not downloaded"}
                </span>
              </SelectItem>
            ))}
          </SelectContent>
        </Select>
        {active && (
          <Text size="caption" tone="muted">
            {active.description} Saved speakers and your recorded voice belong
            to the model they were made with: after switching, record your
            voice again and re-name speakers for them to be recognised.
          </Text>
        )}
      </div>

      {downloading && (
        <div className="space-y-1">
          <Text size="caption" tone="muted">
            Downloading speaker models… {downloadProgress}%
          </Text>
          <Progress value={downloadProgress ?? 0} />
        </div>
      )}

      {missing && !downloading && (
        <div className="
          flex items-center justify-between gap-3 rounded-md border
          border-border p-3
        ">
          <Text size="small">
            {status.is_ready
              ? "The speaker-turn model isn't downloaded, so speaker labels after a recording are less accurate."
              : "The voice model isn't downloaded, so speakers can't be told apart."}
          </Text>
          <Button size="sm" variant="outline" onClick={() => void download()}>
            <Download className="size-4" />
            Download
          </Button>
        </div>
      )}

      <div className="flex items-start justify-between gap-4">
        <div className="space-y-0.5">
          <Label htmlFor="speaker-separate-sources">
            Keep my side and the call separate
          </Label>
          <Text size="caption" tone="muted">
            Never give a voice from your microphone the same label as a voice
            from the call. Turn on when you wear headphones; leave off if your
            microphone can hear your speakers.
          </Text>
        </div>
        <Switch
          id="speaker-separate-sources"
          checked={settings?.separate_sources ?? false}
          disabled={!settings}
          onCheckedChange={(checked) =>
            settings && void save({ ...settings, separate_sources: checked })
          }
        />
      </div>
    </div>
  );
}
