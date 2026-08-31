"use client";

import { useCallback, useEffect, useState } from "react";
import { invoke } from "@tauri-apps/api/core";
import { toast } from "sonner";
import { Eye, EyeOff, Loader2 } from "lucide-react";

import { Input } from "./ui/input";
import { Button } from "./ui/button";
import { Text } from "./ui/typography";
import { RemoteModelManager } from "./RemoteModelManager";
import { SettingsCard } from "@/app/settings/parts/SettingsCard";

/** Mirrors the Rust `RemoteWhisperConfig`. */
export interface RemoteWhisperConfig {
  baseUrl: string;
  apiKey?: string | null;
  model: string;
}

const DEFAULT_MODEL = "whisper-1";

interface RemoteWhisperSettingsProps {
  /** Called after a successful save, with the model that was saved, so the
   *  parent can point the transcript config at this provider. */
  onSaved: (model: string) => void;
}

/**
 * Settings for offloading Whisper decode to another machine.
 *
 * The endpoint is any OpenAI-compatible `/v1/audio/transcriptions` server —
 * whisper.cpp's own `whisper-server` speaks this natively, so the usual setup
 * is just running that on the box with the good GPU.
 *
 * Config lives in its own backend store (`api_*_remote_whisper_config`)
 * rather than the fixed-column transcript settings table, so it's loaded and
 * saved here rather than through the shared transcript config.
 */
export function RemoteWhisperSettings({ onSaved }: RemoteWhisperSettingsProps) {
  const [baseUrl, setBaseUrl] = useState("");
  const [model, setModel] = useState(DEFAULT_MODEL);
  const [apiKey, setApiKey] = useState("");
  const [showApiKey, setShowApiKey] = useState(false);
  const [isTesting, setIsTesting] = useState(false);
  const [isSaving, setIsSaving] = useState(false);

  useEffect(() => {
    const load = async () => {
      try {
        const config = await invoke<RemoteWhisperConfig | null>(
          "api_get_remote_whisper_config",
        );
        if (config) {
          setBaseUrl(config.baseUrl || "");
          setModel(config.model || DEFAULT_MODEL);
          setApiKey(config.apiKey || "");
        }
      } catch (err) {
        console.error("Failed to load remote transcription config:", err);
      }
    };
    load();
  }, []);

  const buildConfig = useCallback(
    (): RemoteWhisperConfig => ({
      baseUrl: baseUrl.trim(),
      model: model.trim() || DEFAULT_MODEL,
      apiKey: apiKey.trim() || null,
    }),
    [baseUrl, model, apiKey],
  );

  const handleTest = async () => {
    if (!baseUrl.trim()) {
      toast.error("Enter the endpoint URL first");
      return;
    }

    setIsTesting(true);
    try {
      // The backend proves the endpoint by pushing a short burst of silence
      // through the real transcription path, so a pass means host, path,
      // auth, and model all actually work.
      const message = await invoke<string>("api_test_remote_whisper", {
        config: buildConfig(),
      });
      toast.success(message);
    } catch (err) {
      toast.error(err instanceof Error ? err.message : String(err));
    } finally {
      setIsTesting(false);
    }
  };

  const handleSave = async () => {
    if (!baseUrl.trim()) {
      toast.error("Enter the endpoint URL first");
      return;
    }

    setIsSaving(true);
    const config = buildConfig();
    try {
      await invoke("api_save_remote_whisper_config", { config });
      // Point the transcript config at this provider too, so recording,
      // import, and retranscription all route here. The key stays in the
      // remote config blob, hence null.
      await invoke("api_save_transcript_config", {
        provider: "remoteWhisper",
        model: config.model,
        apiKey: null,
      });
      onSaved(config.model);
      toast.success("Remote transcription settings saved");
    } catch (err) {
      toast.error(err instanceof Error ? err.message : String(err));
    } finally {
      setIsSaving(false);
    }
  };

  return (
    <SettingsCard
      title="Remote transcription endpoint"
      description="Any OpenAI-compatible /v1/audio/transcriptions server — for example whisper.cpp's whisper-server running on a machine with a faster GPU. Audio is sent to it over the network; everything else still happens locally."
    >
      <div className="space-y-4">
        <div className="space-y-1.5">
          <Text size="small" className="font-medium">
            Endpoint URL
          </Text>
          <Input
            className="focus:border-info focus:ring-1 focus:ring-info"
            value={baseUrl}
            onChange={(e) => setBaseUrl(e.target.value)}
            placeholder="http://gpubox.lan:8080"
            spellCheck={false}
          />
          <Text size="small" tone="muted">
            The server root is enough — /v1/audio/transcriptions is appended
            automatically.
          </Text>
        </div>

        <div className="space-y-1.5">
          <Text size="small" className="font-medium">
            Model
          </Text>
          <Input
            className="focus:border-info focus:ring-1 focus:ring-info"
            value={model}
            onChange={(e) => setModel(e.target.value)}
            placeholder={DEFAULT_MODEL}
            spellCheck={false}
          />
          <Text size="small" tone="muted">
            Servers with a model store list theirs below; simpler ones ignore
            this and use whatever model they were started with.
          </Text>
        </div>

        <div className="space-y-1.5">
          <Text size="small" className="font-medium">
            API key (optional)
          </Text>
          <div className="relative">
            <Input
              type={showApiKey ? "text" : "password"}
              className="
                pr-12
                focus:border-info focus:ring-1 focus:ring-info
              "
              value={apiKey}
              onChange={(e) => setApiKey(e.target.value)}
              placeholder="Leave blank for servers without auth"
            />
            <div className="absolute inset-y-0 right-0 flex items-center pr-1">
              <Button
                type="button"
                variant="ghost"
                size="icon"
                onClick={() => setShowApiKey(!showApiKey)}
              >
                {showApiKey ? (
                  <EyeOff className="size-4" />
                ) : (
                  <Eye className="size-4" />
                )}
              </Button>
            </div>
          </div>
        </div>

        <div className="space-y-1.5">
          <Text size="small" className="font-medium">
            Models on the server
          </Text>
          <RemoteModelManager
            baseUrl={baseUrl}
            apiKey={apiKey.trim() || null}
            selectedModel={model.trim()}
            onModelSelect={setModel}
          />
        </div>

        <div className="flex flex-wrap gap-2">
          <Button
            type="button"
            variant="outline"
            onClick={handleTest}
            disabled={isTesting || isSaving}
          >
            {isTesting && <Loader2 className="mr-2 size-4 animate-spin" />}
            Test connection
          </Button>
          <Button
            type="button"
            onClick={handleSave}
            disabled={isTesting || isSaving}
          >
            {isSaving && <Loader2 className="mr-2 size-4 animate-spin" />}
            Save
          </Button>
        </div>
      </div>
    </SettingsCard>
  );
}
