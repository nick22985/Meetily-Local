"use client";

import { useEffect, useState } from "react";
import { invoke } from "@tauri-apps/api/core";
import { Globe, Mic, Speaker } from "lucide-react";
import { useConfig } from "@/contexts/ConfigContext";
import type {
  AudioDevice,
  SelectedDevices,
} from "@/components/DeviceSelection";
import {
  MIC_MODE_OPTIONS,
  SYSTEM_MODE_OPTIONS,
  appKey,
  appOptions,
  selectionValue,
  storedSelection,
} from "@/lib/audioDeviceModes";
import { useResolvedRoute } from "@/hooks/useResolvedRoute";
import { useAppAudioSources } from "@/hooks/useAppAudioSources";
import { DeviceChip } from "./DeviceChip";

const LANGUAGE_OPTIONS = [
  { value: "auto", label: "Auto detect" },
  { value: "auto-translate", label: "Auto detect + translate to English" },
  { value: "en", label: "English" },
  { value: "fr", label: "French" },
  { value: "de", label: "German" },
  { value: "es", label: "Spanish" },
  { value: "it", label: "Italian" },
  { value: "pt", label: "Portuguese" },
  { value: "nl", label: "Dutch" },
  { value: "ja", label: "Japanese" },
  { value: "ko", label: "Korean" },
  { value: "zh", label: "Chinese" },
  { value: "ru", label: "Russian" },
  { value: "ar", label: "Arabic" },
  { value: "hi", label: "Hindi" },
  { value: "tr", label: "Turkish" },
  { value: "pl", label: "Polish" },
  { value: "sv", label: "Swedish" },
  { value: "da", label: "Danish" },
  { value: "no", label: "Norwegian" },
  { value: "fi", label: "Finnish" },
];

interface DeviceSummaryProps {
  /** Disable selection (e.g. during the "starting recording" flash). */
  disabled?: boolean;
}

/**
 * Three pill chips on the pre-recording hero: microphone, system audio,
 * language. Each opens a popover for changing the value. The state lives
 * in `useConfig()` (`selectedDevices` and `selectedLanguage`); we just
 * read + drive it.
 */
export function DeviceSummary({ disabled = false }: DeviceSummaryProps) {
  const {
    selectedDevices,
    setSelectedDevices,
    selectedLanguage,
    setSelectedLanguage,
  } = useConfig();
  const [devices, setDevices] = useState<AudioDevice[]>([]);

  useEffect(() => {
    let cancelled = false;
    (async () => {
      try {
        const list = await invoke<AudioDevice[]>("get_audio_devices");
        if (!cancelled) setDevices(list);
      } catch (err) {
        console.error("Failed to fetch audio devices:", err);
      }
    })();
    return () => {
      cancelled = true;
    };
  }, []);

  const inputs = devices.filter((d) => d.kind === "microphone");
  const outputs = devices.filter((d) => d.kind === "system");

  // Selections use the sentinels from `@/lib/audioDeviceModes` — the
  // same values DeviceSelection uses on the Settings page. Storing
  // `null` in the context means "system default" (the backend treats
  // null this way), so it collapses onto that sentinel.
  const setMic = (id: string) =>
    setSelectedDevices({
      ...selectedDevices,
      micDevice: storedSelection(id),
    } as SelectedDevices);
  const setSystem = (id: string) =>
    setSelectedDevices({
      ...selectedDevices,
      systemDevice: storedSelection(id),
    } as SelectedDevices);

  const route = useResolvedRoute(
    selectedDevices?.micDevice,
    selectedDevices?.systemDevice,
  );
  const apps = useAppAudioSources();

  // Selections are stored as PipeWire node ids; show the human label.
  // For a selection that follows the graph, the chip shows where it
  // currently lands rather than the mode name — that is the thing the
  // user actually wants to check before hitting record.
  const nodeLabel = (id: string) =>
    devices.find((d) => d.id === id)?.label ?? id;
  const labelFor = (
    id: string | null | undefined,
    resolved: string | null,
    options: { value: string; label: string }[],
  ) => {
    const value = selectionValue(id);
    // An app selection names the app, not wherever it happens to be
    // playing — that is the whole point of picking one.
    const app = appKey(value);
    if (app) {
      return apps.sources.find((s) => s.key === app)?.label ?? app;
    }
    const mode = options.find((o) => o.value === value);
    if (!mode) return nodeLabel(value);
    return resolved ? `${mode.label} · ${nodeLabel(resolved)}` : mode.label;
  };
  const micLabel = labelFor(
    selectedDevices?.micDevice,
    route.microphone,
    MIC_MODE_OPTIONS,
  );
  const systemLabel = labelFor(
    selectedDevices?.systemDevice,
    route.system,
    SYSTEM_MODE_OPTIONS,
  );
  const languageLabel =
    LANGUAGE_OPTIONS.find((l) => l.value === selectedLanguage)?.label ??
    "Auto detect";

  return (
    <div className="flex flex-wrap items-center justify-center gap-2">
      <DeviceChip
        icon={<Mic className="size-3.5" />}
        value={micLabel}
        options={[
          ...MIC_MODE_OPTIONS,
          ...inputs.map((d) => ({ value: d.id, label: d.label })),
        ]}
        selectedValue={selectionValue(selectedDevices?.micDevice)}
        onSelect={setMic}
        emptyState="No microphones detected"
        disabled={disabled}
      />
      <DeviceChip
        icon={<Speaker className="size-3.5" />}
        value={systemLabel}
        options={[
          ...SYSTEM_MODE_OPTIONS,
          ...outputs.map((d) => ({ value: d.id, label: d.label })),
          ...appOptions(apps.sources),
        ]}
        selectedValue={selectionValue(selectedDevices?.systemDevice)}
        onSelect={setSystem}
        emptyState="No output devices detected"
        disabled={disabled}
      />
      <DeviceChip
        icon={<Globe className="size-3.5" />}
        value={languageLabel}
        options={LANGUAGE_OPTIONS}
        selectedValue={selectedLanguage}
        onSelect={setSelectedLanguage}
        disabled={disabled}
      />
    </div>
  );
}
