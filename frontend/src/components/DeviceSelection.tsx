import React, { useState, useEffect, useCallback } from "react";
import { invoke } from "@tauri-apps/api/core";
import { RefreshCw, Mic, Speaker } from "lucide-react";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import { Label } from "@/components/ui/label";
import { Button } from "@/components/ui/button";
import { Heading } from "@/components/ui/typography";
import { Alert, AlertDescription } from "@/components/ui/alert";
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

/**
 * A capture device as reported by the PipeWire registry.
 * `id` is the stable node name (used for selection/preferences),
 * `label` the human-readable description shown in pickers.
 */
export interface AudioDevice {
  id: string;
  label: string;
  kind: "microphone" | "system";
}

export interface SelectedDevices {
  /**
   * PipeWire node id, `null` = system default, or the
   * `"communication"` sentinel = follow the active call. See
   * `@/lib/audioDeviceModes`.
   */
  micDevice: string | null;
  systemDevice: string | null;
}

interface DeviceSelectionProps {
  selectedDevices: SelectedDevices;
  onDeviceChange: (devices: SelectedDevices) => void;
  disabled?: boolean;
}

export function DeviceSelection({
  selectedDevices,
  onDeviceChange,
  disabled = false,
}: DeviceSelectionProps) {
  const [devices, setDevices] = useState<AudioDevice[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [refreshing, setRefreshing] = useState(false);

  const microphones = devices.filter((d) => d.kind === "microphone");
  const systemDevices = devices.filter((d) => d.kind === "system");

  const route = useResolvedRoute(
    selectedDevices.micDevice,
    selectedDevices.systemDevice,
  );
  const apps = useAppAudioSources();

  // Fetch available audio devices
  const fetchDevices = useCallback(async () => {
    try {
      setError(null);
      const result = await invoke<AudioDevice[]>("get_audio_devices");
      setDevices(result);
      console.log("Fetched audio devices:", result);
    } catch (err) {
      console.error("Failed to fetch audio devices:", err);
      setError(
        "Failed to load audio devices. Please check your system audio settings.",
      );
    } finally {
      setLoading(false);
      setRefreshing(false);
    }
  }, []);

  // Load devices on component mount
  useEffect(() => {
    // setState happens after await; the rule cannot see through async boundaries.
    // eslint-disable-next-line react-hooks/set-state-in-effect
    fetchDevices();
  }, [fetchDevices]);

  const handleRefresh = async () => {
    setRefreshing(true);
    apps.refresh();
    await fetchDevices();
  };

  const handleMicDeviceChange = (deviceId: string) => {
    onDeviceChange({
      ...selectedDevices,
      micDevice: storedSelection(deviceId),
    });
  };

  const handleSystemDeviceChange = (deviceId: string) => {
    onDeviceChange({
      ...selectedDevices,
      systemDevice: storedSelection(deviceId),
    });
  };

  /** The label for a node id, once the graph has told us which one. */
  const resolvedLabel = (nodeId: string | null) => {
    if (!nodeId) return null;
    // An app selection resolves to itself, not to a node: the capture
    // follows the app around the graph, so there is no device to name.
    const key = appKey(nodeId);
    if (key) {
      return apps.sources.find((s) => s.key === key)?.label ?? key;
    }
    return devices.find((d) => d.id === nodeId)?.label ?? nodeId;
  };

  const micRouteLabel = resolvedLabel(route.microphone);
  const systemRouteLabel = resolvedLabel(route.system);

  if (loading) {
    return (
      <div className="space-y-4 p-4">
        <div className="animate-pulse">
          <div className="mb-4 h-4 w-1/3 rounded-md bg-muted"></div>
          <div className="mb-3 h-10 rounded-md bg-muted"></div>
          <div className="h-10 rounded-md bg-muted"></div>
        </div>
      </div>
    );
  }

  return (
    <div className="space-y-4">
      <div className="flex items-center justify-between">
        <Heading level={3}>Audio Devices</Heading>
        <Button
          variant="ghost"
          size="icon"
          onClick={handleRefresh}
          disabled={refreshing || disabled}
          className="size-8"
        >
          <RefreshCw
            className={`
              size-4
              ${refreshing ? "animate-spin" : ""}
            `}
          />
        </Button>
      </div>

      {error && (
        <Alert variant="destructive">
          <AlertDescription>{error}</AlertDescription>
        </Alert>
      )}

      <div className="space-y-3">
        {/* Microphone Selection */}
        <div className="space-y-2">
          <div className="flex items-center gap-2">
            <Mic className="size-4 text-muted-foreground" />
            <Label
              htmlFor="mic-selection"
              className="text-sm font-medium text-foreground"
            >
              Microphone
            </Label>
          </div>
          <Select
            value={selectionValue(selectedDevices.micDevice)}
            onValueChange={handleMicDeviceChange}
            disabled={disabled}
          >
            <SelectTrigger id="mic-selection" className="w-full">
              <SelectValue placeholder="Select Microphone" />
            </SelectTrigger>
            <SelectContent>
              {MIC_MODE_OPTIONS.map((option) => (
                <SelectItem key={option.value} value={option.value}>
                  {option.label}
                </SelectItem>
              ))}
              {microphones.map((device) => (
                <SelectItem key={device.id} value={device.id}>
                  {device.label}
                </SelectItem>
              ))}
            </SelectContent>
          </Select>
          {micRouteLabel && (
            <p className="text-sm text-muted-foreground">
              Currently using {micRouteLabel}
            </p>
          )}
          {microphones.length === 0 && (
            <p className="text-sm text-muted-foreground">
              No microphone devices found
            </p>
          )}
        </div>

        {/* System Audio Selection */}
        <div className="space-y-2">
          <div className="flex items-center gap-2">
            <Speaker className="size-4 text-muted-foreground" />
            <Label
              htmlFor="system-selection"
              className="text-sm font-medium text-foreground"
            >
              System Audio
            </Label>
          </div>

          <Select
            value={selectionValue(selectedDevices.systemDevice)}
            onValueChange={handleSystemDeviceChange}
            disabled={disabled}
          >
            <SelectTrigger id="system-selection" className="w-full">
              <SelectValue placeholder="Select System Audio" />
            </SelectTrigger>
            <SelectContent>
              {SYSTEM_MODE_OPTIONS.map((option) => (
                <SelectItem key={option.value} value={option.value}>
                  {option.label}
                </SelectItem>
              ))}
              {systemDevices.map((device) => (
                <SelectItem key={device.id} value={device.id}>
                  {device.label}
                </SelectItem>
              ))}
              {appOptions(apps.sources).map((option) => (
                <SelectItem key={option.value} value={option.value}>
                  {option.label}
                </SelectItem>
              ))}
            </SelectContent>
          </Select>

          {systemRouteLabel && (
            <p className="text-sm text-muted-foreground">
              Currently using {systemRouteLabel}
            </p>
          )}
          {systemDevices.length === 0 && (
            <p className="text-sm text-muted-foreground">
              No system audio devices found
            </p>
          )}
        </div>
      </div>

      {/* Info text */}
      <div className="space-y-1 text-sm text-muted-foreground">
        <p>
          • <strong>Microphone:</strong> Records your voice and ambient sound
        </p>
        <p>
          • <strong>System Audio:</strong> Records computer audio (music, calls,
          etc.) from the selected output device
        </p>
        <p>
          • <strong>Follow the active call:</strong> Tracks whichever device
          your call app is using, and switches with it mid-recording
        </p>
        {apps.enabled && (
          <>
            <p>
              • <strong>Whichever app is in a call:</strong> Records only the
              app you are actually in a call with — Discord, a Meet tab, Zoom —
              and picks it up when the call starts
            </p>
            <p>
              • <strong>A named application:</strong> Records only what that app
              plays, leaving everything else your speakers are doing out of the
              recording
            </p>
          </>
        )}
      </div>
    </div>
  );
}
