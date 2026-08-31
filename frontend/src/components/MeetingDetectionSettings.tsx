"use client";

import { useCallback, useEffect, useState } from "react";
import { invoke } from "@tauri-apps/api/core";
import { listen } from "@tauri-apps/api/event";
import { RefreshCw } from "lucide-react";
import { toast } from "sonner";

import { SettingsCard, SettingsRow } from "@/app/settings/parts/SettingsCard";
import { Alert, AlertDescription } from "@/components/ui/alert";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import { Switch } from "@/components/ui/switch";
import { Text } from "@/components/ui/typography";
import { getErrorMessage } from "@/lib/utils";

interface MeetingDetectionConfig {
  enabled: boolean;
  auto_start_recording: boolean;
  auto_stop_recording: boolean;
  notify_on_detection: boolean;
  ignored_apps: string[];
  detect_unknown_apps: boolean;
  poll_interval_secs: number;
  start_delay_secs: number;
  stop_grace_secs: number;
}

interface MeetingApp {
  key: string;
  label: string;
  note: string | null;
}

interface DetectedMeeting {
  app_key: string | null;
  app_label: string;
  source: string;
  detected_at: string;
  auto_recording: boolean;
}

interface DetectionStatus {
  monitoring: boolean;
  settings: MeetingDetectionConfig;
  active_meeting: DetectedMeeting | null;
  last_error: string | null;
}

interface CaptureStream {
  app_key: string | null;
  label: string;
  node_name: string;
  matched_on: string;
  counts_as_meeting: boolean;
  /** Whether the app is actually listening, rather than sitting on a
   *  capture stream it opened at launch and never closed. */
  live: boolean;
}

const DEFAULTS: MeetingDetectionConfig = {
  enabled: false,
  auto_start_recording: true,
  auto_stop_recording: true,
  notify_on_detection: true,
  ignored_apps: [],
  detect_unknown_apps: false,
  poll_interval_secs: 5,
  start_delay_secs: 10,
  stop_grace_secs: 30,
};

/** Options for "how long before a mic grab counts as a meeting". */
const START_DELAY_OPTIONS = [
  { value: 0, label: "Immediately" },
  { value: 10, label: "After 10 seconds" },
  { value: 30, label: "After 30 seconds" },
  { value: 60, label: "After 1 minute" },
];

/** Options for "how long the mic must stay free before the call is over". */
const STOP_GRACE_OPTIONS = [
  { value: 10, label: "10 seconds" },
  { value: 30, label: "30 seconds" },
  { value: 60, label: "1 minute" },
  { value: 300, label: "5 minutes" },
];

function formatDetectedAt(iso: string): string {
  const parsed = new Date(iso);
  return Number.isNaN(parsed.getTime())
    ? iso
    : parsed.toLocaleTimeString([], { hour: "numeric", minute: "2-digit" });
}

export function MeetingDetectionSettings() {
  const [config, setConfig] = useState<MeetingDetectionConfig>(DEFAULTS);
  const [apps, setApps] = useState<MeetingApp[]>([]);
  const [status, setStatus] = useState<DetectionStatus | null>(null);
  const [streams, setStreams] = useState<CaptureStream[] | null>(null);
  const [loading, setLoading] = useState(true);
  const [saving, setSaving] = useState(false);
  const [probing, setProbing] = useState(false);
  const [unsupported, setUnsupported] = useState<string | null>(null);

  const refreshStatus = useCallback(async () => {
    try {
      const next = await invoke<DetectionStatus>(
        "api_get_meeting_detection_status",
      );
      setStatus(next);
    } catch (error) {
      console.error("Failed to load meeting detection status:", error);
    }
  }, []);

  useEffect(() => {
    const load = async () => {
      try {
        const [initialStatus, catalog] = await Promise.all([
          invoke<DetectionStatus>("api_get_meeting_detection_status"),
          invoke<MeetingApp[]>("api_list_meeting_apps"),
        ]);
        setStatus(initialStatus);
        setConfig(initialStatus.settings);
        setApps(catalog);
      } catch (error) {
        console.error("Failed to load meeting detection settings:", error);
        setUnsupported(getErrorMessage(error));
      } finally {
        setLoading(false);
      }
    };
    load();
  }, []);

  // The detector runs in Rust on its own schedule, so the panel has to be
  // told when something happened rather than deriving it from local state.
  useEffect(() => {
    const unlisteners = ["meeting-detected", "meeting-ended"].map((event) =>
      listen(event, () => {
        void refreshStatus();
      }),
    );
    return () => {
      unlisteners.forEach((pending) => {
        void pending.then((unlisten) => unlisten());
      });
    };
  }, [refreshStatus]);

  const save = useCallback(
    async (next: MeetingDetectionConfig) => {
      const previous = config;
      setConfig(next);
      setSaving(true);
      try {
        // The backend clamps out-of-range values, so take what it returns
        // rather than assuming what we sent survived intact.
        const stored = await invoke<MeetingDetectionConfig>(
          "api_set_meeting_detection_settings",
          { settings: next },
        );
        setConfig(stored);
        await refreshStatus();
      } catch (error) {
        console.error("Failed to save meeting detection settings:", error);
        toast.error("Couldn't save meeting detection settings", {
          description: getErrorMessage(error),
        });
        setConfig(previous);
      } finally {
        setSaving(false);
      }
    },
    [config, refreshStatus],
  );

  const update = useCallback(
    (patch: Partial<MeetingDetectionConfig>) => {
      void save({ ...config, ...patch });
    },
    [config, save],
  );

  const toggleApp = useCallback(
    (key: string, watched: boolean) => {
      const ignored = watched
        ? config.ignored_apps.filter((k) => k !== key)
        : [...config.ignored_apps, key];
      void save({ ...config, ignored_apps: ignored });
    },
    [config, save],
  );

  const probe = useCallback(async () => {
    setProbing(true);
    try {
      const found = await invoke<CaptureStream[]>("api_probe_capture_streams");
      setStreams(found);
    } catch (error) {
      console.error("Failed to probe capture streams:", error);
      toast.error("Couldn't read the audio graph", {
        description: getErrorMessage(error),
      });
    } finally {
      setProbing(false);
    }
  }, []);

  if (loading) {
    return (
      <div className="space-y-4">
        <SettingsCard>
          <div className="h-12 animate-pulse rounded-md bg-muted/40" />
        </SettingsCard>
        <SettingsCard>
          <div className="h-24 animate-pulse rounded-md bg-muted/40" />
        </SettingsCard>
      </div>
    );
  }

  if (unsupported) {
    return (
      <Alert variant="destructive">
        <AlertDescription>
          <p>Meeting detection isn&apos;t available: {unsupported}</p>
        </AlertDescription>
      </Alert>
    );
  }

  const active = status?.active_meeting ?? null;

  return (
    <div className="space-y-4">
      <SettingsCard>
        <SettingsRow
          label="Detect meetings automatically"
          description="Watches which apps have your microphone open and treats a sustained grab by a conferencing app as a meeting."
        >
          <Switch
            checked={config.enabled}
            onCheckedChange={(enabled) => update({ enabled })}
            disabled={saving}
          />
        </SettingsRow>
      </SettingsCard>

      {!config.enabled && (
        <Alert>
          <AlertDescription>
            Nothing is being watched. Turn detection on to have Parley notice
            calls and start recording on its own.
          </AlertDescription>
        </Alert>
      )}

      {config.enabled && (
        <>
          {active && (
            <Alert variant="success">
              <AlertDescription>
                <p>
                  {active.app_label} since {formatDetectedAt(active.detected_at)}
                  {active.auto_recording
                    ? " - recording automatically."
                    : " - not recording."}
                </p>
                {/* Which process actually triggered this. "Browser calls" on
                    its own is unfalsifiable — any tab holding the mic looks
                    the same — so name the thing we saw. */}
                <p className="mt-1 opacity-80">Triggered by {active.source}</p>
              </AlertDescription>
            </Alert>
          )}

          {status?.last_error && (
            <Alert variant="warning">
              <AlertDescription>
                <p>Couldn&apos;t read the audio graph: {status.last_error}</p>
              </AlertDescription>
            </Alert>
          )}

          <SettingsCard
            title="When a meeting is detected"
            description="Auto-stop only ever ends a recording that started automatically — anything you started yourself keeps running until you stop it."
          >
            <div className="space-y-4">
              <SettingsRow
                label="Start recording"
                description="Begin capturing as soon as a call is detected"
              >
                <Switch
                  checked={config.auto_start_recording}
                  onCheckedChange={(value) =>
                    update({ auto_start_recording: value })
                  }
                  disabled={saving}
                />
              </SettingsRow>
              <SettingsRow
                label="Stop when the call ends"
                description="Stop and save once the microphone is released"
              >
                <Switch
                  checked={config.auto_stop_recording}
                  onCheckedChange={(value) =>
                    update({ auto_stop_recording: value })
                  }
                  disabled={saving}
                />
              </SettingsRow>
              <SettingsRow
                label="Notify me"
                description="Show a desktop notification when a meeting is detected"
              >
                <Switch
                  checked={config.notify_on_detection}
                  onCheckedChange={(value) =>
                    update({ notify_on_detection: value })
                  }
                  disabled={saving}
                />
              </SettingsRow>
            </div>
          </SettingsCard>

          <SettingsCard
            title="Apps to watch"
            description="Which applications count as a meeting when they take the microphone."
          >
            <div className="space-y-4">
              {apps.map((app) => (
                <SettingsRow
                  key={app.key}
                  label={app.label}
                  description={app.note ?? undefined}
                >
                  <Switch
                    checked={!config.ignored_apps.includes(app.key)}
                    onCheckedChange={(watched) => toggleApp(app.key, watched)}
                    disabled={saving}
                  />
                </SettingsRow>
              ))}
              <SettingsRow
                label="Anything else that uses the microphone"
                description="Also fires for voice memos, dictation, and games. Off unless you want a wide net."
              >
                <Switch
                  checked={config.detect_unknown_apps}
                  onCheckedChange={(value) =>
                    update({ detect_unknown_apps: value })
                  }
                  disabled={saving}
                />
              </SettingsRow>
            </div>
          </SettingsCard>

          <SettingsCard
            title="Timing"
            description="Apps grab the microphone briefly when you test a device, and let it go for a moment between back-to-back calls. These delays keep both from being mistaken for a meeting."
          >
            <div className="space-y-4">
              <SettingsRow
                label="Wait before starting"
                description="How long an app must hold the microphone first"
              >
                <Select
                  value={String(config.start_delay_secs)}
                  onValueChange={(value) =>
                    update({ start_delay_secs: Number(value) })
                  }
                  disabled={saving}
                >
                  <SelectTrigger className="w-48">
                    <SelectValue />
                  </SelectTrigger>
                  <SelectContent>
                    {START_DELAY_OPTIONS.map((option) => (
                      <SelectItem
                        key={option.value}
                        value={String(option.value)}
                      >
                        {option.label}
                      </SelectItem>
                    ))}
                  </SelectContent>
                </Select>
              </SettingsRow>
              <SettingsRow
                label="Wait before stopping"
                description="How long the microphone must stay free before the call counts as over"
              >
                <Select
                  value={String(config.stop_grace_secs)}
                  onValueChange={(value) =>
                    update({ stop_grace_secs: Number(value) })
                  }
                  disabled={saving}
                >
                  <SelectTrigger className="w-48">
                    <SelectValue />
                  </SelectTrigger>
                  <SelectContent>
                    {STOP_GRACE_OPTIONS.map((option) => (
                      <SelectItem
                        key={option.value}
                        value={String(option.value)}
                      >
                        {option.label}
                      </SelectItem>
                    ))}
                  </SelectContent>
                </Select>
              </SettingsRow>
            </div>
          </SettingsCard>

          <SettingsCard
            title="What has your microphone right now"
            description="A live look at the audio graph, so you can check what the detector sees without having to start a call."
          >
            <div className="space-y-3">
              <Button
                variant="outline"
                size="sm"
                onClick={probe}
                disabled={probing}
              >
                <RefreshCw
                  className={probing ? "size-4 animate-spin" : "size-4"}
                />
                {probing ? "Checking..." : "Check now"}
              </Button>

              {streams !== null && streams.length === 0 && (
                <Text size="small" tone="muted">
                  Nothing is using the microphone.
                </Text>
              )}

              {streams !== null && streams.length > 0 && (
                <ul className="space-y-2">
                  {streams.map((stream) => (
                    <li
                      key={stream.node_name}
                      className="flex items-center justify-between gap-3 rounded-md border border-border px-3 py-2"
                    >
                      <div className="min-w-0">
                        <div className="truncate text-sm font-medium">
                          {stream.label}
                        </div>
                        <Text size="caption" tone="muted" className="truncate">
                          {stream.matched_on || stream.node_name}
                        </Text>
                      </div>
                      <Badge
                        variant={stream.counts_as_meeting ? "success" : "outline"}
                      >
                        {stream.counts_as_meeting
                          ? "Counts as a meeting"
                          : stream.live
                            ? "Ignored"
                            : "Idle"}
                      </Badge>
                    </li>
                  ))}
                </ul>
              )}
            </div>
          </SettingsCard>
        </>
      )}
    </div>
  );
}
