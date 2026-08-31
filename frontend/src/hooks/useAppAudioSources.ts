"use client";

import { useCallback, useEffect, useState } from "react";
import { invoke } from "@tauri-apps/api/core";
import type { AppAudioCatalog, AppAudioSource } from "@/lib/audioDeviceModes";

/** How often the list is re-asked, to keep the "playing" flags honest. */
const POLL_MS = 5000;

/**
 * The applications the system-audio side can be pointed at.
 *
 * Comes back empty with `enabled: false` when per-app capture is
 * switched off, so a picker can render the result unconditionally and
 * simply show nothing extra.
 *
 * Polled rather than pushed: an app opening a playback stream is not an
 * event the Rust side broadcasts, and the only thing that changes
 * between ticks is which entries are marked as playing.
 */
export function useAppAudioSources(): {
  enabled: boolean;
  sources: AppAudioSource[];
  refresh: () => void;
} {
  const [catalog, setCatalog] = useState<AppAudioCatalog>({
    enabled: false,
    sources: [],
  });
  const [nonce, setNonce] = useState(0);

  const refresh = useCallback(() => setNonce((n) => n + 1), []);

  useEffect(() => {
    let cancelled = false;
    const poll = async () => {
      try {
        const next = await invoke<AppAudioCatalog>("list_app_audio_sources");
        if (!cancelled) setCatalog(next);
      } catch (err) {
        console.error("Failed to list application audio sources:", err);
      }
    };

    poll();
    const timer = setInterval(poll, POLL_MS);
    return () => {
      cancelled = true;
      clearInterval(timer);
    };
  }, [nonce]);

  return { ...catalog, refresh };
}
