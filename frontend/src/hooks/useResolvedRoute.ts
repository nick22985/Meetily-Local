"use client";

import { useEffect, useState } from "react";
import { invoke } from "@tauri-apps/api/core";
import {
  type ActiveRoute,
  followsGraph,
  selectionValue,
} from "@/lib/audioDeviceModes";

const POLL_MS = 2000;

/**
 * Where the current selections actually land, for selections that follow
 * the graph.
 *
 * Only worth asking when at least one side is a sentinel — an explicitly
 * named node is its own answer. Polls rather than subscribing: the Rust
 * side re-checks the graph on its own schedule anyway, and a couple of
 * seconds of lag on a label nobody is staring at costs nothing.
 */
export function useResolvedRoute(
  micDevice: string | null | undefined,
  systemDevice: string | null | undefined,
): ActiveRoute {
  const [route, setRoute] = useState<ActiveRoute>({
    microphone: null,
    system: null,
  });

  const watching = followsGraph(micDevice) || followsGraph(systemDevice);

  useEffect(() => {
    if (!watching) return;

    let cancelled = false;
    const poll = async () => {
      try {
        const next = await invoke<ActiveRoute>("resolve_capture_route", {
          microphone: selectionValue(micDevice),
          system: selectionValue(systemDevice),
        });
        if (!cancelled) setRoute(next);
      } catch (err) {
        console.error("Failed to resolve the capture route:", err);
      }
    };

    poll();
    const timer = setInterval(poll, POLL_MS);
    return () => {
      cancelled = true;
      clearInterval(timer);
    };
  }, [watching, micDevice, systemDevice]);

  // Not derived state: a stale route from a previous selection must not
  // outlive it, and the effect is the wrong place to clear it.
  return watching ? route : { microphone: null, system: null };
}
