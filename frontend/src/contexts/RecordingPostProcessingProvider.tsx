"use client";

import React, { useEffect, useRef } from "react";
import { listen } from "@tauri-apps/api/event";
import { useRecordingStop } from "@/hooks/useRecordingStop";

// The global RecordingStateContext already owns the recording flags; the
// hook only takes these for the recording page's local mirrors. Declared at
// module scope so the hook's dependencies stay stable across renders.
const noop = () => {};

/**
 * RecordingPostProcessingProvider
 *
 * This provider handles post-processing when recording stops from any source:
 * - Tray menu stop
 * - Global keyboard shortcut
 * - Overlay stop button
 * - Main UI stop button
 *
 * It listens for the 'recording-stop-complete' event from Rust backend
 * and triggers the full post-processing flow (save to database, navigate, analytics)
 * regardless of which page the user is currently on.
 */
export function RecordingPostProcessingProvider({
  children,
}: {
  children: React.ReactNode;
}) {
  const { handleRecordingStop } = useRecordingStop(noop, noop);

  // `handleRecordingStop` gets a new identity on every render, and this
  // provider re-renders roughly twice a second while a recording is running
  // (RecordingStateContext polls the backend at 500ms). Keying the effect on
  // it would tear the listener down and re-register it that often, and
  // `listen()` is async — a stop landing in the gap is dropped outright,
  // leaving the recording unsaved and the UI stuck in STOPPING. Register
  // once and reach the current handler through a ref instead.
  const handleRecordingStopRef = useRef(handleRecordingStop);
  useEffect(() => {
    handleRecordingStopRef.current = handleRecordingStop;
  });

  useEffect(() => {
    let unlistenFn: (() => void) | undefined;
    let cancelled = false;

    const setupListener = async () => {
      try {
        // Listen for recording-stop-complete event from Rust
        unlistenFn = await listen<boolean>(
          "recording-stop-complete",
          (event) => {
            console.log(
              "[RecordingPostProcessing] Received recording-stop-complete event:",
              event.payload,
            );

            // Call the post-processing handler
            // event.payload is the callApi boolean (true for normal stops)
            handleRecordingStopRef.current(event.payload);
          },
        );

        // Unmounted while `listen` was still resolving.
        if (cancelled) {
          unlistenFn();
          unlistenFn = undefined;
          return;
        }

        console.log(
          "[RecordingPostProcessing] Event listener set up successfully",
        );
      } catch (error) {
        console.error(
          "[RecordingPostProcessing] Failed to set up event listener:",
          error,
        );
      }
    };

    setupListener();

    return () => {
      cancelled = true;
      if (unlistenFn) {
        console.log("[RecordingPostProcessing] Cleaning up event listener");
        unlistenFn();
      }
    };
  }, []);

  return <>{children}</>;
}
