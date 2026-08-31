import { useState, useEffect } from "react";
import { useRecordingState } from "@/contexts/RecordingStateContext";

interface UseRecordingStateSyncReturn {
  isBackendRecording: boolean;
  isRecordingDisabled: boolean;
  setIsRecordingDisabled: (value: boolean) => void;
}

/**
 * Keeps the recording page's local `isRecording` mirror in step with the
 * real recording state, and owns the "disabled while post-processing" flag.
 *
 * The page deliberately keeps its own copy of the flag — several call sites
 * depend on it flipping only after a successful Tauri round-trip — but that
 * copy dies with the component. Navigating away mid-recording and coming
 * back therefore landed on the idle hero with a live recording still running
 * underneath: no stop button to press, and `start_recording` rejecting with
 * "Recording already in progress". Re-seeding from `RecordingStateContext`,
 * which is mounted above the router and already tracks the backend through
 * both events and a 500ms poll, is what lets the page survive navigation.
 *
 * (This used to poll `is_recording` itself, gated on `window.__TAURI__` —
 * a global that only exists when `app.withGlobalTauri` is on, which it
 * isn't. The poll never ran at all.)
 */
export function useRecordingStateSync(
  isRecording: boolean,
  setIsRecording: (value: boolean) => void,
  setIsMeetingActive: (value: boolean) => void,
): UseRecordingStateSyncReturn {
  const [isRecordingDisabled, setIsRecordingDisabled] = useState(false);
  const { isRecording: backendRecording } = useRecordingState();

  useEffect(() => {
    if (backendRecording === isRecording) return;
    setIsRecording(backendRecording);
    // Only ever raised here. The stop flow lowers it once post-processing
    // has finished, which is later than this and shouldn't be pre-empted.
    if (backendRecording) {
      setIsMeetingActive(true);
    }
  }, [backendRecording, isRecording, setIsRecording, setIsMeetingActive]);

  return {
    isBackendRecording: isRecording,
    isRecordingDisabled,
    setIsRecordingDisabled,
  };
}
