"use client";

import { memo, useCallback, useEffect, useMemo, useRef, useState } from "react";
import { invoke } from "@tauri-apps/api/core";
import { toast } from "sonner";
import {
  ChevronLeft,
  ChevronRight,
  Download,
  RefreshCw,
  Search,
  Trash2,
} from "lucide-react";

import {
  Accordion,
  AccordionContent,
  AccordionItem,
  AccordionTrigger,
} from "./ui/accordion";
import { Alert, AlertDescription, AlertTitle } from "./ui/alert";
import { Button } from "./ui/button";
import { Input } from "./ui/input";
import { Spinner } from "./ui/spinner";
import { Heading, Text } from "./ui/typography";
import {
  Dialog,
  DialogContent,
  DialogFooter,
  DialogHeader,
  DialogTitle,
} from "./ui/dialog";
import type { RemoteWhisperConfig } from "./RemoteWhisperSettings";

/** Mirrors the Rust `RemoteModel`. */
export interface RemoteModel {
  id: string;
  owned_by: string;
  task: string | null;
  language: string[] | null;
  created: number | null;
}

/** Mirrors the Rust `RemoteModelCatalog`. */
interface RemoteModelCatalog {
  supported: boolean;
  installed: RemoteModel[];
  registry: RemoteModel[];
  loaded: string[];
}

interface SuggestedModel {
  id: string;
  label: string;
  tagline: string;
  icon: string;
  accuracy: string;
  speed: string;
  recommended?: boolean;
}

/**
 * The remote equivalents of the local manager's basic-model shortlist.
 *
 * A Whisper server's registry is the whole of Hugging Face's speech
 * section — hundreds of community forks, most of them fine-tunes for one
 * language. These are the general-purpose CT2 conversions that correspond
 * to the local `small` / `medium` / `large-v3-turbo` / `large-v3` tiers, so
 * the choice reads the same whichever backend you're on. Anything not
 * listed here is still reachable through the full browser below.
 */
const SUGGESTED_MODELS: SuggestedModel[] = [
  {
    id: "Systran/faster-whisper-small",
    label: "Small",
    tagline: "Quick and light — fine for clear, close-mic speech",
    icon: "🚀",
    accuracy: "Decent",
    speed: "Fast",
  },
  {
    id: "Systran/faster-whisper-medium",
    label: "Medium",
    tagline: "Balanced accuracy and speed",
    icon: "⚡",
    accuracy: "Good",
    speed: "Moderate",
  },
  {
    id: "deepdml/faster-whisper-large-v3-turbo-ct2",
    label: "Large v3 Turbo",
    tagline: "Near large-v3 accuracy, several times faster",
    icon: "🔥",
    accuracy: "High",
    speed: "Fast",
    recommended: true,
  },
  {
    id: "Systran/faster-distil-whisper-large-v3",
    label: "Distil Large v3",
    tagline: "Distilled large-v3 — faster, English-leaning",
    icon: "⚡",
    accuracy: "High",
    speed: "Fast",
  },
  {
    id: "Systran/faster-whisper-large-v3",
    label: "Large v3",
    tagline: "Best accuracy, heaviest to run",
    icon: "🔥",
    accuracy: "High",
    speed: "Slow",
  },
];

const SUGGESTED_IDS = new Set(SUGGESTED_MODELS.map((m) => m.id));

/**
 * Rows per page in the full browser.
 *
 * This is a hard page size, not a "show more" cursor: appending to the
 * rendered list is what made the browser crawl, because every keystroke in
 * the settings form then re-rendered several hundred rows.
 */
const PAGE_SIZE = 10;

interface RemoteModelManagerProps {
  /** Server root, as currently typed in the form. */
  baseUrl: string;
  /** Bearer token for the same, or null. */
  apiKey: string | null;
  /** Model id currently chosen for transcription. */
  selectedModel: string;
  /** Must be referentially stable — this component is memoized. */
  onModelSelect: (modelId: string) => void;
}

/**
 * Browse, download, and delete the Whisper models on a remote endpoint.
 *
 * The counterpart to the local `ModelManager`, for servers that expose their
 * own model store over HTTP (Speaches and API-compatible builds). Everything
 * happens on the server — the weights never touch this machine — so there's
 * no download progress to report: the server pulls synchronously and answers
 * when it's done.
 *
 * Servers that implement transcription but not model management report
 * `supported: false`, and this degrades to a one-line note pointing back at
 * the free-text model field.
 *
 * Memoized on primitive props, so typing in the settings form's other
 * fields doesn't re-render the model lists.
 */
export const RemoteModelManager = memo(function RemoteModelManager({
  baseUrl: rawBaseUrl,
  apiKey,
  selectedModel,
  onModelSelect,
}: RemoteModelManagerProps) {
  const [catalog, setCatalog] = useState<RemoteModelCatalog | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [query, setQuery] = useState("");
  const [page, setPage] = useState(0);
  const [pagedQuery, setPagedQuery] = useState("");
  const [busyModel, setBusyModel] = useState<string | null>(null);
  const [pendingDelete, setPendingDelete] = useState<string | null>(null);

  const baseUrl = rawBaseUrl.trim();

  // The API key is needed for every request but shouldn't itself trigger
  // one, so it's read through a ref rather than a dependency.
  const apiKeyRef = useRef(apiKey);
  useEffect(() => {
    apiKeyRef.current = apiKey;
  }, [apiKey]);

  const requestConfig = useCallback(
    (url: string): RemoteWhisperConfig => ({
      baseUrl: url,
      apiKey: apiKeyRef.current,
      model: selectedModel,
    }),
    [selectedModel],
  );
  const requestConfigRef = useRef(requestConfig);
  useEffect(() => {
    requestConfigRef.current = requestConfig;
  }, [requestConfig]);

  const refresh = useCallback(async (url: string, showErrors: boolean) => {
    if (!url) return;

    setLoading(true);
    setError(null);
    try {
      const result = await invoke<RemoteModelCatalog>(
        "api_remote_whisper_models",
        { config: requestConfigRef.current(url) },
      );
      setCatalog(result);
    } catch (err) {
      const message = err instanceof Error ? err.message : String(err);
      setCatalog(null);
      setError(message);
      if (showErrors) toast.error(message);
    } finally {
      setLoading(false);
    }
  }, []);

  // Debounced: the URL field is typed into character by character, and each
  // intermediate value would otherwise fire a request at a host that
  // doesn't exist yet. An empty URL needs no cleanup — the render path
  // short-circuits to the placeholder before it reads any stale catalog.
  useEffect(() => {
    if (!baseUrl) return;
    const timer = setTimeout(() => refresh(baseUrl, false), 600);
    return () => clearTimeout(timer);
  }, [baseUrl, refresh]);

  const installedIds = useMemo(
    () => new Set((catalog?.installed ?? []).map((m) => m.id)),
    [catalog],
  );
  const loadedIds = useMemo(
    () => new Set(catalog?.loaded ?? []),
    [catalog],
  );

  /** Installed models that aren't already covered by the suggested cards. */
  const otherInstalled = useMemo(
    () => (catalog?.installed ?? []).filter((m) => !SUGGESTED_IDS.has(m.id)),
    [catalog],
  );

  const matches = useMemo(() => {
    const needle = query.trim().toLowerCase();
    const rest = (catalog?.registry ?? []).filter(
      (m) => !installedIds.has(m.id) && !SUGGESTED_IDS.has(m.id),
    );
    if (!needle) return rest;
    return rest.filter((m) => m.id.toLowerCase().includes(needle));
  }, [catalog, installedIds, query]);

  // A new search is a new list, so paging starts over. Adjusting during
  // render rather than in an effect keeps it from painting the old page
  // for a frame first.
  if (pagedQuery !== query) {
    setPagedQuery(query);
    setPage(0);
  }

  const pageCount = Math.max(1, Math.ceil(matches.length / PAGE_SIZE));
  const safePage = Math.min(page, pageCount - 1);
  const pageItems = useMemo(
    () => matches.slice(safePage * PAGE_SIZE, safePage * PAGE_SIZE + PAGE_SIZE),
    [matches, safePage],
  );

  const handleDownload = useCallback(
    async (modelId: string) => {
      setBusyModel(modelId);
      toast.info(
        `Downloading ${modelId} on the server — this can take several minutes.`,
      );
      try {
        const message = await invoke<string>(
          "api_remote_whisper_download_model",
          { config: requestConfigRef.current(baseUrl), modelId },
        );
        toast.success(message);
        await refresh(baseUrl, true);
      } catch (err) {
        toast.error(err instanceof Error ? err.message : String(err));
      } finally {
        setBusyModel(null);
      }
    },
    [baseUrl, refresh],
  );

  const handleDelete = useCallback(
    async (modelId: string) => {
      setPendingDelete(null);
      setBusyModel(modelId);
      try {
        const message = await invoke<string>(
          "api_remote_whisper_delete_model",
          { config: requestConfigRef.current(baseUrl), modelId },
        );
        toast.success(message);
        await refresh(baseUrl, true);
      } catch (err) {
        toast.error(err instanceof Error ? err.message : String(err));
      } finally {
        setBusyModel(null);
      }
    },
    [baseUrl, refresh],
  );

  if (!baseUrl) {
    return (
      <Text size="small" tone="muted">
        Enter the endpoint URL to browse the models installed on the server.
      </Text>
    );
  }

  if (loading && !catalog) {
    return (
      <div className="flex items-center gap-2">
        <Spinner size="sm" />
        <Text size="small" tone="muted">
          Reading the server&rsquo;s model list&hellip;
        </Text>
      </div>
    );
  }

  if (error && !catalog) {
    return (
      <Alert variant="destructive">
        <AlertTitle>Couldn&rsquo;t reach the model API</AlertTitle>
        <AlertDescription>
          <p>{error}</p>
          <div className="mt-2">
            <Button
              type="button"
              variant="outline"
              size="sm"
              onClick={() => refresh(baseUrl, true)}
            >
              <RefreshCw className="mr-2 size-4" />
              Retry
            </Button>
          </div>
        </AlertDescription>
      </Alert>
    );
  }

  if (catalog && !catalog.supported) {
    return (
      <Text size="small" tone="muted">
        This server handles transcription but doesn&rsquo;t expose a
        model-management API, so its models can&rsquo;t be browsed from here.
        Transcription still works — type the model name in the field above, or
        leave the default if the server ignores it.
      </Text>
    );
  }

  if (!catalog) return null;

  return (
    <div className="space-y-3">
      <div className="flex items-center justify-between gap-2">
        <Text size="small" tone="muted">
          {catalog.installed.length} installed &middot; {catalog.registry.length}{" "}
          available
        </Text>
        <Button
          type="button"
          variant="ghost"
          size="sm"
          onClick={() => refresh(baseUrl, true)}
          disabled={loading}
        >
          <RefreshCw
            className={`mr-2 size-4 ${loading ? "animate-spin" : ""}`}
          />
          Refresh
        </Button>
      </div>

      <div className="space-y-3">
        {SUGGESTED_MODELS.map((suggested) => (
          <RemoteModelCard
            key={suggested.id}
            id={suggested.id}
            displayName={suggested.label}
            tagline={suggested.tagline}
            icon={suggested.icon}
            accuracy={suggested.accuracy}
            speed={suggested.speed}
            recommended={suggested.recommended ?? false}
            installed={installedIds.has(suggested.id)}
            loaded={loadedIds.has(suggested.id)}
            selected={selectedModel === suggested.id}
            busy={busyModel === suggested.id}
            disabled={busyModel !== null}
            onSelect={onModelSelect}
            onDownload={handleDownload}
            onDelete={setPendingDelete}
          />
        ))}

        {otherInstalled.map((model) => (
          <RemoteModelCard
            key={model.id}
            id={model.id}
            displayName={model.id.split("/").pop() ?? model.id}
            tagline={
              model.language && model.language.length > 1
                ? `${model.owned_by} · ${model.language.length} languages`
                : model.owned_by
            }
            icon="📊"
            recommended={false}
            installed
            loaded={loadedIds.has(model.id)}
            selected={selectedModel === model.id}
            busy={busyModel === model.id}
            disabled={busyModel !== null}
            onSelect={onModelSelect}
            onDownload={handleDownload}
            onDelete={setPendingDelete}
          />
        ))}
      </div>

      <Accordion type="single" collapsible className="w-full">
        <AccordionItem value="all-models">
          <AccordionTrigger>
            <span className="text-lg">All models on the server</span>
          </AccordionTrigger>
          <AccordionContent>
            <div className="space-y-3 pt-4">
              <div className="relative">
                <Search
                  className="
                    absolute top-1/2 left-3 size-4 -translate-y-1/2
                    text-muted-foreground
                  "
                />
                <Input
                  className="pl-9 focus:border-info focus:ring-1 focus:ring-info"
                  value={query}
                  onChange={(e) => setQuery(e.target.value)}
                  placeholder="Search — e.g. large-v3, distil, tiny, .en"
                  spellCheck={false}
                />
              </div>

              {matches.length === 0 ? (
                <Text size="small" tone="muted">
                  Nothing else matches that search.
                </Text>
              ) : (
                <>
                  <div className="divide-y divide-border rounded-lg border">
                    {pageItems.map((model) => (
                      <RegistryRow
                        key={model.id}
                        id={model.id}
                        busy={busyModel === model.id}
                        disabled={busyModel !== null}
                        onDownload={handleDownload}
                      />
                    ))}
                  </div>

                  <div className="flex items-center justify-between gap-2">
                    <Button
                      type="button"
                      variant="outline"
                      size="sm"
                      disabled={safePage === 0}
                      onClick={() => setPage(safePage - 1)}
                    >
                      <ChevronLeft className="mr-1 size-4" />
                      Previous
                    </Button>
                    <Text size="small" tone="muted">
                      Page {safePage + 1} of {pageCount} &middot;{" "}
                      {matches.length} models
                    </Text>
                    <Button
                      type="button"
                      variant="outline"
                      size="sm"
                      disabled={safePage >= pageCount - 1}
                      onClick={() => setPage(safePage + 1)}
                    >
                      Next
                      <ChevronRight className="ml-1 size-4" />
                    </Button>
                  </div>
                </>
              )}

              <Text size="small" tone="muted">
                Downloads run on the server and are stored there — nothing is
                transferred to this machine. Large models take a few minutes
                and report no progress until they finish.
              </Text>
            </div>
          </AccordionContent>
        </AccordionItem>
      </Accordion>

      {selectedModel && (
        <div className="pt-2 text-center text-sm text-muted-foreground">
          Using {selectedModel} for transcription
        </div>
      )}

      <Dialog
        open={pendingDelete !== null}
        onOpenChange={(open) => {
          if (!open) setPendingDelete(null);
        }}
      >
        <DialogContent>
          <DialogHeader>
            <DialogTitle>Delete this model from the server?</DialogTitle>
          </DialogHeader>
          <p className="text-sm text-muted-foreground">
            <span className="font-medium text-foreground">{pendingDelete}</span>{" "}
            will be removed from {baseUrl}. Any other client using that server
            loses it too, and re-downloading means pulling the weights again.
          </p>
          <DialogFooter>
            <Button variant="ghost" onClick={() => setPendingDelete(null)}>
              Cancel
            </Button>
            <Button
              variant="destructive"
              onClick={() => pendingDelete && handleDelete(pendingDelete)}
            >
              Delete
            </Button>
          </DialogFooter>
        </DialogContent>
      </Dialog>
    </div>
  );
});

interface RemoteModelCardProps {
  id: string;
  displayName: string;
  tagline: string;
  icon: string;
  /** Omitted for models we have no curated characterisation of. */
  accuracy?: string;
  speed?: string;
  recommended: boolean;
  installed: boolean;
  /** Held in the server's memory right now, per `/api/ps`. */
  loaded: boolean;
  selected: boolean;
  busy: boolean;
  /** Another model is mid-download; the server pulls one at a time. */
  disabled: boolean;
  onSelect: (id: string) => void;
  onDownload: (id: string) => void;
  onDelete: (id: string) => void;
}

/** A model tile, laid out like the local manager's `ModelCard`. */
function RemoteModelCard({
  id,
  displayName,
  tagline,
  icon,
  accuracy,
  speed,
  recommended,
  installed,
  loaded,
  selected,
  busy,
  disabled,
  onSelect,
  onDownload,
  onDelete,
}: RemoteModelCardProps) {
  return (
    <div
      role={installed ? "button" : undefined}
      tabIndex={installed ? 0 : undefined}
      aria-pressed={installed ? selected : undefined}
      onKeyDown={
        installed
          ? (e) => {
              if (e.key === "Enter" || e.key === " ") {
                e.preventDefault();
                onSelect(id);
              }
            }
          : undefined
      }
      onClick={() => {
        if (installed) onSelect(id);
      }}
      className={`
        group relative rounded-lg border-2
        ${
          selected && installed
            ? "border-info bg-info/10"
            : installed
              ? "border-border bg-background"
              : "border-border bg-muted"
        }
        ${
          installed
            ? `
              cursor-pointer
              focus-visible:ring-2 focus-visible:ring-ring
              focus-visible:outline-none
            `
            : "cursor-default"
        }
      `}
    >
      {recommended && (
        <div className="
          absolute -top-2 -right-2 rounded-full bg-info px-2 py-0.5 text-sm
          font-medium text-white
        ">
          Recommended
        </div>
      )}

      <div className="p-3">
        <div className="flex items-start justify-between">
          <div className="min-w-0 flex-1">
            <div className="mb-2 flex flex-wrap items-center gap-2">
              <span className="text-2xl">{icon}</span>
              <Heading level={3}>{displayName}</Heading>
              <span className="text-sm text-muted-foreground">•</span>
              <span className="text-sm text-muted-foreground">{tagline}</span>
              {selected && installed && (
                <span className="
                  flex items-center gap-1 rounded-full bg-info px-2 py-0.5
                  text-sm font-medium text-white
                ">
                  ✓
                </span>
              )}
              {loaded && (
                <span className="
                  rounded-full bg-success-muted px-2 py-0.5 text-sm
                  text-success
                ">
                  In memory
                </span>
              )}
            </div>

            <div className="
              mt-1.5 ml-9 flex flex-wrap items-center gap-x-4 text-sm
              text-muted-foreground
            ">
              {accuracy && (
                <span className="flex items-center space-x-1">
                  <span>🎯</span>
                  <span>{accuracy} accuracy</span>
                </span>
              )}
              {speed && (
                <span className="flex items-center space-x-1">
                  <span>⚡</span>
                  <span>{speed} processing</span>
                </span>
              )}
              <span className="truncate" title={id}>
                {id}
              </span>
            </div>
          </div>

          <div className="ml-4 flex shrink-0 items-center gap-2">
            {busy && <Spinner size="sm" />}

            {!busy && installed && (
              <>
                <div className="flex items-center gap-1.5 text-success">
                  <div className="size-2 rounded-full bg-success"></div>
                  <span className="text-sm font-medium">Ready</span>
                </div>
                {/* Hover-revealed delete, as on the local cards: always
                    rendered, visibility flipped by `group-hover` so there's
                    no per-hover React work. */}
                <button
                  type="button"
                  onClick={(e) => {
                    e.stopPropagation();
                    onDelete(id);
                  }}
                  className="
                    pointer-events-none p-1 text-muted-foreground/70 opacity-0
                    hover:text-destructive
                    group-hover:pointer-events-auto group-hover:opacity-100
                  "
                  title="Delete this model from the server"
                >
                  <Trash2 className="size-4" />
                </button>
              </>
            )}

            {!busy && !installed && (
              <Button
                size="sm"
                disabled={disabled}
                onClick={(e) => {
                  e.stopPropagation();
                  onDownload(id);
                }}
                className="bg-info text-white hover:bg-info/90"
              >
                Download
              </Button>
            )}
          </div>
        </div>
      </div>
    </div>
  );
}

interface RegistryRowProps {
  id: string;
  busy: boolean;
  disabled: boolean;
  onDownload: (id: string) => void;
}

/**
 * One line in the full browser.
 *
 * Deliberately not a `RemoteModelCard`: the registry exposes no size,
 * accuracy, or speed for a model it hasn't fetched yet, so a full tile
 * would be three-quarters empty space.
 */
function RegistryRow({ id, busy, disabled, onDownload }: RegistryRowProps) {
  return (
    <div className="flex items-center gap-3 px-3 py-2">
      <span className="min-w-0 flex-1 truncate text-sm" title={id}>
        {id}
      </span>
      {busy ? (
        <Spinner size="sm" />
      ) : (
        <Button
          type="button"
          variant="outline"
          size="sm"
          disabled={disabled}
          onClick={() => onDownload(id)}
        >
          <Download className="mr-2 size-4" />
          Download
        </Button>
      )}
    </div>
  );
}
