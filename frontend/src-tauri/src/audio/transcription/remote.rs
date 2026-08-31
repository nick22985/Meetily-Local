// audio/transcription/remote.rs
//
// Remote Whisper backend: offloads inference to an OpenAI-compatible
// `/v1/audio/transcriptions` endpoint on another machine, so a box with a
// weak GPU can still run the capture/VAD/mixing pipeline locally while a
// beefier machine does the decoding.
//
// The wire format is deliberately the OpenAI audio-transcription shape,
// which whisper.cpp's own `whisper-server` speaks natively (as do Speaches,
// faster-whisper-server, and OpenAI itself). That means "point it at another
// machine" needs no custom server on the other end.
//
// Everything the pipeline hands us is already f32 mono 16 kHz (the same
// buffers the in-process engine gets), so the only conversion here is
// wrapping them in a 16-bit PCM WAV container. At 16 kHz mono that's
// ~32 kB/s of speech, and VAD has already dropped the silence — trivial on
// a LAN.

use anyhow::{anyhow, Result};
use log::{debug, info, warn};
use serde::{Deserialize, Serialize};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};
use std::time::Duration;

/// Provider string persisted in `transcript_settings.provider` for this backend.
pub const REMOTE_WHISPER_PROVIDER: &str = "remoteWhisper";

/// `app_settings` key holding the [`RemoteWhisperConfig`] JSON blob.
///
/// The endpoint URL has nowhere to live in the fixed-column
/// `transcript_settings` table, and adding a column would need a migration
/// for a setting only this provider uses — so the whole config (API key
/// included) goes in the generic key/value store instead.
pub const REMOTE_WHISPER_SETTING_KEY: &str = "remoteWhisper.config";

/// Sample rate every backend receives audio at (the pipeline resamples to
/// this before transcription).
const TRANSCRIBE_SAMPLE_RATE: u32 = 16000;

/// Chunks shorter than this are reported as "partial", matching the local
/// engine's own threshold so downstream consumers behave identically.
const PARTIAL_THRESHOLD_SECONDS: f64 = 15.0;

/// How long a model-download request may run.
///
/// The server pulls multi-GB weights from HuggingFace while holding the
/// connection open, with no progress channel, so this has to be far past any
/// sane per-chunk transcription timeout.
const MODEL_DOWNLOAD_TIMEOUT: Duration = Duration::from_secs(60 * 60);

/// Task filter for the model-management routes. The same server can host
/// text-to-speech models, which are useless here.
const ASR_TASK: &str = "automatic-speech-recognition";

/// Confidence reported when the server returns no decoder signals at all.
///
/// The hallucination screen in [`worker::is_likely_hallucination`] trusts
/// anything at or above 0.5, so this reads as "no evidence either way" and
/// leaves the text intact. Servers that do return `avg_logprob` /
/// `no_speech_prob` (whisper.cpp's `verbose_json` does) get the same
/// confidence math as the in-process engine instead.
///
/// [`worker::is_likely_hallucination`]: super::worker::is_likely_hallucination
const NEUTRAL_CONFIDENCE: f32 = 0.5;

fn default_timeout_secs() -> u64 {
    120
}

/// Persisted configuration for the remote transcription endpoint.
#[derive(Clone, Serialize, Deserialize)]
pub struct RemoteWhisperConfig {
    /// Server root (e.g. `http://gpubox.lan:8080`). The
    /// `/v1/audio/transcriptions` suffix is appended if absent — see
    /// [`RemoteWhisperClient::transcription_url`].
    #[serde(rename = "baseUrl")]
    pub base_url: String,
    /// Sent as `Authorization: Bearer …` when set. Self-hosted
    /// `whisper-server` instances generally don't need one.
    #[serde(rename = "apiKey", default)]
    pub api_key: Option<String>,
    /// Model name passed to the server. Self-hosted servers usually ignore
    /// it (they serve whatever was loaded at startup) but the field is
    /// required by the OpenAI schema.
    #[serde(default = "default_remote_model")]
    pub model: String,
    /// Per-request timeout. Generous by default: a cold server may need to
    /// load the model on the first request.
    #[serde(rename = "timeoutSecs", default = "default_timeout_secs")]
    pub timeout_secs: u64,
}

fn default_remote_model() -> String {
    "whisper-1".to_string()
}

/// Redacts `api_key`, since this config is formatted into error messages
/// and logs on the failure paths.
impl std::fmt::Debug for RemoteWhisperConfig {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("RemoteWhisperConfig")
            .field("base_url", &self.base_url)
            .field(
                "api_key",
                &self
                    .api_key
                    .as_deref()
                    .filter(|k| !k.is_empty())
                    .map(|_| "<set>"),
            )
            .field("model", &self.model)
            .field("timeout_secs", &self.timeout_secs)
            .finish()
    }
}

impl Default for RemoteWhisperConfig {
    fn default() -> Self {
        Self {
            base_url: String::new(),
            api_key: None,
            model: default_remote_model(),
            timeout_secs: default_timeout_secs(),
        }
    }
}

/// One segment of an OpenAI `verbose_json` transcription response.
///
/// Every field is optional: servers vary in how much of the schema they
/// populate, and a missing decoder signal degrades to [`NEUTRAL_CONFIDENCE`]
/// rather than failing the request.
#[derive(Debug, Deserialize)]
struct RemoteSegment {
    #[serde(default)]
    text: String,
    #[serde(default)]
    avg_logprob: Option<f32>,
    #[serde(default)]
    no_speech_prob: Option<f32>,
    #[serde(default)]
    tokens: Vec<i64>,
}

/// An OpenAI-compatible transcription response. `segments` is only present
/// for `response_format=verbose_json`; plain `json` responses carry `text`
/// alone.
#[derive(Debug, Deserialize)]
struct RemoteTranscriptionResponse {
    #[serde(default)]
    text: String,
    #[serde(default)]
    segments: Vec<RemoteSegment>,
}

/// One model as reported by the server's model-management API.
///
/// Passed straight through to the frontend, so the field names double as the
/// TypeScript shape. `additionalProperties` is open on the server side and
/// varies by model type; only the fields worth rendering are captured.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RemoteModel {
    /// HuggingFace-style `owner/name`, e.g. `Systran/faster-whisper-small`.
    pub id: String,
    #[serde(default)]
    pub owned_by: String,
    #[serde(default)]
    pub task: Option<String>,
    /// Language codes the model was trained on; `None` on servers that
    /// don't report it.
    #[serde(default)]
    pub language: Option<Vec<String>>,
    #[serde(default)]
    pub created: Option<i64>,
}

/// Everything the model-management UI needs, in one round trip.
#[derive(Debug, Clone, Serialize)]
pub struct RemoteModelCatalog {
    /// False when the endpoint speaks transcription but not model
    /// management — plain whisper.cpp `whisper-server`, for instance. The
    /// UI hides the browser and falls back to the free-text model field.
    pub supported: bool,
    /// Models already downloaded on the server.
    pub installed: Vec<RemoteModel>,
    /// Models the server offers to download.
    pub registry: Vec<RemoteModel>,
    /// Ids currently held in memory (loaded), a subset of `installed`.
    pub loaded: Vec<String>,
}

/// `{"data": [...]}` — the OpenAI list envelope, shared by `/v1/models`
/// and `/v1/registry`.
#[derive(Debug, Deserialize)]
struct ListModelsResponse {
    #[serde(default)]
    data: Vec<RemoteModel>,
}

/// `{"models": [...]}` from `/api/ps`.
#[derive(Debug, Deserialize)]
struct LoadedModelsResponse {
    #[serde(default)]
    models: Vec<String>,
}

/// HTTP client for an OpenAI-compatible transcription endpoint.
///
/// Cheap to clone-by-`Arc` and safe to share across the live worker, the
/// partial-decode task, and batch jobs — `reqwest::Client` pools connections
/// internally, so concurrent chunks reuse the same keep-alive sockets.
#[derive(Debug)]
pub struct RemoteWhisperClient {
    config: RemoteWhisperConfig,
    client: reqwest::Client,
    /// Separate client for model downloads. Sharing `client` would put a
    /// multi-GB download under the per-chunk transcription timeout.
    download_client: reqwest::Client,
    /// Set once the first response arrives without usable decoder signals,
    /// so the "confidence is synthetic" warning is logged once per session
    /// instead of once per chunk.
    warned_no_logprobs: AtomicBool,
}

impl RemoteWhisperClient {
    pub fn new(config: RemoteWhisperConfig) -> Result<Self> {
        let base = config.base_url.trim();
        if base.is_empty() {
            return Err(anyhow!(
                "Remote transcription endpoint is not configured. Set it in Settings → Transcription."
            ));
        }
        if !base.starts_with("http://") && !base.starts_with("https://") {
            return Err(anyhow!(
                "Remote transcription endpoint must start with http:// or https:// (got '{}')",
                base
            ));
        }

        let client = reqwest::Client::builder()
            .timeout(Duration::from_secs(config.timeout_secs.max(5)))
            // A dead host should fail fast enough to surface in the UI
            // rather than sitting on the request timeout.
            .connect_timeout(Duration::from_secs(5))
            .build()
            .map_err(|e| anyhow!("Failed to build HTTP client: {}", e))?;

        let download_client = reqwest::Client::builder()
            .timeout(MODEL_DOWNLOAD_TIMEOUT)
            .connect_timeout(Duration::from_secs(10))
            .build()
            .map_err(|e| anyhow!("Failed to build download client: {}", e))?;

        Ok(Self {
            config,
            client,
            download_client,
            warned_no_logprobs: AtomicBool::new(false),
        })
    }

    pub fn model(&self) -> &str {
        &self.config.model
    }

    pub fn base_url(&self) -> &str {
        self.config.base_url.trim_end_matches('/')
    }

    /// The OpenAI-style API root (`…/v1`) implied by the configured base.
    ///
    /// Accepts the three things people actually paste: a bare server root
    /// (`http://host:8080`), a root with the API prefix already on it
    /// (`http://host:8080/v1`), or the complete transcription endpoint path.
    fn api_root(&self) -> String {
        let base = self.base_url();
        if let Some(root) = base.strip_suffix("/audio/transcriptions") {
            root.to_string()
        } else if base.ends_with("/v1") {
            base.to_string()
        } else {
            format!("{}/v1", base)
        }
    }

    /// The server root, i.e. the API root minus its version prefix.
    ///
    /// Speaches' load/unload routes sit at `/api/ps`, outside `/v1`, so they
    /// can't be resolved against [`api_root`](Self::api_root).
    fn server_root(&self) -> String {
        let root = self.api_root();
        root.strip_suffix("/v1").unwrap_or(&root).to_string()
    }

    fn transcription_url(&self) -> String {
        format!("{}/audio/transcriptions", self.api_root())
    }

    /// Attach the configured bearer token, if there is one.
    fn authed(&self, request: reqwest::RequestBuilder) -> reqwest::RequestBuilder {
        match self.config.api_key.as_deref().filter(|k| !k.is_empty()) {
            Some(key) => request.bearer_auth(key),
            None => request,
        }
    }

    /// Verify the endpoint is reachable and will accept our requests.
    ///
    /// Sends a short burst of silence through the real transcription path
    /// rather than probing a health URL: OpenAI-compatible servers don't
    /// agree on a health endpoint, but they all agree on this one, so this
    /// validates host, path, auth, and model together. The transcript is
    /// discarded — only the status code matters.
    pub async fn health_check(&self) -> Result<()> {
        let silence = vec![0.0f32; (TRANSCRIBE_SAMPLE_RATE / 10) as usize]; // 100ms
        self.post_audio(&silence, None, None)
            .await
            .map(|_| ())
            .map_err(|e| anyhow!("Remote transcription endpoint unreachable: {}", e))
    }

    /// Transcribe a chunk, returning `(text, confidence, is_partial)` with
    /// the same contract as the in-process engine's method of the same name.
    pub async fn transcribe_audio_with_confidence(
        &self,
        audio_data: Vec<f32>,
        language: Option<String>,
        context_prompt: Option<String>,
    ) -> Result<(String, f32, bool)> {
        let duration_seconds = audio_data.len() as f64 / TRANSCRIBE_SAMPLE_RATE as f64;
        let is_partial = duration_seconds < PARTIAL_THRESHOLD_SECONDS;

        // The local engine maps "auto" to whisper's own language detection
        // and "auto-translate" to translation. The OpenAI schema expresses
        // detection as an absent `language` field; translation is a
        // different endpoint entirely, which self-hosted servers don't
        // reliably expose, so it degrades to detection with a warning.
        let language_param = match language.as_deref() {
            Some("auto") | None => None,
            Some("auto-translate") => {
                warn!(
                    "Remote transcription does not support auto-translate; \
                     falling back to language auto-detection"
                );
                None
            }
            Some(lang) => Some(lang.to_string()),
        };

        let response = self
            .post_audio(&audio_data, language_param, context_prompt)
            .await?;

        let text = response.text.trim().to_string();
        let confidence = self.confidence_from(&response);

        debug!(
            "Remote transcription returned {} chars ({:.2}s audio, confidence {:.2})",
            text.len(),
            duration_seconds,
            confidence
        );

        Ok((text, confidence, is_partial))
    }

    /// Derive a confidence score matching the in-process engine's semantics:
    /// a token-count-weighted mean of `mean_token_probability * (1 -
    /// no_speech_probability)` across segments. `avg_logprob` is the mean
    /// log-probability of a segment's tokens, so `exp(avg_logprob)` is the
    /// mean probability the local path computes directly.
    fn confidence_from(&self, response: &RemoteTranscriptionResponse) -> f32 {
        let mut weighted_sum = 0.0f32;
        let mut total_weight = 0.0f32;

        for segment in &response.segments {
            let Some(avg_logprob) = segment.avg_logprob else {
                continue;
            };
            // Segments carry token ids under verbose_json; when they're
            // absent, weight by text length so long segments still
            // dominate the average the way token counts would.
            let weight = if segment.tokens.is_empty() {
                segment.text.trim().len().max(1) as f32
            } else {
                segment.tokens.len() as f32
            };
            let mean_token_prob = avg_logprob.exp().clamp(0.0, 1.0);
            let no_speech = segment.no_speech_prob.unwrap_or(0.0).clamp(0.0, 1.0);
            weighted_sum += mean_token_prob * (1.0 - no_speech) * weight;
            total_weight += weight;
        }

        if total_weight > 0.0 {
            (weighted_sum / total_weight).clamp(0.0, 1.0)
        } else {
            if !self.warned_no_logprobs.swap(true, Ordering::Relaxed) {
                warn!(
                    "Remote transcription server returns no decoder signals \
                     (avg_logprob/no_speech_prob); confidence scores and \
                     low-confidence hallucination filtering are unavailable"
                );
            }
            NEUTRAL_CONFIDENCE
        }
    }

    /// POST one WAV-wrapped chunk and parse the response.
    async fn post_audio(
        &self,
        samples: &[f32],
        language: Option<String>,
        prompt: Option<String>,
    ) -> Result<RemoteTranscriptionResponse> {
        let wav = encode_wav_pcm16(samples, TRANSCRIBE_SAMPLE_RATE);

        let file_part = reqwest::multipart::Part::bytes(wav)
            .file_name("audio.wav")
            .mime_str("audio/wav")
            .map_err(|e| anyhow!("Failed to build audio part: {}", e))?;

        let mut form = reqwest::multipart::Form::new()
            .part("file", file_part)
            .text("model", self.config.model.clone())
            // verbose_json is what carries avg_logprob / no_speech_prob;
            // servers that don't implement it still return a `text` field,
            // which is all the parse strictly requires.
            .text("response_format", "verbose_json")
            // Greedy decode, matching the deterministic settings the local
            // engine uses for its first pass.
            .text("temperature", "0");

        if let Some(lang) = language {
            form = form.text("language", lang);
        }
        if let Some(prompt) = prompt {
            form = form.text("prompt", prompt);
        }

        let response = self
            .authed(self.client.post(self.transcription_url()).multipart(form))
            .send()
            .await
            .map_err(|e| anyhow!("Request to {} failed: {}", self.transcription_url(), e))?;

        let status = response.status();
        let body = response
            .text()
            .await
            .map_err(|e| anyhow!("Failed to read response body: {}", e))?;

        if !status.is_success() {
            // Server error bodies are often multi-line HTML; keep the log
            // readable while preserving enough to diagnose auth/model errors.
            let snippet: String = body.chars().take(500).collect();
            return Err(anyhow!(
                "Remote transcription failed with HTTP {}: {}",
                status,
                snippet.trim()
            ));
        }

        serde_json::from_str::<RemoteTranscriptionResponse>(&body).map_err(|e| {
            let snippet: String = body.chars().take(200).collect();
            anyhow!(
                "Could not parse response from {} ({}). Body started: {}",
                self.transcription_url(),
                e,
                snippet.trim()
            )
        })
    }

    // ----- model management -------------------------------------------
    //
    // Speaches (and API-compatible servers) expose the server's own model
    // store over HTTP, so the remote endpoint can be managed from here the
    // same way the local Whisper models are — browse what's installed,
    // download something bigger, drop what isn't used. Servers that only
    // implement transcription 404 on these routes; see
    // [`supports_model_management`](Self::supports_model_management).

    /// Fetch everything the model browser needs in one round trip.
    ///
    /// A server without the management routes yields
    /// `supported: false` and empty lists rather than an error — the
    /// endpoint is still perfectly usable for transcription, it just can't
    /// be managed from here.
    pub async fn model_catalog(&self) -> Result<RemoteModelCatalog> {
        let registry = match self.list_registry_models().await {
            Ok(models) => models,
            Err(e) => {
                info!(
                    "Remote endpoint {} has no model-management API ({}); \
                     model browsing disabled",
                    self.base_url(),
                    e
                );
                return Ok(RemoteModelCatalog {
                    supported: false,
                    installed: Vec::new(),
                    registry: Vec::new(),
                    loaded: Vec::new(),
                });
            }
        };

        // Installed and loaded are independent of each other; fetch
        // concurrently. `loaded` is an experimental route on Speaches, so a
        // failure there degrades to "nothing known to be loaded" rather
        // than sinking the whole catalog.
        let (installed, loaded) =
            tokio::join!(self.list_installed_models(), self.loaded_model_ids());

        Ok(RemoteModelCatalog {
            supported: true,
            installed: installed?,
            registry,
            loaded: loaded.unwrap_or_default(),
        })
    }

    /// Models already downloaded on the server.
    pub async fn list_installed_models(&self) -> Result<Vec<RemoteModel>> {
        self.list_models_at(&format!("{}/models", self.api_root()))
            .await
    }

    /// Models the server offers to download.
    pub async fn list_registry_models(&self) -> Result<Vec<RemoteModel>> {
        self.list_models_at(&format!("{}/registry", self.api_root()))
            .await
    }

    async fn list_models_at(&self, url: &str) -> Result<Vec<RemoteModel>> {
        let response = self
            .authed(self.client.get(url).query(&[("task", ASR_TASK)]))
            .send()
            .await
            .map_err(|e| anyhow!("Request to {} failed: {}", url, e))?;

        let body = read_body(response, "Listing models").await?;
        let parsed: ListModelsResponse = serde_json::from_str(&body).map_err(|e| {
            let snippet: String = body.chars().take(200).collect();
            anyhow!(
                "Could not parse the model list from {} ({}). Body started: {}",
                url,
                e,
                snippet.trim()
            )
        })?;

        // The `task` query param is advisory — servers that ignore it hand
        // back text-to-speech models too, which would be nonsense here.
        Ok(parsed
            .data
            .into_iter()
            .filter(|m| m.task.as_deref().map_or(true, |t| t == ASR_TASK))
            .collect())
    }

    /// Ids currently held in memory on the server.
    pub async fn loaded_model_ids(&self) -> Result<Vec<String>> {
        let url = format!("{}/api/ps", self.server_root());
        let response = self
            .authed(self.client.get(&url))
            .send()
            .await
            .map_err(|e| anyhow!("Request to {} failed: {}", url, e))?;

        let body = read_body(response, "Listing loaded models").await?;
        let parsed: LoadedModelsResponse = serde_json::from_str(&body)
            .map_err(|e| anyhow!("Could not parse loaded models from {}: {}", url, e))?;
        Ok(parsed.models)
    }

    /// Ask the server to download `model_id` into its own model store.
    ///
    /// The server pulls the weights synchronously and reports no progress,
    /// so this holds the connection for however long that takes — minutes,
    /// for a multi-GB model. Re-downloading something already present is a
    /// no-op the server answers immediately.
    pub async fn download_model(&self, model_id: &str) -> Result<()> {
        validate_model_id(model_id)?;
        let url = format!("{}/models/{}", self.api_root(), model_id);
        info!(
            "⬇️  Asking {} to download model '{}'",
            self.base_url(),
            model_id
        );

        let response = self
            .authed(self.download_client.post(&url))
            .send()
            .await
            .map_err(|e| anyhow!("Download request to {} failed: {}", url, e))?;

        read_body(response, "Model download").await.map(|_| ())
    }

    /// Delete `model_id` from the server's model store.
    pub async fn delete_model(&self, model_id: &str) -> Result<()> {
        validate_model_id(model_id)?;
        let url = format!("{}/models/{}", self.api_root(), model_id);
        warn!(
            "🗑️  Asking {} to delete model '{}'",
            self.base_url(),
            model_id
        );

        let response = self
            .authed(self.client.delete(&url))
            .send()
            .await
            .map_err(|e| anyhow!("Delete request to {} failed: {}", url, e))?;

        read_body(response, "Model deletion").await.map(|_| ())
    }
}

/// Read a management-API response, turning a non-2xx status into an error
/// that prefers the server's own `{"detail": …}` message over the raw body.
///
/// Success bodies are returned as text rather than parsed: the routes are
/// inconsistent about it — `POST /v1/models/{id}` answers an already-present
/// model with `201` and a bare sentence, not JSON.
async fn read_body(response: reqwest::Response, what: &str) -> Result<String> {
    let status = response.status();
    let body = response
        .text()
        .await
        .map_err(|e| anyhow!("Failed to read response body: {}", e))?;

    if status.is_success() {
        return Ok(body);
    }

    let detail = serde_json::from_str::<serde_json::Value>(&body)
        .ok()
        .and_then(|v| v.get("detail")?.as_str().map(str::to_string));

    Err(match detail {
        Some(detail) => anyhow!("{} failed (HTTP {}): {}", what, status, detail),
        None => {
            let snippet: String = body.chars().take(300).collect();
            anyhow!("{} failed (HTTP {}): {}", what, status, snippet.trim())
        }
    })
}

/// Reject model ids that can't be safely interpolated into a URL path.
///
/// Ids are HuggingFace-style `owner/name`, and the slash is part of the
/// route — percent-encoding it would 404 — so the id goes in verbatim and
/// has to be screened for anything that could escape the path instead.
fn validate_model_id(model_id: &str) -> Result<()> {
    let invalid = model_id.is_empty()
        || model_id.starts_with('/')
        || model_id.contains("..")
        || model_id.contains('?')
        || model_id.contains('#')
        || model_id
            .chars()
            .any(|c| c.is_whitespace() || c.is_control());

    if invalid {
        return Err(anyhow!(
            "'{}' is not a valid model id (expected something like 'Systran/faster-whisper-small')",
            model_id
        ));
    }
    Ok(())
}

/// Wrap mono f32 samples in a 16-bit PCM WAV container.
///
/// Written by hand rather than via a crate for the same reason the recording
/// saver does it: the format we emit is fixed (mono, 16-bit, known rate), so
/// a header builder is shorter than a dependency.
fn encode_wav_pcm16(samples: &[f32], sample_rate: u32) -> Vec<u8> {
    const CHANNELS: u16 = 1;
    const BITS_PER_SAMPLE: u16 = 16;

    let data_len = samples.len() * 2;
    let byte_rate = sample_rate * CHANNELS as u32 * (BITS_PER_SAMPLE as u32 / 8);
    let block_align = CHANNELS * (BITS_PER_SAMPLE / 8);

    let mut out = Vec::with_capacity(44 + data_len);
    out.extend_from_slice(b"RIFF");
    out.extend_from_slice(&((36 + data_len) as u32).to_le_bytes());
    out.extend_from_slice(b"WAVE");
    out.extend_from_slice(b"fmt ");
    out.extend_from_slice(&16u32.to_le_bytes()); // PCM fmt chunk size
    out.extend_from_slice(&1u16.to_le_bytes()); // format tag: PCM
    out.extend_from_slice(&CHANNELS.to_le_bytes());
    out.extend_from_slice(&sample_rate.to_le_bytes());
    out.extend_from_slice(&byte_rate.to_le_bytes());
    out.extend_from_slice(&block_align.to_le_bytes());
    out.extend_from_slice(&BITS_PER_SAMPLE.to_le_bytes());
    out.extend_from_slice(b"data");
    out.extend_from_slice(&(data_len as u32).to_le_bytes());

    for &sample in samples {
        // Clamp before scaling: samples arriving hot from the mixer would
        // otherwise wrap around to the opposite polarity and click.
        let scaled = (sample.clamp(-1.0, 1.0) * i16::MAX as f32) as i16;
        out.extend_from_slice(&scaled.to_le_bytes());
    }

    out
}

/// The remote client for the current session, or `None` when the configured
/// provider is local Whisper.
///
/// Mirrors the existing `WHISPER_ENGINE` global so that code paths which
/// can't be handed an engine directly (notably the streaming partial-decode
/// task) can reach the active backend. Set by
/// [`super::engine::get_or_init_transcription_engine`] at recording start.
pub static REMOTE_ENGINE: Mutex<Option<Arc<RemoteWhisperClient>>> = Mutex::new(None);

/// Install `client` as the session's active remote backend.
pub fn set_active_remote(client: Option<Arc<RemoteWhisperClient>>) {
    let describe = client
        .as_ref()
        .map(|c| format!("{} (model '{}')", c.base_url(), c.model()));
    let mut guard = REMOTE_ENGINE.lock().unwrap_or_else(|e| e.into_inner());
    *guard = client;
    match describe {
        Some(target) => info!("🌐 Remote transcription backend active: {}", target),
        None => debug!("Remote transcription backend cleared (using local Whisper)"),
    }
}

/// Get the session's active remote backend, if any.
pub fn active_remote() -> Option<Arc<RemoteWhisperClient>> {
    REMOTE_ENGINE
        .lock()
        .unwrap_or_else(|e| e.into_inner())
        .as_ref()
        .cloned()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn client_with_base(base: &str) -> RemoteWhisperClient {
        RemoteWhisperClient::new(RemoteWhisperConfig {
            base_url: base.to_string(),
            ..Default::default()
        })
        .expect("valid base url")
    }

    #[test]
    fn url_resolution_accepts_the_three_paste_shapes() {
        assert_eq!(
            client_with_base("http://gpubox:8080").transcription_url(),
            "http://gpubox:8080/v1/audio/transcriptions"
        );
        assert_eq!(
            client_with_base("http://gpubox:8080/").transcription_url(),
            "http://gpubox:8080/v1/audio/transcriptions"
        );
        assert_eq!(
            client_with_base("http://gpubox:8080/v1").transcription_url(),
            "http://gpubox:8080/v1/audio/transcriptions"
        );
        assert_eq!(
            client_with_base("http://gpubox:8080/v1/audio/transcriptions").transcription_url(),
            "http://gpubox:8080/v1/audio/transcriptions"
        );
    }

    #[test]
    fn management_routes_resolve_from_any_paste_shape() {
        // /api/ps lives outside /v1, so it has to come off the server root
        // no matter which of the three shapes was configured.
        for base in [
            "http://gpu.lan:8080",
            "http://gpu.lan:8080/",
            "http://gpu.lan:8080/v1",
            "http://gpu.lan:8080/v1/audio/transcriptions",
        ] {
            let client = client_with_base(base);
            assert_eq!(client.api_root(), "http://gpu.lan:8080/v1", "base {}", base);
            assert_eq!(client.server_root(), "http://gpu.lan:8080", "base {}", base);
        }
    }

    #[test]
    fn model_ids_keep_their_slash_in_the_path() {
        // Percent-encoding the owner/name separator would 404: the slash is
        // part of the server's route, not a value to escape.
        let client = client_with_base("http://gpu.lan:8080");
        assert_eq!(
            format!(
                "{}/models/{}",
                client.api_root(),
                "Systran/faster-whisper-small"
            ),
            "http://gpu.lan:8080/v1/models/Systran/faster-whisper-small"
        );
    }

    #[test]
    fn model_id_validation_rejects_path_escapes() {
        assert!(validate_model_id("Systran/faster-whisper-small").is_ok());
        assert!(validate_model_id("deepdml/faster-whisper-large-v3-turbo-ct2").is_ok());

        for bad in [
            "",
            "/etc/passwd",
            "../../secret",
            "a b",
            "x?y",
            "x#y",
            "a\nb",
        ] {
            assert!(validate_model_id(bad).is_err(), "should reject {:?}", bad);
        }
    }

    #[test]
    fn non_asr_models_are_filtered_out() {
        // Servers that ignore the `task` query param hand back their
        // text-to-speech models too.
        let listing = r#"{"data":[
            {"id":"Systran/faster-whisper-small","owned_by":"Systran","task":"automatic-speech-recognition"},
            {"id":"speaches-ai/piper-en","owned_by":"speaches-ai","task":"text-to-speech"},
            {"id":"legacy/no-task-field","owned_by":"legacy"}
        ]}"#;
        let parsed: ListModelsResponse = serde_json::from_str(listing).unwrap();
        let kept: Vec<String> = parsed
            .data
            .into_iter()
            .filter(|m| m.task.as_deref().map_or(true, |t| t == ASR_TASK))
            .map(|m| m.id)
            .collect();
        // A server that reports no task at all is given the benefit of the
        // doubt — filtering it out would empty the list on older builds.
        assert_eq!(
            kept,
            vec!["Systran/faster-whisper-small", "legacy/no-task-field"]
        );
    }

    #[test]
    fn error_bodies_surface_the_servers_own_detail() {
        // Speaches answers an unknown model with {"detail": "..."}; that
        // sentence is far more useful than the status code alone.
        let body = r#"{"detail":"Model 'nick/nope' not found"}"#;
        let detail = serde_json::from_str::<serde_json::Value>(body)
            .ok()
            .and_then(|v| v.get("detail")?.as_str().map(str::to_string));
        assert_eq!(detail.as_deref(), Some("Model 'nick/nope' not found"));
    }

    #[test]
    fn rejects_endpoints_without_a_scheme() {
        let err = RemoteWhisperClient::new(RemoteWhisperConfig {
            base_url: "gpubox:8080".to_string(),
            ..Default::default()
        })
        .unwrap_err()
        .to_string();
        assert!(err.contains("http://"), "unexpected error: {}", err);

        let err = RemoteWhisperClient::new(RemoteWhisperConfig::default())
            .unwrap_err()
            .to_string();
        assert!(err.contains("not configured"), "unexpected error: {}", err);
    }

    #[test]
    fn wav_header_describes_the_payload() {
        let wav = encode_wav_pcm16(&[0.0, 1.0, -1.0], 16000);
        assert_eq!(&wav[0..4], b"RIFF");
        assert_eq!(&wav[8..12], b"WAVE");
        assert_eq!(wav.len(), 44 + 6);
        // RIFF size counts everything after the first 8 bytes.
        assert_eq!(u32::from_le_bytes(wav[4..8].try_into().unwrap()), 36 + 6);
        assert_eq!(u32::from_le_bytes(wav[24..28].try_into().unwrap()), 16000);
        assert_eq!(u32::from_le_bytes(wav[40..44].try_into().unwrap()), 6);
    }

    #[test]
    fn wav_samples_clamp_instead_of_wrapping() {
        // 1.5 must saturate to i16::MAX, not wrap to a negative value.
        let wav = encode_wav_pcm16(&[1.5, -1.5], 16000);
        assert_eq!(
            i16::from_le_bytes(wav[44..46].try_into().unwrap()),
            i16::MAX
        );
        assert_eq!(
            i16::from_le_bytes(wav[46..48].try_into().unwrap()),
            -i16::MAX
        );
    }

    #[test]
    fn confidence_matches_the_local_weighting() {
        let client = client_with_base("http://gpubox:8080");
        // exp(-0.1) ≈ 0.9048 discounted by 10% no-speech ≈ 0.8143.
        let response = RemoteTranscriptionResponse {
            text: "hello".into(),
            segments: vec![RemoteSegment {
                text: "hello".into(),
                avg_logprob: Some(-0.1),
                no_speech_prob: Some(0.1),
                tokens: vec![1, 2, 3],
            }],
        };
        let confidence = client.confidence_from(&response);
        assert!((confidence - 0.8143).abs() < 0.001, "got {}", confidence);
    }

    #[test]
    fn confidence_weights_segments_by_token_count() {
        let client = client_with_base("http://gpubox:8080");
        // A long confident segment should outweigh a short unconfident one.
        let response = RemoteTranscriptionResponse {
            text: "…".into(),
            segments: vec![
                RemoteSegment {
                    text: "long".into(),
                    avg_logprob: Some(0.0), // probability 1.0
                    no_speech_prob: Some(0.0),
                    tokens: vec![0; 9],
                },
                RemoteSegment {
                    text: "short".into(),
                    avg_logprob: Some(-10.0), // probability ~0
                    no_speech_prob: Some(0.0),
                    tokens: vec![0; 1],
                },
            ],
        };
        let confidence = client.confidence_from(&response);
        assert!((confidence - 0.9).abs() < 0.001, "got {}", confidence);
    }

    #[test]
    fn confidence_falls_back_when_the_server_reports_nothing() {
        let client = client_with_base("http://gpubox:8080");
        let response = RemoteTranscriptionResponse {
            text: "hello".into(),
            segments: Vec::new(),
        };
        assert_eq!(client.confidence_from(&response), NEUTRAL_CONFIDENCE);

        // Segments present but with no decoder signals is the same case.
        let response = RemoteTranscriptionResponse {
            text: "hello".into(),
            segments: vec![RemoteSegment {
                text: "hello".into(),
                avg_logprob: None,
                no_speech_prob: None,
                tokens: Vec::new(),
            }],
        };
        assert_eq!(client.confidence_from(&response), NEUTRAL_CONFIDENCE);
    }

    #[test]
    fn plain_json_responses_parse_without_segments() {
        let response: RemoteTranscriptionResponse =
            serde_json::from_str(r#"{"text":"hello world"}"#).expect("parses");
        assert_eq!(response.text, "hello world");
        assert!(response.segments.is_empty());
    }
}
