//! Score speaker diarization against hand-labelled recordings.
//!
//! ```text
//! cargo run --release --bin diarization_eval -- \
//!     --audio meeting.wav --reference meeting.rttm \
//!     [--manifest eval.tsv] [--models-dir DIR] \
//!     [--embedding-model FILE|all]... [--method online|hac|offline|app|all] \
//!     [--threshold 0.55] [--refine-threshold 0.55] [--offline-threshold 0.5] \
//!     [--collar 0.25] [--rttm-out DIR]
//! ```
//!
//! A manifest is a file of `audio<TAB>reference` lines (blank lines and `#`
//! comments ignored; relative paths resolve against the manifest). The
//! reference is RTTM (`.rttm`) or an Audacity label export (anything else),
//! so a meeting can be labelled in Audacity: one label per speaker turn,
//! named after the speaker.
//!
//! Methods, mirroring the app:
//! - `online` — the live clusterer, one VAD segment at a time.
//! - `hac`    — the post-recording HAC refinement over the online history.
//! - `offline` — sherpa-onnx pyannote diarization of the whole file.
//! - `app`    — offline turns regrouping the online history: what the app
//!   saves after a recording when the segmentation model is present.
//!
//! Models come from `--models-dir` (default: the app's
//! `~/.local/share/com.meetily.ai/speaker_models`), which must also hold
//! `silero_vad.onnx`. `--embedding-model all` runs every downloaded
//! embedding model, for picking one.

use anyhow::{anyhow, Context, Result};
use std::collections::BTreeMap;
use std::path::{Path, PathBuf};
use std::sync::Arc;

use app_lib::audio::{decode_audio_file, vad};
use app_lib::speaker_diarization::der::{compute_der, load_reference, to_rttm, DerReport, LabeledSegment};
use app_lib::speaker_diarization::model::{self, EMBEDDING_MODELS};
use app_lib::speaker_diarization::offline::{self, OFFLINE_CLUSTER_THRESHOLD};
use app_lib::speaker_diarization::{
    Diarizer, DiarizerOptions, SegmentInfo, SpeakerEmbedder, DEFAULT_CLUSTER_THRESHOLD,
    REFINE_CLUSTER_THRESHOLD,
};

const METHODS: [&str; 4] = ["online", "hac", "offline", "app"];
/// Same VAD hangover as retranscription/import.
const VAD_REDEMPTION_TIME_MS: u32 = 800;
/// Long VAD segments are cut into pieces of at most this length before
/// embedding, roughly as the live pipeline's chunking does.
const MAX_SEGMENT_SECS: f64 = 15.0;

struct Args {
    items: Vec<(PathBuf, PathBuf)>,
    models_dir: PathBuf,
    embedding_models: Vec<String>,
    methods: Vec<String>,
    threshold: f32,
    refine_threshold: f32,
    offline_threshold: f32,
    collar: f64,
    rttm_out: Option<PathBuf>,
}

fn usage() -> ! {
    eprintln!("{}", include_str!("diarization_eval.rs")
        .lines()
        .take_while(|l| l.starts_with("//!"))
        .map(|l| l.trim_start_matches("//!").strip_prefix(' ').unwrap_or(""))
        .collect::<Vec<_>>()
        .join("\n"));
    std::process::exit(2);
}

fn parse_args() -> Result<Args> {
    let mut audio = None;
    let mut reference = None;
    let mut manifest = None;
    let mut models_dir = dirs::data_dir().map(|d| d.join("com.meetily.ai").join("speaker_models"));
    let mut embedding_models = Vec::new();
    let mut methods = Vec::new();
    let mut threshold = DEFAULT_CLUSTER_THRESHOLD;
    let mut refine_threshold = REFINE_CLUSTER_THRESHOLD;
    let mut offline_threshold = OFFLINE_CLUSTER_THRESHOLD;
    let mut collar = 0.25;
    let mut rttm_out = None;

    let mut it = std::env::args().skip(1);
    while let Some(flag) = it.next() {
        if flag == "-h" || flag == "--help" {
            usage();
        }
        let mut value = || it.next().ok_or_else(|| anyhow!("{} needs a value", flag));
        match flag.as_str() {
            "--audio" => audio = Some(PathBuf::from(value()?)),
            "--reference" => reference = Some(PathBuf::from(value()?)),
            "--manifest" => manifest = Some(PathBuf::from(value()?)),
            "--models-dir" => models_dir = Some(PathBuf::from(value()?)),
            "--embedding-model" => embedding_models.push(value()?),
            "--method" => methods.push(value()?),
            "--threshold" => threshold = value()?.parse()?,
            "--refine-threshold" => refine_threshold = value()?.parse()?,
            "--offline-threshold" => offline_threshold = value()?.parse()?,
            "--collar" => collar = value()?.parse()?,
            "--rttm-out" => rttm_out = Some(PathBuf::from(value()?)),
            other => return Err(anyhow!("Unknown argument {} (see --help)", other)),
        }
    }

    let mut items = Vec::new();
    match (audio, reference) {
        (Some(a), Some(r)) => items.push((a, r)),
        (None, None) => {}
        _ => return Err(anyhow!("--audio and --reference go together")),
    }
    if let Some(manifest) = manifest {
        items.extend(read_manifest(&manifest)?);
    }
    if items.is_empty() {
        usage();
    }

    let models_dir = models_dir.ok_or_else(|| anyhow!("No --models-dir and no data dir"))?;
    model::set_models_dir_path(models_dir.clone());
    if embedding_models.is_empty() {
        embedding_models.push(model::DEFAULT_MODEL_FILENAME.to_string());
    }
    if embedding_models.iter().any(|m| m == "all") {
        embedding_models = EMBEDDING_MODELS
            .iter()
            .map(|m| m.filename.to_string())
            .filter(|f| model::model_is_ready(&models_dir.join(f)))
            .collect();
    }
    if methods.is_empty() || methods.iter().any(|m| m == "all") {
        methods = METHODS.iter().map(|m| m.to_string()).collect();
    }
    if let Some(bad) = methods.iter().find(|m| !METHODS.contains(&m.as_str())) {
        return Err(anyhow!("Unknown method {}", bad));
    }

    Ok(Args {
        items,
        models_dir,
        embedding_models,
        methods,
        threshold,
        refine_threshold,
        offline_threshold,
        collar,
        rttm_out,
    })
}

fn read_manifest(path: &Path) -> Result<Vec<(PathBuf, PathBuf)>> {
    let base = path.parent().unwrap_or(Path::new("."));
    let text = std::fs::read_to_string(path)
        .with_context(|| format!("Failed to read manifest {}", path.display()))?;
    text.lines()
        .map(str::trim)
        .filter(|l| !l.is_empty() && !l.starts_with('#'))
        .map(|l| {
            let (a, r) = l
                .split_once('\t')
                .ok_or_else(|| anyhow!("Manifest line needs audio<TAB>reference: {}", l))?;
            Ok((base.join(a.trim()), base.join(r.trim())))
        })
        .collect()
}

/// VAD segments of the file, cut to at most [`MAX_SEGMENT_SECS`].
fn speech_segments(samples: &[f32]) -> Result<Vec<(f64, f64, Vec<f32>)>> {
    let max = (MAX_SEGMENT_SECS * 16_000.0) as usize;
    let mut out = Vec::new();
    for s in vad::get_speech_chunks(samples, VAD_REDEMPTION_TIME_MS)? {
        let start = s.start_timestamp_ms / 1000.0;
        for (i, piece) in s.samples.chunks(max).enumerate() {
            let a = start + (i * max) as f64 / 16_000.0;
            out.push((a, a + piece.len() as f64 / 16_000.0, piece.to_vec()));
        }
    }
    Ok(out)
}

struct FileData {
    name: String,
    samples: Vec<f32>,
    segments: Vec<(f64, f64, Vec<f32>)>,
    reference: Vec<LabeledSegment>,
}

fn evaluate_model(
    args: &Args,
    file: &FileData,
    embedding_model: &str,
) -> Result<Vec<(String, DerReport, Vec<LabeledSegment>)>> {
    let path = args.models_dir.join(embedding_model);
    let wants = |m: &str| args.methods.iter().any(|x| x == m);
    let mut results = Vec::new();

    let needs_offline = wants("offline") || wants("app");
    let turns = if needs_offline {
        let seg = args.models_dir.join(model::SEGMENTATION_MODEL_FILENAME);
        Some(offline::diarize_samples_with(
            &file.samples,
            &seg,
            &path,
            args.offline_threshold,
        )?)
    } else {
        None
    };
    if wants("offline") {
        let hyp: Vec<LabeledSegment> = turns
            .iter()
            .flatten()
            .map(|t| LabeledSegment::new(t.start, t.end, format!("spk{}", t.speaker)))
            .collect();
        results.push(("offline".to_string(), compute_der(&file.reference, &hyp, args.collar), hyp));
    }

    if wants("online") || wants("hac") || wants("app") {
        let embedder = SpeakerEmbedder::from_path(&path, 4)?;
        let options = DiarizerOptions {
            cluster_threshold: args.threshold,
            refine_threshold: args.refine_threshold,
            ..DiarizerOptions::default()
        };
        let diarizer = Diarizer::new(Arc::new(embedder), embedding_model.to_string(), options, None);
        let mut times = BTreeMap::new();
        for (seq, (start, end, samples)) in file.segments.iter().enumerate() {
            let info = SegmentInfo {
                start_secs: *start,
                end_secs: *end,
                source: None,
            };
            if diarizer.process(seq as u64, samples, None, info).is_ok() {
                times.insert(seq as u64, (*start, *end));
            }
        }
        let labelled = |labels: Vec<(u64, String)>| -> Vec<LabeledSegment> {
            labels
                .into_iter()
                .filter_map(|(seq, label)| {
                    times.get(&seq).map(|(a, b)| LabeledSegment::new(*a, *b, label))
                })
                .collect()
        };
        if wants("online") {
            let hyp = labelled(
                diarizer
                    .export_history()
                    .into_iter()
                    .map(|r| (r.sequence_id, r.label))
                    .collect(),
            );
            results.push(("online".to_string(), compute_der(&file.reference, &hyp, args.collar), hyp));
        }
        if wants("hac") {
            let hyp = labelled(
                diarizer
                    .refine_with_turns(None)
                    .into_iter()
                    .map(|a| (a.sequence_id, a.speaker))
                    .collect(),
            );
            results.push(("hac".to_string(), compute_der(&file.reference, &hyp, args.collar), hyp));
        }
        if wants("app") {
            let hyp = labelled(
                diarizer
                    .refine_with_turns(turns.as_deref())
                    .into_iter()
                    .map(|a| (a.sequence_id, a.speaker))
                    .collect(),
            );
            results.push(("app".to_string(), compute_der(&file.reference, &hyp, args.collar), hyp));
        }
    }
    Ok(results)
}

fn print_row(file: &str, model: &str, method: &str, r: &DerReport) {
    println!(
        "{:<24} {:<44} {:<8} {:>6.1}% {:>6.1}% {:>6.1}% {:>6.1}%  {:>2}/{:<2}",
        file,
        model,
        method,
        r.der() * 100.0,
        r.missed_secs / r.reference_secs.max(1e-9) * 100.0,
        r.false_alarm_secs / r.reference_secs.max(1e-9) * 100.0,
        r.confusion_rate() * 100.0,
        r.hypothesis_speakers,
        r.reference_speakers,
    );
}

fn main() -> Result<()> {
    env_logger::Builder::from_env(env_logger::Env::default().default_filter_or("warn")).init();
    let args = parse_args()?;
    if !model::model_is_ready(&args.models_dir.join(model::SILERO_VAD_FILENAME)) {
        return Err(anyhow!(
            "{} not found in {}",
            model::SILERO_VAD_FILENAME,
            args.models_dir.display()
        ));
    }
    if let Some(dir) = &args.rttm_out {
        std::fs::create_dir_all(dir)?;
    }

    println!(
        "{:<24} {:<44} {:<8} {:>7} {:>7} {:>7} {:>7}  {}",
        "file", "embedding model", "method", "DER", "miss", "FA", "conf", "hyp/ref spk"
    );
    let mut totals: BTreeMap<(String, String), DerReport> = BTreeMap::new();
    for (audio, reference) in &args.items {
        let name = audio
            .file_stem()
            .map(|s| s.to_string_lossy().into_owned())
            .unwrap_or_default();
        let samples = decode_audio_file(audio)
            .with_context(|| format!("Failed to decode {}", audio.display()))?
            .to_whisper_format();
        let file = FileData {
            segments: speech_segments(&samples)?,
            reference: load_reference(reference)?,
            samples,
            name,
        };

        for embedding_model in &args.embedding_models {
            let results = match evaluate_model(&args, &file, embedding_model) {
                Ok(r) => r,
                Err(e) => {
                    eprintln!("{} / {}: {:#}", file.name, embedding_model, e);
                    continue;
                }
            };
            for (method, report, hyp) in results {
                print_row(&file.name, embedding_model, &method, &report);
                if let Some(dir) = &args.rttm_out {
                    let stem = embedding_model.trim_end_matches(".onnx");
                    let out = dir.join(format!("{}.{}.{}.rttm", file.name, stem, method));
                    std::fs::write(out, to_rttm(&file.name, &hyp))?;
                }
                let t = totals
                    .entry((embedding_model.clone(), method))
                    .or_default();
                t.reference_secs += report.reference_secs;
                t.missed_secs += report.missed_secs;
                t.false_alarm_secs += report.false_alarm_secs;
                t.confusion_secs += report.confusion_secs;
            }
        }
    }

    if args.items.len() > 1 {
        println!();
        for ((embedding_model, method), r) in &totals {
            print_row("TOTAL", embedding_model, method, r);
        }
    }
    Ok(())
}
