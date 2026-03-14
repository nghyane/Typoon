pub mod lazy;

use std::path::PathBuf;

use anyhow::{Context, Result};
use hf_hub::api::tokio::Api;

const HF_REPO: &str = "nghyane/typoon-models";

/// Model files available in the hub.
pub enum Model {
    ComicTextDetector,
    PpocrDet,
    PpocrRec,
    PpocrKeys,
    PpocrRecMultilingual,
    MangaOcrEncoder,
    MangaOcrDecoder,
    MangaOcrVocab,
    Lama,
}

impl Model {
    /// Returns `(hf_repo, filename)` for this model.
    fn source(&self) -> (&'static str, &'static str) {
        (HF_REPO, self.default_filename())
    }

    fn default_filename(&self) -> &'static str {
        match self {
            Self::ComicTextDetector => "comic-text-detector.onnx",
            Self::PpocrDet => "ppocr_det.onnx",
            Self::PpocrRec => "ppocr_rec.onnx",
            Self::PpocrKeys => "ppocr_keys.txt",
            Self::PpocrRecMultilingual => "ppocr_rec_multilingual.onnx",
            Self::MangaOcrEncoder => "encoder_model.onnx",
            Self::MangaOcrDecoder => "decoder_model.onnx",
            Self::MangaOcrVocab => "vocab.txt",
            Self::Lama => "lama_fp32.onnx",
        }
    }
}

/// Resolve a model file: check local `models_dir` first, fall back to HuggingFace Hub download.
pub async fn resolve(models_dir: &str, model: Model) -> Result<PathBuf> {
    let (hf_repo, filename) = model.source();

    // 1. Check local override
    let local = std::path::Path::new(models_dir).join(filename);
    if local.exists() {
        tracing::debug!("Model {filename}: using local {}", local.display());
        return Ok(local);
    }

    // 2. Download from HuggingFace Hub (cached automatically)
    tracing::info!("Model {filename}: downloading from {hf_repo}...");
    let api = Api::new().context("Failed to create HF Hub API client")?;
    let repo = api.model(hf_repo.to_string());
    let path = repo
        .get(filename)
        .await
        .with_context(|| format!("Failed to download {filename} from {hf_repo}"))?;
    tracing::info!("Model {filename}: cached at {}", path.display());
    Ok(path)
}

/// Resolve a model file, returning None if unavailable (for optional models).
pub async fn resolve_optional(models_dir: &str, model: Model) -> Option<PathBuf> {
    match resolve(models_dir, model).await {
        Ok(path) => Some(path),
        Err(e) => {
            tracing::warn!("Optional model unavailable: {e}");
            None
        }
    }
}


