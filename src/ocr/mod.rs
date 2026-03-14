mod ppocr;
mod manga_ocr;

use anyhow::Result;

use crate::detection::TextRegion;
use crate::model_hub::lazy::LazySession;
use crate::model_hub::{self, Model};

pub struct OcrResult {
    pub text: String,
    pub confidence: f64,
}

pub struct OcrEngine {
    ppocr: Option<ppocr::PpOcrAdapter>,
    manga_ocr: Option<manga_ocr::MangaOcrAdapter>,
}

impl OcrEngine {
    pub async fn new(models_dir: &str) -> Result<Self> {
        let ppocr = match Self::load_ppocr(models_dir).await {
            Ok(adapter) => Some(adapter),
            Err(e) => {
                tracing::warn!("PP-OCR not loaded: {e}");
                None
            }
        };
        let manga_ocr = match Self::load_manga_ocr(models_dir).await {
            Ok(adapter) => Some(adapter),
            Err(e) => {
                tracing::warn!("manga-ocr not loaded: {e}");
                None
            }
        };
        Ok(Self { ppocr, manga_ocr })
    }

    async fn load_ppocr(models_dir: &str) -> Result<ppocr::PpOcrAdapter> {
        let rec_path = model_hub::resolve(models_dir, Model::PpocrRec).await?;
        let dict_path = model_hub::resolve(models_dir, Model::PpocrKeys).await?;
        let det_path = model_hub::resolve_optional(models_dir, Model::PpocrDet).await;
        ppocr::PpOcrAdapter::new(
            LazySession::new(rec_path),
            &dict_path,
            det_path.map(LazySession::new),
        )
    }

    async fn load_manga_ocr(models_dir: &str) -> Result<manga_ocr::MangaOcrAdapter> {
        let encoder = model_hub::resolve(models_dir, Model::MangaOcrEncoder).await?;
        let decoder = model_hub::resolve(models_dir, Model::MangaOcrDecoder).await?;
        let vocab = model_hub::resolve(models_dir, Model::MangaOcrVocab).await?;
        manga_ocr::MangaOcrAdapter::new(
            LazySession::new(encoder),
            LazySession::new(decoder),
            &vocab,
        )
    }

    pub fn is_loaded(&self) -> bool {
        self.ppocr.is_some() || self.manga_ocr.is_some()
    }

    /// Whether PP-OCR detection model is available
    pub fn can_detect(&self) -> bool {
        self.ppocr.as_ref().is_some_and(|p| p.can_detect())
    }

    /// Detect text regions using PP-OCR's DB detection model
    pub fn detect(&self, img: &image::DynamicImage) -> Result<Vec<TextRegion>> {
        let ppocr = self.ppocr.as_ref()
            .ok_or_else(|| anyhow::anyhow!("PP-OCR adapter not loaded"))?;
        ppocr.detect(img)
    }

    /// Select provider by language and run OCR
    pub fn recognize(&self, image: &image::DynamicImage, lang: &str) -> Result<OcrResult> {
        match lang {
            "ja" => {
                let provider = self.manga_ocr.as_ref()
                    .ok_or_else(|| anyhow::anyhow!("manga-ocr model not loaded"))?;
                provider.recognize(image)
            }
            _ => {
                let provider = self.ppocr.as_ref()
                    .ok_or_else(|| anyhow::anyhow!("PP-OCR model not loaded"))?;
                provider.recognize(image)
            }
        }
    }
}
