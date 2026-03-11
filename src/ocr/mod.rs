mod ppocr;
mod manga_ocr;

use anyhow::Result;

use crate::detection::TextRegion;

/// Create a standalone MangaOcrAdapter (for testing)
pub fn manga_ocr_adapter(models_dir: &str) -> Result<impl OcrProvider> {
    manga_ocr::MangaOcrAdapter::new(models_dir)
}

pub struct OcrResult {
    pub text: String,
    pub confidence: f64,
}

/// Abstract OCR provider trait
pub trait OcrProvider: Send + Sync {
    fn recognize(&self, image: &image::DynamicImage) -> Result<OcrResult>;
    fn name(&self) -> &str;
}

pub struct OcrEngine {
    ppocr: Option<ppocr::PpOcrAdapter>,
    manga_ocr: Option<manga_ocr::MangaOcrAdapter>,
}

impl OcrEngine {
    pub fn new(models_dir: &str) -> Result<Self> {
        let ppocr = ppocr::PpOcrAdapter::new(models_dir).ok();
        let manga_ocr = manga_ocr::MangaOcrAdapter::new(models_dir).ok();
        Ok(Self { ppocr, manga_ocr })
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
