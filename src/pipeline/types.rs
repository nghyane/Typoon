use serde::{Deserialize, Serialize};

use crate::render::layout::DrawableArea;
use crate::vision::detection::LocalTextMask;

/// Raw output from detect_and_ocr (before border detection + area computation).
#[derive(Debug, Clone)]
pub struct RawBubble {
    pub source_text: String,
    pub polygon: Vec<[f64; 2]>,
    pub det_confidence: f64,
    pub ocr_confidence: f64,
    pub mask: Option<LocalTextMask>,
}

/// Phase 1 output: a single detected + OCR'd bubble with layout info.
#[derive(Debug, Clone)]
pub struct DetectedBubble {
    pub idx: usize,
    pub source_text: String,
    pub polygon: Vec<[f64; 2]>,
    pub area: DrawableArea,
    pub det_confidence: f64,
    pub ocr_confidence: f64,
    pub mask: Option<LocalTextMask>,
}

/// Phase 1 output: all detected bubbles on a page.
pub struct PageDetections {
    pub page_index: usize,
    pub bubbles: Vec<DetectedBubble>,
}

/// Phase 2 output: a single translated + fitted bubble.
/// Contains everything needed for rendering — no Option fields.
#[derive(Debug, Clone)]
pub struct TranslatedBubble {
    pub idx: usize,
    pub source_text: String,
    pub translated_text: String,
    pub polygon: Vec<[f64; 2]>,
    pub area: DrawableArea,
    pub mask: Option<LocalTextMask>,
    pub font_size_px: u32,
    pub line_height: f64,
    pub overflow: bool,
}

/// Phase 2 output: all translated bubbles on a page.
pub struct PageTranslations {
    pub page_index: usize,
    pub bubbles: Vec<TranslatedBubble>,
}

/// Chapter-level output after full pipeline (translate + render).
pub struct ChapterOutput {
    pub pages: Vec<RenderedPage>,
}

/// A rendered page with its image.
pub struct RenderedPage {
    pub page_index: usize,
    pub bubbles: Vec<TranslatedBubble>,
    pub image: image::RgbaImage,
}

/// All parameters for a chapter translation job.
/// Replaces 8 loose params in the old `translate_inner`.
pub struct TranslateJob<'a> {
    pub detections: &'a [PageDetections],
    pub images: &'a [image::DynamicImage],
    pub target_lang: &'a str,
    pub source_lang: &'a str,
    pub chapter_index: Option<usize>,
}

/// API-only response type. Serializable, no render-internal fields.
#[derive(Debug, Serialize, Deserialize)]
pub struct BubbleResult {
    pub bubble_id: String,
    pub polygon: Vec<[f64; 2]>,
    pub source_text: String,
    pub translated_text: String,
    pub font_size_px: u32,
    pub line_height: f64,
    pub overflow: bool,
    #[serde(default = "default_align")]
    pub align: String,
}

fn default_align() -> String {
    "center".into()
}

impl BubbleResult {
    pub fn from_translated(b: &TranslatedBubble, page_index: usize) -> Self {
        Self {
            bubble_id: format!("p{}_b{}", page_index, b.idx),
            polygon: b.polygon.clone(),
            source_text: b.source_text.clone(),
            translated_text: b.translated_text.clone(),
            font_size_px: b.font_size_px,
            line_height: b.line_height,
            overflow: b.overflow,
            align: "center".into(),
        }
    }
}
