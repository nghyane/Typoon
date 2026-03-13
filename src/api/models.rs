use serde::{Deserialize, Serialize};

use crate::detection::LocalTextMask;
use crate::text_layout::DrawableArea;

// --- Request ---

#[derive(Debug, Deserialize)]
pub struct TranslateImageRequest {
    pub image_id: String,
    pub image_blob_b64: String,
    pub source_lang: Option<String>,
    pub target_lang: String,
    pub ocr_provider: Option<String>,
    pub translation_provider: Option<String>,
    pub provider_config: Option<ProviderConfig>,
    pub context_hint: Option<ContextHint>,
}

#[derive(Debug, Deserialize)]
pub struct ProviderConfig {
    pub endpoint: Option<String>,
    pub api_key: Option<String>,
    pub model: Option<String>,
}

#[derive(Debug, Deserialize)]
pub struct ContextHint {
    pub chapter_id: Option<String>,
    pub image_index: u32,
    pub previous_translations: Vec<PreviousTranslation>,
}

#[derive(Debug, Deserialize)]
pub struct PreviousTranslation {
    pub image_index: u32,
    pub bubbles: Vec<BubbleContext>,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct BubbleContext {
    pub source_text: String,
    pub translated_text: String,
}

// --- Response ---

#[derive(Debug, Serialize, Deserialize)]
pub struct TranslateImageResponse {
    pub image_id: String,
    pub status: String,
    pub bubbles: Vec<BubbleResult>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub rendered_image_png_b64: Option<String>,
}

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
    /// Canonical drawable area (computed once, used by fit + render).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub drawable_area: Option<DrawableArea>,
    /// ML-generated text mask for erasing (render-only, not serialized).
    #[serde(skip, default)]
    pub text_mask: Option<LocalTextMask>,
}

fn default_align() -> String { "center".into() }

#[derive(Debug, Serialize, Deserialize)]
pub struct HealthResponse {
    pub ready: bool,
    pub detection_model_loaded: bool,
    pub ocr_model_loaded: bool,
    pub translation_configured: bool,
}
