mod openai_compatible;

use anyhow::Result;
use image::DynamicImage;
use serde::{Deserialize, Serialize};

use crate::config::TranslationConfig;
use crate::glossary::{Glossary, GlossaryEntry};

// ── Page-grouped input ──

/// A single page's worth of bubbles.
#[derive(Debug, Clone, Serialize)]
pub struct PageInput {
    pub page_index: usize,
    pub bubbles: Vec<BubbleInput>,
}

#[derive(Debug, Clone, Serialize)]
pub struct BubbleInput {
    pub id: String,
    pub source_text: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub position: Option<(i32, i32)>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BubbleTranslated {
    pub id: String,
    pub source_text: String,
    pub translated_text: String,
}

// ── Request ──

/// Translation request supporting both single-page and chapter mode.
pub struct TranslateRequest {
    pub pages: Vec<PageInput>,
    pub source_lang: String,
    pub target_lang: String,
    pub context: Vec<BubbleTranslated>,
    pub glossary: Vec<GlossaryEntry>,
}

impl TranslateRequest {
    /// Convenience: build a single-page request (backward compat).
    pub fn single_page(
        bubbles: Vec<BubbleInput>,
        source_lang: String,
        target_lang: String,
        context: Vec<BubbleTranslated>,
    ) -> Self {
        Self {
            pages: vec![PageInput { page_index: 0, bubbles }],
            source_lang,
            target_lang,
            context,
            glossary: vec![],
        }
    }

    /// All bubbles across all pages, flattened.
    pub fn all_bubbles(&self) -> impl Iterator<Item = &BubbleInput> {
        self.pages.iter().flat_map(|p| &p.bubbles)
    }
}

// ── Runtime context passed alongside the request ──

/// External resources available to the translation agent during execution.
pub struct TranslateContext<'a> {
    pub page_images: &'a [DynamicImage],
    pub glossary: Option<&'a Glossary>,
}

impl Default for TranslateContext<'_> {
    fn default() -> Self {
        Self { page_images: &[], glossary: None }
    }
}

// ── Engine ──

pub struct TranslationEngine {
    provider: openai_compatible::OpenAICompatibleAdapter,
}

impl TranslationEngine {
    pub fn new(config: &TranslationConfig) -> Result<Self> {
        let provider = openai_compatible::OpenAICompatibleAdapter::new(config)?;
        Ok(Self { provider })
    }

    pub async fn translate(
        &self,
        req: &TranslateRequest,
        ctx: &TranslateContext<'_>,
    ) -> Result<Vec<BubbleTranslated>> {
        self.provider.translate(req, ctx).await
    }
}
