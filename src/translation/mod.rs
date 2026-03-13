mod agent;
mod prompt;
mod tools;

use anyhow::Result;
use image::DynamicImage;
use serde::{Deserialize, Serialize};

use crate::agent::Provider;
use crate::context::ContextStore;
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

/// A continuity note from a previous chapter (proactively injected).
#[derive(Debug, Clone)]
pub struct ContextNote {
    pub note_type: String,
    pub content: String,
}

// ── Request ──

/// Translation request supporting both single-page and chapter mode.
pub struct TranslateRequest {
    pub pages: Vec<PageInput>,
    pub source_lang: String,
    pub target_lang: String,
    pub context: Vec<BubbleTranslated>,
    pub glossary: Vec<GlossaryEntry>,
    /// Proactive continuity notes from previous chapters.
    pub notes: Vec<ContextNote>,
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
            notes: vec![],
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
    pub context_store: Option<&'a ContextStore>,
    pub context_agent: Option<&'a dyn Provider>,
    pub project_id: Option<&'a str>,
    pub chapter_index: Option<usize>,
}

impl Default for TranslateContext<'_> {
    fn default() -> Self {
        Self {
            page_images: &[],
            glossary: None,
            context_store: None,
            context_agent: None,
            project_id: None,
            chapter_index: None,
        }
    }
}

// ── Engine ──

pub struct TranslationEngine {
    provider: Box<dyn Provider>,
}

impl TranslationEngine {
    pub fn new(provider: Box<dyn Provider>) -> Self {
        Self { provider }
    }

    pub async fn translate(
        &self,
        req: &TranslateRequest,
        ctx: &TranslateContext<'_>,
    ) -> Result<Vec<BubbleTranslated>> {
        agent::run(&*self.provider, req, ctx).await
    }
}
