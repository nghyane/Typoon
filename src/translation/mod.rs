pub mod prompt;
pub mod tools;

use anyhow::Result;
use image::DynamicImage;
use serde::{Deserialize, Serialize};

use crate::llm::Provider;
use crate::pipeline::types::{DetectedBubble, PageDetections};
use crate::storage::project::ProjectStore;

// ── Translation request (references detection output directly) ──

/// Translation request — references detections instead of copying into separate types.
pub struct TranslateRequest<'a> {
    pub detections: &'a [PageDetections],
    pub source_lang: &'a str,
    pub target_lang: &'a str,
    pub glossary: Vec<crate::storage::project::GlossaryEntry>,
    pub knowledge_snapshot: Option<String>,
}

impl<'a> TranslateRequest<'a> {
    /// All detected bubbles across all pages, flattened.
    pub fn all_bubbles(&self) -> impl Iterator<Item = (usize, &DetectedBubble)> {
        self.detections
            .iter()
            .flat_map(|pd| pd.bubbles.iter().map(move |b| (pd.page_index, b)))
    }

    /// Generate the string ID for a bubble (only for LLM prompt/response matching).
    pub fn bubble_id(page_index: usize, bubble_idx: usize) -> String {
        format!("p{}_b{}", page_index, bubble_idx)
    }
}

/// A previously translated bubble (for context injection from prior pages/chapters).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PreviousTranslation {
    pub source_text: String,
    pub translated_text: String,
}

// ── LLM output (string ID because LLM returns it) ──

/// Raw output from the LLM agent loop. String ID is matched back to bubble index.
#[derive(Debug, Clone)]
pub struct BubbleTranslated {
    pub id: String,
    pub source_text: String,
    pub translated_text: String,
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
        req: &TranslateRequest<'_>,
        page_images: &[DynamicImage],
        project: Option<&ProjectStore>,
        context_provider: Option<&dyn Provider>,
    ) -> Result<Vec<BubbleTranslated>> {
        let agent = crate::agent::translation::TranslationAgent::new(
            req,
            page_images,
            project,
            context_provider,
        );
        crate::agent::run(&*self.provider, agent).await
    }
}
