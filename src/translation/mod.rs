mod openai_compatible;

use anyhow::Result;
use serde::{Deserialize, Serialize};

use crate::config::TranslationConfig;

#[derive(Debug, Serialize)]
pub struct TranslateRequest {
    pub bubbles: Vec<BubbleInput>,
    pub target_lang: String,
    pub context: Vec<BubbleTranslated>,
}

#[derive(Debug, Serialize)]
pub struct BubbleInput {
    pub id: String,
    pub source_text: String,
    /// Position (x, y) in image pixels — helps LLM merge fragments and filter noise
    #[serde(skip_serializing_if = "Option::is_none")]
    pub position: Option<(i32, i32)>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BubbleTranslated {
    pub id: String,
    pub source_text: String,
    pub translated_text: String,
}

/// Abstract translation provider trait
pub trait TranslationProvider: Send + Sync {
    fn translate(&self, req: &TranslateRequest) -> impl Future<Output = Result<Vec<BubbleTranslated>>> + Send;
    fn name(&self) -> &str;
}

use std::future::Future;

pub struct TranslationEngine {
    provider: openai_compatible::OpenAICompatibleAdapter,
}

impl TranslationEngine {
    pub fn new(config: &TranslationConfig) -> Result<Self> {
        let provider = openai_compatible::OpenAICompatibleAdapter::new(config)?;
        Ok(Self { provider })
    }

    pub fn is_configured(&self) -> bool {
        true
    }

    pub async fn translate(&self, req: &TranslateRequest) -> Result<Vec<BubbleTranslated>> {
        self.provider.translate(req).await
    }
}
