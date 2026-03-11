use anyhow::{Context, Result};
use async_openai::Client;
use async_openai::config::OpenAIConfig;
use futures::StreamExt;

use super::{BubbleTranslated, TranslateRequest, TranslationProvider};
use crate::config::TranslationConfig;

pub struct OpenAICompatibleAdapter {
    client: Client<OpenAIConfig>,
    model: String,
}

impl OpenAICompatibleAdapter {
    pub fn new(config: &TranslationConfig) -> Result<Self> {
        let openai_config = OpenAIConfig::new()
            .with_api_base(&config.endpoint)
            .with_api_key(config.api_key.as_deref().unwrap_or("not-needed"));

        Ok(Self {
            client: Client::with_config(openai_config),
            model: config.model.clone(),
        })
    }

    fn build_prompt(&self, req: &TranslateRequest) -> String {
        let mut prompt = format!(
            "You are processing OCR results from a manga/manhwa page. The text was extracted by OCR and may contain errors.\n\
             \n\
             Your tasks:\n\
             1. SKIP watermarks, scan group names, URLs, and non-dialogue text (do NOT include them)\n\
             2. FIX OCR errors (misspellings, garbled characters)\n\
             3. MERGE bubbles that are fragments of the same sentence (same Y position = same row, adjacent bubbles)\n\
             4. TRANSLATE the cleaned dialogue to {}\n\
             \n\
             Output JSON: {{\"bubbles\": [{{\"id\": \"b0\", \"source_text\": \"cleaned original\", \"translated_text\": \"translation\"}}]}}\n\
             - For merged bubbles, use the first bubble's id\n\
             - Omit skipped bubbles entirely\n\
             - Keep translations natural and conversational. Preserve tone and emotion.\n",
            req.target_lang
        );

        if !req.context.is_empty() {
            prompt.push_str("\nPrevious translations for context:\n");
            for ctx in &req.context {
                prompt.push_str(&format!("  {} → {}\n", ctx.source_text, ctx.translated_text));
            }
        }

        prompt.push_str("\nOCR bubbles (id, position, text):\n");
        for bubble in &req.bubbles {
            if let Some((x, y)) = bubble.position {
                prompt.push_str(&format!("  [{}] pos=({},{}) \"{}\"\n", bubble.id, x, y, bubble.source_text));
            } else {
                prompt.push_str(&format!("  [{}] \"{}\"\n", bubble.id, bubble.source_text));
            }
        }

        prompt
    }
}

impl TranslationProvider for OpenAICompatibleAdapter {
    async fn translate(&self, req: &TranslateRequest) -> Result<Vec<BubbleTranslated>> {
        let prompt = self.build_prompt(req);

        // Stream with json_object response format
        let mut stream = self.client.chat().create_stream_byot(serde_json::json!({
            "model": self.model,
            "messages": [
                {"role": "system", "content": "You are a professional manga/manhwa translator. Output valid JSON only."},
                {"role": "user", "content": prompt}
            ],
            "response_format": {"type": "json_object"},
            "stream": true
        })).await.context("LLM stream request failed")?;

        let mut content = String::new();
        while let Some(chunk) = stream.next().await {
            let chunk: async_openai::types::chat::CreateChatCompletionStreamResponse =
                chunk.context("Stream chunk error")?;
            for choice in &chunk.choices {
                if let Some(c) = &choice.delta.content {
                    content.push_str(c);
                }
            }
        }

        if content.is_empty() {
            anyhow::bail!("LLM stream returned no content");
        }

        // Try {"bubbles": [...]} wrapper first, then bare array
        if let Ok(wrapper) = serde_json::from_str::<BubblesWrapper>(&content) {
            return Ok(wrapper.bubbles);
        }

        serde_json::from_str::<Vec<BubbleTranslated>>(&content)
            .with_context(|| format!("Failed to parse LLM JSON: {content}"))
    }

    fn name(&self) -> &str {
        "openai_compatible"
    }
}

#[derive(serde::Deserialize)]
struct BubblesWrapper {
    bubbles: Vec<BubbleTranslated>,
}
