use anyhow::{Context, Result};
use async_openai::Client;
use async_openai::config::OpenAIConfig;

use super::{BubbleTranslated, TranslateRequest, TranslationProvider};
use crate::config::TranslationConfig;

pub struct OpenAICompatibleAdapter {
    client: Client<OpenAIConfig>,
    model: String,
    reasoning_effort: Option<String>,
}

impl OpenAICompatibleAdapter {
    pub fn new(config: &TranslationConfig) -> Result<Self> {
        let openai_config = OpenAIConfig::new()
            .with_api_base(&config.endpoint)
            .with_api_key(config.api_key.as_deref().unwrap_or("not-needed"));

        Ok(Self {
            client: Client::with_config(openai_config),
            model: config.model.clone(),
            reasoning_effort: config.reasoning_effort.clone(),
        })
    }

    fn build_prompt(&self, req: &TranslateRequest) -> String {
        let mut prompt = format!(
            "Translate each bubble to {}. \
             Call translate() once for EVERY bubble. Preserve tone and emotion. \
             Fix obvious OCR errors in source_text.\n",
            req.target_lang
        );

        if !req.context.is_empty() {
            prompt.push_str("\nPrevious translations for context:\n");
            for ctx in &req.context {
                prompt.push_str(&format!("  {} → {}\n", ctx.source_text, ctx.translated_text));
            }
        }

        prompt.push_str("\nBubbles:\n");
        for bubble in &req.bubbles {
            prompt.push_str(&format!("  [{}] \"{}\"\n", bubble.id, bubble.source_text));
        }

        prompt
    }
}

fn translate_tool_def() -> serde_json::Value {
    serde_json::json!([{
        "type": "function",
        "function": {
            "name": "translate",
            "description": "Submit the translation for one bubble.",
            "strict": true,
            "parameters": {
                "type": "object",
                "required": ["id", "source_text", "translated_text"],
                "additionalProperties": false,
                "properties": {
                    "id": {
                        "type": "string",
                        "description": "Bubble ID (e.g. b0)"
                    },
                    "source_text": {
                        "type": "string",
                        "description": "Cleaned/corrected source text"
                    },
                    "translated_text": {
                        "type": "string",
                        "description": "Translated text"
                    }
                }
            }
        }
    }])
}

#[derive(serde::Deserialize)]
struct TranslateArgs {
    id: String,
    source_text: String,
    translated_text: String,
}

impl TranslationProvider for OpenAICompatibleAdapter {
    async fn translate(&self, req: &TranslateRequest) -> Result<Vec<BubbleTranslated>> {
        let prompt = self.build_prompt(req);

        let resp: async_openai::types::chat::CreateChatCompletionResponse = self
            .client
            .chat()
            .create_byot({
                let mut body = serde_json::json!({
                    "model": self.model,
                    "messages": [
                        {"role": "system", "content": "You are a professional manga/manhwa translator."},
                        {"role": "user", "content": prompt}
                    ],
                    "tools": translate_tool_def(),
                    "parallel_tool_calls": true
                });
                if let Some(effort) = &self.reasoning_effort {
                    body["reasoning_effort"] = serde_json::json!(effort);
                }
                body
            })
            .await
            .context("Translation request failed")?;

        let choice = resp
            .choices
            .first()
            .ok_or_else(|| anyhow::anyhow!("No choices returned"))?;

        let tool_calls = match &choice.message.tool_calls {
            Some(calls) => calls,
            None => anyhow::bail!("LLM returned no tool calls"),
        };

        let mut results = Vec::new();
        for call in tool_calls {
            let tc = match call {
                async_openai::types::chat::ChatCompletionMessageToolCalls::Function(f) => f,
                _ => continue,
            };
            if tc.function.name != "translate" {
                continue;
            }
            let args: TranslateArgs = serde_json::from_str(&tc.function.arguments)
                .with_context(|| format!("Bad translate args: {}", tc.function.arguments))?;
            results.push(BubbleTranslated {
                id: args.id,
                source_text: args.source_text,
                translated_text: args.translated_text,
            });
        }

        Ok(results)
    }

    fn name(&self) -> &str {
        "openai_compatible"
    }
}
