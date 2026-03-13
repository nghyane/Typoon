/// OpenAI-compatible provider implementation.
///
/// Serializes `Message` IR → OpenAI chat JSON, calls the API via `async-openai`,
/// and parses the response back to `ToolCallMsg` IR.
use std::time::Duration;

use anyhow::{Context, Result};
use async_openai::Client;
use async_openai::config::OpenAIConfig;
use async_openai::types::chat::ChatCompletionMessageToolCalls;

use super::ir::{ContentPart, Message, ToolCallMsg};
use super::provider::Provider;
use super::tool::ToolDef;

const LLM_TIMEOUT_SECS: u64 = 180;

pub struct OpenAIProvider {
    client: Client<OpenAIConfig>,
    model: String,
    reasoning_effort: Option<String>,
}

impl OpenAIProvider {
    pub fn new(endpoint: &str, api_key: Option<&str>, model: &str) -> Result<Self> {
        let config = OpenAIConfig::new()
            .with_api_base(endpoint)
            .with_api_key(api_key.unwrap_or("not-needed"));

        Ok(Self {
            client: Client::with_config(config),
            model: model.to_string(),
            reasoning_effort: None,
        })
    }

    pub fn with_reasoning_effort(mut self, effort: Option<String>) -> Self {
        self.reasoning_effort = effort;
        self
    }
}

impl Provider for OpenAIProvider {
    fn call<'a>(
        &'a self,
        messages: &'a [Message],
        tools: &'a [ToolDef],
    ) -> std::pin::Pin<Box<dyn std::future::Future<Output = Result<Vec<ToolCallMsg>>> + Send + 'a>> {
        Box::pin(async move {
            let messages_json = serialize_messages(messages);
            let tools_json = serialize_tools(tools);

            let mut body = serde_json::json!({
                "model": self.model,
                "messages": messages_json,
                "tools": tools_json,
                "parallel_tool_calls": true
            });
            if let Some(effort) = &self.reasoning_effort {
                body["reasoning_effort"] = serde_json::json!(effort);
            }

            let chat = self.client.chat();
            let raw: async_openai::types::chat::CreateChatCompletionResponse =
                tokio::time::timeout(
                    Duration::from_secs(LLM_TIMEOUT_SECS),
                    chat.create_byot(body),
                )
                .await
                .map_err(|_| {
                    anyhow::anyhow!("Translation LLM call timed out ({}s)", LLM_TIMEOUT_SECS)
                })?
                .context("Translation LLM call failed")?;

            let choice = raw
                .choices
                .first()
                .ok_or_else(|| anyhow::anyhow!("No choices returned"))?;

            let tool_calls = match &choice.message.tool_calls {
                Some(calls) if !calls.is_empty() => calls
                    .iter()
                    .filter_map(|c| {
                        let ChatCompletionMessageToolCalls::Function(f) = c else {
                            return None;
                        };
                        Some(ToolCallMsg {
                            id: f.id.clone(),
                            name: f.function.name.clone(),
                            arguments: f.function.arguments.clone(),
                        })
                    })
                    .collect(),
                _ => vec![],
            };

            Ok(tool_calls)
        })
    }
}

// ── IR → OpenAI JSON serialization ──

fn serialize_messages(messages: &[Message]) -> Vec<serde_json::Value> {
    messages.iter().map(serialize_message).collect()
}

fn serialize_message(msg: &Message) -> serde_json::Value {
    match msg {
        Message::System(text) => {
            serde_json::json!({ "role": "system", "content": text })
        }
        Message::User(parts) => {
            if parts.len() == 1 {
                if let ContentPart::Text(text) = &parts[0] {
                    return serde_json::json!({ "role": "user", "content": text });
                }
            }
            serde_json::json!({
                "role": "user",
                "content": parts.iter().map(serialize_content_part).collect::<Vec<_>>()
            })
        }
        Message::Assistant { tool_calls } => {
            serde_json::json!({
                "role": "assistant",
                "tool_calls": tool_calls.iter().map(|tc| {
                    serde_json::json!({
                        "id": tc.id,
                        "type": "function",
                        "function": {
                            "name": tc.name,
                            "arguments": tc.arguments
                        }
                    })
                }).collect::<Vec<_>>()
            })
        }
        Message::ToolResult { tool_call_id, content } => {
            if content.len() == 1 {
                if let ContentPart::Text(text) = &content[0] {
                    return serde_json::json!({
                        "role": "tool",
                        "tool_call_id": tool_call_id,
                        "content": text
                    });
                }
            }
            serde_json::json!({
                "role": "tool",
                "tool_call_id": tool_call_id,
                "content": content.iter().map(serialize_content_part).collect::<Vec<_>>()
            })
        }
    }
}

fn serialize_content_part(part: &ContentPart) -> serde_json::Value {
    match part {
        ContentPart::Text(text) => {
            serde_json::json!({ "type": "text", "text": text })
        }
        ContentPart::Image { data_uri } => {
            serde_json::json!({
                "type": "image_url",
                "image_url": { "url": data_uri, "detail": "low" }
            })
        }
    }
}

fn serialize_tools(tools: &[ToolDef]) -> serde_json::Value {
    let arr: Vec<serde_json::Value> = tools
        .iter()
        .map(|t| {
            let mut func = serde_json::json!({
                "name": t.name,
                "description": t.description,
                "parameters": t.parameters,
            });
            if t.strict {
                func["strict"] = serde_json::json!(true);
            }
            serde_json::json!({
                "type": "function",
                "function": func
            })
        })
        .collect();
    serde_json::json!(arr)
}
