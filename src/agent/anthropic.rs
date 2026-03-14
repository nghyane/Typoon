/// Anthropic provider implementation.
///
/// Serializes `Message` IR → Anthropic messages API format, calls via reqwest,
/// and parses the response back to `ToolCallMsg` IR.
use std::time::Duration;

use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};

use super::ir::{ContentPart, Message, ToolCallMsg};
use super::provider::{CallResponse, Provider};
use super::tool::ToolDef;

const ANTHROPIC_VERSION: &str = "2023-06-01";
const DEFAULT_TIMEOUT_SECS: u64 = 60;
const DEFAULT_MAX_TOKENS: u32 = 64000;

pub struct AnthropicProvider {
    client: reqwest::Client,
    endpoint: String,
    api_key: String,
    model: String,
    max_tokens: u32,
}

impl AnthropicProvider {
    pub fn new(endpoint: &str, api_key: &str, model: &str) -> Result<Self> {
        let client = reqwest::Client::builder()
            .timeout(Duration::from_secs(DEFAULT_TIMEOUT_SECS))
            .build()
            .context("Failed to create HTTP client")?;

        Ok(Self {
            client,
            endpoint: endpoint.trim_end_matches('/').to_string(),
            api_key: api_key.to_string(),
            model: model.to_string(),
            max_tokens: DEFAULT_MAX_TOKENS,
        })
    }

    pub fn with_max_tokens(mut self, max_tokens: u32) -> Self {
        self.max_tokens = max_tokens;
        self
    }
}

impl Provider for AnthropicProvider {
    fn call<'a>(
        &'a self,
        messages: &'a [Message],
        tools: &'a [ToolDef],
    ) -> std::pin::Pin<Box<dyn std::future::Future<Output = Result<CallResponse>> + Send + 'a>> {
        Box::pin(async move {
            let system = messages.iter().find_map(|m| match m {
                Message::System(text) => Some(serde_json::json!([{
                    "type": "text",
                    "text": text,
                    "cache_control": {"type": "ephemeral"}
                }])),
                _ => None,
            });

            let api_messages = serialize_messages(messages);
            let api_tools = serialize_tools(tools);

            let body = ApiRequest {
                model: self.model.clone(),
                max_tokens: self.max_tokens,
                system,
                messages: api_messages,
                tools: api_tools,
            };

            let url = format!("{}/messages", self.endpoint);
            let resp = self
                .client
                .post(&url)
                .header("x-api-key", &self.api_key)
                .header("anthropic-version", ANTHROPIC_VERSION)
                .header("content-type", "application/json")
                .json(&body)
                .send()
                .await
                .context("Anthropic API request failed")?;

            let status = resp.status();
            let body_text = resp
                .text()
                .await
                .context("Failed to read Anthropic response")?;

            if !status.is_success() {
                if let Ok(err) = serde_json::from_str::<ErrorResponse>(&body_text) {
                    anyhow::bail!("Anthropic API error ({}): {}", status, err.error.message);
                }
                anyhow::bail!("Anthropic API error ({}): {}", status, body_text);
            }

            let response: ApiResponse =
                serde_json::from_str(&body_text).with_context(|| {
                    format!(
                        "Failed to parse Anthropic response: {}",
                        &body_text[..200.min(body_text.len())]
                    )
                })?;

            let tool_calls = response
                .content
                .iter()
                .filter_map(|block| match block {
                    ResponseBlock::ToolUse { id, name, input } => Some(ToolCallMsg {
                        id: id.clone(),
                        name: name.clone(),
                        arguments: input.to_string(),
                    }),
                    _ => None,
                })
                .collect();

            let text: String = response
                .content
                .iter()
                .filter_map(|block| match block {
                    ResponseBlock::Text { text } => Some(text.as_str()),
                    _ => None,
                })
                .collect::<Vec<_>>()
                .join("");
            let text = if text.is_empty() { None } else { Some(text) };

            Ok(CallResponse { tool_calls, text })
        })
    }
}

// ── Anthropic API wire types (private) ──

#[derive(Serialize)]
struct ApiRequest {
    model: String,
    max_tokens: u32,
    #[serde(skip_serializing_if = "Option::is_none")]
    system: Option<serde_json::Value>,
    messages: Vec<serde_json::Value>,
    #[serde(skip_serializing_if = "Vec::is_empty")]
    tools: Vec<serde_json::Value>,
}

#[derive(Deserialize, Debug)]
struct ApiResponse {
    content: Vec<ResponseBlock>,
    #[allow(dead_code)]
    stop_reason: Option<String>,
}

#[derive(Deserialize, Debug)]
#[serde(tag = "type")]
enum ResponseBlock {
    #[serde(rename = "text")]
    Text {
        #[allow(dead_code)]
        text: String,
    },
    #[serde(rename = "tool_use")]
    ToolUse {
        id: String,
        name: String,
        input: serde_json::Value,
    },
}

#[derive(Deserialize, Debug)]
struct ErrorResponse {
    error: ErrorDetail,
}

#[derive(Deserialize, Debug)]
struct ErrorDetail {
    message: String,
}

// ── IR → Anthropic JSON serialization ──

fn serialize_messages(messages: &[Message]) -> Vec<serde_json::Value> {
    messages
        .iter()
        .filter_map(|msg| match msg {
            Message::System(_) => None, // handled separately
            Message::User(parts) => {
                Some(serde_json::json!({
                    "role": "user",
                    "content": serialize_content_parts(parts)
                }))
            }
            Message::Assistant { text, tool_calls } => {
                let mut blocks: Vec<serde_json::Value> = Vec::new();
                if let Some(text) = text {
                    if !text.is_empty() {
                        blocks.push(serde_json::json!({
                            "type": "text",
                            "text": text
                        }));
                    }
                }
                for tc in tool_calls {
                    let input: serde_json::Value =
                        serde_json::from_str(&tc.arguments).unwrap_or_default();
                    blocks.push(serde_json::json!({
                        "type": "tool_use",
                        "id": tc.id,
                        "name": tc.name,
                        "input": input
                    }));
                }
                Some(serde_json::json!({
                    "role": "assistant",
                    "content": blocks
                }))
            }
            Message::ToolResult { tool_call_id, content } => {
                // Anthropic: tool results go in a "user" message with tool_result blocks
                let text = content
                    .iter()
                    .filter_map(|p| match p {
                        ContentPart::Text(t) => Some(t.as_str()),
                        _ => None,
                    })
                    .collect::<Vec<_>>()
                    .join("\n");
                Some(serde_json::json!({
                    "role": "user",
                    "content": [{
                        "type": "tool_result",
                        "tool_use_id": tool_call_id,
                        "content": text
                    }]
                }))
            }
        })
        .collect()
}

fn serialize_content_parts(parts: &[ContentPart]) -> serde_json::Value {
    if parts.len() == 1 {
        if let ContentPart::Text(text) = &parts[0] {
            return serde_json::json!(text);
        }
    }
    let blocks: Vec<serde_json::Value> = parts
        .iter()
        .map(|p| match p {
            ContentPart::Text(text) => serde_json::json!({ "type": "text", "text": text }),
            ContentPart::Image { data_uri } => {
                // Anthropic uses base64 source with media_type
                // For data URIs, extract the base64 data
                if let Some(rest) = data_uri.strip_prefix("data:image/jpeg;base64,") {
                    serde_json::json!({
                        "type": "image",
                        "source": {
                            "type": "base64",
                            "media_type": "image/jpeg",
                            "data": rest
                        }
                    })
                } else {
                    serde_json::json!({ "type": "text", "text": "[image]" })
                }
            }
        })
        .collect();
    serde_json::json!(blocks)
}

fn serialize_tools(tools: &[ToolDef]) -> Vec<serde_json::Value> {
    let len = tools.len();
    tools
        .iter()
        .enumerate()
        .map(|(i, t)| {
            let mut tool = serde_json::json!({
                "name": t.name,
                "description": t.description,
                "input_schema": t.parameters,
            });
            // Cache breakpoint on last tool — caches system + all tools prefix
            if i == len - 1 {
                tool["cache_control"] = serde_json::json!({"type": "ephemeral"});
            }
            tool
        })
        .collect()
}
