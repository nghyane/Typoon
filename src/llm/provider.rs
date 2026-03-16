/// Provider trait — abstracts over LLM API differences.
///
/// Each provider (OpenAI, Anthropic) implements this trait to:
/// 1. Serialize `Message` IR → provider-specific wire format
/// 2. Serialize `ToolDef` → provider-specific tool format
/// 3. Call the API and parse response back to `CallResponse` IR
///
/// Uses `Pin<Box<dyn Future>>` return type for dyn-compatibility
/// (`Box<dyn Provider>` must work for runtime dispatch).
use std::future::Future;
use std::pin::Pin;

use anyhow::Result;

use super::ir::{Message, ToolCallMsg};
use super::tool::ToolDef;

/// Response from a provider call — may contain tool calls, text, or both.
pub struct CallResponse {
    pub tool_calls: Vec<ToolCallMsg>,
    pub text: Option<String>,
}

pub trait Provider: Send + Sync {
    fn call<'a>(
        &'a self,
        messages: &'a [Message],
        tools: &'a [ToolDef],
    ) -> Pin<Box<dyn Future<Output = Result<CallResponse>> + Send + 'a>>;
}
