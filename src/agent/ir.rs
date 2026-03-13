/// Typed intermediate representation for LLM messages.
///
/// Provider-agnostic — each provider serializes these to its own wire format.

/// A message in the conversation.
#[derive(Debug, Clone)]
pub enum Message {
    System(String),
    User(Vec<ContentPart>),
    Assistant { tool_calls: Vec<ToolCallMsg> },
    ToolResult {
        tool_call_id: String,
        content: Vec<ContentPart>,
    },
}

/// A content part within a message.
#[derive(Debug, Clone)]
pub enum ContentPart {
    Text(String),
    Image { data_uri: String },
}

/// A tool call returned by the LLM.
#[derive(Debug, Clone)]
pub struct ToolCallMsg {
    pub id: String,
    pub name: String,
    pub arguments: String,
}

// ── Convenience constructors ──

impl Message {
    pub fn system(text: impl Into<String>) -> Self {
        Self::System(text.into())
    }

    pub fn user_text(text: impl Into<String>) -> Self {
        Self::User(vec![ContentPart::Text(text.into())])
    }

    pub fn user_parts(parts: Vec<ContentPart>) -> Self {
        Self::User(parts)
    }

    pub fn tool_result_text(tool_call_id: impl Into<String>, text: impl Into<String>) -> Self {
        Self::ToolResult {
            tool_call_id: tool_call_id.into(),
            content: vec![ContentPart::Text(text.into())],
        }
    }

    pub fn tool_result_parts(
        tool_call_id: impl Into<String>,
        parts: Vec<ContentPart>,
    ) -> Self {
        Self::ToolResult {
            tool_call_id: tool_call_id.into(),
            content: parts,
        }
    }
}
