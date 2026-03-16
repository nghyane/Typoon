/// Generic agent infrastructure — IR, tool definitions, provider trait, and adapters.
///
/// This module is provider-agnostic. Domain-specific code (translation, context)
/// uses these building blocks without knowing which LLM provider is in use.
pub mod anthropic;
pub mod ir;
pub mod openai;
pub mod provider;
pub mod tool;

pub use ir::{ContentPart, Message, ToolCallMsg};
pub use provider::{CallResponse, Provider};
pub use tool::{ToolDef, ToolResponse};
