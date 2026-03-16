/// Translation agent tool definitions.
///
/// Each module exports: `def() -> ToolDef`, `Args` struct.
/// Tool dispatch is handled by `agent::translation::TranslationAgent`.
pub mod get_context;
pub mod search_glossary;
pub mod translate;
pub mod view_page;

use crate::llm::ToolDef;

/// Build the translation agent's tool list based on available capabilities.
pub fn build_tools(has_images: bool, has_glossary: bool, has_context: bool) -> Vec<ToolDef> {
    let mut tools = vec![translate::def()];

    if has_images {
        tools.push(view_page::def());
    }

    if has_glossary {
        tools.push(search_glossary::def());
    }

    if has_context {
        tools.push(get_context::def());
    }

    tools
}
