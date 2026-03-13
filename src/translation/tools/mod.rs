/// Translation agent tool definitions — one file per tool.
///
/// Each module exports: `def() -> ToolDef`, `Args` struct, and `handle()` where applicable.
/// `translate` and `done` are special (handled inline by the agent loop).
pub mod add_note;
pub mod done;
pub mod get_context;
pub mod search_glossary;
pub mod translate;
pub mod update_glossary;
pub mod view_page;

use crate::agent;

/// Build the translation agent's tool list based on available capabilities.
pub fn build_tools(has_images: bool, has_glossary: bool, has_context: bool) -> Vec<agent::ToolDef> {
    let mut tools = vec![translate::def(), done::def()];

    if has_images {
        tools.push(view_page::def());
    }

    if has_glossary {
        tools.push(search_glossary::def());
        tools.push(update_glossary::def());
    }

    if has_context {
        tools.push(get_context::def());
        tools.push(add_note::def());
    }

    tools
}
