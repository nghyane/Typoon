use crate::llm::{ToolDef, ToolResponse};
use crate::translation::TranslateContext;

#[derive(serde::Deserialize)]
pub struct Args {
    pub source_term: String,
    pub target_term: String,
    pub category: String,
    pub notes: String,
}

pub fn def() -> ToolDef {
    ToolDef::new(
        "update_glossary",
        "Save a high-confidence canonical term to the persistent glossary for future chapters.\n\n\
            Behavior:\n\
            - Use only for stable, recurring items: character names, titles, places, techniques.\n\
            - Save only when confident the term should be reused later.\n\
            - Translation should not wait on this; it is optional.\n\n\
            When to use: new recurring term that should stay consistent in future chapters.\n\
            When NOT to use: casual dialogue, uncertain pronouns, one-off phrasing.",
        serde_json::json!({
            "type": "object",
            "required": ["source_term", "target_term", "category", "notes"],
            "additionalProperties": false,
            "properties": {
                "source_term": {
                    "type": "string",
                    "description": "Original term in source language"
                },
                "target_term": {
                    "type": "string",
                    "description": "Canonical translated term"
                },
                "category": {
                    "type": "string",
                    "enum": ["character", "term", "title", "sfx", "other"],
                    "description": "character=names, term=techniques/orgs/places, title=honorifics/ranks, sfx=sound effects"
                },
                "notes": {
                    "type": "string",
                    "description": "Brief context for future reference"
                }
            }
        }),
    )
    .strict()
}

pub fn handle(args: &Args, ctx: &TranslateContext<'_>) -> ToolResponse {
    let notes = format!("[{}] {}", args.category, args.notes);
    let response = if let Some(glossary) = ctx.glossary {
        match glossary.upsert(&args.source_term, &args.target_term, Some(&notes)) {
            Ok(()) => {
                tracing::info!(
                    "Glossary updated: {} → {} ({})",
                    args.source_term,
                    args.target_term,
                    notes
                );
                format!("Saved: {} → {}", args.source_term, args.target_term)
            }
            Err(e) => format!("Failed to save: {e}"),
        }
    } else {
        "Glossary not available.".to_string()
    };

    ToolResponse::Text(response)
}
