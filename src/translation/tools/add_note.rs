use crate::llm::{ToolDef, ToolResponse};
use crate::translation::TranslateContext;

#[derive(serde::Deserialize)]
pub struct Args {
    pub note_type: String,
    pub content: String,
}

pub fn def() -> ToolDef {
    ToolDef::new(
        "add_note",
        "Record a chapter observation for future reference.\n\n\
            Behavior:\n\
            - Save important observations about this chapter: key events, character introductions,\n\
              relationship changes, setting descriptions.\n\
            - These notes are searchable by the context agent in future chapters.\n\
            - Multiple notes per chapter are fine \u{2014} one per distinct observation.\n\n\
            When to use: significant plot events, new characters, relationship reveals, setting changes.\n\
            When NOT to use: trivial dialogue or routine scene descriptions.",
        serde_json::json!({
            "type": "object",
            "required": ["note_type", "content"],
            "additionalProperties": false,
            "properties": {
                "note_type": {
                    "type": "string",
                    "enum": ["event", "character", "relationship", "setting"],
                    "description": "event=plot points, character=introductions/descriptions, relationship=who knows whom, setting=locations"
                },
                "content": {
                    "type": "string",
                    "description": "Concise observation in the target language"
                }
            }
        }),
    )
    .strict()
}

pub fn handle(args: &Args, ctx: &TranslateContext<'_>) -> ToolResponse {
    let response = if let (Some(store), Some(chapter_idx)) = (ctx.project, ctx.chapter_index) {
        match store.add_note(chapter_idx, &args.note_type, &args.content) {
            Ok(()) => {
                let preview = match args.content.char_indices().find(|&(i, _)| i >= 80) {
                    Some((i, _)) => format!("{}...", &args.content[..i]),
                    None => args.content.clone(),
                };
                tracing::info!("Note added [{}]: {}", args.note_type, preview);
                "ok".to_string()
            }
            Err(e) => format!("Failed to save note: {e}"),
        }
    } else {
        "Context store not available.".to_string()
    };

    ToolResponse::Text(response)
}
