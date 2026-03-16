use crate::llm::{ToolDef, ToolResponse};
use crate::translation::TranslateContext;

#[derive(serde::Deserialize)]
pub struct Args {
    pub question: String,
}

pub fn def() -> ToolDef {
    ToolDef::new(
        "get_context",
        "Search previous chapters for translation history and chapter notes.\n\n\
            Behavior:\n\
            - A sub-agent searches the project database which contains:\n\
              \u{2022} Prior translations: source \u{2192} translated text for every bubble in past chapters.\n\
              \u{2022} Chapter notes (saved by add_note): character introductions, relationships, \
                plot events, setting descriptions.\n\
            - Ask a specific question mentioning names, terms, or relationships you need clarified.\n\
            - Returns a concise factual answer from database results.\n\n\
            When to use: encountering a character/term/relationship that may have appeared in earlier chapters.\n\
            When NOT to use: first chapter of a project (no prior context exists).",
        serde_json::json!({
            "type": "object",
            "required": ["question"],
            "additionalProperties": false,
            "properties": {
                "question": {
                    "type": "string",
                    "description": "Specific question about the project's translation history"
                }
            }
        }),
    )
    .strict()
}

pub async fn handle(args: &Args, ctx: &TranslateContext<'_>) -> ToolResponse {
    let response = if let (Some(store), Some(agent)) = (ctx.project, ctx.context_agent) {
        // Early exit: skip LLM sub-agent if DB has no data
        match store.has_data() {
            Ok(false) => {
                tracing::info!("get_context: no data in project store, skipping sub-agent");
                return ToolResponse::Text(
                    "No prior context exists for this project.".to_string(),
                );
            }
            Err(e) => {
                tracing::warn!("Context store check failed: {e}");
                return ToolResponse::Text(format!("Context check failed: {e}"));
            }
            Ok(true) => {}
        }

        tracing::info!("get_context: {:?}", args.question);
        match crate::storage::context::agent::answer_context_question(
            agent,
            store,
            &args.question,
        )
        .await
        {
            Ok(answer) if !answer.is_empty() => answer,
            Ok(_) => "No relevant context found.".to_string(),
            Err(e) => {
                tracing::warn!("Context agent failed: {e}");
                format!("Context search failed: {e}")
            }
        }
    } else {
        "Context agent not available.".to_string()
    };

    ToolResponse::Text(response)
}
