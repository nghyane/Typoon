use crate::agent::{self, ToolResponse};
use crate::translation::TranslateContext;

#[derive(serde::Deserialize)]
pub struct Args {
    pub question: String,
}

pub fn def() -> agent::ToolDef {
    agent::ToolDef::new(
        "get_context",
        "Search previous chapters for translation history and chapter notes.\n\n\
            Behavior:\n\
            - A sub-agent searches the project database which contains:\n\
              • Prior translations: source → translated text for every bubble in past chapters.\n\
              • Chapter notes (saved by add_note): character introductions, relationships, \
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
    let response = if let (Some(store), Some(agent), Some(project_id)) =
        (ctx.context_store, ctx.context_agent, ctx.project_id)
    {
        // Early exit: skip LLM sub-agent if DB has no data for this project
        match store.has_data(project_id) {
            Ok(false) => {
                tracing::info!("get_context: no data for project {project_id}, skipping sub-agent");
                return ToolResponse::Text("No prior context exists for this project.".to_string());
            }
            Err(e) => {
                tracing::warn!("Context store check failed: {e}");
                return ToolResponse::Text(format!("Context check failed: {e}"));
            }
            Ok(true) => {}
        }

        tracing::info!("get_context: {:?}", args.question);
        match crate::context::agent::answer_context_question(
            agent,
            store,
            project_id,
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
