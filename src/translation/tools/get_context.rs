use crate::agent::{self, ToolResponse};
use crate::translation::TranslateContext;

#[derive(serde::Deserialize)]
pub struct Args {
    pub question: String,
}

pub fn def() -> agent::ToolDef {
    agent::ToolDef::new(
        "get_context",
        "Ask the context agent to search previous chapters for relevant information.\n\n\
            Behavior:\n\
            - A sub-agent searches the project's translation history and chapter notes.\n\
            - Ask specific questions: character names, relationships, past events, established terms.\n\
            - Returns a focused answer based on database search results.\n\n\
            When to use: encountering a character/term/event that may have appeared in earlier chapters.\n\
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
        tracing::info!("get_context: {:?}", args.question);
        match crate::context::agent::answer_context_question(
            agent, store, project_id, &args.question,
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
