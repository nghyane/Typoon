use crate::llm::ToolDef;

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
              \u{2022} Chapter notes: character introductions, relationships, plot events, settings.\n\
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
