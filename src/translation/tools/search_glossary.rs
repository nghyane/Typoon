use crate::llm::ToolDef;

#[derive(serde::Deserialize)]
pub struct Args {
    pub query: String,
}

pub fn def() -> ToolDef {
    ToolDef::new(
        "search_glossary",
        "Search the persistent glossary for canonical translations.\n\n\
            Behavior:\n\
            - Search BEFORE inventing a translation for names, titles, places, \
            organizations, abilities, items, or recurring terms.\n\
            - Batch multiple search_glossary() calls in one message.\n\
            - If no useful entry found, translate naturally and continue.\n\n\
            When to use: proper nouns or recurring terms likely to appear again.\n\
            When NOT to use: ordinary words or one-off everyday phrases.",
        serde_json::json!({
            "type": "object",
            "required": ["query"],
            "additionalProperties": false,
            "properties": {
                "query": {
                    "type": "string",
                    "description": "Source-language term to look up"
                }
            }
        }),
    )
    .strict()
}
