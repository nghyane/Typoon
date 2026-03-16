use crate::agent::{self, ToolResponse};
use crate::translation::TranslateContext;

#[derive(serde::Deserialize)]
pub struct Args {
    pub query: String,
}

pub fn def() -> agent::ToolDef {
    agent::ToolDef::new(
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

pub fn handle(args: &Args, ctx: &TranslateContext<'_>) -> ToolResponse {
    let response = if let Some(glossary) = ctx.glossary {
        match glossary.search(&args.query) {
            Ok(entries) if entries.is_empty() => "No matching glossary entries found.".to_string(),
            Ok(entries) => {
                let mut out = format!("Found {} entries:\n", entries.len());
                for e in &entries {
                    out.push_str(&format!("  {} → {}", e.source_term, e.target_term));
                    if let Some(n) = &e.notes {
                        out.push_str(&format!(" ({})", n));
                    }
                    out.push('\n');
                }
                out
            }
            Err(e) => format!("Search error: {e}"),
        }
    } else {
        "Glossary not available.".to_string()
    };

    tracing::info!(
        "search_glossary({:?}) → {} chars",
        args.query,
        response.len()
    );
    ToolResponse::Text(response)
}
