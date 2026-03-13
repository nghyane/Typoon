use anyhow::Result;

use crate::agent::{Message, Provider, ToolDef};
use crate::context::{ChapterTranslation, NoteMatch, TranslationMatch};
use crate::context::ContextStore;

const MAX_TURNS: usize = 5;

const SYSTEM_PROMPT: &str = "\
You are a context retrieval agent for a manga translation project.
Your job: answer the translation agent's question by searching the project database.

Tools available:
- search_translations(query): Full-text search over all translated bubbles
- search_notes(query): Search chapter notes (events, characters, relationships)
- read_chapter(chapter_index): Get all translations for a specific chapter

Workflow:
1. Analyze the question
2. Search relevant data using tools
3. Call answer() with a concise, focused response

Keep answers short and relevant. Only include information that directly answers the question.
Do not add speculation or information not found in the database.";

fn tool_defs() -> Vec<ToolDef> {
    vec![
        ToolDef::new(
            "search_translations",
            "Full-text search over all translated bubbles in the project",
            serde_json::json!({
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "Search terms"}
                },
                "required": ["query"]
            }),
        ),
        ToolDef::new(
            "search_notes",
            "Search chapter notes (events, characters, relationships)",
            serde_json::json!({
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "Search terms"}
                },
                "required": ["query"]
            }),
        ),
        ToolDef::new(
            "read_chapter",
            "Get all translations for a specific chapter",
            serde_json::json!({
                "type": "object",
                "properties": {
                    "chapter_index": {"type": "integer", "description": "Chapter number (0-based)"}
                },
                "required": ["chapter_index"]
            }),
        ),
        ToolDef::new(
            "answer",
            "Submit the final answer to the translation agent's question",
            serde_json::json!({
                "type": "object",
                "properties": {
                    "text": {"type": "string", "description": "The answer"}
                },
                "required": ["text"]
            }),
        ),
    ]
}

pub async fn answer_context_question(
    provider: &dyn Provider,
    store: &ContextStore,
    project_id: &str,
    question: &str,
) -> Result<String> {
    let tools = tool_defs();
    let mut messages = vec![
        Message::system(SYSTEM_PROMPT),
        Message::user_text(question),
    ];

    for turn in 0..MAX_TURNS {
        tracing::debug!("Context agent turn {turn}");

        let tool_calls = provider.call(&messages, &tools).await?;

        if tool_calls.is_empty() {
            break;
        }

        messages.push(Message::Assistant {
            tool_calls: tool_calls.clone(),
        });

        for tc in &tool_calls {
            // Parse arguments as JSON Value for field access
            let input: serde_json::Value =
                serde_json::from_str(&tc.arguments).unwrap_or_default();

            let result = match tc.name.as_str() {
                "search_translations" => {
                    let query = input["query"].as_str().unwrap_or("");
                    tracing::debug!("search_translations({:?})", query);
                    match store.search_translations(query, project_id, 20) {
                        Ok(matches) => format_translation_matches(&matches),
                        Err(e) => format!("Search error: {e}"),
                    }
                }
                "search_notes" => {
                    let query = input["query"].as_str().unwrap_or("");
                    tracing::debug!("search_notes({:?})", query);
                    match store.search_notes(query, project_id, 20) {
                        Ok(matches) => format_note_matches(&matches),
                        Err(e) => format!("Search error: {e}"),
                    }
                }
                "read_chapter" => {
                    let chapter_index = input["chapter_index"].as_u64().unwrap_or(0) as usize;
                    tracing::debug!("read_chapter({})", chapter_index);
                    match store.get_chapter_translations(project_id, chapter_index) {
                        Ok(translations) => format_chapter_translations(&translations),
                        Err(e) => format!("Read error: {e}"),
                    }
                }
                "answer" => {
                    let text = input["text"].as_str().unwrap_or("").to_string();
                    tracing::info!("Context agent answered: {} chars", text.len());
                    return Ok(text);
                }
                other => {
                    tracing::warn!("Unknown tool call: {other}");
                    "Unknown tool.".to_string()
                }
            };

            messages.push(Message::tool_result_text(&tc.id, result));
        }
    }

    tracing::warn!("Context agent exhausted turns without calling answer()");
    Ok(String::new())
}

fn format_translation_matches(matches: &[TranslationMatch]) -> String {
    if matches.is_empty() {
        return "No results found.".to_string();
    }
    let mut out = format!("Found {} results:\n", matches.len());
    for m in matches {
        out.push_str(&format!(
            "  [Ch{} p{} {}] {} → {}\n",
            m.chapter_index, m.page_index, m.bubble_id, m.source_text, m.translated_text
        ));
    }
    out
}

fn format_note_matches(matches: &[NoteMatch]) -> String {
    if matches.is_empty() {
        return "No results found.".to_string();
    }
    let mut out = format!("Found {} results:\n", matches.len());
    for m in matches {
        out.push_str(&format!(
            "  [Ch{} {}] {}\n",
            m.chapter_index, m.note_type, m.content
        ));
    }
    out
}

fn format_chapter_translations(translations: &[ChapterTranslation]) -> String {
    if translations.is_empty() {
        return "No translations found for this chapter.".to_string();
    }
    let mut out = format!("Chapter has {} translations:\n", translations.len());
    for t in translations {
        out.push_str(&format!(
            "  [p{} {}] {} → {}\n",
            t.page_index, t.bubble_id, t.source_text, t.translated_text
        ));
    }
    out
}
