use anyhow::Result;

use crate::agent::{Message, Provider, ToolDef};
use crate::context::{ChapterTranslation, ContextHit, ContextStore, SearchScope};

const SYSTEM_PROMPT: &str = "\
You are a context retrieval sub-agent for a manga translation project.

The database contains two types of records from previous chapters:
- translations: source text → translated text for every bubble.
- chapter notes (4 types): character (introductions), relationship (who knows whom), \
  event (plot points), setting (locations/workplace/school).

You have 2 tools:
- search(queries, scope): batch search across translations and/or notes. \
  Pass 2-6 short keyword queries (names, terms, relationship pairs) in one call. \
  Use scope='notes' for character/relationship/event/setting info, \
  scope='translations' for prior wording, scope='all' if unsure.
- read_chapter(chapter_index): get all translations for a specific chapter. \
  Use only if search results point to a chapter you need full context from.

Rules:
- Call search() ONCE with ALL relevant queries in a single message.
- After reviewing results, respond with a concise text answer (no tool calls).
- Only your last text message is returned to the caller.
- Plain text only, no markdown. Answer directly with facts from the database.
- Do not speculate or add information not found in search results.";

fn tool_defs() -> Vec<ToolDef> {
    vec![
        ToolDef::new(
            "search",
            "Batch search across translations and/or notes. Pass 2-6 short keyword queries.",
            serde_json::json!({
                "type": "object",
                "properties": {
                    "queries": {
                        "type": "array",
                        "items": { "type": "string" },
                        "description": "Short keyword queries, e.g. [\"Max\", \"Joy\", \"team leader\", \"office setting\"]"
                    },
                    "scope": {
                        "type": "string",
                        "enum": ["all", "translations", "notes"],
                        "description": "Search scope: 'translations' for prior wording, 'notes' for relationships/events, 'all' if unsure"
                    }
                },
                "required": ["queries"]
            }),
        ),
        ToolDef::new(
            "read_chapter",
            "Get all translations for a specific chapter. Use only when search points to a chapter needing full context.",
            serde_json::json!({
                "type": "object",
                "properties": {
                    "chapter_index": {"type": "integer", "description": "Chapter number (0-based)"}
                },
                "required": ["chapter_index"]
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
    // Early exit: no data in DB for this project
    if !store.has_data(project_id)? {
        tracing::info!("Context agent: no data for project {project_id}, skipping");
        return Ok(String::new());
    }

    let tools = tool_defs();
    let mut messages = vec![
        Message::system(SYSTEM_PROMPT),
        Message::user_text(question),
    ];

    loop {
        let resp = provider.call(&messages, &tools).await?;

        // No tool calls → text is the final answer
        if resp.tool_calls.is_empty() {
            let answer = resp.text.unwrap_or_default();
            tracing::info!("Context agent answered ({} chars):\n{}", answer.len(), answer);
            return Ok(answer);
        }

        // Has tool calls → execute them, continue loop
        messages.push(Message::Assistant {
            text: resp.text,
            tool_calls: resp.tool_calls.clone(),
        });

        for tc in &resp.tool_calls {
            let input: serde_json::Value =
                serde_json::from_str(&tc.arguments).unwrap_or_default();

            let result = match tc.name.as_str() {
                "search" => {
                    let queries: Vec<String> = input["queries"]
                        .as_array()
                        .map(|arr| {
                            arr.iter()
                                .filter_map(|v| v.as_str().map(String::from))
                                .collect()
                        })
                        .unwrap_or_default();
                    let scope: SearchScope = input
                        .get("scope")
                        .and_then(|v| serde_json::from_value(v.clone()).ok())
                        .unwrap_or_default();

                    tracing::info!("search({:?}, {:?})", queries, scope);

                    if queries.is_empty() {
                        "No queries provided.".to_string()
                    } else {
                        match store.batch_search(project_id, &queries, scope, 12) {
                            Ok(hits) => format_hits(&hits),
                            Err(e) => format!("Search error: {e}"),
                        }
                    }
                }
                "read_chapter" => {
                    let chapter_index = input["chapter_index"].as_u64().unwrap_or(0) as usize;
                    tracing::info!("read_chapter({})", chapter_index);
                    match store.get_chapter_translations(project_id, chapter_index) {
                        Ok(translations) => format_chapter_translations(&translations),
                        Err(e) => format!("Read error: {e}"),
                    }
                }
                other => {
                    tracing::warn!("Unknown tool call: {other}");
                    "Unknown tool.".to_string()
                }
            };

            messages.push(Message::tool_result_text(&tc.id, result));
        }
    }
}

fn format_hits(hits: &[ContextHit]) -> String {
    if hits.is_empty() {
        return "No results found.".to_string();
    }
    let mut out = format!("Found {} results:\n", hits.len());
    for h in hits {
        let kind = match h.kind {
            crate::context::ContextHitKind::Translation => "translation",
            crate::context::ContextHitKind::Note => "note",
        };
        out.push_str(&format!("  [Ch{} {}] {}\n", h.chapter_index, kind, h.summary));
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
