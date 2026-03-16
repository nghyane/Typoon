/// Context retrieval agent — searches project DB to answer questions
/// about characters, relationships, terms, and prior translations.
///
/// Used as a sub-agent by the translation agent (via `get_context` tool).
use std::future::Future;

use crate::llm::{Message, ToolCallMsg, ToolDef, ToolResponse};
use crate::storage::project::ProjectStore;

use super::Agent;

pub struct ContextAgent<'a> {
    store: &'a ProjectStore,
    question: String,
    answer: Option<String>,
}

impl<'a> ContextAgent<'a> {
    pub fn new(store: &'a ProjectStore, question: &str) -> Self {
        Self {
            store,
            question: question.to_string(),
            answer: None,
        }
    }
}

const SYSTEM_PROMPT: &str = "\
You are a context retrieval sub-agent for a manga translation project.

The database contains two types of records from previous chapters:
- translations: source text -> translated text for every bubble.
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
                        "description": "Short keyword queries, e.g. [\"Max\", \"Joy\", \"team leader\"]"
                    },
                    "scope": {
                        "type": "string",
                        "enum": ["all", "translations", "notes"],
                        "description": "Search scope"
                    }
                },
                "required": ["queries"]
            }),
        ),
        ToolDef::new(
            "read_chapter",
            "Get all translations for a specific chapter.",
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

impl Agent for ContextAgent<'_> {
    type Output = String;

    fn name(&self) -> &'static str {
        "context"
    }

    fn system_prompt(&self) -> String {
        SYSTEM_PROMPT.to_string()
    }

    fn user_message(&self) -> Message {
        Message::user_text(&self.question)
    }

    fn tools(&self) -> Vec<ToolDef> {
        tool_defs()
    }

    fn dispatch<'a>(
        &'a mut self,
        call: &'a ToolCallMsg,
    ) -> impl Future<Output = ToolResponse> + Send + 'a {
        async move {
            let input: serde_json::Value =
                serde_json::from_str(&call.arguments).unwrap_or_default();

            match call.name.as_str() {
                "search" => {
                    let queries: Vec<String> = input["queries"]
                        .as_array()
                        .map(|arr| {
                            arr.iter()
                                .filter_map(|v| v.as_str().map(String::from))
                                .collect()
                        })
                        .unwrap_or_default();
                    let scope = input
                        .get("scope")
                        .and_then(|v| v.as_str())
                        .unwrap_or("all");

                    tracing::info!("context search({:?}, {:?})", queries, scope);

                    if queries.is_empty() {
                        ToolResponse::Text("No queries provided.".into())
                    } else {
                        match self.store.batch_search_context(&queries, scope, 12) {
                            Ok(hits) => ToolResponse::Text(format_hits(&hits)),
                            Err(e) => ToolResponse::Text(format!("Search error: {e}")),
                        }
                    }
                }
                "read_chapter" => {
                    let chapter = input["chapter_index"].as_u64().unwrap_or(0) as usize;
                    tracing::info!("context read_chapter({})", chapter);
                    match self.store.get_chapter_pairs(chapter) {
                        Ok(pairs) => ToolResponse::Text(format_chapter_pairs(&pairs)),
                        Err(e) => ToolResponse::Text(format!("Read error: {e}")),
                    }
                }
                other => {
                    tracing::warn!("Context agent: unknown tool {other}");
                    ToolResponse::Text("Unknown tool.".into())
                }
            }
        }
    }

    fn on_text(&mut self, text: Option<&str>) {
        self.answer = text.map(|s| s.to_string());
    }

    fn into_output(self) -> String {
        self.answer.unwrap_or_default()
    }
}

fn format_hits(hits: &[String]) -> String {
    if hits.is_empty() {
        return "No results found.".to_string();
    }
    let mut out = format!("Found {} results:\n", hits.len());
    for h in hits {
        out.push_str(&format!("  {h}\n"));
    }
    out
}

fn format_chapter_pairs(pairs: &[(String, String)]) -> String {
    if pairs.is_empty() {
        return "No translations found for this chapter.".to_string();
    }
    let mut out = format!("Chapter has {} translations:\n", pairs.len());
    for (source, translated) in pairs {
        out.push_str(&format!("  {source} -> {translated}\n"));
    }
    out
}
