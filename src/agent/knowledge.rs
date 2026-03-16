/// Knowledge consolidation agent — runs post-translation to extract and
/// persist series knowledge (characters, relationships, terms, events).
///
/// Spawned alongside render for zero added latency.
/// Uses `Arc<ProjectStore>` since it outlives the caller (spawned on tokio).
use std::future::Future;
use std::sync::Arc;

use crate::llm::{Message, ToolCallMsg, ToolDef, ToolResponse};
use crate::storage::project::ProjectStore;

use super::Agent;

pub struct KnowledgeAgent {
    store: Arc<ProjectStore>,
    chapter_index: usize,
    source_lang: String,
    target_lang: String,
    previous_snapshot: Option<String>,
    pairs: Vec<(String, String)>,
    done: bool,
}

impl KnowledgeAgent {
    pub fn new(
        store: Arc<ProjectStore>,
        chapter_index: usize,
        source_lang: &str,
        target_lang: &str,
        previous_snapshot: Option<String>,
        pairs: Vec<(String, String)>,
    ) -> Self {
        Self {
            store,
            chapter_index,
            source_lang: source_lang.to_string(),
            target_lang: target_lang.to_string(),
            previous_snapshot,
            pairs,
            done: false,
        }
    }
}

impl Agent for KnowledgeAgent {
    type Output = ();

    fn name(&self) -> &'static str {
        "knowledge"
    }

    fn system_prompt(&self) -> String {
        format!(
            "\
You are a knowledge extraction agent for a manga/manhwa/manhua translation project ({} → {}).

Your task: given this chapter's translated dialogue, extract and consolidate series knowledge.

You have 3 tools:
- save_snapshot: save the COMPLETE updated knowledge state as a single text block.
  This snapshot is injected into future chapter prompts, so it must be self-contained.
- upsert_glossary: add/update a canonical recurring term (names, places, techniques).
- append_note: record a chapter-specific event or setting change.

Workflow:
1. Read all translated dialogue below.
2. Identify: characters (with descriptions, speech style), relationships, recurring terms, key events.
3. Merge with previous snapshot if available — update existing entries, add new ones, remove outdated info.
4. Call save_snapshot() with the complete knowledge state.
5. Call upsert_glossary() for recurring terms (character names, places, organizations, techniques).
6. Call append_note() for significant events or setting changes.

Snapshot format (keep compact, ~500-800 chars):
Characters:
- Name (original): age/role, speech style, pronouns
Relationships:
- A ↔ B: type, dynamics
Recent events (last 3-5 chapters):
- ChN: brief summary

Rules:
- Snapshot must be self-contained — a new reader should understand the series state.
- Keep it concise. Drop old events (>5 chapters ago) unless plot-critical.
- All names should include both original and translated forms.
- Call all tools in ONE message. Do not trickle across turns.",
            self.source_lang, self.target_lang,
        )
    }

    fn user_message(&self) -> Message {
        let mut prompt = String::new();

        match &self.previous_snapshot {
            Some(snapshot) => {
                prompt.push_str("Previous knowledge snapshot:\n");
                prompt.push_str(snapshot);
                prompt.push_str("\n\n");
            }
            None => {
                prompt.push_str("Previous knowledge snapshot: (none — first chapter)\n\n");
            }
        }

        prompt.push_str(&format!(
            "Chapter {} translations ({} → {}):\n",
            self.chapter_index, self.source_lang, self.target_lang,
        ));
        for (source, translated) in &self.pairs {
            prompt.push_str(&format!("  \"{}\" → \"{}\"\n", source, translated));
        }

        Message::user_text(prompt)
    }

    fn tools(&self) -> Vec<ToolDef> {
        vec![
            ToolDef::new(
                "save_snapshot",
                "Save the complete updated knowledge snapshot for this series.",
                serde_json::json!({
                    "type": "object",
                    "required": ["snapshot"],
                    "additionalProperties": false,
                    "properties": {
                        "snapshot": {
                            "type": "string",
                            "description": "Complete knowledge state as formatted text"
                        }
                    }
                }),
            )
            .strict(),
            ToolDef::new(
                "upsert_glossary",
                "Add or update a canonical recurring term.",
                serde_json::json!({
                    "type": "object",
                    "required": ["source_term", "target_term", "notes"],
                    "additionalProperties": false,
                    "properties": {
                        "source_term": {
                            "type": "string",
                            "description": "Original term in source language"
                        },
                        "target_term": {
                            "type": "string",
                            "description": "Canonical translated term"
                        },
                        "notes": {
                            "type": "string",
                            "description": "Brief context (e.g., character name, place, technique)"
                        }
                    }
                }),
            )
            .strict(),
            ToolDef::new(
                "append_note",
                "Record a chapter-specific event or setting detail.",
                serde_json::json!({
                    "type": "object",
                    "required": ["note_type", "content"],
                    "additionalProperties": false,
                    "properties": {
                        "note_type": {
                            "type": "string",
                            "enum": ["event", "setting"],
                            "description": "event=plot points, setting=locations/environment"
                        },
                        "content": {
                            "type": "string",
                            "description": "Concise observation"
                        }
                    }
                }),
            )
            .strict(),
        ]
    }

    fn dispatch<'a>(
        &'a mut self,
        call: &'a ToolCallMsg,
    ) -> impl Future<Output = ToolResponse> + Send + 'a {
        async move {
            match call.name.as_str() {
                "save_snapshot" => {
                    #[derive(serde::Deserialize)]
                    struct Args {
                        snapshot: String,
                    }
                    match serde_json::from_str::<Args>(&call.arguments) {
                        Ok(args) => {
                            match self
                                .store
                                .save_snapshot(self.chapter_index, &args.snapshot)
                            {
                                Ok(()) => {
                                    self.done = true;
                                    tracing::info!(
                                        "Knowledge snapshot saved for chapter {} ({} chars)",
                                        self.chapter_index,
                                        args.snapshot.len(),
                                    );
                                    ToolResponse::Text("Snapshot saved.".into())
                                }
                                Err(e) => ToolResponse::Text(format!("Save failed: {e}")),
                            }
                        }
                        Err(e) => ToolResponse::Text(format!("Bad args: {e}")),
                    }
                }
                "upsert_glossary" => {
                    #[derive(serde::Deserialize)]
                    struct Args {
                        source_term: String,
                        target_term: String,
                        notes: String,
                    }
                    match serde_json::from_str::<Args>(&call.arguments) {
                        Ok(args) => {
                            match self.store.glossary_upsert(
                                &args.source_term,
                                &args.target_term,
                                Some(&args.notes),
                            ) {
                                Ok(()) => {
                                    tracing::debug!(
                                        "Glossary: {} → {}",
                                        args.source_term,
                                        args.target_term,
                                    );
                                    ToolResponse::Text(format!(
                                        "Saved: {} → {}",
                                        args.source_term, args.target_term,
                                    ))
                                }
                                Err(e) => ToolResponse::Text(format!("Glossary error: {e}")),
                            }
                        }
                        Err(e) => ToolResponse::Text(format!("Bad args: {e}")),
                    }
                }
                "append_note" => {
                    #[derive(serde::Deserialize)]
                    struct Args {
                        note_type: String,
                        content: String,
                    }
                    match serde_json::from_str::<Args>(&call.arguments) {
                        Ok(args) => {
                            match self
                                .store
                                .add_note(self.chapter_index, &args.note_type, &args.content)
                            {
                                Ok(()) => {
                                    tracing::debug!(
                                        "Note [{}]: {}",
                                        args.note_type,
                                        &args.content[..args.content.len().min(80)],
                                    );
                                    ToolResponse::Text("Note saved.".into())
                                }
                                Err(e) => ToolResponse::Text(format!("Note error: {e}")),
                            }
                        }
                        Err(e) => ToolResponse::Text(format!("Bad args: {e}")),
                    }
                }
                other => {
                    tracing::warn!("Knowledge agent: unknown tool {other}");
                    ToolResponse::Text("Unknown tool.".into())
                }
            }
        }
    }

    fn is_done(&self) -> bool {
        self.done
    }

    fn into_output(self) {}
}
