/// Translation agent — orchestrates LLM tool-calling loop for manga/manhwa translation.
///
/// Implements the `Agent` trait. Completeness checking is handled via `retry_prompt`.
use std::collections::HashMap;
use std::future::Future;

use anyhow::Context;
use image::DynamicImage;

use crate::llm::{ContentPart, Message, Provider, ToolCallMsg, ToolDef, ToolResponse};
use crate::storage::project::ProjectStore;
use crate::translation::prompt::PromptBuilder;
use crate::translation::tools;
use crate::translation::{BubbleTranslated, TranslateRequest};

use super::Agent;

const MAX_RETRY_ROUNDS: usize = 5;

pub struct TranslationAgent<'a> {
    // Dispatch resources
    page_images: &'a [DynamicImage],
    project: Option<&'a ProjectStore>,
    context_provider: Option<&'a dyn Provider>,

    // Result accumulation
    results: Vec<BubbleTranslated>,
    source_lookup: HashMap<String, String>,
    total_bubbles: usize,
    retry_count: usize,

    // Cached prompt data
    system: String,
    user_msg: Message,
    tool_defs: Vec<ToolDef>,
}

impl<'a> TranslationAgent<'a> {
    pub fn new(
        req: &TranslateRequest<'_>,
        page_images: &'a [DynamicImage],
        project: Option<&'a ProjectStore>,
        context_provider: Option<&'a dyn Provider>,
    ) -> Self {
        let has_images = !page_images.is_empty();
        let has_glossary = project.is_some();
        let has_context = project.is_some() && context_provider.is_some();

        let builder = PromptBuilder::new(req, page_images.len(), has_glossary, has_context);
        let system = builder.system_prompt();
        let user_prompt = builder.user_prompt();

        let is_single_page = req.detections.len() == 1;
        let user_msg = if is_single_page && page_images.len() == 1 {
            let data_uri = tools::view_page::encode_page_jpeg(&page_images[0]);
            Message::user_parts(vec![
                ContentPart::Text(user_prompt),
                ContentPart::Image { data_uri },
            ])
        } else {
            Message::user_text(user_prompt)
        };

        let tool_defs = tools::build_tools(has_images, has_glossary, has_context);

        let source_lookup: HashMap<String, String> = req
            .all_bubbles()
            .map(|(page_idx, b)| {
                (
                    TranslateRequest::bubble_id(page_idx, b.idx),
                    b.source_text.clone(),
                )
            })
            .collect();

        let total_bubbles: usize = req.detections.iter().map(|pd| pd.bubbles.len()).sum();

        Self {
            page_images,
            project,
            context_provider,
            results: Vec::new(),
            source_lookup,
            total_bubbles,
            retry_count: 0,
            system,
            user_msg,
            tool_defs,
        }
    }

    fn missing_ids(&self) -> Vec<String> {
        let translated: std::collections::HashSet<&str> =
            self.results.iter().map(|r| r.id.as_str()).collect();
        self.source_lookup
            .keys()
            .filter(|id| !translated.contains(id.as_str()))
            .cloned()
            .collect()
    }
}

impl Agent for TranslationAgent<'_> {
    type Output = Vec<BubbleTranslated>;

    fn name(&self) -> &'static str {
        "translate"
    }

    fn system_prompt(&self) -> String {
        self.system.clone()
    }

    fn user_message(&self) -> Message {
        self.user_msg.clone()
    }

    fn tools(&self) -> Vec<ToolDef> {
        self.tool_defs.clone()
    }

    fn dispatch<'a>(
        &'a mut self,
        call: &'a ToolCallMsg,
    ) -> impl Future<Output = ToolResponse> + Send + 'a {
        async move {
            match call.name.as_str() {
                "translate" => {
                    match serde_json::from_str::<tools::translate::Args>(&call.arguments)
                        .with_context(|| format!("Bad translate args: {}", call.arguments))
                    {
                        Ok(args) => {
                            let count = args.translations.len();
                            for item in args.translations {
                                tracing::debug!(
                                    "translate [{}] → {:?}",
                                    item.id,
                                    item.translated_text
                                );
                                let source = self
                                    .source_lookup
                                    .get(&item.id)
                                    .cloned()
                                    .unwrap_or_default();
                                self.results.push(BubbleTranslated {
                                    id: item.id,
                                    source_text: source,
                                    translated_text: item.translated_text,
                                });
                            }
                            tracing::info!(
                                "Translated {count} bubbles ({}/{})",
                                self.results.len(),
                                self.total_bubbles,
                            );
                            ToolResponse::Text(format!("ok ({count} bubbles)"))
                        }
                        Err(e) => ToolResponse::Text(format!("Error: {e}")),
                    }
                }

                "view_page" => {
                    match serde_json::from_str::<tools::view_page::Args>(&call.arguments) {
                        Ok(args) => {
                            if args.page_index < self.page_images.len() {
                                tracing::info!("Agent viewing page {}", args.page_index);
                                let data_uri = tools::view_page::encode_page_jpeg(
                                    &self.page_images[args.page_index],
                                );
                                ToolResponse::ImageContent {
                                    text: format!("Page {} image:", args.page_index),
                                    data_uri,
                                }
                            } else {
                                ToolResponse::Text(format!(
                                    "Error: page_index {} out of range (0..{})",
                                    args.page_index,
                                    self.page_images.len(),
                                ))
                            }
                        }
                        Err(e) => ToolResponse::Text(format!("Bad args: {e}")),
                    }
                }

                "search_glossary" => {
                    match serde_json::from_str::<tools::search_glossary::Args>(&call.arguments) {
                        Ok(args) => {
                            let response = if let Some(project) = self.project {
                                match project.glossary_search(&args.query) {
                                    Ok(entries) if entries.is_empty() => {
                                        "No matching glossary entries found.".to_string()
                                    }
                                    Ok(entries) => {
                                        let mut out =
                                            format!("Found {} entries:\n", entries.len());
                                        for e in &entries {
                                            out.push_str(&format!(
                                                "  {} -> {}",
                                                e.source_term, e.target_term
                                            ));
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
                                "Glossary not available.".into()
                            };
                            tracing::info!(
                                "search_glossary({:?}) → {} chars",
                                args.query,
                                response.len(),
                            );
                            ToolResponse::Text(response)
                        }
                        Err(e) => ToolResponse::Text(format!("Bad args: {e}")),
                    }
                }

                "get_context" => {
                    match serde_json::from_str::<tools::get_context::Args>(&call.arguments) {
                        Ok(args) => {
                            if let (Some(store), Some(provider)) =
                                (self.project, self.context_provider)
                            {
                                match store.has_data() {
                                    Ok(false) => {
                                        tracing::info!(
                                            "get_context: no data in store, skipping"
                                        );
                                        ToolResponse::Text(
                                            "No prior context exists for this project."
                                                .to_string(),
                                        )
                                    }
                                    Err(e) => {
                                        ToolResponse::Text(format!("Context check failed: {e}"))
                                    }
                                    Ok(true) => {
                                        tracing::info!("get_context: {:?}", args.question);
                                        let ctx_agent = super::context::ContextAgent::new(
                                            store,
                                            &args.question,
                                        );
                                        match super::run(provider, ctx_agent).await {
                                            Ok(answer) if !answer.is_empty() => {
                                                ToolResponse::Text(answer)
                                            }
                                            Ok(_) => ToolResponse::Text(
                                                "No relevant context found.".into(),
                                            ),
                                            Err(e) => {
                                                tracing::warn!("Context agent failed: {e}");
                                                ToolResponse::Text(format!(
                                                    "Context search failed: {e}"
                                                ))
                                            }
                                        }
                                    }
                                }
                            } else {
                                ToolResponse::Text("Context agent not available.".into())
                            }
                        }
                        Err(e) => ToolResponse::Text(format!("Bad args: {e}")),
                    }
                }

                other => {
                    tracing::warn!("Unknown tool call: {other}");
                    ToolResponse::Text("unknown tool".into())
                }
            }
        }
    }

    fn is_done(&self) -> bool {
        self.results.len() >= self.total_bubbles
    }

    fn retry_prompt(&mut self) -> Option<String> {
        if self.retry_count >= MAX_RETRY_ROUNDS {
            return None;
        }
        let missing = self.missing_ids();
        if missing.is_empty() {
            return None;
        }

        self.retry_count += 1;
        tracing::warn!(
            "Completeness retry {}: {} bubbles missing",
            self.retry_count,
            missing.len(),
        );

        let mut prompt = format!(
            "You missed {} bubbles. Translate these now:\n\n",
            missing.len(),
        );
        for id in &missing {
            let source = self.source_lookup.get(id).map(|s| s.as_str()).unwrap_or("");
            prompt.push_str(&format!("[{id}] \"{source}\"\n"));
        }
        Some(prompt)
    }

    fn into_output(mut self) -> Vec<BubbleTranslated> {
        dedup(&mut self.results);
        self.results
    }
}

fn dedup(results: &mut Vec<BubbleTranslated>) {
    let mut seen = std::collections::HashSet::new();
    results.reverse();
    results.retain(|r| seen.insert(r.id.clone()));
    results.reverse();
}
