/// Translation agent loop — orchestrates prompt building, LLM calls via Provider,
/// tool dispatch, completeness guardrails, and dedup.
///
/// Uses `agent::ir::Message` (typed IR) instead of raw JSON.
use anyhow::{Context, Result};

use crate::llm::{ContentPart, Message, Provider, ToolDef, ToolResponse};

use super::prompt::PromptBuilder;
use super::tools;
use super::{BubbleTranslated, TranslateContext, TranslateRequest};

/// Run the translation agent loop using the given provider.
pub async fn run(
    provider: &dyn Provider,
    req: &TranslateRequest,
    ctx: &TranslateContext<'_>,
) -> Result<Vec<BubbleTranslated>> {
    let builder = PromptBuilder::new(req, ctx);
    let system_prompt = builder.system_prompt();
    let user_prompt = builder.user_prompt();

    let has_images = !ctx.page_images.is_empty();
    let has_glossary = ctx.glossary.is_some();
    let has_context = ctx.context_store.is_some() && ctx.context_agent.is_some();
    let tool_defs = tools::build_tools(has_images, has_glossary, has_context);

    // Build initial messages
    let is_single_page = req.pages.len() == 1;
    let user_message = if is_single_page && ctx.page_images.len() == 1 {
        let data_uri = tools::view_page::encode_page_jpeg(&ctx.page_images[0]);
        Message::user_parts(vec![
            ContentPart::Text(user_prompt),
            ContentPart::Image { data_uri },
        ])
    } else {
        Message::user_text(user_prompt)
    };

    let mut messages = vec![Message::system(system_prompt), user_message];
    let mut results: Vec<BubbleTranslated> = Vec::new();

    // Lookup: id → source_text (for populating results without LLM echoing source)
    let source_lookup: std::collections::HashMap<&str, &str> = req
        .all_bubbles()
        .map(|b| (b.id.as_str(), b.source_text.as_str()))
        .collect();

    let total_bubbles = req.all_bubbles().count();

    // ── Main agent loop ──
    let mut turn = 0usize;
    loop {
        turn += 1;
        tracing::info!(
            "Agent turn {} — {}/{} bubbles translated",
            turn,
            results.len(),
            total_bubbles,
        );

        let resp = provider.call(&messages, &tool_defs).await?;

        if resp.tool_calls.is_empty() {
            break;
        }
        let tool_calls = resp.tool_calls;

        messages.push(Message::Assistant {
            text: resp.text,
            tool_calls: tool_calls.clone(),
        });

        for tc in &tool_calls {
            match tc.name.as_str() {
                "translate" => {
                    let args: tools::translate::Args = serde_json::from_str(&tc.arguments)
                        .with_context(|| format!("Bad translate args: {}", tc.arguments))?;
                    let count = args.translations.len();
                    for item in args.translations {
                        tracing::debug!("translate [{}] → {:?}", item.id, item.translated_text);
                        let source = source_lookup
                            .get(item.id.as_str())
                            .unwrap_or(&"")
                            .to_string();
                        results.push(BubbleTranslated {
                            id: item.id,
                            source_text: source,
                            translated_text: item.translated_text,
                        });
                    }
                    tracing::info!(
                        "Translated {count} bubbles ({}/{total_bubbles})",
                        results.len()
                    );
                    messages.push(Message::tool_result_text(
                        &tc.id,
                        format!("ok ({count} bubbles)"),
                    ));
                }

                "view_page" => {
                    let args: tools::view_page::Args = serde_json::from_str(&tc.arguments)
                        .with_context(|| format!("Bad view_page args: {}", tc.arguments))?;
                    let resp = tools::view_page::handle(&args, ctx);
                    messages.push(tool_response_to_message(&tc.id, resp));
                }

                "search_glossary" => {
                    let args: tools::search_glossary::Args = serde_json::from_str(&tc.arguments)
                        .with_context(|| format!("Bad search_glossary args: {}", tc.arguments))?;
                    let resp = tools::search_glossary::handle(&args, ctx);
                    messages.push(tool_response_to_message(&tc.id, resp));
                }

                "update_glossary" => {
                    let args: tools::update_glossary::Args = serde_json::from_str(&tc.arguments)
                        .with_context(|| format!("Bad update_glossary args: {}", tc.arguments))?;
                    let resp = tools::update_glossary::handle(&args, ctx);
                    messages.push(tool_response_to_message(&tc.id, resp));
                }

                "get_context" => {
                    let args: tools::get_context::Args = serde_json::from_str(&tc.arguments)
                        .with_context(|| format!("Bad get_context args: {}", tc.arguments))?;
                    let resp = tools::get_context::handle(&args, ctx).await;
                    messages.push(tool_response_to_message(&tc.id, resp));
                }

                "add_note" => {
                    let args: tools::add_note::Args = serde_json::from_str(&tc.arguments)
                        .with_context(|| format!("Bad add_note args: {}", tc.arguments))?;
                    let resp = tools::add_note::handle(&args, ctx);
                    messages.push(tool_response_to_message(&tc.id, resp));
                }

                other => {
                    tracing::warn!("Unknown tool call: {other}");
                    messages.push(Message::tool_result_text(&tc.id, "unknown tool"));
                }
            }
        }

        if results.len() >= total_bubbles {
            break;
        }
    }

    // ── Completeness guardrail ──
    completeness_check(
        provider,
        req,
        &mut messages,
        &tool_defs,
        &mut results,
        &source_lookup,
    )
    .await?;

    // ── Dedup: keep last translation per ID ──
    dedup(&mut results);

    Ok(results)
}

const MAX_COMPLETION_ROUNDS: usize = 5;

async fn completeness_check(
    provider: &dyn Provider,
    req: &TranslateRequest,
    messages: &mut Vec<Message>,
    tool_defs: &[ToolDef],
    results: &mut Vec<BubbleTranslated>,
    source_lookup: &std::collections::HashMap<&str, &str>,
) -> Result<()> {
    let expected_ids: std::collections::HashSet<&str> =
        req.all_bubbles().map(|b| b.id.as_str()).collect();

    for round in 0..MAX_COMPLETION_ROUNDS {
        let translated_ids: std::collections::HashSet<&str> =
            results.iter().map(|r| r.id.as_str()).collect();
        let missing: Vec<&str> = expected_ids.difference(&translated_ids).copied().collect();

        if missing.is_empty() {
            return Ok(());
        }

        tracing::warn!(
            "Completeness round {}: {} bubbles missing",
            round + 1,
            missing.len(),
        );

        // Re-send missing bubbles with source text so LLM doesn't need to search history.
        let mut retry_prompt = format!(
            "You missed {} bubbles. Translate these now, then call done():\n\n",
            missing.len(),
        );
        for id in &missing {
            let source = source_lookup.get(id).unwrap_or(&"");
            retry_prompt.push_str(&format!("[{id}] \"{source}\"\n"));
        }
        messages.push(Message::user_text(retry_prompt));

        match provider.call(messages, tool_defs).await {
            Ok(resp) if !resp.tool_calls.is_empty() => {
                let tool_calls = resp.tool_calls;
                messages.push(Message::Assistant {
                    text: resp.text,
                    tool_calls: tool_calls.clone(),
                });

                for tc in &tool_calls {
                    if tc.name == "translate" {
                        if let Ok(args) =
                            serde_json::from_str::<tools::translate::Args>(&tc.arguments)
                        {
                            for item in args.translations {
                                let source = source_lookup
                                    .get(item.id.as_str())
                                    .unwrap_or(&"")
                                    .to_string();
                                results.push(BubbleTranslated {
                                    id: item.id,
                                    source_text: source,
                                    translated_text: item.translated_text,
                                });
                            }
                        }
                    }
                    messages.push(Message::tool_result_text(&tc.id, "ok"));
                }
            }
            _ => break,
        }
    }

    Ok(())
}

fn dedup(results: &mut Vec<BubbleTranslated>) {
    let mut seen = std::collections::HashSet::new();
    results.reverse();
    results.retain(|r| seen.insert(r.id.clone()));
    results.reverse();
}

fn tool_response_to_message(tool_call_id: &str, response: ToolResponse) -> Message {
    match response {
        ToolResponse::Text(text) => Message::tool_result_text(tool_call_id, text),
        ToolResponse::ImageContent { text, data_uri } => Message::tool_result_parts(
            tool_call_id,
            vec![ContentPart::Text(text), ContentPart::Image { data_uri }],
        ),
    }
}
