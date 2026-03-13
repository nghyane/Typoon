/// Translation agent loop — orchestrates prompt building, LLM calls via Provider,
/// tool dispatch, completeness guardrails, and dedup.
///
/// Uses `agent::ir::Message` (typed IR) instead of raw JSON.
use anyhow::{Context, Result};

use crate::agent::{self, ContentPart, Message, Provider, ToolResponse};

use super::{BubbleTranslated, TranslateContext, TranslateRequest};
use super::prompt::PromptBuilder;
use super::tools;

const MAX_AGENT_TURNS: usize = 15;

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

    // ── Main agent loop ──
    for turn in 0..MAX_AGENT_TURNS {
        tracing::debug!("Translation agent turn {turn}");

        let tool_calls = provider.call(&messages, &tool_defs).await?;

        if tool_calls.is_empty() {
            break;
        }

        messages.push(Message::Assistant {
            tool_calls: tool_calls.clone(),
        });

        let mut got_done = false;

        for tc in &tool_calls {
            match tc.name.as_str() {
                "translate" => {
                    let args: tools::translate::Args =
                        serde_json::from_str(&tc.arguments)
                            .with_context(|| format!("Bad translate args: {}", tc.arguments))?;
                    tracing::debug!("translate [{}] → {:?}", args.id, args.translated_text);
                    results.push(BubbleTranslated {
                        id: args.id,
                        source_text: args.source_text,
                        translated_text: args.translated_text,
                    });
                    messages.push(Message::tool_result_text(&tc.id, "ok"));
                }

                "view_page" => {
                    let args: tools::view_page::Args =
                        serde_json::from_str(&tc.arguments)
                            .with_context(|| format!("Bad view_page args: {}", tc.arguments))?;
                    let resp = tools::view_page::handle(&args, ctx);
                    messages.push(tool_response_to_message(&tc.id, resp));
                }

                "search_glossary" => {
                    let args: tools::search_glossary::Args =
                        serde_json::from_str(&tc.arguments)
                            .with_context(|| format!("Bad search_glossary args: {}", tc.arguments))?;
                    let resp = tools::search_glossary::handle(&args, ctx);
                    messages.push(tool_response_to_message(&tc.id, resp));
                }

                "update_glossary" => {
                    let args: tools::update_glossary::Args =
                        serde_json::from_str(&tc.arguments)
                            .with_context(|| format!("Bad update_glossary args: {}", tc.arguments))?;
                    let resp = tools::update_glossary::handle(&args, ctx);
                    messages.push(tool_response_to_message(&tc.id, resp));
                }

                "get_context" => {
                    let args: tools::get_context::Args =
                        serde_json::from_str(&tc.arguments)
                            .with_context(|| format!("Bad get_context args: {}", tc.arguments))?;
                    let resp = tools::get_context::handle(&args, ctx).await;
                    messages.push(tool_response_to_message(&tc.id, resp));
                }

                "add_note" => {
                    let args: tools::add_note::Args =
                        serde_json::from_str(&tc.arguments)
                            .with_context(|| format!("Bad add_note args: {}", tc.arguments))?;
                    let resp = tools::add_note::handle(&args, ctx);
                    messages.push(tool_response_to_message(&tc.id, resp));
                }

                "done" => {
                    tracing::info!("Translation agent done ({} results)", results.len());
                    messages.push(Message::tool_result_text(&tc.id, "ok"));
                    got_done = true;
                }

                other => {
                    tracing::warn!("Unknown tool call: {other}");
                    messages.push(Message::tool_result_text(&tc.id, "unknown tool"));
                }
            }
        }

        if got_done {
            break;
        }
    }

    // ── Completeness guardrail ──
    completeness_check(provider, req, &mut messages, &tool_defs, &mut results).await?;

    // ── Dedup: keep last translation per ID ──
    dedup(&mut results);

    Ok(results)
}

async fn completeness_check(
    provider: &dyn Provider,
    req: &TranslateRequest,
    messages: &mut Vec<Message>,
    tool_defs: &[agent::ToolDef],
    results: &mut Vec<BubbleTranslated>,
) -> Result<()> {
    let expected_ids: std::collections::HashSet<&str> =
        req.all_bubbles().map(|b| b.id.as_str()).collect();
    let translated_ids: std::collections::HashSet<&str> =
        results.iter().map(|r| r.id.as_str()).collect();
    let missing: Vec<&str> = expected_ids.difference(&translated_ids).copied().collect();

    if missing.is_empty() {
        return Ok(());
    }

    tracing::warn!(
        "Translation agent missed {} bubbles: {:?}. Requesting completion.",
        missing.len(),
        missing
    );

    messages.push(Message::user_text(format!(
        "You missed {} bubbles. Translate these remaining IDs now, then call done(): {}",
        missing.len(),
        missing.join(", ")
    )));

    if let Ok(tool_calls) = provider.call(messages, tool_defs).await {
        for tc in &tool_calls {
            if tc.name == "translate" {
                if let Ok(args) = serde_json::from_str::<tools::translate::Args>(&tc.arguments) {
                    results.push(BubbleTranslated {
                        id: args.id,
                        source_text: args.source_text,
                        translated_text: args.translated_text,
                    });
                }
            }
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
