use std::time::Duration;

use anyhow::{Context, Result};
use async_openai::Client;
use async_openai::config::OpenAIConfig;

use async_openai::types::chat::ChatCompletionMessageToolCalls;

use super::{BubbleTranslated, TranslateContext, TranslateRequest};
use crate::config::TranslationConfig;

const MAX_AGENT_TURNS: usize = 15;
const LLM_TIMEOUT_SECS: u64 = 180;

pub struct OpenAICompatibleAdapter {
    client: Client<OpenAIConfig>,
    model: String,
    reasoning_effort: Option<String>,
}

impl OpenAICompatibleAdapter {
    pub fn new(config: &TranslationConfig) -> Result<Self> {
        let openai_config = OpenAIConfig::new()
            .with_api_base(&config.endpoint)
            .with_api_key(config.api_key.as_deref().unwrap_or("not-needed"));

        Ok(Self {
            client: Client::with_config(openai_config),
            model: config.model.clone(),
            reasoning_effort: config.reasoning_effort.clone(),
        })
    }
}

// ── Prompt building ──

fn build_system_prompt(req: &TranslateRequest) -> String {
    let comic_type = match req.source_lang.as_str() {
        "ja" => "manga",
        "ko" => "manhwa",
        "zh" => "manhua",
        _ => "comic",
    };

    let mut sys = format!(
        "You are ComicScan's tool-only translation agent for {comic_type} ({} → {}).\n\n",
        req.source_lang, req.target_lang
    );

    // Output contract
    sys.push_str("\
## Output contract
- Your ONLY valid outputs are function/tool calls. No prose, markdown, explanations, or status text.
- Every assistant message must contain one or more tool calls.
- Batch many tool calls in one message when possible.
- If you need more context, call a tool (view_page, search_glossary). Never ask in prose.\n\n");

    // Workflow
    sys.push_str("\
## Workflow
1. Identify every required bubble ID from the user task.
2. Visual context (if page image is attached or view_page is available):
   - Inspect the page to determine speaker age, clothing, setting, power dynamics, relationships.
   - For Vietnamese target: ALWAYS inspect the image before choosing xưng hô.
3. Terminology: call search_glossary() for names, titles, places, recurring terms.
4. Translation: call translate() for EVERY required bubble ID.
   - source_text = cleaned/corrected OCR text.
   - translated_text = final localized text only.
   - Never skip a bubble — translate best-effort even if short, unclear, or SFX.
   - To revise: call translate() again for the same ID (latest wins).
5. Completion: call done() only after every required ID has a translate() call.
   - Prefer sending final translate() calls and done() in the same message.\n\n");

    sys.push_str(source_policy(&req.source_lang));
    sys.push_str(target_policy(&req.target_lang));

    sys
}

/// Source-language notes for the system prompt.
fn source_policy(lang: &str) -> &'static str {
    match lang {
        "ja" => "\
## Source: Japanese manga
- Reading order: right-to-left. Bubble order follows this.
- Japanese often omits subject — infer from context/visuals.
- Keigo levels signal character relationships — reflect in translation register.
- Honorifics (-san, -kun, -chan, -sama, -sensei, -senpai): preserve or adapt per target.
- SFX onomatopoeia (ドキドキ, ゴゴゴ): translate or transliterate per target norms.\n\n",

        "ko" => "\
## Source: Korean manhwa
- Reading order: left-to-right. Webtoon format (vertical scroll).
- Speech levels (존댓말/반말) signal power dynamics — reflect in translation register.
- Honorifics (-님, -씨, 선배, 형/오빠/언니/누나): adapt per target conventions.
- Onomatopoeia (쿵, 두근두근): translate or adapt.\n\n",

        "zh" => "\
## Source: Chinese manhua
- Reading order: left-to-right (modern).
- Chengyu (成语): translate meaning, not literally.
- Cultivation/xianxia terms (修炼, 气, 丹田, 境界): use glossary if available.
- Honorifics (先生, 大人, 小姐): contextual — adapt per target.\n\n",

        _ => "",
    }
}

/// Target-language notes for the system prompt.
fn target_policy(lang: &str) -> &'static str {
    match lang {
        "vi" => "\
## Target: Vietnamese
- Xưng hô is chosen from the speaker–addressee RELATIONSHIP, not literal source pronouns.
- USE THE IMAGE to classify: child, teen/student, adult peer, younger→older, subordinate→superior, \
civilian→officer, employee→manager, teacher↔student, family, hostile speaker, etc.
- Adults in uniform / workplace / official / police / military / medical / office scenes: \
default to tôi/anh/chị/em/ông/bà as appropriate. NEVER cậu/tớ for adults.
- cậu/tớ: ONLY for young peers or intimate youthful speech clearly visible in the image.
- Teacher↔student: thầy/cô ↔ em. Adult staff in school/work: adult pronouns, not cậu/tớ.
- Rough/aggressive: tao/mày. Formal/respectful: tôi/ngài.
- Keep xưng hô consistent within the same scene unless the relationship visibly shifts.
- When the image contradicts a text-only assumption, trust the image.
- Prefer natural Vietnamese; use Hán-Việt for formal/cultivation terms when natural.\n\n",

        "en" => "\
## Target: English
- Natural, fluent English. Avoid translationese.
- Preserve character voice: tough characters sound tough, polite ones sound polite.
- Honorifics: keep -san/-kun/-senpai for manga audience, or naturalize — be consistent.
- SFX: use English equivalents where natural (THUD, CRACK), keep original for iconic ones.\n\n",

        _ => "",
    }
}

fn build_user_prompt(req: &TranslateRequest, ctx: &TranslateContext<'_>) -> String {
    let mut prompt = String::new();

    // Task header with completion requirements
    let all_ids: Vec<&str> = req.all_bubbles().map(|b| b.id.as_str()).collect();
    prompt.push_str(&format!(
        "Task: Translate every listed bubble from {} to {} using tool calls only.\n\n",
        req.source_lang, req.target_lang
    ));
    prompt.push_str(&format!(
        "Completion requirements:\n- Total bubbles: {}\n- Required IDs: {}\n- Each ID must get a translate() call before done().\n\n",
        all_ids.len(),
        all_ids.join(", ")
    ));

    // Available context summary
    prompt.push_str("Available context:\n");
    let is_single_page = req.pages.len() == 1;
    if is_single_page && !ctx.page_images.is_empty() {
        prompt.push_str("- Page image: attached below (inspect directly)\n");
    } else if !ctx.page_images.is_empty() {
        prompt.push_str(&format!(
            "- Page images: {} pages available via view_page(index)\n",
            ctx.page_images.len()
        ));
    }
    if ctx.glossary.is_some() {
        prompt.push_str("- Glossary search: available via search_glossary()\n");
    }
    prompt.push('\n');

    // Glossary entries provided with request
    if !req.glossary.is_empty() {
        prompt.push_str("Glossary (canon — use these):\n");
        for entry in &req.glossary {
            prompt.push_str(&format!("  {} → {}", entry.source_term, entry.target_term));
            if let Some(notes) = &entry.notes {
                prompt.push_str(&format!("  ({})", notes));
            }
            prompt.push('\n');
        }
        prompt.push('\n');
    }

    // Previous translations for context continuity
    if !req.context.is_empty() {
        prompt.push_str("Previous translations for context:\n");
        for c in &req.context {
            prompt.push_str(&format!("  {} → {}\n", c.source_text, c.translated_text));
        }
        prompt.push('\n');
    }

    // Recommended workflow
    if req.target_lang == "vi" && (is_single_page || !ctx.page_images.is_empty()) {
        prompt.push_str(
            "Recommended: Inspect the page image first to determine xưng hô, then translate all bubbles, then done().\n\n",
        );
    }

    // Bubble list per page
    for page in &req.pages {
        if req.pages.len() > 1 {
            let img_note = if page.page_index < ctx.page_images.len() {
                "available via view_page"
            } else {
                "no image"
            };
            prompt.push_str(&format!("── Page {} ({}) ──\n", page.page_index, img_note));
        }
        for bubble in &page.bubbles {
            prompt.push_str(&format!("[{}] \"{}\"\n", bubble.id, bubble.source_text));
        }
        prompt.push('\n');
    }

    prompt
}

// ── Tool definitions ──
// Amp-style: clear description, when to use / when NOT to use, examples.

fn tools_def(has_images: bool, has_glossary: bool) -> serde_json::Value {
    let mut tools = vec![
        serde_json::json!({
            "type": "function",
            "function": {
                "name": "translate",
                "description": "Submit the final translation for exactly one bubble.\n\n\
                    Behavior:\n\
                    - Call once for every required bubble ID. Batch many translate() calls in one message.\n\
                    - id must exactly match a required bubble ID from the task.\n\
                    - source_text = cleaned/corrected OCR text for that bubble.\n\
                    - translated_text = final localized text only — no notes or explanations.\n\
                    - To revise: call translate() again with the same id (latest wins).\n\n\
                    When to use: whenever you are ready to submit a bubble translation.\n\
                    When NOT to use: do not skip any listed bubble, even if short or SFX.\n\n\
                    Examples:\n\
                    - {\"id\":\"b0\",\"source_text\":\"おはよう\",\"translated_text\":\"Chào buổi sáng.\"}\n\
                    - {\"id\":\"p0_b3\",\"source_text\":\"What?!\",\"translated_text\":\"Cái gì?!\"}",
                "strict": true,
                "parameters": {
                    "type": "object",
                    "required": ["id", "source_text", "translated_text"],
                    "additionalProperties": false,
                    "properties": {
                        "id": {
                            "type": "string",
                            "description": "Bubble ID exactly as listed in the task (e.g. b0, p0_b1)"
                        },
                        "source_text": {
                            "type": "string",
                            "description": "Cleaned/corrected source text for this bubble"
                        },
                        "translated_text": {
                            "type": "string",
                            "description": "Final translated text — no notes or explanations"
                        }
                    }
                }
            }
        }),
        serde_json::json!({
            "type": "function",
            "function": {
                "name": "done",
                "description": "Signal that the translation job is complete.\n\n\
                    Behavior:\n\
                    - Call ONLY after every required bubble ID has at least one translate() call.\n\
                    - Prefer sending final translate() calls and done() in the same message.\n\
                    - Before calling, verify coverage against the required ID list.\n\n\
                    When to use: once, when nothing remains untranslated.\n\
                    When NOT to use: do not call after partial work or after only viewing/searching.",
                "strict": true,
                "parameters": {
                    "type": "object",
                    "additionalProperties": false,
                    "properties": {}
                }
            }
        }),
    ];

    if has_images {
        tools.push(serde_json::json!({
            "type": "function",
            "function": {
                "name": "view_page",
                "description": "View a comic page image for visual context.\n\n\
                    Behavior:\n\
                    - Use BEFORE translating when speaker identity, age, rank, uniform, \
                    workplace/school setting, or tone affects the translation.\n\
                    - For Vietnamese target: inspect the page before choosing xưng hô \
                    whenever adult/student/formal-role distinctions may matter.\n\
                    - If the page image is already attached in the user message, inspect it directly — \
                    do not call view_page() for the same page.\n\n\
                    When to use: pronouns/register depend on who is talking to whom; \
                    need to distinguish adult vs student, boss vs employee, officer vs civilian.\n\
                    When NOT to use: page image already attached and context is clear from text.",
                "strict": true,
                "parameters": {
                    "type": "object",
                    "required": ["page_index"],
                    "additionalProperties": false,
                    "properties": {
                        "page_index": {
                            "type": "integer",
                            "description": "Zero-based page index"
                        }
                    }
                }
            }
        }));
    }

    if has_glossary {
        tools.push(serde_json::json!({
            "type": "function",
            "function": {
                "name": "search_glossary",
                "description": "Search the persistent glossary for canonical translations.\n\n\
                    Behavior:\n\
                    - Search BEFORE inventing a translation for names, titles, places, \
                    organizations, abilities, items, or recurring terms.\n\
                    - Batch multiple search_glossary() calls in one message.\n\
                    - If no useful entry found, translate naturally and continue.\n\n\
                    When to use: proper nouns or recurring terms likely to appear again.\n\
                    When NOT to use: ordinary words or one-off everyday phrases.",
                "strict": true,
                "parameters": {
                    "type": "object",
                    "required": ["query"],
                    "additionalProperties": false,
                    "properties": {
                        "query": {
                            "type": "string",
                            "description": "Source-language term to look up"
                        }
                    }
                }
            }
        }));
        tools.push(serde_json::json!({
            "type": "function",
            "function": {
                "name": "update_glossary",
                "description": "Save a high-confidence canonical term to the persistent glossary for future chapters.\n\n\
                    Behavior:\n\
                    - Use only for stable, recurring items: character names, titles, places, techniques.\n\
                    - Save only when confident the term should be reused later.\n\
                    - Translation should not wait on this; it is optional.\n\n\
                    When to use: new recurring term that should stay consistent in future chapters.\n\
                    When NOT to use: casual dialogue, uncertain pronouns, one-off phrasing.",
                "strict": true,
                "parameters": {
                    "type": "object",
                    "required": ["source_term", "target_term", "category", "notes"],
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
                        "category": {
                            "type": "string",
                            "enum": ["character", "term", "title", "sfx", "other"],
                            "description": "character=names, term=techniques/orgs/places, title=honorifics/ranks, sfx=sound effects"
                        },
                        "notes": {
                            "type": "string",
                            "description": "Brief context for future reference"
                        }
                    }
                }
            }
        }));
    }

    serde_json::json!(tools)
}

// ── Arg structs ──

#[derive(serde::Deserialize)]
struct TranslateArgs {
    id: String,
    source_text: String,
    translated_text: String,
}

#[derive(serde::Deserialize)]
struct ViewPageArgs {
    page_index: usize,
}

#[derive(serde::Deserialize)]
struct SearchGlossaryArgs {
    query: String,
}

#[derive(serde::Deserialize)]
struct UpdateGlossaryArgs {
    source_term: String,
    target_term: String,
    category: String,
    notes: String,
}

// ── Helpers ──

fn encode_page_jpeg(img: &image::DynamicImage) -> String {
    crate::overlay::encode_jpeg_data_uri(img, 80)
}

fn tool_response(tool_call_id: &str, content: &str) -> serde_json::Value {
    serde_json::json!({
        "role": "tool",
        "tool_call_id": tool_call_id,
        "content": content
    })
}

impl OpenAICompatibleAdapter {
    pub async fn translate(
        &self,
        req: &TranslateRequest,
        ctx: &TranslateContext<'_>,
    ) -> Result<Vec<BubbleTranslated>> {
        let has_images = !ctx.page_images.is_empty();
        let has_glossary = ctx.glossary.is_some();
        let system_prompt = build_system_prompt(req);
        let user_prompt = build_user_prompt(req, ctx);
        let tools = tools_def(has_images, has_glossary);

        // For single-page requests with an image available, embed it directly in the
        // user message so the agent has visual context from the start (character age,
        // clothing, setting) without needing to call view_page().
        let is_single_page = req.pages.len() == 1;
        let user_message = if is_single_page && ctx.page_images.len() == 1 {
            let image_uri = encode_page_jpeg(&ctx.page_images[0]);
            serde_json::json!({
                "role": "user",
                "content": [
                    { "type": "text", "text": user_prompt },
                    { "type": "image_url", "image_url": { "url": image_uri, "detail": "high" } }
                ]
            })
        } else {
            serde_json::json!({ "role": "user", "content": user_prompt })
        };

        let mut messages: Vec<serde_json::Value> = vec![
            serde_json::json!({ "role": "system", "content": system_prompt }),
            user_message,
        ];

        let mut results: Vec<BubbleTranslated> = Vec::new();

        for turn in 0..MAX_AGENT_TURNS {
            tracing::debug!("Translation agent turn {turn}");

            let resp = self.call_llm(&messages, &tools).await?;
            let choice = resp
                .choices
                .first()
                .ok_or_else(|| anyhow::anyhow!("No choices returned on turn {turn}"))?;

            let tool_calls = match &choice.message.tool_calls {
                Some(calls) if !calls.is_empty() => calls,
                _ => break,
            };

            // Append assistant message with tool_calls
            messages.push(serde_json::json!({
                "role": "assistant",
                "tool_calls": tool_calls.iter().filter_map(|c| {
                    let ChatCompletionMessageToolCalls::Function(f) = c else { return None };
                    Some(serde_json::json!({
                        "id": f.id,
                        "type": "function",
                        "function": {
                            "name": f.function.name,
                            "arguments": f.function.arguments
                        }
                    }))
                }).collect::<Vec<_>>()
            }));

            let mut got_done = false;

            for call in tool_calls {
                let ChatCompletionMessageToolCalls::Function(tc) = call else { continue };

                match tc.function.name.as_str() {
                    "translate" => {
                        let args: TranslateArgs =
                            serde_json::from_str(&tc.function.arguments)
                                .with_context(|| format!("Bad translate args: {}", tc.function.arguments))?;
                        tracing::debug!("translate [{}] → {:?}", args.id, args.translated_text);
                        results.push(BubbleTranslated {
                            id: args.id,
                            source_text: args.source_text,
                            translated_text: args.translated_text,
                        });
                        messages.push(tool_response(&tc.id, "ok"));
                    }

                    "view_page" => {
                        let args: ViewPageArgs =
                            serde_json::from_str(&tc.function.arguments)
                                .with_context(|| format!("Bad view_page args: {}", tc.function.arguments))?;

                        if args.page_index < ctx.page_images.len() {
                            tracing::info!("Agent viewing page {}", args.page_index);
                            let image_uri = encode_page_jpeg(&ctx.page_images[args.page_index]);
                            messages.push(serde_json::json!({
                                "role": "tool",
                                "tool_call_id": tc.id,
                                "content": [
                                    { "type": "text", "text": format!("Page {} image:", args.page_index) },
                                    { "type": "image_url", "image_url": { "url": image_uri, "detail": "high" } }
                                ]
                            }));
                        } else {
                            messages.push(tool_response(
                                &tc.id,
                                &format!("Error: page_index {} out of range (0..{})", args.page_index, ctx.page_images.len()),
                            ));
                        }
                    }

                    "search_glossary" => {
                        let args: SearchGlossaryArgs =
                            serde_json::from_str(&tc.function.arguments)
                                .with_context(|| format!("Bad search_glossary args: {}", tc.function.arguments))?;

                        let response = if let Some(glossary) = ctx.glossary {
                            match glossary.search(&args.query) {
                                Ok(entries) if entries.is_empty() => {
                                    "No matching glossary entries found.".to_string()
                                }
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

                        tracing::debug!("search_glossary({:?}) → {} chars", args.query, response.len());
                        messages.push(tool_response(&tc.id, &response));
                    }

                    "update_glossary" => {
                        let args: UpdateGlossaryArgs =
                            serde_json::from_str(&tc.function.arguments)
                                .with_context(|| format!("Bad update_glossary args: {}", tc.function.arguments))?;

                        let notes = format!("[{}] {}", args.category, args.notes);
                        let response = if let Some(glossary) = ctx.glossary {
                            match glossary.upsert(&args.source_term, &args.target_term, Some(&notes)) {
                                Ok(()) => {
                                    tracing::info!(
                                        "Glossary updated: {} → {} ({})",
                                        args.source_term, args.target_term, notes
                                    );
                                    format!("Saved: {} → {}", args.source_term, args.target_term)
                                }
                                Err(e) => format!("Failed to save: {e}"),
                            }
                        } else {
                            "Glossary not available.".to_string()
                        };

                        messages.push(tool_response(&tc.id, &response));
                    }

                    "done" => {
                        tracing::info!("Translation agent done ({} results)", results.len());
                        messages.push(tool_response(&tc.id, "ok"));
                        got_done = true;
                    }

                    other => {
                        tracing::warn!("Unknown tool call: {other}");
                        messages.push(tool_response(&tc.id, "unknown tool"));
                    }
                }
            }

            if got_done {
                break;
            }
        }

        // ── Completeness guardrail ──
        let expected_ids: std::collections::HashSet<&str> =
            req.all_bubbles().map(|b| b.id.as_str()).collect();
        let translated_ids: std::collections::HashSet<&str> =
            results.iter().map(|r| r.id.as_str()).collect();
        let missing: Vec<&str> = expected_ids.difference(&translated_ids).copied().collect();

        if !missing.is_empty() {
            tracing::warn!(
                "Translation agent missed {} bubbles: {:?}. Requesting completion.",
                missing.len(),
                missing
            );

            messages.push(serde_json::json!({
                "role": "user",
                "content": format!(
                    "You missed {} bubbles. Translate these remaining IDs now, then call done(): {}",
                    missing.len(),
                    missing.join(", ")
                )
            }));

            if let Ok(resp) = self.call_llm(&messages, &tools).await {
                if let Some(choice) = resp.choices.first() {
                    if let Some(calls) = &choice.message.tool_calls {
                        for call in calls {
                            let ChatCompletionMessageToolCalls::Function(tc) = call else { continue };
                            if tc.function.name == "translate" {
                                if let Ok(args) = serde_json::from_str::<TranslateArgs>(&tc.function.arguments) {
                                    results.push(BubbleTranslated {
                                        id: args.id,
                                        source_text: args.source_text,
                                        translated_text: args.translated_text,
                                    });
                                }
                            }
                        }
                    }
                }
            }
        }

        // Deduplicate: keep last translation for each ID
        let mut seen = std::collections::HashSet::new();
        results.reverse();
        results.retain(|r| seen.insert(r.id.clone()));
        results.reverse();

        Ok(results)
    }

    async fn call_llm(
        &self,
        messages: &[serde_json::Value],
        tools: &serde_json::Value,
    ) -> Result<async_openai::types::chat::CreateChatCompletionResponse> {
        let mut body = serde_json::json!({
            "model": self.model,
            "messages": messages,
            "tools": tools,
            "parallel_tool_calls": true
        });
        if let Some(effort) = &self.reasoning_effort {
            body["reasoning_effort"] = serde_json::json!(effort);
        }

        let chat = self.client.chat();
        let resp = tokio::time::timeout(
            Duration::from_secs(LLM_TIMEOUT_SECS),
            chat.create_byot(body),
        )
        .await
        .map_err(|_| anyhow::anyhow!("Translation LLM call timed out ({}s)", LLM_TIMEOUT_SECS))?
        .context("Translation LLM call failed")?;

        Ok(resp)
    }
}
