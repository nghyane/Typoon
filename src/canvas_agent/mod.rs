use std::collections::HashMap;
use std::time::Duration;

use anyhow::{Context, Result};
use async_openai::Client;
use async_openai::config::OpenAIConfig;
use async_openai::types::chat::ChatCompletionMessageToolCalls;
use image::{DynamicImage, Rgba, RgbaImage};
use imageproc::drawing::{draw_filled_rect_mut, draw_hollow_rect_mut, draw_text_mut};
use imageproc::rect::Rect;
use ab_glyph::{FontRef, PxScale};
use serde::Deserialize;

use crate::config::TranslationConfig;
use crate::text_layout;
use crate::text_layout::DrawableArea;

const SYSTEM_PROMPT: &str = r#"You are a professional manga/manhwa typesetter. You receive a comic page with detected text bubbles annotated (colored rectangles with ID labels).

Your job: for each bubble, specify how to erase the original text and typeset the translation.

The detection model's bounding boxes may be slightly off — use the crop parameters to adjust:
- crop_left/right/top/bottom: pixels to trim FROM EACH SIDE of the detected bbox.
  Positive = shrink inward (e.g. bbox extends outside bubble border → crop to fix).
  Negative = expand outward (e.g. bbox too tight, text not fully covered → expand).
  The erase and text placement both use the adjusted bbox.
  Typical values: 0-15px. Look at the image carefully to judge.

For each bubble, call typeset() with:
- bubble_id: the ID shown on the annotated image
- text: the translated text (use the provided translation, fix obvious translation errors if needed)
- font_size: font size in pixels. Each bubble has a "suggested" size computed by the engine — use it as baseline. You may adjust ±2-4px for consistency or emphasis, but stay close to the suggestion.
- align: "center" for round/oval speech bubbles, "left" for rectangular narration boxes
- crop_left, crop_right, crop_top, crop_bottom: bbox adjustment in pixels (default 0)

Rules:
- Process ALL bubbles — don't skip any unless it's clearly SFX/watermark/non-dialogue
- For SFX or sound effects, call skip() with the bubble_id
- After processing all bubbles, call done()"#;

const BBOX_COLORS: [Rgba<u8>; 6] = [
    Rgba([255, 0, 0, 255]),     // red
    Rgba([0, 255, 0, 255]),     // green
    Rgba([0, 0, 255, 255]),     // blue
    Rgba([255, 0, 255, 255]),   // magenta
    Rgba([0, 255, 255, 255]),   // cyan
    Rgba([255, 255, 0, 255]),   // yellow
];

// ── Public types ──

/// Input bubble data for the canvas agent.
pub struct CanvasBubble {
    pub id: String,
    pub polygon: Vec<[f64; 2]>,
    pub source_text: String,
    pub translated_text: String,
    /// Base drawable area from border detection.
    pub drawable_area: DrawableArea,
}

/// A single command from the canvas agent.
pub enum CanvasCommand {
    Typeset {
        bubble_id: String,
        text: String,
        font_size: u32,
        align: String,
        crop: [f64; 4], // [left, right, top, bottom]
        /// Final drawable area after applying agent crop.
        drawable_area: DrawableArea,
    },
}

/// Output from the canvas agent.
pub struct CanvasAgentOutput {
    pub commands: Vec<CanvasCommand>,
    pub image: RgbaImage,
}

// ── Arg structs ──

#[derive(Deserialize)]
struct TypesetArgs {
    bubble_id: String,
    text: String,
    font_size: u32,
    align: String,
    #[serde(default)]
    crop_left: f64,
    #[serde(default)]
    crop_right: f64,
    #[serde(default)]
    crop_top: f64,
    #[serde(default)]
    crop_bottom: f64,
}

#[derive(Deserialize)]
struct SkipArgs {
    bubble_id: String,
}

// ── CanvasAgent ──

pub struct CanvasAgent {
    client: Client<OpenAIConfig>,
    model: String,
    reasoning_effort: Option<String>,
}

impl CanvasAgent {
    pub fn new(translation: &TranslationConfig) -> Result<Self> {
        let config = OpenAIConfig::new()
            .with_api_base(&translation.endpoint)
            .with_api_key(translation.api_key.as_deref().unwrap_or("not-needed"));

        Ok(Self {
            client: Client::with_config(config),
            model: translation.model.clone(),
            reasoning_effort: translation.reasoning_effort.clone(),
        })
    }

    /// Save the annotated image (what the LLM sees) to a path for debugging.
    pub fn save_annotated(&self, img: &DynamicImage, bubbles: &[CanvasBubble], path: &std::path::Path) -> Result<()> {
        let annotated = annotate_image(img, bubbles);
        annotated.save(path).context("Failed to save annotated image")?;
        Ok(())
    }

    pub async fn run(
        &self,
        img: &DynamicImage,
        bubbles: &[CanvasBubble],
    ) -> Result<CanvasAgentOutput> {
        if bubbles.is_empty() {
            return Ok(CanvasAgentOutput {
                commands: Vec::new(),
                image: img.to_rgba8(),
            });
        }

        let font = text_layout::get_font();
        let mut canvas = img.to_rgba8();
        let mut all_commands: Vec<CanvasCommand> = Vec::new();

        let bubble_map: HashMap<&str, (usize, &CanvasBubble)> =
            bubbles.iter().enumerate().map(|(i, b)| (b.id.as_str(), (i, b))).collect();

        // Build prompt with annotated image
        let annotated = annotate_image(img, bubbles);
        let annotated_uri = encode_canvas_jpeg(&annotated);
        let user_text = build_user_text(bubbles, img.width(), img.height());

        tracing::debug!("Canvas agent prompt:\n{user_text}");

        let tools = tools_def();

        let messages = vec![
            serde_json::json!({
                "role": "system",
                "content": SYSTEM_PROMPT
            }),
            serde_json::json!({
                "role": "user",
                "content": [
                    { "type": "text", "text": user_text },
                    {
                        "type": "image_url",
                        "image_url": { "url": annotated_uri, "detail": "high" }
                    }
                ]
            }),
        ];

        // Single LLM call — all bubbles processed at once via parallel tool calls
        tracing::info!("Canvas agent: calling LLM for {} bubbles", bubbles.len());
        let chat = self.client.chat();
        let llm_future = chat.create_byot({
            let mut body = serde_json::json!({
                "model": self.model,
                "messages": messages,
                "tools": tools,
                "parallel_tool_calls": true
            });
            if let Some(effort) = &self.reasoning_effort {
                body["reasoning_effort"] = serde_json::json!(effort);
            }
            body
        });
        let resp: async_openai::types::chat::CreateChatCompletionResponse =
            tokio::time::timeout(Duration::from_secs(120), llm_future)
                .await
                .map_err(|_| anyhow::anyhow!("Canvas agent LLM call timed out (120s)"))?
                .context("Canvas agent LLM call failed")?;

        let choice = resp
            .choices
            .first()
            .ok_or_else(|| anyhow::anyhow!("Canvas agent: no choices returned"))?;

        let tool_calls = match &choice.message.tool_calls {
            Some(calls) if !calls.is_empty() => calls,
            _ => {
                tracing::warn!("Canvas agent: no tool calls returned");
                return Ok(CanvasAgentOutput { commands: all_commands, image: canvas });
            }
        };

        for call in tool_calls {
            let ChatCompletionMessageToolCalls::Function(tc) = call else { continue };

            match tc.function.name.as_str() {
                "typeset" => {
                    let args: TypesetArgs =
                        serde_json::from_str(&tc.function.arguments).with_context(|| {
                            format!("Bad typeset args: {}", tc.function.arguments)
                        })?;

                    let Some(&(_idx, bubble)) = bubble_map.get(args.bubble_id.as_str()) else {
                        tracing::warn!("Canvas agent: unknown bubble_id: {}", args.bubble_id);
                        continue;
                    };

                    // Derive final drawable area: base insets clamped to at least agent crop
                    let final_area = bubble.drawable_area.with_crop_min([
                        args.crop_left, args.crop_right, args.crop_top, args.crop_bottom,
                    ]);

                    tracing::info!(
                        "Canvas agent: typeset [{}] size={} align={} crop=[{},{},{},{}]",
                        args.bubble_id, args.font_size, args.align,
                        args.crop_left, args.crop_right, args.crop_top, args.crop_bottom
                    );

                    execute_typeset(&mut canvas, &final_area, &args, font);
                    all_commands.push(CanvasCommand::Typeset {
                        bubble_id: args.bubble_id,
                        text: args.text,
                        font_size: args.font_size,
                        align: args.align,
                        crop: [args.crop_left, args.crop_right, args.crop_top, args.crop_bottom],
                        drawable_area: final_area,
                    });
                }
                "skip" => {
                    let args: SkipArgs =
                        serde_json::from_str(&tc.function.arguments).unwrap_or(SkipArgs {
                            bubble_id: String::new(),
                        });
                    tracing::info!("Canvas agent: skip [{}]", args.bubble_id);
                }
                "done" => {
                    tracing::info!("Canvas agent: done");
                }
                other => {
                    tracing::warn!("Canvas agent: unknown tool call: {other}");
                }
            }
        }

        Ok(CanvasAgentOutput {
            commands: all_commands,
            image: canvas,
        })
    }
}

// ── Tool definitions ──

fn tools_def() -> serde_json::Value {
    serde_json::json!([
        {
            "type": "function",
            "function": {
                "name": "typeset",
                "description": "Erase original text and render translated text in a bubble. Uses the detected bbox adjusted by crop offsets. Positive crop = shrink inward, negative = expand outward.",
                "strict": true,
                "parameters": {
                    "type": "object",
                    "required": ["bubble_id", "text", "font_size", "align", "crop_left", "crop_right", "crop_top", "crop_bottom"],
                    "additionalProperties": false,
                    "properties": {
                        "bubble_id": { "type": "string", "description": "ID of the bubble (e.g. 'b0', 'b1')" },
                        "text": { "type": "string", "description": "Translated text to render (fix translation errors if needed)" },
                        "font_size": { "type": "integer", "description": "Font size in pixels. Use the suggested_font value ± 2-4px." },
                        "align": { "type": "string", "enum": ["center", "left", "right"] },
                        "crop_left": { "type": "number", "description": "Pixels to trim from left side of detected bbox (negative to expand)" },
                        "crop_right": { "type": "number", "description": "Pixels to trim from right side of detected bbox (negative to expand)" },
                        "crop_top": { "type": "number", "description": "Pixels to trim from top of detected bbox (negative to expand)" },
                        "crop_bottom": { "type": "number", "description": "Pixels to trim from bottom of detected bbox (negative to expand)" }
                    }
                }
            }
        },
        {
            "type": "function",
            "function": {
                "name": "skip",
                "description": "Skip a bubble (for SFX, watermarks, or non-dialogue elements that should not be translated).",
                "strict": true,
                "parameters": {
                    "type": "object",
                    "required": ["bubble_id"],
                    "additionalProperties": false,
                    "properties": {
                        "bubble_id": { "type": "string", "description": "ID of the bubble to skip" }
                    }
                }
            }
        },
        {
            "type": "function",
            "function": {
                "name": "done",
                "description": "Signal that all bubbles have been processed.",
                "strict": true,
                "parameters": {
                    "type": "object",
                    "additionalProperties": false,
                    "properties": {}
                }
            }
        }
    ])
}

// ── Annotate image ──

fn annotate_image(img: &DynamicImage, bubbles: &[CanvasBubble]) -> RgbaImage {
    let mut canvas = img.to_rgba8();
    let font = text_layout::get_font();
    let label_scale = PxScale::from(16.0);

    for (i, bubble) in bubbles.iter().enumerate() {
        let color = BBOX_COLORS[i % BBOX_COLORS.len()];
        let (x1, y1, x2, y2) = text_layout::polygon_bbox(&bubble.polygon);

        let rx = x1.max(0.0) as i32;
        let ry = y1.max(0.0) as i32;
        let rw = ((x2 - x1).max(0.0) as u32).min(canvas.width().saturating_sub(rx as u32));
        let rh = ((y2 - y1).max(0.0) as u32).min(canvas.height().saturating_sub(ry as u32));

        if rw > 2 && rh > 2 {
            // Draw 3px thick border for visibility
            let rect = Rect::at(rx, ry).of_size(rw, rh);
            draw_hollow_rect_mut(&mut canvas, rect, color);
            if rw > 4 && rh > 4 {
                let r1 = Rect::at(rx + 1, ry + 1).of_size(rw - 2, rh - 2);
                draw_hollow_rect_mut(&mut canvas, r1, color);
            }
            if rw > 6 && rh > 6 {
                let r2 = Rect::at(rx + 2, ry + 2).of_size(rw - 4, rh - 4);
                draw_hollow_rect_mut(&mut canvas, r2, color);
            }
        }

        // Draw ID label near top-left
        let label = format!("[{}]", bubble.id);
        draw_text_mut(
            &mut canvas,
            color,
            rx + 4,
            ry + 2,
            label_scale,
            &font,
            &label,
        );
    }

    canvas
}

// ── User text building ──

fn build_user_text(bubbles: &[CanvasBubble], img_w: u32, img_h: u32) -> String {
    use crate::fit_engine::FitEngine;

    let mut msg = format!(
        "Comic page with {} detected text bubbles. Image dimensions: {}x{}px.\n\nBubbles:\n",
        bubbles.len(),
        img_w,
        img_h
    );

    // Compute suggested font sizes via FitEngine using canonical DrawableAreas
    let fit_items: Vec<(&str, &DrawableArea)> = bubbles
        .iter()
        .map(|b| (b.translated_text.as_str(), &b.drawable_area))
        .collect();
    let fits = FitEngine::fit_page_areas(&fit_items, img_w).unwrap_or_default();

    for (i, bubble) in bubbles.iter().enumerate() {
        let [x1, y1, x2, y2] = bubble.drawable_area.bbox;
        let suggested = fits.get(i).map(|f| f.font_size_px).unwrap_or(16);
        msg.push_str(&format!(
            "[{}] bbox=({:.0},{:.0})-({:.0},{:.0}) suggested_font={}px source=\"{}\" translated=\"{}\"\n",
            bubble.id,
            x1, y1, x2, y2,
            suggested,
            bubble.source_text.replace('\n', " "),
            bubble.translated_text.replace('\n', " "),
        ));
    }

    msg.push_str(
        "\nCall typeset() for each bubble, skip() for SFX/watermarks, then done().",
    );

    msg
}

// ── Command execution ──

/// Erase original text and render translated text using the canonical DrawableArea.
fn execute_typeset(
    canvas: &mut RgbaImage,
    area: &DrawableArea,
    args: &TypesetArgs,
    font: &FontRef<'_>,
) {
    let (draw_x, draw_y, draw_w, draw_h) = area.rect();

    let img_w = canvas.width() as i32;
    let img_h = canvas.height() as i32;

    // 1. Erase: fill drawable rect with white
    let rx = (draw_x as i32).max(0);
    let ry = (draw_y as i32).max(0);
    let rw = (draw_w as i32).max(0).min(img_w - rx) as u32;
    let rh = (draw_h as i32).max(0).min(img_h - ry) as u32;
    if rw > 0 && rh > 0 {
        let white = Rgba([255u8, 255, 255, 255]);
        draw_filled_rect_mut(canvas, Rect::at(rx, ry).of_size(rw, rh), white);
    }

    // 2. Draw text within the same drawable area (no additional padding)
    let black = Rgba([0u8, 0, 0, 255]);
    let scale = PxScale::from(args.font_size as f32);
    let line_spacing = args.font_size as f64 * text_layout::LINE_HEIGHT_MULTIPLIER;

    let lines = text_layout::wrap_text(&args.text, draw_w, args.font_size, font);
    let total_text_h = lines.len() as f64 * line_spacing;
    let start_y = (draw_y + (draw_h - total_text_h) / 2.0).max(draw_y);

    for (i, line) in lines.iter().enumerate() {
        let line_w = text_layout::measure_text_width(line, args.font_size, font);
        let x = match args.align.as_str() {
            "left" => draw_x,
            "right" => draw_x + draw_w - line_w,
            _ => draw_x + (draw_w - line_w) / 2.0,
        };
        let y = start_y + i as f64 * line_spacing;
        if args.font_size <= 16 {
            draw_text_mut(canvas, black, x as i32 + 1, y as i32, scale, font, line);
            draw_text_mut(canvas, black, x as i32, y as i32 + 1, scale, font, line);
        }
        draw_text_mut(canvas, black, x as i32, y as i32, scale, font, line);
    }
}

// ── Encoding ──

fn encode_canvas_jpeg(canvas: &RgbaImage) -> String {
    crate::overlay::encode_jpeg_data_uri(&DynamicImage::ImageRgba8(canvas.clone()), 85)
}


