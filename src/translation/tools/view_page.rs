use crate::llm::{ToolDef, ToolResponse};
use crate::translation::TranslateContext;

#[derive(serde::Deserialize)]
pub struct Args {
    pub page_index: usize,
}

pub fn def() -> ToolDef {
    ToolDef::new(
        "view_page",
        "View a comic page image for visual context.\n\n\
            Behavior:\n\
            - This is a SELECTIVE tool — only call when the OCR text is insufficient.\n\
            - Use when you cannot determine speaker identity, age, or relationship from text alone.\n\
            - If the page image is already attached in the user message, do not call again.\n\n\
            When to use: unnamed speaker, ambiguous age/rank not clear from dialogue.\n\
            When NOT to use: text already reveals the speaker and relationship \
            (names, honorifics, terms of address).",
        serde_json::json!({
            "type": "object",
            "required": ["page_index"],
            "additionalProperties": false,
            "properties": {
                "page_index": {
                    "type": "integer",
                    "description": "Zero-based page index"
                }
            }
        }),
    )
    .strict()
}

pub fn handle(args: &Args, ctx: &TranslateContext<'_>) -> ToolResponse {
    if args.page_index < ctx.page_images.len() {
        tracing::info!("Agent viewing page {}", args.page_index);
        let data_uri = encode_page_jpeg(&ctx.page_images[args.page_index]);
        ToolResponse::ImageContent {
            text: format!("Page {} image:", args.page_index),
            data_uri,
        }
    } else {
        ToolResponse::Text(format!(
            "Error: page_index {} out of range (0..{})",
            args.page_index,
            ctx.page_images.len()
        ))
    }
}

/// Images sent to the LLM use `detail: "low"` (85 tokens flat, 512×512).
/// We resize to 512px and use low JPEG quality to minimize upload bandwidth.
const AGENT_IMAGE_MAX_DIM: u32 = 512;
const AGENT_IMAGE_JPEG_QUALITY: u8 = 60;

pub fn encode_page_jpeg(img: &image::DynamicImage) -> String {
    use image::GenericImageView;
    let (w, h) = img.dimensions();
    let longest = w.max(h);
    if longest > AGENT_IMAGE_MAX_DIM {
        let resized = img.resize(
            AGENT_IMAGE_MAX_DIM,
            AGENT_IMAGE_MAX_DIM,
            image::imageops::FilterType::Triangle,
        );
        crate::render::overlay::encode_jpeg_data_uri(&resized, AGENT_IMAGE_JPEG_QUALITY)
    } else {
        crate::render::overlay::encode_jpeg_data_uri(img, AGENT_IMAGE_JPEG_QUALITY)
    }
}
