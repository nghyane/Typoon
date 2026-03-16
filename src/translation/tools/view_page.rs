use crate::llm::ToolDef;

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
