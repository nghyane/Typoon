pub mod chapter;
pub mod common;
pub mod merge;

use std::sync::Arc;

use anyhow::Result;
use base64::Engine;
use base64::engine::general_purpose::STANDARD;
use image::{DynamicImage, RgbaImage};

use crate::api::{AppState, BubbleResult, TranslateImageRequest, TranslateImageResponse};
use crate::border_detect;
use crate::text_layout::DrawableArea;
use crate::translation::{BubbleInput, BubbleTranslated, TranslateContext};

/// Internal pipeline output: bubbles + optional rendered image from canvas agent.
pub struct PipelineOutput {
    pub bubbles: Vec<BubbleResult>,
    pub rendered_image: Option<RgbaImage>,
}

/// Main entry point: decode → detect → OCR → translate → fit → render
pub async fn process_image(
    state: &Arc<AppState>,
    req: &TranslateImageRequest,
) -> Result<TranslateImageResponse> {
    let image_bytes = STANDARD.decode(&req.image_blob_b64)?;
    let img = image::load_from_memory(&image_bytes)?;

    let source_lang = common::detect_source_lang(req.source_lang.as_deref(), &req.target_lang);
    let context = common::build_context(req);

    let output = run_pipeline(state, req, &img, source_lang, &context).await?;

    let rendered_image_png_b64 = output.rendered_image.as_ref().map(|rgba| {
        let png_bytes = crate::overlay::encode_png(rgba);
        STANDARD.encode(&png_bytes)
    });

    Ok(TranslateImageResponse {
        image_id: req.image_id.clone(),
        status: "ok".into(),
        bubbles: output.bubbles,
        rendered_image_png_b64,
    })
}

/// Unified pipeline: detect → OCR → translate → fit → render
///
/// source_lang drives component selection:
///   "ja" → comic-text-detector + manga-ocr
///   _    → PP-OCR det (line merge) + PP-OCR rec
async fn run_pipeline(
    state: &Arc<AppState>,
    req: &TranslateImageRequest,
    img: &DynamicImage,
    source_lang: &str,
    context: &[BubbleTranslated],
) -> Result<PipelineOutput> {
    // 1. Detect + OCR (lang-specific)
    let (inputs, polygons) = detect_and_ocr(Arc::clone(state), img.clone(), source_lang.to_string()).await?;

    if inputs.is_empty() {
        return Ok(PipelineOutput { bubbles: Vec::new(), rendered_image: None });
    }

    // 2. Detect border thickness per bubble → compute DrawableAreas once
    let areas: Vec<DrawableArea> = polygons
        .iter()
        .map(|poly| {
            let inset = border_detect::detect_inset(img, poly);
            DrawableArea::from_polygon(poly, inset)
        })
        .collect();

    let engine = common::resolve_engine(state, req)?;
    let page_images = [img.clone()];
    let translate_ctx = TranslateContext {
        page_images: &page_images,
        glossary: state.glossary.as_ref(),
    };

    // 3. Try canvas agent path (optional)
    if let Some(agent) = &state.canvas_agent {
        let canvas_bubbles = common::translate_only(
            &engine, inputs.clone(), &polygons, source_lang, &req.target_lang, context.to_vec(), &areas, &translate_ctx,
        ).await?;
        match agent.run(img, &canvas_bubbles).await {
            Ok(output) => {
                let bubbles = common::bubbles_from_canvas(&canvas_bubbles, &output.commands);
                return Ok(PipelineOutput {
                    bubbles,
                    rendered_image: Some(output.image),
                });
            }
            Err(e) => {
                tracing::warn!("Canvas agent failed, falling back to fit_engine: {e}");
            }
        }
    }

    // 4. Translate + fit (default path)
    let bubbles = common::translate_and_fit(
        &engine, inputs, &polygons, source_lang, &req.target_lang, context.to_vec(), img.width(), &areas, &translate_ctx,
    ).await?;
    Ok(PipelineOutput { bubbles, rendered_image: None })
}

/// Detect text regions and OCR them. Strategy based on source_lang.
/// Runs synchronous ONNX inference inside spawn_blocking to avoid blocking the tokio runtime.
async fn detect_and_ocr(
    state: Arc<AppState>,
    img: DynamicImage,
    source_lang: String,
) -> Result<(Vec<BubbleInput>, Vec<Vec<[f64; 2]>>)> {
    tokio::task::spawn_blocking(move || {
        match source_lang.as_str() {
            "ja" => detect_ocr_manga(&state, &img, &source_lang),
            _ if state.ocr.can_detect() => detect_ocr_ppocr(&state, &img, &source_lang),
            _ => detect_ocr_manga(&state, &img, &source_lang),
        }
    }).await?
}

/// comic-text-detector: detects whole bubble polygons, OCR each region
fn detect_ocr_manga(
    state: &AppState,
    img: &DynamicImage,
    lang: &str,
) -> Result<(Vec<BubbleInput>, Vec<Vec<[f64; 2]>>)> {
    let regions = state.detector.lock().unwrap().detect(img)?;
    let mut inputs = Vec::new();
    let mut polygons = Vec::new();

    for (i, region) in regions.iter().enumerate() {
        let result = state.ocr.recognize(&region.crop, lang)?;
        if !result.text.trim().is_empty() {
            let pos = region.polygon.first().map(|p| (p[0] as i32, p[1] as i32));
            inputs.push(BubbleInput {
                id: format!("b{i}"),
                source_text: result.text,
                position: pos,
            });
            polygons.push(region.polygon.clone());
        }
    }

    Ok((inputs, polygons))
}

/// PP-OCR: detect text lines → merge into bubbles → OCR each line → concat
fn detect_ocr_ppocr(
    state: &AppState,
    img: &DynamicImage,
    lang: &str,
) -> Result<(Vec<BubbleInput>, Vec<Vec<[f64; 2]>>)> {
    let lines = state.ocr.detect(img)?;
    let merged = merge::group_lines(lines);

    let mut inputs = Vec::new();
    let mut polygons = Vec::new();

    for (i, bubble) in merged.iter().enumerate() {
        let mut texts = Vec::new();
        for line in &bubble.lines {
            let result = state.ocr.recognize(&line.crop, lang)?;
            let text = result.text.trim().to_string();
            if text.is_empty() || (text.chars().count() <= 2 && result.confidence < 0.5) {
                continue;
            }
            texts.push(text);
        }
        let joined = texts.join(" ");
        if joined.is_empty() {
            continue;
        }

        // Filter watermarks
        let lower = joined.to_lowercase();
        if lower.contains(".com") || lower.contains(".net") || lower.contains(".org") {
            continue;
        }

        let pos = bubble.polygon.first().map(|p| (p[0] as i32, p[1] as i32));
        inputs.push(BubbleInput {
            id: format!("b{i}"),
            source_text: joined,
            position: pos,
        });
        polygons.push(bubble.polygon.clone());
    }

    Ok((inputs, polygons))
}
