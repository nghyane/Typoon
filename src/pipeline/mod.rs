pub mod chapter;
pub mod merge;

use std::sync::Arc;

use anyhow::Result;
use base64::Engine;
use base64::engine::general_purpose::STANDARD;
use image::DynamicImage;

use crate::api::{AppState, TranslateImageRequest, TranslateImageResponse};
use crate::detection::{LocalTextMask, TextDetector};
use crate::ocr::OcrEngine;
use crate::translation::{BubbleInput, BubbleTranslated};

/// Main HTTP entry point: decode → detect → OCR → translate → fit → render.
///
/// Delegates to the chapter pipeline, treating a single image as a 1-page chapter.
pub async fn process_image(
    state: &Arc<AppState>,
    req: &TranslateImageRequest,
) -> Result<TranslateImageResponse> {
    let image_bytes = STANDARD.decode(&req.image_blob_b64)?;
    let img = image::load_from_memory(&image_bytes)?;

    let source_lang = detect_source_lang(req.source_lang.as_deref(), &req.target_lang);

    // Resolve translation engine (handles per-request provider override)
    let engine = crate::api::resolve_engine(state, req)?;

    // Detect + OCR as a 1-page chapter
    let runner = &state.runner;
    let det = Arc::clone(&runner.detector);
    let ocr = Arc::clone(&runner.ocr);
    let images = vec![img];

    let detections = tokio::task::spawn_blocking({
        let images = images.clone();
        let lang = source_lang.to_string();
        move || chapter::detect_chapter(&det, &ocr, &images, &lang)
    }).await??;

    // Build context from request hints
    let context = build_context(req);

    // Translate + fit + render using chapter pipeline
    let output = chapter::translate_and_render_with_engine(
        runner,
        engine.as_ref(),
        detections,
        &images,
        &req.target_lang,
        source_lang,
        context,
    ).await?;

    // Build response
    let mut bubbles = Vec::new();
    let mut rendered_image_png_b64 = None;

    for page in output.pages {
        bubbles.extend(page.bubbles);
        if let Some(rgba) = page.rendered_image {
            let png_bytes = crate::overlay::encode_png(&rgba);
            rendered_image_png_b64 = Some(STANDARD.encode(&png_bytes));
        }
    }

    Ok(TranslateImageResponse {
        image_id: req.image_id.clone(),
        status: "ok".into(),
        bubbles,
        rendered_image_png_b64,
    })
}

// ═══════════════════════════════════════════════════════════════════════
// Shared detect + OCR (used by both single-page HTTP and chapter CLI)
// ═══════════════════════════════════════════════════════════════════════

/// Synchronous detect + OCR dispatch. Strategy based on source_lang:
///   "ja" → comic-text-detector + manga-ocr
///   _    → PP-OCR det (line merge) + PP-OCR rec
pub(crate) fn detect_and_ocr(
    detector: &TextDetector,
    ocr: &OcrEngine,
    img: &DynamicImage,
    source_lang: &str,
) -> Result<(Vec<BubbleInput>, Vec<Vec<[f64; 2]>>, Vec<Option<LocalTextMask>>)> {
    match source_lang {
        "ja" => detect_and_ocr_manga(detector, ocr, img, source_lang),
        _ if ocr.can_detect() => detect_and_ocr_ppocr(ocr, img, source_lang),
        _ => detect_and_ocr_manga(detector, ocr, img, source_lang),
    }
}

/// comic-text-detector: detects whole bubble polygons, OCR each region
fn detect_and_ocr_manga(
    detector: &TextDetector,
    ocr: &OcrEngine,
    img: &DynamicImage,
    lang: &str,
) -> Result<(Vec<BubbleInput>, Vec<Vec<[f64; 2]>>, Vec<Option<LocalTextMask>>)> {
    let regions = detector.detect(img)?;
    let mut inputs = Vec::new();
    let mut polygons = Vec::new();
    let mut masks = Vec::new();

    for region in &regions {
        let result = ocr.recognize(&region.crop, lang)?;
        if !result.text.trim().is_empty() {
            let pos = region.polygon.first().map(|p| (p[0] as i32, p[1] as i32));
            inputs.push(BubbleInput {
                id: format!("b{}", inputs.len()),
                source_text: result.text,
                position: pos,
            });
            polygons.push(region.polygon.clone());
            masks.push(region.mask.clone());
        }
    }

    Ok((inputs, polygons, masks))
}

/// PP-OCR: detect text lines → merge into bubbles → OCR each line → concat
fn detect_and_ocr_ppocr(
    ocr: &OcrEngine,
    img: &DynamicImage,
    lang: &str,
) -> Result<(Vec<BubbleInput>, Vec<Vec<[f64; 2]>>, Vec<Option<LocalTextMask>>)> {
    let lines = ocr.detect(img)?;
    let merged = merge::group_lines(lines);

    let mut inputs = Vec::new();
    let mut polygons = Vec::new();
    let mut masks = Vec::new();

    for bubble in &merged {
        let mut texts = Vec::new();
        let mut total_conf = 0.0_f64;
        let mut conf_count = 0usize;
        for line in &bubble.lines {
            let result = ocr.recognize(&line.crop, lang)?;
            let text = result.text.trim().to_string();
            if text.is_empty() || (text.chars().count() <= 2 && result.confidence < 0.5) {
                continue;
            }
            total_conf += result.confidence;
            conf_count += 1;
            texts.push(text);
        }
        let joined = texts.join(" ");
        if joined.is_empty() {
            continue;
        }

        // SFX filter: combine OCR confidence + geometry.
        // SFX (sound effects) produce portrait-ish lines with low OCR confidence
        // because they are stylized text the model can't read well.
        // Normal dialogue has high confidence even with large fonts.
        let avg_conf = if conf_count > 0 { total_conf / conf_count as f64 } else { 0.0 };
        let all_portrait = bubble.lines.iter().all(|l| {
            let (lx1, ly1, lx2, ly2) = merge::line_bbox(&l.polygon);
            let w = lx2 - lx1;
            let h = ly2 - ly1;
            h >= w * 0.8
        });
        if all_portrait && avg_conf < 0.7 {
            continue;
        }

        // Filter watermarks
        let lower = joined.to_lowercase();
        if lower.contains(".com") || lower.contains(".net") || lower.contains(".org") {
            continue;
        }

        let pos = bubble.polygon.first().map(|p| (p[0] as i32, p[1] as i32));
        inputs.push(BubbleInput {
            id: format!("b{}", inputs.len()),
            source_text: joined,
            position: pos,
        });
        polygons.push(bubble.polygon.clone());
        masks.push(bubble.mask.clone());
    }

    Ok((inputs, polygons, masks))
}

// ═══════════════════════════════════════════════════════════════════════
// Shared helpers (used by process_image and examples)
// ═══════════════════════════════════════════════════════════════════════

pub fn build_context(req: &TranslateImageRequest) -> Vec<BubbleTranslated> {
    let Some(hint) = &req.context_hint else {
        return vec![];
    };

    hint.previous_translations
        .iter()
        .flat_map(|pt| {
            pt.bubbles.iter().map(|b| BubbleTranslated {
                id: String::new(),
                source_text: b.source_text.clone(),
                translated_text: b.translated_text.clone(),
            })
        })
        .collect()
}

const KNOWN_LANGS: &[&str] = &["ja", "ko", "zh", "en", "vi"];

pub fn detect_source_lang(explicit: Option<&str>, target_lang: &str) -> &'static str {
    if let Some(lang) = explicit {
        return KNOWN_LANGS.iter().find(|&&k| k == lang).copied().unwrap_or("en");
    }
    match target_lang {
        "en" | "vi" => "ja",
        "ja" => "en",
        _ => "en",
    }
}
