pub mod merge;

use anyhow::Result;
use base64::Engine;
use base64::engine::general_purpose::STANDARD;

use crate::api::{
    AppState, BubbleResult, TranslateImageRequest, TranslateImageResponse,
};
use crate::config::TranslationConfig;
use crate::translation::{BubbleInput, BubbleTranslated, TranslateRequest, TranslationEngine};

type OcrOutput = (Vec<BubbleInput>, Vec<Vec<[f64; 2]>>);

/// Main pipeline: detect → OCR → translate → fit
pub async fn process_image(
    state: &AppState,
    req: &TranslateImageRequest,
) -> Result<TranslateImageResponse> {
    // 1. Decode image
    let image_bytes = STANDARD.decode(&req.image_blob_b64)?;
    let img = image::load_from_memory(&image_bytes)?;

    // 2. Check cache
    let model = req.provider_config.as_ref()
        .and_then(|c| c.model.as_deref())
        .unwrap_or(&state.config.translation.model);
    let cache_key = crate::cache::DiskCache::key(&req.image_id, &req.target_lang, model);

    if let Some(cached) = state.cache.get(&cache_key)? {
        let bubbles: Vec<BubbleResult> = serde_json::from_slice(&cached)?;
        return Ok(TranslateImageResponse {
            image_id: req.image_id.clone(),
            status: "ok (cached)".into(),
            bubbles,
        });
    }

    // 3. Detect + OCR (route by source language)
    let source_lang = detect_source_lang(req.source_lang.as_deref(), &req.target_lang);
    let (bubble_inputs, bubble_polygons) = match source_lang {
        "ja" => {
            // Japanese: comic-text-detector detects whole bubbles → OCR each
            let regions = state.detector.lock().unwrap().detect(&img)?;
            ocr_regions(&state.ocr, &regions, source_lang)?
        }
        _ if state.ocr.can_detect() => {
            // Non-Japanese: PP-OCR text lines → proximity merge into bubbles → OCR
            let lines = state.ocr.detect(&img)?;
            let merged = merge::group_lines(lines);
            ocr_merged_bubbles(&state.ocr, &merged, source_lang)?
        }
        _ => {
            // Fallback: comic-text-detector only
            let regions = state.detector.lock().unwrap().detect(&img)?;
            ocr_regions(&state.ocr, &regions, source_lang)?
        }
    };

    if bubble_inputs.is_empty() {
        return Ok(TranslateImageResponse {
            image_id: req.image_id.clone(),
            status: "ok".into(),
            bubbles: vec![],
        });
    }

    // 4. Build context from previous translations
    let context = build_context(req);

    // 5. Batch translate
    let translate_req = TranslateRequest {
        bubbles: bubble_inputs,
        target_lang: req.target_lang.clone(),
        context,
    };

    let has_override = req.provider_config.as_ref().is_some_and(|c| {
        c.endpoint.is_some() || c.api_key.is_some() || c.model.is_some()
    });
    let temp_engine;
    let engine: &TranslationEngine = if has_override {
        let pc = req.provider_config.as_ref().unwrap();
        let cfg = TranslationConfig {
            endpoint: pc.endpoint.clone().unwrap_or_else(|| state.config.translation.endpoint.clone()),
            api_key: pc.api_key.clone().or_else(|| state.config.translation.api_key.clone()),
            model: pc.model.clone().unwrap_or_else(|| state.config.translation.model.clone()),
        };
        temp_engine = TranslationEngine::new(&cfg)?;
        &temp_engine
    } else {
        &state.translation
    };
    let translated = engine.translate(&translate_req).await?;

    // 6. Fit text into bubbles
    //    LLM may return fewer results (merged/skipped), so look up polygon by bubble id
    let id_to_polygon: std::collections::HashMap<String, usize> = translate_req.bubbles
        .iter()
        .enumerate()
        .map(|(i, b)| (b.id.clone(), i))
        .collect();

    let mut results = Vec::new();
    for translated_bubble in &translated {
        let polygon = match id_to_polygon.get(&translated_bubble.id) {
            Some(&idx) => &bubble_polygons[idx],
            None => continue, // LLM returned an id we don't recognize
        };
        let fit = crate::fit_engine::FitEngine::fit(&translated_bubble.translated_text, polygon)?;
        results.push(BubbleResult {
            bubble_id: translated_bubble.id.clone(),
            polygon: polygon.clone(),
            source_text: translated_bubble.source_text.clone(),
            translated_text: fit.text,
            font_size_px: fit.font_size_px,
            line_height: fit.line_height,
            overflow: fit.overflow,
        });
    }

    // 7. Cache result
    let serialized = serde_json::to_vec(&results)?;
    state.cache.set(&cache_key, &serialized)?;

    Ok(TranslateImageResponse {
        image_id: req.image_id.clone(),
        status: "ok".into(),
        bubbles: results,
    })
}

/// OCR individual regions (for Japanese / fallback path)
fn ocr_regions(
    ocr: &crate::ocr::OcrEngine,
    regions: &[crate::detection::TextRegion],
    lang: &str,
) -> Result<OcrOutput> {
    let mut inputs = Vec::new();
    let mut polygons = Vec::new();
    for (i, region) in regions.iter().enumerate() {
        let result = ocr.recognize(&region.crop, lang)?;
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

/// OCR merged bubbles: OCR each text line, concat within bubble
pub fn ocr_merged_bubbles(
    ocr: &crate::ocr::OcrEngine,
    merged: &[merge::MergedBubble],
    lang: &str,
) -> Result<OcrOutput> {
    let mut inputs = Vec::new();
    let mut polygons = Vec::new();
    for (i, bubble) in merged.iter().enumerate() {
        let mut texts = Vec::new();
        for line in &bubble.lines {
            let result = ocr.recognize(&line.crop, lang)?;
            let text = result.text.trim().to_string();
            // Filter noise: very short text with low confidence is likely SFX/artifact
            if text.is_empty() || (text.chars().count() <= 2 && result.confidence < 0.5) {
                continue;
            }
            texts.push(text);
        }
        let joined = texts.join(" ");
        if joined.is_empty() {
            continue;
        }

        // Filter watermarks: text containing URL-like patterns
        let lower = joined.to_lowercase();
        if lower.contains(".com") || lower.contains(".net") || lower.contains(".org") {
            continue;
        }

        // Filter noise: very short text relative to bounding box area
        let poly = &bubble.polygon;
        if poly.len() >= 4 {
            let bw = poly[1][0] - poly[0][0];
            let bh = poly[2][1] - poly[0][1];
            let area = bw * bh;
            let char_count = joined.chars().count() as f64;
            if char_count < 6.0 && area > char_count * 2000.0 {
                continue;
            }
        }

        // Position for LLM context (top-left corner)
        let (px, py) = if bubble.polygon.len() >= 2 {
            (bubble.polygon[0][0] as i32, bubble.polygon[0][1] as i32)
        } else {
            (0, 0)
        };

        inputs.push(BubbleInput {
            id: format!("b{i}"),
            source_text: joined,
            position: Some((px, py)),
        });
        polygons.push(bubble.polygon.clone());
    }
    Ok((inputs, polygons))
}

fn detect_source_lang(explicit: Option<&str>, target_lang: &str) -> &'static str {
    if let Some(lang) = explicit {
        // Map known values to static strs
        return match lang {
            "ja" => "ja",
            "ko" => "ko",
            "zh" => "zh",
            "en" => "en",
            "vi" => "vi",
            _ => "en",
        };
    }
    // Heuristic: infer source from target
    match target_lang {
        "en" | "vi" => "ja",  // translating to EN/VI → likely Japanese manga
        "ja" => "en",         // translating to JA → likely English source
        _ => "en",
    }
}

fn build_context(req: &TranslateImageRequest) -> Vec<BubbleTranslated> {
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
