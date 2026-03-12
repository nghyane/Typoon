use anyhow::Result;

use crate::api::{AppState, BubbleResult, TranslateImageRequest};
use crate::config::TranslationConfig;
use crate::fit_engine::FitEngine;

use crate::translation::{BubbleInput, BubbleTranslated, TranslateRequest, TranslationEngine};

/// Resolve the translation engine: use provider_config override if present, else default.
pub fn resolve_engine<'a>(
    state: &'a AppState,
    req: &TranslateImageRequest,
) -> Result<ResolvedEngine<'a>> {
    let has_override = req.provider_config.as_ref().is_some_and(|c| {
        c.endpoint.is_some() || c.api_key.is_some() || c.model.is_some()
    });

    if has_override {
        let pc = req.provider_config.as_ref().unwrap();
        let cfg = TranslationConfig {
            endpoint: pc
                .endpoint
                .clone()
                .unwrap_or_else(|| state.config.translation.endpoint.clone()),
            api_key: pc
                .api_key
                .clone()
                .or_else(|| state.config.translation.api_key.clone()),
            model: pc
                .model
                .clone()
                .unwrap_or_else(|| state.config.translation.model.clone()),
            reasoning_effort: state.config.translation.reasoning_effort.clone(),
        };
        Ok(ResolvedEngine::Owned(TranslationEngine::new(&cfg)?))
    } else {
        Ok(ResolvedEngine::Borrowed(&state.translation))
    }
}

pub enum ResolvedEngine<'a> {
    Borrowed(&'a TranslationEngine),
    Owned(TranslationEngine),
}

impl ResolvedEngine<'_> {
    pub async fn translate(&self, req: &TranslateRequest) -> Result<Vec<BubbleTranslated>> {
        match self {
            ResolvedEngine::Borrowed(e) => e.translate(req).await,
            ResolvedEngine::Owned(e) => e.translate(req).await,
        }
    }
}

/// Batch translate bubbles and fit into polygons.
pub async fn translate_and_fit(
    engine: &ResolvedEngine<'_>,
    inputs: Vec<BubbleInput>,
    polygons: &[Vec<[f64; 2]>],
    target_lang: &str,
    context: Vec<BubbleTranslated>,
    page_width: u32,
    insets: &[f64],
) -> Result<Vec<BubbleResult>> {
    let id_to_idx: std::collections::HashMap<String, usize> = inputs
        .iter()
        .enumerate()
        .map(|(i, b)| (b.id.clone(), i))
        .collect();

    let translate_req = TranslateRequest {
        bubbles: inputs,
        target_lang: target_lang.to_string(),
        context,
    };
    let translated = engine.translate(&translate_req).await?;

    // Build matched pairs for page-level fitting
    let mut fit_meta: Vec<(&BubbleTranslated, &Vec<[f64; 2]>)> = Vec::new();
    for t in &translated {
        if let Some(&idx) = id_to_idx.get(&t.id) {
            fit_meta.push((t, &polygons[idx]));
        }
    }

    let page_items: Vec<(&str, &[[f64; 2]])> = fit_meta
        .iter()
        .map(|(t, poly)| (t.translated_text.as_str(), poly.as_slice()))
        .collect();
    let fits = FitEngine::fit_page(&page_items, page_width)?;

    let results = fit_meta
        .iter()
        .zip(fits.into_iter())
        .map(|((t, polygon), fit)| {
            let idx = id_to_idx.get(&t.id).copied().unwrap_or(0);
            let inset = insets.get(idx).copied().unwrap_or(3.0);
            BubbleResult {
                bubble_id: t.id.clone(),
                polygon: (*polygon).clone(),
                source_text: t.source_text.clone(),
                translated_text: fit.text,
                font_size_px: fit.font_size_px,
                line_height: fit.line_height,
                overflow: fit.overflow,
                align: "center".to_string(),
                inset,
            }
        })
        .collect();
    Ok(results)
}

/// Batch translate bubbles only (no fitting). For use with canvas agent.
pub async fn translate_only(
    engine: &ResolvedEngine<'_>,
    inputs: Vec<BubbleInput>,
    polygons: &[Vec<[f64; 2]>],
    target_lang: &str,
    context: Vec<BubbleTranslated>,
) -> Result<Vec<crate::canvas_agent::CanvasBubble>> {
    let id_to_idx: std::collections::HashMap<String, usize> = inputs
        .iter()
        .enumerate()
        .map(|(i, b)| (b.id.clone(), i))
        .collect();

    let translate_req = TranslateRequest {
        bubbles: inputs,
        target_lang: target_lang.to_string(),
        context,
    };
    let translated = engine.translate(&translate_req).await?;

    let canvas_bubbles = translated
        .into_iter()
        .filter_map(|t| {
            id_to_idx.get(&t.id).map(|&idx| crate::canvas_agent::CanvasBubble {
                id: t.id,
                polygon: polygons[idx].clone(),
                source_text: t.source_text,
                translated_text: t.translated_text,
            })
        })
        .collect();

    Ok(canvas_bubbles)
}

/// Convert canvas agent output to BubbleResult for API response.
/// Wraps translated text to fit polygon so overlay rendering works correctly.
pub fn bubbles_from_canvas(
    canvas_bubbles: &[crate::canvas_agent::CanvasBubble],
    commands: &[crate::canvas_agent::CanvasCommand],
    _polygons: &[Vec<[f64; 2]>],
    insets: &[f64],
) -> Vec<BubbleResult> {
    let font = crate::text_layout::get_font();
    let bubble_map: std::collections::HashMap<&str, (&crate::canvas_agent::CanvasBubble, usize)> =
        canvas_bubbles.iter().enumerate().map(|(i, b)| (b.id.as_str(), (b, i))).collect();

    let mut results = Vec::new();
    for cmd in commands {
        let crate::canvas_agent::CanvasCommand::Typeset {
            bubble_id, text, font_size, align, crop: _,
        } = cmd;
        let Some(&(bubble, idx)) = bubble_map.get(bubble_id.as_str()) else {
            continue;
        };
        let inset = insets.get(idx).copied().unwrap_or(3.0);

        // Wrap text to fit within the polygon bbox (matching overlay render logic)
        let (bx1, _, bx2, _) = crate::text_layout::polygon_bbox(&bubble.polygon);
        let safe_w = (bx2 - bx1 - 2.0 * inset).max(1.0);
        let wrapped = crate::text_layout::wrap_text(text, safe_w, *font_size, font);

        results.push(BubbleResult {
            bubble_id: bubble_id.clone(),
            polygon: bubble.polygon.clone(),
            source_text: bubble.source_text.clone(),
            translated_text: wrapped.join("\n"),
            font_size_px: *font_size,
            line_height: crate::text_layout::LINE_HEIGHT_MULTIPLIER,
            overflow: false,
            align: align.clone(),
            inset,
        });
    }
    results
}

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

pub fn detect_source_lang(explicit: Option<&str>, target_lang: &str) -> &'static str {
    if let Some(lang) = explicit {
        return match lang {
            "ja" => "ja",
            "ko" => "ko",
            "zh" => "zh",
            "en" => "en",
            "vi" => "vi",
            _ => "en",
        };
    }
    match target_lang {
        "en" | "vi" => "ja",
        "ja" => "en",
        _ => "en",
    }
}
