use std::sync::Arc;

use anyhow::Result;
use image::DynamicImage;

use crate::api::{AppState, BubbleResult};
use crate::border_detect;
use crate::glossary::Glossary;
use crate::text_layout::DrawableArea;
use crate::translation::{BubbleInput, BubbleTranslated, PageInput, TranslateContext, TranslateRequest};

use super::common::{self, ResolvedEngine};

/// Per-page detection output, kept around for fit+render after translation.
pub struct PageDetection {
    pub page_index: usize,
    pub inputs: Vec<BubbleInput>,
    pub polygons: Vec<Vec<[f64; 2]>>,
    pub areas: Vec<DrawableArea>,
}

/// Chapter-level output: per-page bubble results + optional rendered images.
pub struct ChapterOutput {
    pub pages: Vec<ChapterPageOutput>,
}

pub struct ChapterPageOutput {
    pub page_index: usize,
    pub bubbles: Vec<BubbleResult>,
    pub rendered_image: Option<image::RgbaImage>,
}

/// Translate an entire chapter (1..N pages) in a single agentic LLM call.
///
/// The LLM receives all OCR text grouped by page and can call `view_page(idx)`
/// to see any page image on demand, enabling accurate pronoun/relationship choices.
pub async fn process_chapter(
    state: &Arc<AppState>,
    images: &[DynamicImage],
    target_lang: &str,
    source_lang: Option<&str>,
    glossary: Option<&Glossary>,
    engine: &ResolvedEngine<'_>,
) -> Result<ChapterOutput> {
    let source = common::detect_source_lang(source_lang, target_lang);

    // ── Phase 1: Detect + OCR all pages ──
    let mut page_detections: Vec<PageDetection> = Vec::new();

    for (page_idx, img) in images.iter().enumerate() {
        let (raw_inputs, polygons) = super::detect_and_ocr(Arc::clone(state), img.clone(), source.to_string()).await?;

        // Prefix bubble IDs with page index for chapter-wide uniqueness
        let inputs: Vec<BubbleInput> = raw_inputs
            .into_iter()
            .map(|mut b| {
                b.id = format!("p{}_{}", page_idx, b.id);
                b
            })
            .collect();

        let areas: Vec<DrawableArea> = polygons
            .iter()
            .map(|poly| {
                let inset = border_detect::detect_inset(img, poly);
                DrawableArea::from_polygon(poly, inset)
            })
            .collect();

        page_detections.push(PageDetection {
            page_index: page_idx,
            inputs,
            polygons,
            areas,
        });
    }

    // ── Phase 2: Build chapter-level translate request ──
    let pages: Vec<PageInput> = page_detections
        .iter()
        .filter(|pd| !pd.inputs.is_empty())
        .map(|pd| PageInput {
            page_index: pd.page_index,
            bubbles: pd.inputs.clone(),
        })
        .collect();

    if pages.iter().all(|p| p.bubbles.is_empty()) {
        return Ok(ChapterOutput {
            pages: page_detections
                .iter()
                .map(|pd| ChapterPageOutput {
                    page_index: pd.page_index,
                    bubbles: vec![],
                    rendered_image: None,
                })
                .collect(),
        });
    }

    // Auto-match glossary entries from OCR text
    let glossary_entries = if let Some(glossary) = glossary {
        let all_texts: Vec<&str> = pages
            .iter()
            .flat_map(|p| p.bubbles.iter().map(|b| b.source_text.as_str()))
            .collect();
        glossary.search_batch(&all_texts).unwrap_or_default()
    } else {
        vec![]
    };

    let translate_req = TranslateRequest {
        pages,
        source_lang: source.to_string(),
        target_lang: target_lang.to_string(),
        context: vec![],
        glossary: glossary_entries,
    };

    // ── Phase 3: Single agentic LLM call with view_page access ──
    tracing::info!(
        "Chapter translation: {} pages, {} total bubbles",
        images.len(),
        translate_req.pages.iter().map(|p| p.bubbles.len()).sum::<usize>()
    );

    let ctx = TranslateContext {
        page_images: images,
        glossary,
    };
    let translated = engine.translate(&translate_req, &ctx).await?;

    // ── Phase 4: Group results back by page and fit ──
    let translated_map: std::collections::HashMap<&str, &BubbleTranslated> =
        translated.iter().map(|t| (t.id.as_str(), t)).collect();

    let mut chapter_pages = Vec::new();

    for pd in &page_detections {
        let img = &images[pd.page_index];

        let mut fit_items: Vec<(&BubbleTranslated, usize)> = Vec::new();
        for (local_idx, input) in pd.inputs.iter().enumerate() {
            if let Some(&t) = translated_map.get(input.id.as_str()) {
                fit_items.push((t, local_idx));
            }
        }

        if fit_items.is_empty() {
            chapter_pages.push(ChapterPageOutput {
                page_index: pd.page_index,
                bubbles: vec![],
                rendered_image: None,
            });
            continue;
        }

        let page_items: Vec<(&str, &DrawableArea)> = fit_items
            .iter()
            .map(|(t, idx)| (t.translated_text.as_str(), &pd.areas[*idx]))
            .collect();

        let fits = crate::fit_engine::FitEngine::fit_page_areas(&page_items, img.width())?;

        let bubbles: Vec<BubbleResult> = fit_items
            .iter()
            .zip(fits)
            .map(|((t, idx), fit)| {
                let area = &pd.areas[*idx];
                BubbleResult {
                    bubble_id: t.id.clone(),
                    polygon: pd.polygons[*idx].clone(),
                    source_text: t.source_text.clone(),
                    translated_text: fit.text,
                    font_size_px: fit.font_size_px,
                    line_height: fit.line_height,
                    overflow: fit.overflow,
                    align: "center".to_string(),
                    drawable_area: Some(area.clone()),
                }
            })
            .collect();

        chapter_pages.push(ChapterPageOutput {
            page_index: pd.page_index,
            bubbles,
            rendered_image: None,
        });
    }

    Ok(ChapterOutput { pages: chapter_pages })
}
