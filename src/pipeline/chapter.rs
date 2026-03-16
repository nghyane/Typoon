use anyhow::Result;
use image::DynamicImage;

use crate::vision::border;
use crate::vision::detection::TextDetector;
use crate::vision::ocr::OcrEngine;
use crate::runner::TranslationRunner;
use crate::translation::{
    BubbleInput, BubbleTranslated, ContextNote, PageInput, TranslateContext, TranslateRequest,
    TranslationEngine,
};

use super::types::*;

// ═══════════════════════════════════════════════════════════════════════
// Phase 1: Detect + OCR
// ═══════════════════════════════════════════════════════════════════════

/// Detect and OCR all pages in a chapter.
/// Enriches raw detection with border detection + drawable area.
pub fn detect_chapter(
    detector: &TextDetector,
    ocr: &OcrEngine,
    images: &[DynamicImage],
    source_lang: &str,
) -> Result<Vec<PageDetections>> {
    let mut pages = Vec::with_capacity(images.len());

    for (page_idx, img) in images.iter().enumerate() {
        let raw = super::detect_and_ocr(detector, ocr, img, source_lang)?;

        let bubbles: Vec<DetectedBubble> = raw
            .into_iter()
            .enumerate()
            .map(|(idx, r)| {
                let inset = border::detect_inset(img, &r.polygon);
                let area = crate::render::layout::DrawableArea::from_polygon(&r.polygon, inset);
                DetectedBubble {
                    idx,
                    source_text: r.source_text,
                    polygon: r.polygon,
                    area,
                    det_confidence: r.det_confidence,
                    ocr_confidence: r.ocr_confidence,
                    mask: r.mask,
                }
            })
            .collect();

        pages.push(PageDetections {
            page_index: page_idx,
            bubbles,
        });
    }

    Ok(pages)
}

// ═══════════════════════════════════════════════════════════════════════
// Phase 2: Translate + Fit
// ═══════════════════════════════════════════════════════════════════════

/// Translate and fit a chapter from detections. No rendering.
pub async fn translate_chapter(
    runner: &TranslationRunner,
    detections: &[PageDetections],
    images: &[DynamicImage],
    target_lang: &str,
    source_lang: &str,
    chapter_index: Option<usize>,
) -> Result<Vec<PageTranslations>> {
    translate_inner(
        runner,
        &runner.translation,
        detections,
        images,
        target_lang,
        source_lang,
        vec![],
        chapter_index,
    )
    .await
}

/// Translate + fit + render (convenience for single chapter / HTTP).
pub async fn translate_and_render(
    runner: &TranslationRunner,
    detections: Vec<PageDetections>,
    images: &[DynamicImage],
    target_lang: &str,
    source_lang: &str,
    chapter_index: Option<usize>,
) -> Result<ChapterOutput> {
    let pages = translate_chapter(
        runner, &detections, images, target_lang, source_lang, chapter_index,
    )
    .await?;

    let rendered = render_pages(pages, images, runner);
    Ok(ChapterOutput { pages: rendered })
}

/// Translate + fit + render with custom engine and context (HTTP API).
pub async fn translate_and_render_with_engine(
    runner: &TranslationRunner,
    engine: &TranslationEngine,
    detections: Vec<PageDetections>,
    images: &[DynamicImage],
    target_lang: &str,
    source_lang: &str,
    context: Vec<BubbleTranslated>,
) -> Result<ChapterOutput> {
    let pages = translate_inner(
        runner, engine, &detections, images, target_lang, source_lang, context, None,
    )
    .await?;

    let rendered = render_pages(pages, images, runner);
    Ok(ChapterOutput { pages: rendered })
}

async fn translate_inner(
    runner: &TranslationRunner,
    engine: &TranslationEngine,
    detections: &[PageDetections],
    images: &[DynamicImage],
    target_lang: &str,
    source_lang: &str,
    context: Vec<BubbleTranslated>,
    chapter_index: Option<usize>,
) -> Result<Vec<PageTranslations>> {
    let pages: Vec<PageInput> = detections
        .iter()
        .filter(|pd| !pd.bubbles.is_empty())
        .map(|pd| PageInput {
            page_index: pd.page_index,
            bubbles: pd
                .bubbles
                .iter()
                .map(|b| BubbleInput {
                    id: format!("p{}_b{}", pd.page_index, b.idx),
                    source_text: b.source_text.clone(),
                    position: b.polygon.first().map(|p| (p[0] as i32, p[1] as i32)),
                    det_confidence: b.det_confidence,
                    ocr_confidence: b.ocr_confidence,
                })
                .collect(),
        })
        .collect();

    if pages.iter().all(|p| p.bubbles.is_empty()) {
        return Ok(detections
            .iter()
            .map(|pd| PageTranslations {
                page_index: pd.page_index,
                bubbles: vec![],
            })
            .collect());
    }

    let glossary_entries = if let Some(project) = &runner.default_project {
        let all_texts: Vec<&str> = pages
            .iter()
            .flat_map(|p| p.bubbles.iter().map(|b| b.source_text.as_str()))
            .collect();
        project.glossary_search_batch(&all_texts).unwrap_or_default()
    } else {
        vec![]
    };

    let notes = fetch_previous_notes(runner, chapter_index);

    let translate_req = TranslateRequest {
        pages,
        source_lang: source_lang.to_string(),
        target_lang: target_lang.to_string(),
        context,
        glossary: glossary_entries,
        notes,
    };

    let total_bubbles: usize = translate_req.pages.iter().map(|p| p.bubbles.len()).sum();
    tracing::info!("Chapter translation: {} pages, {total_bubbles} bubbles", images.len());

    let page_widths: Vec<u32> = images.iter().map(|img| img.width()).collect();

    let ctx = TranslateContext {
        page_images: images,
        project: runner.default_project.as_deref(),
        context_agent: runner
            .context_agent
            .as_ref()
            .map(|a| &**a as &dyn crate::llm::Provider),
        chapter_index,
    };

    let t_phase = std::time::Instant::now();
    let translated = engine.translate(&translate_req, &ctx).await?;
    tracing::info!("Phase translate: {:.1}s", t_phase.elapsed().as_secs_f64());

    let t_phase = std::time::Instant::now();
    let result = fit_results(detections, &page_widths, &translated)?;
    tracing::info!("Phase fit: {:.1}s", t_phase.elapsed().as_secs_f64());

    Ok(result)
}

// ═══════════════════════════════════════════════════════════════════════
// Phase 3: Render
// ═══════════════════════════════════════════════════════════════════════

/// Render translated pages onto source images.
pub fn render_pages(
    pages: Vec<PageTranslations>,
    images: &[DynamicImage],
    runner: &TranslationRunner,
) -> Vec<RenderedPage> {
    let inpainter = runner.inpainter.as_ref();
    tracing::debug!("Render phase using {} worker(s)", runner.render_workers());

    runner.install_render(|| {
        use rayon::prelude::*;

        pages
            .into_par_iter()
            .map(|page| {
                let image = if page.bubbles.is_empty() {
                    images[page.page_index].to_rgba8()
                } else {
                    let img = &images[page.page_index];
                    crate::render::overlay::render(img, &page.bubbles, inpainter)
                };
                RenderedPage {
                    page_index: page.page_index,
                    bubbles: page.bubbles,
                    image,
                }
            })
            .collect()
    })
}

// ═══════════════════════════════════════════════════════════════════════
// Internal helpers
// ═══════════════════════════════════════════════════════════════════════

fn fit_results(
    detections: &[PageDetections],
    page_widths: &[u32],
    translated: &[BubbleTranslated],
) -> Result<Vec<PageTranslations>> {
    let translated_map: std::collections::HashMap<&str, &BubbleTranslated> =
        translated.iter().map(|t| (t.id.as_str(), t)).collect();

    let mut result = Vec::new();

    for pd in detections {
        let page_w = page_widths[pd.page_index];

        let matched: Vec<(&DetectedBubble, &BubbleTranslated)> = pd
            .bubbles
            .iter()
            .filter_map(|b| {
                let id = format!("p{}_b{}", pd.page_index, b.idx);
                translated_map.get(id.as_str()).map(|t| (b, *t))
            })
            .collect();

        if matched.is_empty() {
            result.push(PageTranslations {
                page_index: pd.page_index,
                bubbles: vec![],
            });
            continue;
        }

        let page_items: Vec<(&str, &crate::render::layout::DrawableArea)> = matched
            .iter()
            .map(|(b, t)| (t.translated_text.as_str(), &b.area))
            .collect();

        let fits = crate::render::fit::FitEngine::fit_page_areas(&page_items, page_w)?;

        let bubbles: Vec<TranslatedBubble> = matched
            .iter()
            .zip(fits)
            .map(|((det, trans), fit)| TranslatedBubble {
                idx: det.idx,
                source_text: trans.source_text.clone(),
                translated_text: fit.text,
                polygon: det.polygon.clone(),
                area: det.area.clone(),
                mask: det.mask.clone(),
                font_size_px: fit.font_size_px,
                line_height: fit.line_height,
                overflow: fit.overflow,
            })
            .collect();

        result.push(PageTranslations {
            page_index: pd.page_index,
            bubbles,
        });
    }

    Ok(result)
}

// ── Notes injection ──

const NOTES_BUDGET_CHARS: usize = 2000;

fn note_priority(note_type: &str) -> u8 {
    match note_type {
        "relationship" => 0,
        "character" => 1,
        _ => 2,
    }
}

fn fetch_previous_notes(
    runner: &TranslationRunner,
    chapter_index: Option<usize>,
) -> Vec<ContextNote> {
    let (Some(store), Some(ch_idx)) = (&runner.default_project, chapter_index) else {
        return vec![];
    };
    if ch_idx == 0 {
        return vec![];
    }

    let raw = match store.get_notes_before(ch_idx) {
        Ok(notes) => notes,
        Err(e) => {
            tracing::warn!("Failed to fetch previous notes: {e}");
            return vec![];
        }
    };
    if raw.is_empty() {
        return vec![];
    }

    let stable: Vec<_> = raw
        .into_iter()
        .filter(|n| matches!(n.note_type.as_str(), "relationship" | "character"))
        .collect();

    let mut seen = std::collections::HashSet::new();
    let mut notes: Vec<_> = stable
        .into_iter()
        .rev()
        .filter(|n| seen.insert(n.content.clone()))
        .collect();

    notes.sort_by(|a, b| note_priority(&a.note_type).cmp(&note_priority(&b.note_type)));

    let mut total = 0;
    let result: Vec<ContextNote> = notes
        .into_iter()
        .take_while(|n| {
            total += n.content.len();
            total <= NOTES_BUDGET_CHARS
        })
        .map(|n| ContextNote {
            note_type: n.note_type,
            content: n.content,
        })
        .collect();

    if !result.is_empty() {
        tracing::info!(
            "Injecting {} continuity notes ({} chars)",
            result.len(),
            total
        );
    }

    result
}
