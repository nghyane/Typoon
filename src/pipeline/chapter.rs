use anyhow::Result;
use image::DynamicImage;
use serde::{Deserialize, Serialize};

use crate::vision::border;
use crate::storage::context::ChapterTranslation;
use crate::vision::detection::{LocalTextMask, TextDetector};
use crate::vision::ocr::OcrEngine;
use crate::runner::TranslationRunner;
use crate::render::layout::DrawableArea;
use crate::translation::{
    BubbleInput, BubbleTranslated, ContextNote, PageInput, TranslateContext, TranslateRequest,
    TranslationEngine,
};

/// A single translated bubble with fit results and rendering data.
#[derive(Debug, Serialize, Deserialize)]
pub struct BubbleResult {
    pub bubble_id: String,
    pub polygon: Vec<[f64; 2]>,
    pub source_text: String,
    pub translated_text: String,
    pub font_size_px: u32,
    pub line_height: f64,
    pub overflow: bool,
    #[serde(default = "default_align")]
    pub align: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub drawable_area: Option<DrawableArea>,
    #[serde(skip, default)]
    pub text_mask: Option<LocalTextMask>,
}

fn default_align() -> String {
    "center".into()
}

/// Per-page detection output, kept around for fit+render after translation.
pub struct PageDetection {
    pub page_index: usize,
    pub inputs: Vec<BubbleInput>,
    pub polygons: Vec<Vec<[f64; 2]>>,
    pub areas: Vec<DrawableArea>,
    pub masks: Vec<Option<LocalTextMask>>,
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

// ═══════════════════════════════════════════════════════════════════════
// Phase 1: Detect + OCR (CPU-bound, no context dependency)
// ═══════════════════════════════════════════════════════════════════════

/// Detect and OCR all pages in a chapter. CPU-bound, can run in parallel
/// with translation of another chapter.
///
/// Uses a pipelined approach: detection (Mutex-bound) runs ahead of OCR +
/// border detection so the detector lock is released as early as possible,
/// allowing the next chapter's detection to proceed.
pub fn detect_chapter(
    detector: &TextDetector,
    ocr: &OcrEngine,
    images: &[DynamicImage],
    source_lang: &str,
) -> Result<Vec<PageDetection>> {
    let mut detections = Vec::with_capacity(images.len());

    for (page_idx, img) in images.iter().enumerate() {
        let (raw_inputs, polygons, masks) = super::detect_and_ocr(detector, ocr, img, source_lang)?;

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
                let inset = border::detect_inset(img, poly);
                DrawableArea::from_polygon(poly, inset)
            })
            .collect();

        detections.push(PageDetection {
            page_index: page_idx,
            inputs,
            polygons,
            areas,
            masks,
        });
    }

    Ok(detections)
}

// ═══════════════════════════════════════════════════════════════════════
// Phase 2: Translate + Fit + Save (needs context from prior chapters)
// ═══════════════════════════════════════════════════════════════════════

/// Translate, fit, and save chapter context from pre-computed detections.
/// Must run sequentially per chapter (depends on context from prior chapters).
pub async fn translate_and_prepare(
    runner: &TranslationRunner,
    detections: Vec<PageDetection>,
    images: &[DynamicImage],
    target_lang: &str,
    source_lang: &str,
    project_id: Option<&str>,
    chapter_index: Option<usize>,
) -> Result<Vec<ChapterPageOutput>> {
    translate_and_prepare_inner(
        runner,
        &runner.translation,
        detections,
        images,
        target_lang,
        source_lang,
        vec![],
        project_id,
        chapter_index,
    )
    .await
}

/// Translate, fit, save context, and render a chapter from pre-computed detections.
/// Must run sequentially per chapter (depends on context from prior chapters).
pub async fn translate_and_render(
    runner: &TranslationRunner,
    detections: Vec<PageDetection>,
    images: &[DynamicImage],
    target_lang: &str,
    source_lang: &str,
    project_id: Option<&str>,
    chapter_index: Option<usize>,
) -> Result<ChapterOutput> {
    let chapter_pages = translate_and_prepare(
        runner,
        detections,
        images,
        target_lang,
        source_lang,
        project_id,
        chapter_index,
    )
    .await?;

    let t_phase = std::time::Instant::now();
    let chapter_pages = render_prepared_pages(chapter_pages, images, runner);
    tracing::info!("Phase render: {:.1}s", t_phase.elapsed().as_secs_f64());

    Ok(ChapterOutput {
        pages: chapter_pages,
    })
}

/// Translate + fit + render using a caller-provided engine and context.
///
/// Used by the HTTP single-page pipeline which may override the translation
/// provider per-request and supply context hints from previous translations.
pub async fn translate_and_render_with_engine(
    runner: &TranslationRunner,
    engine: &TranslationEngine,
    detections: Vec<PageDetection>,
    images: &[DynamicImage],
    target_lang: &str,
    source_lang: &str,
    context: Vec<BubbleTranslated>,
) -> Result<ChapterOutput> {
    let chapter_pages = translate_and_prepare_inner(
        runner,
        engine,
        detections,
        images,
        target_lang,
        source_lang,
        context,
        None,
        None,
    )
    .await?;

    let t_phase = std::time::Instant::now();
    let chapter_pages = render_prepared_pages(chapter_pages, images, runner);
    tracing::info!("Phase render: {:.1}s", t_phase.elapsed().as_secs_f64());

    Ok(ChapterOutput {
        pages: chapter_pages,
    })
}

async fn translate_and_prepare_inner(
    runner: &TranslationRunner,
    engine: &TranslationEngine,
    detections: Vec<PageDetection>,
    images: &[DynamicImage],
    target_lang: &str,
    source_lang: &str,
    context: Vec<BubbleTranslated>,
    project_id: Option<&str>,
    chapter_index: Option<usize>,
) -> Result<Vec<ChapterPageOutput>> {
    // ── Build chapter-level translate request ──
    let pages: Vec<PageInput> = detections
        .iter()
        .filter(|pd| !pd.inputs.is_empty())
        .map(|pd| PageInput {
            page_index: pd.page_index,
            bubbles: pd.inputs.clone(),
        })
        .collect();

    if pages.iter().all(|p| p.bubbles.is_empty()) {
        return Ok(detections
            .iter()
            .map(|pd| ChapterPageOutput {
                page_index: pd.page_index,
                bubbles: vec![],
                rendered_image: None,
            })
            .collect());
    }

    let glossary_entries = if let Some(glossary) = &runner.glossary {
        let all_texts: Vec<&str> = pages
            .iter()
            .flat_map(|p| p.bubbles.iter().map(|b| b.source_text.as_str()))
            .collect();
        glossary.search_batch(&all_texts).unwrap_or_default()
    } else {
        vec![]
    };

    let notes = fetch_previous_notes(runner, project_id, chapter_index);

    let translate_req = TranslateRequest {
        pages,
        source_lang: source_lang.to_string(),
        target_lang: target_lang.to_string(),
        context,
        glossary: glossary_entries,
        notes,
    };

    // ── Single agentic LLM call ──
    tracing::info!(
        "Chapter translation: {} pages, {} bubbles",
        images.len(),
        translate_req
            .pages
            .iter()
            .map(|p| p.bubbles.len())
            .sum::<usize>()
    );

    // Capture page widths before translate (fit only needs widths, not full images).
    let page_widths: Vec<u32> = images.iter().map(|img| img.width()).collect();

    let ctx = TranslateContext {
        page_images: images,
        glossary: runner.glossary.as_ref(),
        context_store: runner.context_store.as_deref(),
        context_agent: runner
            .context_agent
            .as_ref()
            .map(|a| &**a as &dyn crate::llm::Provider),
        project_id,
        chapter_index,
    };
    let t_phase = std::time::Instant::now();
    let translated = engine.translate(&translate_req, &ctx).await?;
    tracing::info!("Phase translate: {:.1}s", t_phase.elapsed().as_secs_f64());

    // ── Fit text into bubbles ──
    let t_phase = std::time::Instant::now();
    let chapter_pages = fit_results(&detections, &page_widths, &translated)?;
    tracing::info!("Phase fit: {:.1}s", t_phase.elapsed().as_secs_f64());

    // ── Save translations to ContextStore ──
    save_context(
        &chapter_pages,
        runner,
        source_lang,
        target_lang,
        project_id,
        chapter_index,
    );

    Ok(chapter_pages)
}

// ═══════════════════════════════════════════════════════════════════════
// Internal helpers
// ═══════════════════════════════════════════════════════════════════════

// ── Fit results into bubbles ──

fn fit_results(
    detections: &[PageDetection],
    page_widths: &[u32],
    translated: &[BubbleTranslated],
) -> Result<Vec<ChapterPageOutput>> {
    let translated_map: std::collections::HashMap<&str, &BubbleTranslated> =
        translated.iter().map(|t| (t.id.as_str(), t)).collect();

    let mut chapter_pages = Vec::new();

    for pd in detections {
        let page_w = page_widths[pd.page_index];

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

        let fits = crate::render::fit::FitEngine::fit_page_areas(&page_items, page_w)?;

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
                    text_mask: pd.masks.get(*idx).cloned().flatten(),
                }
            })
            .collect();

        chapter_pages.push(ChapterPageOutput {
            page_index: pd.page_index,
            bubbles,
            rendered_image: None,
        });
    }

    Ok(chapter_pages)
}

// ── Save context ──

/// Save chapter translations to the context store (SQLite + FTS5).
fn save_context(
    chapter_pages: &[ChapterPageOutput],
    runner: &TranslationRunner,
    source_lang: &str,
    target_lang: &str,
    project_id: Option<&str>,
    chapter_index: Option<usize>,
) {
    let (Some(store), Some(pid), Some(ch_idx)) = (&runner.context_store, project_id, chapter_index)
    else {
        return;
    };

    let translations: Vec<ChapterTranslation> = chapter_pages
        .iter()
        .flat_map(|page| {
            page.bubbles.iter().map(move |b| ChapterTranslation {
                page_index: page.page_index,
                bubble_id: b.bubble_id.clone(),
                source_text: b.source_text.clone(),
                translated_text: b.translated_text.clone(),
                source_lang: source_lang.to_string(),
                target_lang: target_lang.to_string(),
            })
        })
        .collect();

    if translations.is_empty() {
        return;
    }

    if let Err(e) = store.save_chapter(pid, ch_idx, &translations) {
        tracing::warn!("Failed to save chapter context: {e}");
    }
}

// ── Render ──

/// Render translated text onto page images.
///
/// Pages are rendered in a dedicated, bounded thread pool configured via
/// `config.runtime.*`. This avoids unbounded contention when LaMa/CoreML is enabled.
pub fn render_prepared_pages(
    chapter_pages: Vec<ChapterPageOutput>,
    images: &[DynamicImage],
    runner: &TranslationRunner,
) -> Vec<ChapterPageOutput> {
    let inpainter = runner.inpainter.as_ref();
    tracing::debug!("Render phase using {} worker(s)", runner.render_workers());

    runner.install_render(|| {
        use rayon::prelude::*;

        chapter_pages
            .into_par_iter()
            .map(|mut page| {
                if !page.bubbles.is_empty() {
                    let img = &images[page.page_index];
                    page.rendered_image =
                        Some(crate::render::overlay::render(img, &page.bubbles, inpainter));
                }
                page
            })
            .collect()
    })
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
    project_id: Option<&str>,
    chapter_index: Option<usize>,
) -> Vec<ContextNote> {
    let (Some(store), Some(pid), Some(ch_idx)) = (&runner.context_store, project_id, chapter_index)
    else {
        return vec![];
    };
    if ch_idx == 0 {
        return vec![];
    }

    let raw = match store.get_notes_before(pid, ch_idx) {
        Ok(notes) => notes,
        Err(e) => {
            tracing::warn!("Failed to fetch previous notes: {e}");
            return vec![];
        }
    };
    if raw.is_empty() {
        return vec![];
    }

    // Only proactively inject stable facts (relationship, character).
    // Event/setting notes are ephemeral — available reactively via get_context().
    let stable: Vec<_> = raw
        .into_iter()
        .filter(|n| matches!(n.note_type.as_str(), "relationship" | "character"))
        .collect();

    // Dedup: keep latest occurrence (later chapter wins).
    let mut seen = std::collections::HashSet::new();
    let mut notes: Vec<_> = stable
        .into_iter()
        .rev()
        .filter(|n| seen.insert(n.content.clone()))
        .collect();

    // Stable sort: priority first, then recency (rev order preserved within same priority).
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
