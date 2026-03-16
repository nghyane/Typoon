use anyhow::Result;
use image::DynamicImage;

use crate::runner::Session;
use crate::translation::{BubbleTranslated, TranslateRequest};
use crate::vision::border;
use crate::vision::detection::TextDetector;
use crate::vision::ocr::OcrEngine;

use super::types::*;

// ═══════════════════════════════════════════════════════════════════════
// Phase 1: Detect + OCR
// ═══════════════════════════════════════════════════════════════════════

pub fn detect_chapter(
    detector: &TextDetector,
    ocr: &OcrEngine,
    images: &[DynamicImage],
    source_lang: &str,
) -> Result<Vec<PageDetections>> {
    let use_ppocr = source_lang != "ja" && ocr.can_detect();

    if !use_ppocr {
        // manga-ocr is autoregressive — can't split det/rec, run sequentially.
        return detect_chapter_sequential(detector, ocr, images, source_lang);
    }

    // PP-OCR pipeline: det and rec use separate ONNX sessions.
    // While rec model processes page N (batch inference),
    // det model runs on page N+1 — true parallelism, no session contention.
    let mut pages = Vec::with_capacity(images.len());
    let mut pending_det: Option<Result<super::PpocrDetected>> = None;

    for page_idx in 0..images.len() {
        let detected = pending_det.take()
            .unwrap_or_else(|| super::ppocr_detect(ocr, &images[page_idx]));

        let next_idx = page_idx + 1;

        // Overlap: rec(page N) ‖ det(page N+1)
        let (raw, next_det) = if next_idx < images.len() {
            std::thread::scope(|s| {
                let next_handle = s.spawn(|| super::ppocr_detect(ocr, &images[next_idx]));
                let raw = detected.and_then(|d| super::ppocr_recognize(ocr, &d, source_lang));
                let next = next_handle.join().unwrap();
                (raw, Some(next))
            })
        } else {
            let raw = detected.and_then(|d| super::ppocr_recognize(ocr, &d, source_lang));
            (raw, None)
        };

        let img = &images[page_idx];
        let bubbles: Vec<DetectedBubble> = raw?
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

        pending_det = next_det;
    }

    Ok(pages)
}

/// Sequential fallback for manga-ocr (autoregressive decoder can't pipeline).
fn detect_chapter_sequential(
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

/// Translate a chapter. All per-series resources come from `session`.
pub async fn translate_chapter(
    session: &Session,
    job: &TranslateJob<'_>,
) -> Result<Vec<PageTranslations>> {
    let non_empty = job.detections.iter().any(|pd| !pd.bubbles.is_empty());

    if !non_empty {
        return Ok(job
            .detections
            .iter()
            .map(|pd| PageTranslations {
                page_index: pd.page_index,
                bubbles: vec![],
            })
            .collect());
    }

    let knowledge_snapshot = session.project.as_ref().and_then(|store| {
        job.chapter_index
            .and_then(|ch| store.get_latest_snapshot(ch).ok().flatten())
    });

    let glossary_entries = match &session.project {
        Some(store) => {
            let all_texts: Vec<&str> = job
                .detections
                .iter()
                .flat_map(|pd| pd.bubbles.iter().map(|b| b.source_text.as_str()))
                .collect();
            store.glossary_search_batch(&all_texts).unwrap_or_default()
        }
        None => vec![],
    };

    let translate_req = TranslateRequest {
        detections: job.detections,
        source_lang: job.source_lang,
        target_lang: job.target_lang,
        glossary: glossary_entries,
        knowledge_snapshot,
    };

    let total_bubbles: usize = job.detections.iter().map(|pd| pd.bubbles.len()).sum();
    tracing::info!(
        "Chapter translation: {} pages, {total_bubbles} bubbles",
        job.images.len()
    );

    let t_phase = std::time::Instant::now();
    let translated = session
        .engine()
        .translate(
            &translate_req,
            job.images,
            session.project.as_deref(),
            session.context_provider(),
        )
        .await?;
    tracing::info!("Phase translate: {:.1}s", t_phase.elapsed().as_secs_f64());

    let page_widths: Vec<u32> = job.images.iter().map(|img| img.width()).collect();

    let t_phase = std::time::Instant::now();
    let result = fit_results(job.detections, &page_widths, &translated)?;
    tracing::info!("Phase fit: {:.1}s", t_phase.elapsed().as_secs_f64());

    Ok(result)
}

// ═══════════════════════════════════════════════════════════════════════
// Phase 3: Render
// ═══════════════════════════════════════════════════════════════════════

pub fn render_pages(
    pages: Vec<PageTranslations>,
    images: &[DynamicImage],
    runner: &crate::runner::TranslationRunner,
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
    use rayon::prelude::*;

    let translated_map: std::collections::HashMap<String, &BubbleTranslated> = translated
        .iter()
        .map(|t| (t.id.clone(), t))
        .collect();

    detections
        .par_iter()
        .map(|pd| {
            let page_w = page_widths[pd.page_index];

            let matched: Vec<(&DetectedBubble, &BubbleTranslated)> = pd
                .bubbles
                .iter()
                .filter_map(|b| {
                    let id = TranslateRequest::bubble_id(pd.page_index, b.idx);
                    translated_map.get(&id).map(|t| (b, *t))
                })
                .collect();

            if matched.is_empty() {
                return Ok(PageTranslations {
                    page_index: pd.page_index,
                    bubbles: vec![],
                });
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

            Ok(PageTranslations {
                page_index: pd.page_index,
                bubbles,
            })
        })
        .collect()
}
