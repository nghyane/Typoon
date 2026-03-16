pub mod chapter;
pub mod merge;
pub mod series;
pub mod types;

use anyhow::Result;
use image::DynamicImage;

use crate::vision::detection::TextDetector;
use crate::vision::ocr::{DetectionOutput, OcrEngine};

use types::RawBubble;

// ═══════════════════════════════════════════════════════════════════════
// Shared detect + OCR (used by both single-page HTTP and chapter CLI)
// ═══════════════════════════════════════════════════════════════════════

/// Synchronous detect + OCR dispatch. Returns raw bubbles (no layout info yet).
///
/// Strategy based on source_lang:
///   "ja" → comic-text-detector + manga-ocr
///   _    → PP-OCR det (line merge) + PP-OCR rec
pub(crate) fn detect_and_ocr(
    detector: &TextDetector,
    ocr: &OcrEngine,
    img: &DynamicImage,
    source_lang: &str,
) -> Result<Vec<RawBubble>> {
    match source_lang {
        "ja" => detect_and_ocr_manga(detector, ocr, img, source_lang),
        _ if ocr.can_detect() => detect_and_ocr_ppocr(ocr, img, source_lang),
        _ => detect_and_ocr_manga(detector, ocr, img, source_lang),
    }
}

// ── PP-OCR two-phase pipeline ──

/// Phase 1 (det model only): detect text lines + merge into bubbles.
/// Returns intermediate state needed by phase 2.
pub(crate) struct PpocrDetected {
    pub merged: Vec<merge::MergedBubble>,
}

fn ppocr_detect(ocr: &OcrEngine, img: &DynamicImage) -> Result<PpocrDetected> {
    let DetectionOutput {
        regions: lines,
        prob_image,
    } = ocr.detect(img)?;
    let merged = merge::group_lines(lines, img, prob_image.as_ref());
    Ok(PpocrDetected { merged })
}

/// Phase 2 (rec model only): batch OCR all line crops → assemble bubbles.
fn ppocr_recognize(ocr: &OcrEngine, detected: &PpocrDetected, lang: &str) -> Result<Vec<RawBubble>> {
    let merged = &detected.merged;

    // Collect all line crops for a single batch inference.
    let mut all_crops: Vec<&DynamicImage> = Vec::new();
    let mut line_to_bubble: Vec<usize> = Vec::new();
    for (bi, bubble) in merged.iter().enumerate() {
        for line in &bubble.lines {
            all_crops.push(&line.crop);
            line_to_bubble.push(bi);
        }
    }

    let all_results = ocr.recognize_batch(&all_crops, lang)?;

    // Distribute results back to their bubbles.
    let mut bubble_texts: Vec<Vec<(String, f64, f64)>> = vec![Vec::new(); merged.len()];
    for (idx, result) in all_results.into_iter().enumerate() {
        let bi = line_to_bubble[idx];
        if let Ok(r) = result {
            let text = r.text.trim().to_string();
            if text.is_empty() || (text.chars().count() <= 2 && r.confidence < 0.5) {
                continue;
            }
            if r.min_char_confidence < 0.15 && text.chars().count() <= 4 {
                continue;
            }
            bubble_texts[bi].push((text, r.confidence, r.min_char_confidence));
        }
    }

    let mut bubbles = Vec::new();
    for (bi, bubble) in merged.iter().enumerate() {
        let line_results = &bubble_texts[bi];
        if line_results.is_empty() {
            continue;
        }

        let joined: String = line_results.iter().map(|(t, _, _)| t.as_str()).collect::<Vec<_>>().join(" ");
        let conf_count = line_results.len();
        let total_conf: f64 = line_results.iter().map(|(_, c, _)| c).sum();
        let avg_conf = total_conf / conf_count as f64;

        let all_portrait = bubble.lines.iter().all(|l| {
            let (lx1, ly1, lx2, ly2) = merge::line_bbox(&l.polygon);
            let w = lx2 - lx1;
            let h = ly2 - ly1;
            h >= w * 0.8
        });
        if all_portrait && avg_conf < 0.7 {
            continue;
        }

        let lower = joined.to_lowercase();
        if lower.contains(".com") || lower.contains(".net") || lower.contains(".org") {
            continue;
        }

        bubbles.push(RawBubble {
            source_text: joined,
            polygon: bubble.polygon.clone(),
            det_confidence: bubble.confidence,
            ocr_confidence: avg_conf,
            mask: bubble.mask.clone(),
        });
    }

    Ok(bubbles)
}

/// Combined PP-OCR: detect + recognize (single-page convenience).
fn detect_and_ocr_ppocr(
    ocr: &OcrEngine,
    img: &DynamicImage,
    lang: &str,
) -> Result<Vec<RawBubble>> {
    let detected = ppocr_detect(ocr, img)?;
    ppocr_recognize(ocr, &detected, lang)
}

// ── Manga pipeline (not split — autoregressive decoder can't pipeline) ──

/// comic-text-detector: detects whole bubble polygons, OCR each region
fn detect_and_ocr_manga(
    detector: &TextDetector,
    ocr: &OcrEngine,
    img: &DynamicImage,
    lang: &str,
) -> Result<Vec<RawBubble>> {
    let regions = detector.detect(img)?;
    let mut bubbles = Vec::new();

    for region in &regions {
        let result = ocr.recognize(&region.crop, lang)?;
        if !result.text.trim().is_empty() {
            bubbles.push(RawBubble {
                source_text: result.text,
                polygon: region.polygon.clone(),
                det_confidence: region.confidence,
                ocr_confidence: result.confidence,
                mask: region.mask.clone(),
            });
        }
    }

    Ok(bubbles)
}

// ═══════════════════════════════════════════════════════════════════════
// Language helpers
// ═══════════════════════════════════════════════════════════════════════

const KNOWN_LANGS: &[&str] = &["ja", "ko", "zh", "en", "vi"];

pub fn detect_source_lang(explicit: Option<&str>, target_lang: &str) -> &'static str {
    if let Some(lang) = explicit {
        return KNOWN_LANGS
            .iter()
            .find(|&&k| k == lang)
            .copied()
            .unwrap_or("en");
    }
    match target_lang {
        "en" | "vi" => "ja",
        "ja" => "en",
        _ => "en",
    }
}
