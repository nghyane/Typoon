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

/// PP-OCR: detect text lines → merge into bubbles → OCR each line → concat
fn detect_and_ocr_ppocr(
    ocr: &OcrEngine,
    img: &DynamicImage,
    lang: &str,
) -> Result<Vec<RawBubble>> {
    let DetectionOutput {
        regions: lines,
        prob_image,
    } = ocr.detect(img)?;
    let merged = merge::group_lines(lines, img, prob_image.as_ref());

    let mut bubbles = Vec::new();

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
            if result.min_char_confidence < 0.15 && text.chars().count() <= 4 {
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

        let avg_conf = if conf_count > 0 {
            total_conf / conf_count as f64
        } else {
            0.0
        };
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
