use anyhow::Result;

use crate::layout;
use crate::layout::DrawableArea;

/// Minimum font size (px) before declaring overflow.
const MIN_FONT_SIZE: u32 = 8;
/// Absolute floor for the per-page maximum font size.
const ABS_MAX_FONT_SIZE: u32 = 96;
/// Page-width scaling factor for max font size: typical manga at 1755px
/// → max ≈ 87px, webtoon at 720px → max ≈ 36px (matches the typical
/// large narration / SFX font Lens reports for that resolution).
const MAX_FONT_PAGE_FRACTION: f64 = 0.05;

/// Tolerance around the source bubble's font size (in steps of 1px) to
/// consider before falling back to pure binary search.
const HINT_SEED_RADIUS: u32 = 3;

/// When a hint specifies an aspect ratio, layouts whose actual aspect
/// is within this fraction of the target get a small score bonus.
const HINT_ASPECT_TOLERANCE: f64 = 0.25;

/// Minimum acceptable ratio of fitted-to-source font size. Below this,
/// the layout is considered too cramped relative to the original
/// typography — falls back to a layout that respects the source size
/// even if it means fewer lines.
const MIN_FITTED_TO_SOURCE_RATIO: f64 = 0.65;

/// Allow the rendered text block to overshoot the bubble height by this
/// fraction. Manga bubbles have curved ascenders/descenders; a small
/// overshoot reads as natural typesetting rather than overflow, and
/// shrinking the font 30% just to gain 2px of height makes the bubble
/// look anemic.
const HEIGHT_OVERFLOW_TOLERANCE: f64 = 0.08;

/// Aspect ratio threshold above which a bubble is considered "tall narrow"
/// (oriented_h / oriented_w). In these bubbles each word becomes one line;
/// we fit against the long axis so individual words never char-break.
const TALL_NARROW_ASPECT_THRESHOLD: f64 = 2.5;

/// Fit prior derived from the source detector (Lens detailed output).
///
/// All fields are in page pixels / counts. `font_size_px` seeds the
/// binary search around the source font; `line_count` tilts the
/// candidate scoring so layouts with a similar number of lines win
/// ties; `avg_chars_per_line` informs the wrap target width.
#[derive(Clone, Copy, Debug)]
pub struct FitHint {
    pub font_size_px:       u32,
    pub line_count:         u32,
    pub avg_chars_per_line: f64,
}

#[derive(Clone)]
pub struct FitResult {
    pub text:         String,
    pub font_size_px: u32,
    pub line_height:  f64,
    pub overflow:     bool,
}

pub struct FitEngine;

impl FitEngine {
    /// Fit a page of bubbles. Each bubble gets the largest font that fits
    /// its own drawable area; when a hint is supplied, fit biases toward
    /// matching the source bubble's typesetting.
    pub fn fit_page_areas(
        items: &[(&str, &DrawableArea, Option<FitHint>)],
        page_width: u32,
    ) -> Result<Vec<FitResult>> {
        let max_font = max_font_for_page(page_width);
        items
            .iter()
            .map(|(text, area, hint)| Self::fit_area(text, area, *hint, max_font))
            .collect()
    }

    /// Convenience: fit one bubble with no hint.
    pub fn fit(translated_text: &str, polygon: &[[f64; 2]]) -> Result<FitResult> {
        let area = DrawableArea::from_polygon(polygon, layout::DEFAULT_INSET);
        let max_font = max_font_for_page(0);
        Self::fit_area(translated_text, &area, None, max_font)
    }

    /// Estimate how many characters fit in a drawable area at readable font size.
    pub fn char_budget(area: &DrawableArea) -> usize {
        let (w, h) = area.size();
        if w < 1.0 || h < 1.0 {
            return 0;
        }
        let font_size = (h / 5.0).clamp(MIN_FONT_SIZE as f64, ABS_MAX_FONT_SIZE as f64) as u32;
        let font = layout::get_font();
        let line_h = font_size as f64 * layout::LINE_HEIGHT_MULTIPLIER;
        let max_lines = ((h - font_size as f64) / line_h + 1.0).floor().max(1.0) as usize;

        let sample = "abcdefghijklmnopqrstuvwxyz àáảãạ ăắẳẵặ đ êếểễệ ôốổỗộ ưứửữự";
        let sample_w = layout::measure_text_width(sample, font_size, font);
        let char_count = sample.chars().filter(|c| !c.is_whitespace()).count();
        let avg_char_w = sample_w / char_count as f64;

        let chars_per_line = (w / avg_char_w).floor() as usize;
        chars_per_line * max_lines
    }

    fn fit_area(
        translated_text: &str,
        area: &DrawableArea,
        hint: Option<FitHint>,
        max_font: u32,
    ) -> Result<FitResult> {
        let text = normalize_text(translated_text);
        if text.is_empty() {
            return Ok(FitResult {
                text,
                font_size_px: MIN_FONT_SIZE,
                line_height: layout::LINE_HEIGHT_MULTIPLIER,
                overflow: false,
            });
        }

        let (safe_w, safe_h) = area.size();
        if safe_w < 1.0 || safe_h < 1.0 {
            return Ok(FitResult {
                text,
                font_size_px: MIN_FONT_SIZE,
                line_height: layout::LINE_HEIGHT_MULTIPLIER,
                overflow: true,
            });
        }

        // Tall-narrow bubbles (vertical label style — aspect > threshold):
        // fit against safe_h as the wrap width so words are never wider
        // than the long axis and char-break is impossible.
        let aspect = if safe_w > 0.0 { safe_h / safe_w } else { 0.0 };
        let (fit_w, fit_h) = if aspect > TALL_NARROW_ASPECT_THRESHOLD {
            (safe_h, safe_w)
        } else {
            (safe_w, safe_h)
        };

        let font = layout::get_font();
        let hi_bound = (fit_h as u32).min(max_font);
        let tolerant_h = fit_h * (1.0 + HEIGHT_OVERFLOW_TOLERANCE);

        // Soft-break path: the translator emitted explicit \n anchors.
        // Try to honour them at the largest font where every segment fits
        // within fit_w. If at the binary-search best_size the soft-break
        // layout fits in height, prefer it — it reflects the translator's
        // semantic intent. Otherwise fall through to the free-wrap path.
        let has_soft_breaks = text.contains('\n');
        let soft_result = if has_soft_breaks {
            try_soft_break_fit(&text, fit_w, fit_h, tolerant_h, hi_bound, font)
        } else {
            None
        };

        // Binary search for the largest size that fits (free wrap fallback).
        let mut lo = MIN_FONT_SIZE;
        let mut hi = hi_bound;
        let mut best_size = MIN_FONT_SIZE;
        let mut best_wrapped = layout::wrap_text(&text, fit_w, MIN_FONT_SIZE, font);

        while lo <= hi {
            let mid = (lo + hi) / 2;
            let wrapped = layout::wrap_text(&text, fit_w, mid, font);
            let total_h = text_block_height(wrapped.len(), mid);

            if total_h <= tolerant_h {
                best_size = mid;
                best_wrapped = wrapped;
                lo = mid + 1;
            } else if mid == 0 {
                break;
            } else {
                hi = mid - 1;
            }
        }

        // Hint-guided refinement (only for free-wrap path).
        let (free_size, free_wrapped) = match hint {
            Some(h) => refine_with_hint(
                &text, fit_w, fit_h, font, h, best_size, best_wrapped.clone(),
            ),
            None => (best_size, best_wrapped),
        };

        // Choose between soft-break and free-wrap results.
        // Prefer soft-break when it produces a font >= free-wrap result
        // (translator breaks are "free") or when it's within 15% smaller
        // (small size cost is worth keeping semantic breaks).
        let (final_size, final_wrapped) = match soft_result {
            Some((sb_size, sb_lines))
                if sb_size >= free_size
                    || sb_size as f64 >= free_size as f64 * 0.85 =>
            {
                (sb_size, sb_lines)
            }
            _ => (free_size, free_wrapped),
        };

        let total_h = text_block_height(final_wrapped.len(), final_size);
        let overflow = total_h > tolerant_h || final_size < MIN_FONT_SIZE;

        Ok(FitResult {
            text: final_wrapped.join("\n"),
            font_size_px: final_size,
            line_height: layout::LINE_HEIGHT_MULTIPLIER,
            overflow,
        })
    }
}

/// Pick the best font size in a small radius around `hint.font_size_px`,
/// scoring by how closely the resulting **aspect ratio** matches the
/// source (lines×font_size as proxy for filled bubble area).
///
/// Important: we DON'T blindly target the source line count. A Vietnamese
/// translation of an English bubble may need 2 lines where the original
/// had 6 vertical lines — forcing 6 lines would shrink the font absurdly.
/// Instead we score by how well the rendered block fills the bubble.
///
/// Returns the binary-search baseline unchanged when no candidate within
/// the hint radius scores higher.
fn refine_with_hint(
    text: &str,
    safe_w: f64,
    safe_h: f64,
    font: &ab_glyph::FontRef<'_>,
    hint: FitHint,
    baseline_size: u32,
    baseline_wrapped: Vec<String>,
) -> (u32, Vec<String>) {
    let source_size = hint.font_size_px.max(MIN_FONT_SIZE);

    // Source bubble's intrinsic aspect ratio: how wide each line is
    // relative to its height. Translations should try to match this
    // so the rendered block fills the same proportion of the bubble.
    let source_aspect = if hint.line_count > 0 {
        hint.avg_chars_per_line / hint.line_count as f64
    } else {
        1.0
    };

    // Scan a window centred on the source size, extending up to
    // baseline_size on the high side (allowing fits LARGER than the
    // source when translation happens to be shorter).
    let lo_size = source_size
        .saturating_sub(HINT_SEED_RADIUS)
        .max(MIN_FONT_SIZE);
    let hi_size = baseline_size.max(source_size + HINT_SEED_RADIUS);

    let mut best_score = score_layout(
        baseline_wrapped.len() as u32,
        baseline_size,
        source_size,
        source_aspect,
        text,
    );
    let mut best_size = baseline_size;
    let mut best_wrapped = baseline_wrapped;

    for candidate in lo_size..=hi_size {
        let wrapped = layout::wrap_text(text, safe_w, candidate, font);
        let total_h = text_block_height(wrapped.len(), candidate);
        let tolerant_h = safe_h * (1.0 + HEIGHT_OVERFLOW_TOLERANCE);
        if total_h > tolerant_h {
            continue;
        }
        let score = score_layout(
            wrapped.len() as u32, candidate, source_size, source_aspect, text,
        );
        if score > best_score {
            best_score = score;
            best_size = candidate;
            best_wrapped = wrapped;
        }
    }

    // Safety net: if the chosen font is too small relative to the
    // source typography, the bubble will look visually cramped. Trust
    // the binary-search baseline in that case — it picked the largest
    // font that fits, which is the better fallback.
    let ratio = best_size as f64 / source_size as f64;
    if ratio < MIN_FITTED_TO_SOURCE_RATIO && best_size < baseline_size {
        return (
            baseline_size,
            layout::wrap_text(text, safe_w, baseline_size, font),
        );
    }

    (best_size, best_wrapped)
}

/// Score a candidate layout. Higher = better.
///
/// Two axes:
///   - Font size: bigger is better (primary).
///   - Aspect-ratio match: layouts whose chars-per-line / lines ratio
///     matches the source get a bonus. This replaces the old hard
///     line-count match, which over-penalised short translations.
fn score_layout(
    lines: u32,
    size: u32,
    target_size: u32,
    target_aspect: f64,
    text: &str,
) -> i64 {
    let size_score = (size as i64) * 100;

    // Bonus when the candidate font is close to source size. Penalty
    // grows linearly with deviation.
    let size_diff = (size as i64 - target_size as i64).abs();
    let size_proximity_bonus = (50 - size_diff * 8).max(-200);

    // Aspect ratio of the rendered block, in same units as target.
    let chars = text.chars().filter(|c| !c.is_whitespace()).count() as f64;
    let actual_aspect = if lines > 0 {
        (chars / lines as f64) / lines as f64
    } else {
        1.0
    };
    let aspect_diff = (actual_aspect - target_aspect).abs() / target_aspect.max(0.01);
    let aspect_bonus = if aspect_diff < HINT_ASPECT_TOLERANCE {
        ((HINT_ASPECT_TOLERANCE - aspect_diff) * 100.0) as i64
    } else {
        -((aspect_diff * 30.0) as i64).min(150)
    };

    size_score + size_proximity_bonus + aspect_bonus
}

/// Per-page maximum font size — scales with image resolution so webtoon
/// strips don't get manga-sized 72px text and large native scans aren't
/// capped at a tiny font.
fn max_font_for_page(page_width: u32) -> u32 {
    let scaled = (page_width as f64 * MAX_FONT_PAGE_FRACTION) as u32;
    scaled.clamp(48, ABS_MAX_FONT_SIZE)
}

/// Try to fit text honouring the translator's explicit `\n` soft-break anchors.
///
/// Each `\n`-delimited segment is treated as a fixed line. We binary-search
/// for the largest font where every segment fits within `fit_w` (with the
/// normal width tolerance) AND the total block fits within `fit_h`.
///
/// Returns `Some((font_size, lines))` when a valid fit exists above
/// `MIN_FONT_SIZE`, or `None` when the soft breaks produce a layout that
/// is too wide or too tall at any readable size.
fn try_soft_break_fit(
    text: &str,
    fit_w: f64,
    fit_h: f64,
    tolerant_h: f64,
    hi_bound: u32,
    font: &ab_glyph::FontRef<'_>,
) -> Option<(u32, Vec<String>)> {
    let segments: Vec<&str> = text.lines().collect();
    if segments.is_empty() {
        return None;
    }

    let mut best: Option<(u32, Vec<String>)> = None;
    let mut lo = MIN_FONT_SIZE;
    let mut hi = hi_bound;

    while lo <= hi {
        let mid = (lo + hi) / 2;
        // Each segment becomes exactly one line — no further wrapping.
        // Check that no segment exceeds fit_w by more than the tolerance.
        let tolerance = layout::WIDTH_OVERFLOW_TOLERANCE;
        let tolerant_w = fit_w * (1.0 + tolerance);
        let all_fit = segments.iter().all(|seg| {
            layout::measure_text_width(seg, mid, font) <= tolerant_w
        });
        let total_h = text_block_height(segments.len(), mid);

        if all_fit && total_h <= tolerant_h {
            best = Some((mid, segments.iter().map(|s| s.to_string()).collect()));
            lo = mid + 1;
        } else if mid == 0 {
            break;
        } else {
            hi = mid - 1;
        }
    }

    best
}

/// Total height of a text block.
fn text_block_height(n_lines: usize, font_size_px: u32) -> f64 {
    if n_lines == 0 {
        return 0.0;
    }
    let spacing = font_size_px as f64 * layout::LINE_HEIGHT_MULTIPLIER;
    (n_lines - 1) as f64 * spacing + font_size_px as f64
}

fn normalize_text(text: &str) -> String {
    // Preserve explicit newlines from the translator as soft break hints.
    // Only collapse intra-line whitespace; do not flatten across lines.
    text.lines()
        .map(|line| line.split_whitespace().collect::<Vec<_>>().join(" "))
        .filter(|line| !line.is_empty())
        .collect::<Vec<_>>()
        .join("\n")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_fit_empty_text() {
        let polygon = vec![[0.0, 0.0], [200.0, 0.0], [200.0, 100.0], [0.0, 100.0]];
        let result = FitEngine::fit("", &polygon).unwrap();
        assert!(!result.overflow);
        assert!(result.text.is_empty());
    }

    #[test]
    fn test_fit_short_text() {
        let polygon = vec![[0.0, 0.0], [300.0, 0.0], [300.0, 200.0], [0.0, 200.0]];
        let result = FitEngine::fit("Hello world", &polygon).unwrap();
        assert!(!result.overflow);
        assert!(result.font_size_px >= MIN_FONT_SIZE);
        assert!(result.text.contains("Hello"));
    }

    #[test]
    fn test_fit_overflow() {
        let polygon = vec![[0.0, 0.0], [20.0, 0.0], [20.0, 10.0], [0.0, 10.0]];
        let result =
            FitEngine::fit("This is a very long text that should overflow", &polygon).unwrap();
        assert!(result.overflow);
    }

    #[test]
    fn test_wrap_text_tolerates_small_width_overshoot() {
        // Direct test of the wrap function: a single word ~5% wider than
        // the bubble (within tolerance) should stay on one line, not
        // char-break into pieces.
        use crate::layout::{measure_text_width, wrap_text};
        let font = crate::layout::get_font();

        let word = "CÙNG";
        let font_size = 30u32;
        let word_w = measure_text_width(word, font_size, font);
        // bubble width 5% smaller than word → inside 8% tolerance
        let bubble_w = word_w / 1.05;
        let lines = wrap_text(word, bubble_w, font_size, font);
        assert_eq!(
            lines.len(),
            1,
            "single short word should fit within tolerance: got {:?}",
            lines,
        );
        assert_eq!(lines[0], word);
    }

    #[test]
    fn test_wrap_text_char_breaks_when_well_beyond_tolerance() {
        // 50% overshoot is far beyond the 8% tolerance → must char-break.
        use crate::layout::{measure_text_width, wrap_text};
        let font = crate::layout::get_font();

        let word = "TRANSLATION";
        let font_size = 30u32;
        let word_w = measure_text_width(word, font_size, font);
        let bubble_w = word_w / 2.0;
        let lines = wrap_text(word, bubble_w, font_size, font);
        assert!(
            lines.len() > 1,
            "word 50% too wide should char-break: got {:?}",
            lines,
        );
    }

    #[test]
    fn test_max_font_scales_with_page_width() {
        // Webtoon strip ~720px wide
        assert_eq!(max_font_for_page(720), 48);
        // Manga page ~1755px wide → 87, capped at ABS_MAX
        assert_eq!(max_font_for_page(1755), 87);
        // Huge native scan
        assert_eq!(max_font_for_page(4000), ABS_MAX_FONT_SIZE);
    }

    #[test]
    fn test_hint_keeps_baseline_when_aspect_already_matches() {
        // Bubble with aspect ~3:1 (300×100), text fits cleanly in 1-2 lines.
        // Source hint: 1 line, char-density similar → score should not
        // shrink the baseline result.
        let area = DrawableArea::from_polygon(
            &[[0.0, 0.0], [300.0, 0.0], [300.0, 100.0], [0.0, 100.0]],
            2.0,
        );
        let text = "Short line";

        let no_hint = FitEngine::fit_page_areas(
            &[(text, &area, None)], 1755,
        ).unwrap();

        let with_hint = FitEngine::fit_page_areas(
            &[(
                text, &area,
                Some(FitHint {
                    font_size_px: no_hint[0].font_size_px,
                    line_count: 1,
                    avg_chars_per_line: 10.0,
                }),
            )],
            1755,
        ).unwrap();

        // Hint matches baseline → result should be same or larger,
        // never shrunk.
        assert!(with_hint[0].font_size_px >= no_hint[0].font_size_px);
        assert!(!with_hint[0].overflow);
    }

    #[test]
    fn test_hint_does_not_shrink_short_translation() {
        // Source bubble was 6 vertical lines of small text (Japanese
        // SFX feel). Translation is 2 words \u2014 should NOT shrink to
        // satisfy the source line count.
        let area = DrawableArea::from_polygon(
            &[[0.0, 0.0], [200.0, 0.0], [200.0, 180.0], [0.0, 180.0]],
            2.0,
        );
        let text = "BLUSH";

        let no_hint = FitEngine::fit_page_areas(
            &[(text, &area, None)], 1755,
        ).unwrap();

        let with_hint = FitEngine::fit_page_areas(
            &[(
                text, &area,
                Some(FitHint {
                    font_size_px: 30,         // source was small per-line
                    line_count: 6,             // source had 6 lines
                    avg_chars_per_line: 1.0,   // source was 1 char/line (vertical)
                }),
            )],
            1755,
        ).unwrap();

        // Critical: hint must NOT force the font below baseline just
        // because the source had more lines. The MIN_FITTED_TO_SOURCE_RATIO
        // safety net protects baseline.
        let ratio = with_hint[0].font_size_px as f64 / no_hint[0].font_size_px as f64;
        assert!(
            ratio >= 0.85,
            "translation should not shrink: no_hint={}px with_hint={}px",
            no_hint[0].font_size_px, with_hint[0].font_size_px,
        );
    }

    #[test]
    fn test_hint_falls_back_when_unfittable() {
        // Hint requests font size larger than the area can hold —
        // refinement must keep the baseline binary-search result.
        let area = DrawableArea::from_polygon(
            &[[0.0, 0.0], [200.0, 0.0], [200.0, 40.0], [0.0, 40.0]],
            2.0,
        );
        let result = FitEngine::fit_page_areas(
            &[(
                "Hello", &area,
                Some(FitHint {
                    font_size_px: 200, // absurdly large
                    line_count: 1,
                    avg_chars_per_line: 5.0,
                }),
            )],
            1755,
        ).unwrap();

        assert!(!result[0].overflow);
        assert!(result[0].font_size_px >= MIN_FONT_SIZE);
    }

    #[test]
    fn test_tall_narrow_bubble_does_not_char_break() {
        // Reproduces the "MỘ LÂM ĐỨC HỮU" bug: a vertical name banner
        // (~35px wide, ~220px tall). Before the fix ĐỨC was char-broken
        // into ĐỨ / C because safe_w was narrower than the word.
        // After the fix fit uses safe_h as wrap width → each word = 1 line.
        let area = DrawableArea::from_polygon(
            &[[0.0, 0.0], [35.0, 0.0], [35.0, 220.0], [0.0, 220.0]],
            2.0,
        );
        let text = "MỘ LÂM ĐỨC HỮU";
        let result = FitEngine::fit_page_areas(&[(text, &area, None)], 800).unwrap();

        // Must not overflow.
        assert!(
            !result[0].overflow,
            "tall-narrow bubble should not overflow"
        );
        // Font must be readable (not collapsed to minimum).
        assert!(
            result[0].font_size_px >= 12,
            "font too small: {}px", result[0].font_size_px,
        );
        // No line should be a single character (char-break sign).
        for line in result[0].text.lines() {
            assert!(
                line.chars().count() >= 2,
                "char-broken line detected: {:?} in {:?}", line, result[0].text,
            );
        }
    }
}
    #[test]
    fn test_normalize_text_preserves_soft_breaks() {
        let text = "Không…\nkhông có gì.";
        let normalized = normalize_text(text);
        assert!(normalized.contains('\n'), "soft break lost: {:?}", normalized);
        assert_eq!(normalized, "Không…\nkhông có gì.");
    }

    #[test]
    fn test_soft_break_respected_when_fits() {
        let area = DrawableArea::from_polygon(
            &[[0.0, 0.0], [300.0, 0.0], [300.0, 100.0], [0.0, 100.0]],
            2.0,
        );
        let text = "Anh không thể\nlàm được điều đó.";
        let result = FitEngine::fit_page_areas(&[(text, &area, None)], 1200).unwrap();
        assert!(result[0].text.contains('\n'), "soft break not preserved: {:?}", result[0].text);
        assert_eq!(result[0].text.lines().count(), 2);
        assert!(!result[0].overflow);
    }

    #[test]
    fn test_soft_break_fallback_when_too_wide() {
        let area = DrawableArea::from_polygon(
            &[[0.0, 0.0], [60.0, 0.0], [60.0, 120.0], [0.0, 120.0]],
            2.0,
        );
        let text = "Anh không thể làm được\nđiều đó cả đời này đâu.";
        let result = FitEngine::fit_page_areas(&[(text, &area, None)], 800).unwrap();
        assert!(result[0].font_size_px >= MIN_FONT_SIZE);
    }
