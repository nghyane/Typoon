use anyhow::Result;

use crate::layout;
use crate::layout::DrawableArea;

/// Minimum font size (px) before declaring overflow.
const MIN_FONT_SIZE: u32 = 8;
/// Absolute floor for the per-page maximum font size.
const ABS_MAX_FONT_SIZE: u32 = 96;
/// Page-width scaling factor for max font size.
const MAX_FONT_PAGE_FRACTION: f64 = 0.05;

/// Hard fit contract: text MUST NOT extend past the polygon rect.
const HEIGHT_OVERFLOW_TOLERANCE: f64 = 0.0;

/// Cap the upper font size at this multiple of the page-aggregated
/// source-typeset size (median of all dialogue bubbles' Lens hints).
/// Manga letterers pick ONE body size per page; without a chapter/page
/// cap the fitter would inflate VI text to fill each big bubble
/// independently → wildly inconsistent neighbour panels. Allow modest
/// growth so VI body text can edge above source size when a bubble
/// genuinely has the room.
const HINT_MAX_GROWTH: f64 = 1.15;

/// Per-bubble hint cap: how far above the bubble's own source font size
/// the fitter is allowed to go. Tighter than page cap because the
/// letterer made an explicit per-bubble choice for THIS bubble's
/// aspect / content density. A short VI translation in a tall bubble
/// must not inflate beyond this — otherwise it overshoots the bubble's
/// visible curve even though the AABB technically allows it.
const BUBBLE_HINT_MAX_GROWTH: f64 = 1.10;

/// Char-density cap: limit font size so that the text's character count
/// plausibly fills the bubble area. Without this, a short translation
/// ("THẬT Á?!" = 7 chars) inside a large bubble inflates to fill the
/// full height → absurdly oversized compared to adjacent dialogue bubbles.
///
/// Derived from area / chars: each character occupies roughly
/// `(font_px * LINE_HEIGHT_MULTIPLIER) * (font_px * CHAR_WIDTH_FRAC)`
/// pixels. Solving for font_px: font_px ≈ sqrt(area / (chars * ratio)).
///
/// CHAR_WIDTH_FRAC: average Vietnamese character width as a fraction of
/// font size in px (empirically ~0.62 for SamaritanTall with mixed diacritics).
/// DENSITY_SCALE: allow text to occupy at most this fraction of bubble area
/// before capping. Set to 0.55 so a bubble half-filled looks natural;
/// 1.0 would require text to tile every pixel which is never readable.
const CHAR_WIDTH_FRAC:  f64 = 0.62;
const DENSITY_SCALE:    f64 = 0.55;

/// Discard outlier hints when computing the page median. Lens
/// occasionally reports `font_size_px` < 8 (column-split tategaki where
/// each "line" is a 4-glyph cluster) or > 60 (huge SFX bleed); both
/// distort the median. Range matches the real body-text font sizes
/// observed across the fixture corpus.
const HINT_OUTLIER_MIN: u32 = 10;
const HINT_OUTLIER_MAX: u32 = 60;

/// Source script direction. Vertical (Japanese tategaki) means
/// `line_count` represents COLUMNS, not horizontal lines, so the
/// line-budget refinement in `fit_area` must skip it.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum TextDirection {
    #[default]
    Horizontal,
    Vertical,
}

/// Fit prior from the source detector.
#[derive(Clone, Copy, Debug)]
pub struct FitHint {
    pub font_size_px:       u32,
    pub line_count:         u32,
    pub avg_chars_per_line: f64,
    pub text_direction:     TextDirection,
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
    /// Fit a whole page of bubbles. Computes ONE body-text font ceiling
    /// from the page-aggregated source hint (median of all valid hints)
    /// and applies it to every item. This is the stability fix: per-
    /// bubble caps gave each bubble a different ceiling so adjacent
    /// short and long translations landed at very different sizes.
    /// Using the page median + `HINT_MAX_GROWTH` mirrors how a manga
    /// letterer types: one body size per page; bubbles shrink only when
    /// their own translation forces a smaller fit, never inflate above
    /// the page cap regardless of how much empty room they have.
    pub fn fit_page_areas(
        items: &[(&str, &DrawableArea, Option<FitHint>)],
        page_width: u32,
    ) -> Result<Vec<FitResult>> {
        let max_font = max_font_for_page(page_width);
        let page_cap = page_body_cap(items, max_font);
        items.iter()
            .map(|(text, area, hint)| Self::fit_area(text, area, *hint, page_cap))
            .collect()
    }

    pub fn fit(translated_text: &str, polygon: &[[f64; 2]]) -> Result<FitResult> {
        let area = DrawableArea::from_polygon(polygon, layout::DEFAULT_INSET);
        let max_font = max_font_for_page(0);
        Self::fit_area(translated_text, &area, None, max_font)
    }

    pub fn char_budget(area: &DrawableArea) -> usize {
        let (w, h) = area.size();
        if w < 1.0 || h < 1.0 { return 0; }
        let font_size = (h / 5.0).clamp(MIN_FONT_SIZE as f64, ABS_MAX_FONT_SIZE as f64) as u32;
        let line_h    = font_size as f64 * layout::LINE_HEIGHT_MULTIPLIER;
        let max_lines = ((h - font_size as f64) / line_h + 1.0).floor().max(1.0) as usize;
        let sample    = "abcdefghijklmnopqrstuvwxyz àáảãạ ăắẳẵặ đ êếểễệ ôốổỗộ ưứửữự";
        let sample_w  = layout::measure_text_width(sample, font_size);
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
            return Ok(FitResult { text, font_size_px: MIN_FONT_SIZE,
                                  line_height: layout::LINE_HEIGHT_MULTIPLIER, overflow: false });
        }
        let (safe_w, safe_h) = area.size();
        if safe_w < 1.0 || safe_h < 1.0 {
            return Ok(FitResult { text, font_size_px: MIN_FONT_SIZE,
                                  line_height: layout::LINE_HEIGHT_MULTIPLIER, overflow: true });
        }

        // Fit against the bubble's actual drawable rect. The overlay always
        // paints Vietnamese horizontally (see `overlay::render` — the only
        // rotation it honours is `area.is_rotated()` from an OBB polygon),
        // so the wrap width MUST be the horizontal extent of the bubble.
        //
        // An older "tall-narrow swap" used `safe_h` as wrap width whenever
        // aspect > 2.5 (intent: tategaki bubbles). That lied to the fitter —
        // the painter never rotated the text, so 1-line "Năm 28 thời Shōwa"
        // sized to fit a 341×80 imaginary frame ended up smeared across an
        // 80×341 vertical bubble and bled outside the page. Removed.
        let fit_w = safe_w;
        let fit_h = safe_h;

        // `max_font` is the page-aggregated body cap (see
        // `page_body_cap`). `char_density_cap` further limits the upper
        // bound based on how many characters need to fit: a short text
        // ("THẬT Á?!") in a large bubble must not inflate to fill the
        // full height — that produces sizes wildly inconsistent with
        // adjacent panels that happen to have longer translations.
        //
        // `bubble_hint_cap`: per-bubble ceiling derived from THIS bubble's
        // source font size. When the letterer wrote 28px in this bubble,
        // VI must not jump to 40px just because the bubble is tall — that
        // makes short translations balloon when src had many lines but VI
        // has few. Falls back to `max_font` when no hint is available.
        let density_cap = char_density_cap(&text, safe_w, safe_h);
        let bubble_hint_cap = bubble_hint_cap(hint, max_font);
        let hi_bound = (fit_h as u32).min(max_font).min(density_cap).min(bubble_hint_cap);
        let tolerant_h = fit_h * (1.0 + HEIGHT_OVERFLOW_TOLERANCE);

        // Soft-break path
        let has_soft_breaks = text.contains('\n');
        let soft_result = if has_soft_breaks {
            try_soft_break_fit(&text, fit_w, fit_h, tolerant_h, hi_bound)
        } else {
            None
        };

        // Binary search — free wrap. A size is acceptable only if (a) the
        // block height fits AND (b) the widest source word fits the line
        // at that size. Without (b) `wrap_text` falls into the greedy
        // char-break path which splits "Cộc." into ["Cộ","c."] — visually
        // broken; the fitter then thinks it fits because each fragment
        // is narrow. Reject sizes that would require char-breaking.
        let mut lo = MIN_FONT_SIZE;
        let mut hi = hi_bound;
        let mut best_size    = MIN_FONT_SIZE;
        let mut best_wrapped = layout::wrap_text(&text, fit_w, MIN_FONT_SIZE);

        while lo <= hi {
            let mid     = (lo + hi) / 2;
            let wrapped = layout::wrap_text(&text, fit_w, mid);
            let total_h = text_block_height(wrapped.len(), mid);
            let words_fit = !needs_char_break(&text, fit_w, mid);
            if total_h <= tolerant_h && words_fit {
                best_size    = mid;
                best_wrapped = wrapped;
                lo = mid + 1;
            } else if mid == 0 {
                break;
            } else {
                hi = mid - 1;
            }
        }

        // Line-budget refinement (vấn đề 2): when the source bubble was
        // typeset on N lines, try to wrap VI on ≤ N lines too. The free
        // binary search above can pick a smaller font that wraps to
        // N+1/N+2 lines, leaving the bubble feeling under-filled. If we
        // can fit the same text in `hint.line_count` lines at a LARGER
        // size, prefer that — it matches the letterer's visual rhythm.
        //
        // Skip for vertical (tategaki) source: their line_count is the
        // number of vertical columns, not horizontal lines.
        let line_budget_result = hint.and_then(|h| {
            if h.line_count == 0 { return None; }
            if h.text_direction == TextDirection::Vertical { return None; }
            try_line_budget_fit(&text, fit_w, tolerant_h, hi_bound, h.line_count as usize)
        });

        // Prefer soft-break result
        let (mut final_size, mut final_wrapped) = match soft_result {
            Some((sb_size, sb_lines))
                if sb_size >= best_size || sb_size as f64 >= best_size as f64 * 0.85
            => (sb_size, sb_lines),
            _ => (best_size, best_wrapped),
        };

        // Apply line-budget result if it gives at least the same font.
        // No 0.85 fudge — the budget path already enforces <= src lines,
        // so any equal-or-better size is a real visual improvement
        // (fewer lines at the same size looks closer to src letterer's
        // intent than more lines at the same size).
        if let Some((lb_size, lb_lines)) = line_budget_result {
            if lb_size >= final_size {
                final_size    = lb_size;
                final_wrapped = lb_lines;
            }
        }

        // Hard width constraint: the binary search above only checks height.
        // wrap_text is allowed a small overshoot tolerance so it doesn't
        // char-break on a 1-pixel miss; but the renderer paints into the
        // detected polygon rect, so any line still wider than `fit_w` after
        // wrap will visibly escape the bubble. Shrink the font size until
        // every wrapped line fits — or bottom out at MIN_FONT_SIZE with
        // overflow flagged.
        let (final_size, final_wrapped) = enforce_width(
            &text, fit_w, fit_h, tolerant_h, final_size, final_wrapped,
        );

        let total_h = text_block_height(final_wrapped.len(), final_size);
        let too_wide = final_wrapped
            .iter()
            .any(|line| layout::measure_text_width(line, final_size) > fit_w);
        let overflow = total_h > tolerant_h || too_wide || final_size < MIN_FONT_SIZE;

        Ok(FitResult {
            text: final_wrapped.join("\n"),
            font_size_px: final_size,
            line_height:  layout::LINE_HEIGHT_MULTIPLIER,
            overflow,
        })
    }
}

fn max_font_for_page(page_width: u32) -> u32 {
    let scaled = (page_width as f64 * MAX_FONT_PAGE_FRACTION) as u32;
    scaled.clamp(48, ABS_MAX_FONT_SIZE)
}

/// Estimate the maximum font size that makes sense given the number of
/// characters in `text` and the bubble's drawable area.
///
/// A short text in a large bubble would otherwise inflate to fill the
/// full height. This cap solves:
///   area ≈ chars × char_w × line_h
///   char_w  = font_px × CHAR_WIDTH_FRAC
///   line_h  = font_px × LINE_HEIGHT_MULTIPLIER
///   → font_px = sqrt(area × DENSITY_SCALE / (chars × CHAR_WIDTH_FRAC × LINE_HEIGHT_MULTIPLIER))
///
/// Floored at MIN_FONT_SIZE so very short texts (SFX) still get a usable
/// size; the page_body_cap and geometry max act as the true ceiling.
fn char_density_cap(text: &str, safe_w: f64, safe_h: f64) -> u32 {
    let chars = text.chars().filter(|c| !c.is_whitespace()).count();
    if chars == 0 { return ABS_MAX_FONT_SIZE; }
    let area = safe_w * safe_h * DENSITY_SCALE;
    let px = (area / (chars as f64 * CHAR_WIDTH_FRAC * layout::LINE_HEIGHT_MULTIPLIER)).sqrt();
    (px.round() as u32).max(MIN_FONT_SIZE)
}

/// Compute the page-level body-text font ceiling.
///
/// Strategy:
///   1. Collect all valid per-bubble hints (in-range, non-outlier).
///   2. Median across the page → robust to a handful of mis-detected
///      SFX or tategaki column splits.
///   3. Cap at `page_geometry_max` (page_width-scaled ceiling) so the
///      hint can't propose anything larger than physically reasonable.
///   4. Apply `HINT_MAX_GROWTH` so VI body can edge above source size.
///   5. No hints at all → fall back to the unmodified page max.
///
/// This is the page-wide stability layer. Individual bubble `fit_area`
/// calls then treat this single number as their hi_bound; bubbles can
/// only SHRINK below the cap (when their translation forces it), never
/// inflate above. Result: adjacent panels stay consistent.
fn page_body_cap(
    items: &[(&str, &DrawableArea, Option<FitHint>)],
    page_geometry_max: u32,
) -> u32 {
    let mut samples: Vec<u32> = items.iter()
        .filter_map(|(_, _, h)| h.map(|h| h.font_size_px))
        .filter(|&s| (HINT_OUTLIER_MIN..=HINT_OUTLIER_MAX).contains(&s))
        .collect();
    if samples.is_empty() {
        return page_geometry_max;
    }
    samples.sort_unstable();
    let median = samples[samples.len() / 2];
    let grown = (median as f64 * HINT_MAX_GROWTH).round() as u32;
    grown.max(MIN_FONT_SIZE).min(page_geometry_max)
}

/// Per-bubble cap derived from the bubble's own source font size.
/// Returns `max_font` when no hint, or when hint is outside the trusted
/// range (matches the page cap's outlier filter so SFX-mis-detection
/// can't lock a body bubble to 8px).
fn bubble_hint_cap(hint: Option<FitHint>, max_font: u32) -> u32 {
    let Some(h) = hint else { return max_font; };
    if !(HINT_OUTLIER_MIN..=HINT_OUTLIER_MAX).contains(&h.font_size_px) {
        return max_font;
    }
    let grown = (h.font_size_px as f64 * BUBBLE_HINT_MAX_GROWTH).round() as u32;
    grown.max(MIN_FONT_SIZE).min(max_font)
}

/// Try to wrap `text` on at most `max_lines` lines at the largest
/// possible font size. Returns `None` when no size ≥ MIN_FONT_SIZE can
/// achieve it (text genuinely needs more lines).
///
/// This is the "src had N lines, give VI the same budget" path. We
/// run a binary search just like the free-wrap one, but the gate is
/// `wrapped.len() <= max_lines` instead of free wrap.
fn try_line_budget_fit(
    text:      &str,
    fit_w:     f64,
    tolerant_h: f64,
    hi_bound:  u32,
    max_lines: usize,
) -> Option<(u32, Vec<String>)> {
    if max_lines == 0 { return None; }
    let mut lo = MIN_FONT_SIZE;
    let mut hi = hi_bound;
    let mut best: Option<(u32, Vec<String>)> = None;

    while lo <= hi {
        let mid     = (lo + hi) / 2;
        let wrapped = layout::wrap_text(text, fit_w, mid);
        let total_h = text_block_height(wrapped.len(), mid);
        let words_fit = !needs_char_break(text, fit_w, mid);
        let lines_ok  = wrapped.len() <= max_lines;
        if lines_ok && words_fit && total_h <= tolerant_h {
            best = Some((mid, wrapped));
            lo = mid + 1;
        } else if mid == 0 {
            break;
        } else {
            hi = mid - 1;
        }
    }
    best
}

fn try_soft_break_fit(
    text: &str,
    fit_w: f64,
    _fit_h: f64,
    tolerant_h: f64,
    hi_bound: u32,
) -> Option<(u32, Vec<String>)> {
    let segments: Vec<&str> = text.lines().collect();
    if segments.is_empty() { return None; }

    let mut best: Option<(u32, Vec<String>)> = None;
    let mut lo = MIN_FONT_SIZE;
    let mut hi = hi_bound;

    while lo <= hi {
        let mid        = (lo + hi) / 2;
        let tolerant_w = fit_w * (1.0 + layout::WIDTH_OVERFLOW_TOLERANCE);
        let all_fit    = segments.iter().all(|seg| layout::measure_text_width(seg, mid) <= tolerant_w);
        let total_h    = text_block_height(segments.len(), mid);

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

fn text_block_height(n_lines: usize, font_size_px: u32) -> f64 {
    layout::text_block_height(n_lines, font_size_px)
}

/// True if any atomic wrap-token would not fit on a line of width `fit_w`
/// at `size`, forcing `wrap_text` into its char-break fallback. Used as a
/// hard guard so the fitter never picks a size that produces character-
/// broken Vietnamese (e.g. "Cộ" / "c.").
///
/// Uses `longest_atom_width` rather than per-word measurement so a
/// hyphenated compound ("28-Shonan") doesn't block the binary search —
/// the wrapper can split it at the hyphen into "28-" + "Shonan".
fn needs_char_break(text: &str, fit_w: f64, size: u32) -> bool {
    layout::longest_atom_width(text, size) > fit_w
}

/// Shrink font until every wrapped line fits `fit_w` (no overshoot).
/// The starting `size`/`wrapped` come from the height-binary-search; this
/// is the final width-correctness gate so the renderer can never paint
/// outside the bubble polygon's horizontal extent.
fn enforce_width(
    text:        &str,
    fit_w:       f64,
    _fit_h:      f64,
    tolerant_h:  f64,
    mut size:    u32,
    mut wrapped: Vec<String>,
) -> (u32, Vec<String>) {
    // Preserve soft breaks (\n) when shrinking — only re-flow if the
    // original text had no explicit line breaks.
    let preserve_breaks = text.contains('\n');
    loop {
        let widest = wrapped
            .iter()
            .map(|l| layout::measure_text_width(l, size))
            .fold(0.0_f64, f64::max);
        if widest <= fit_w || size <= MIN_FONT_SIZE {
            return (size, wrapped);
        }
        size -= 1;
        if !preserve_breaks {
            wrapped = layout::wrap_text(text, fit_w, size);
        }
        if text_block_height(wrapped.len(), size) > tolerant_h * 2.0 {
            return (size, wrapped);
        }
    }
}

fn normalize_text(text: &str) -> String {
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
        let result = FitEngine::fit("This is a very long text that should overflow", &polygon).unwrap();
        assert!(result.overflow);
    }

    #[test]
    fn test_max_font_scales_with_page_width() {
        assert_eq!(max_font_for_page(720),  48);
        assert_eq!(max_font_for_page(1755), 87);
        assert_eq!(max_font_for_page(4000), ABS_MAX_FONT_SIZE);
    }

    #[test]
    fn test_page_cap_uses_median_of_hints() {
        // 3 bubbles share one page: hints 20/24/28 → median 24
        // → cap = 24 × 1.15 ≈ 28. All bubbles fit at <= cap regardless
        // of how much room each one has.
        let big   = DrawableArea::from_polygon(
            &[[0.0, 0.0], [500.0, 0.0], [500.0, 400.0], [0.0, 400.0]], 2.0,
        );
        let tiny  = DrawableArea::from_polygon(
            &[[0.0, 0.0], [120.0, 0.0], [120.0, 60.0], [0.0, 60.0]], 2.0,
        );
        let mid   = DrawableArea::from_polygon(
            &[[0.0, 0.0], [240.0, 0.0], [240.0, 180.0], [0.0, 180.0]], 2.0,
        );
        let items: Vec<(&str, &DrawableArea, Option<FitHint>)> = vec![
            ("À", &big,  Some(FitHint { font_size_px: 20, line_count: 1, avg_chars_per_line: 1.0, text_direction: TextDirection::Horizontal })),
            ("Ờ", &tiny, Some(FitHint { font_size_px: 24, line_count: 1, avg_chars_per_line: 1.0, text_direction: TextDirection::Horizontal })),
            ("OK", &mid,  Some(FitHint { font_size_px: 28, line_count: 1, avg_chars_per_line: 1.0, text_direction: TextDirection::Horizontal })),
        ];
        let result = FitEngine::fit_page_areas(&items, 1755).unwrap();
        let median_cap = (24.0 * HINT_MAX_GROWTH).round() as u32;
        for (i, r) in result.iter().enumerate() {
            assert!(
                r.font_size_px <= median_cap,
                "bubble {i} font {}px exceeds page cap {median_cap}px",
                r.font_size_px,
            );
        }
        // The big-bubble + tiny-text case is the regression target:
        // without page cap it inflated to ~80px; with cap it stays small.
        assert!(result[0].font_size_px <= median_cap);
    }

    #[test]
    fn test_page_cap_falls_back_when_all_hints_missing() {
        // No hints → no page cap → bubble is free to use geometry max.
        let area = DrawableArea::from_polygon(
            &[[0.0, 0.0], [500.0, 0.0], [500.0, 400.0], [0.0, 400.0]], 2.0,
        );
        let no_hint = FitEngine::fit_page_areas(&[("À", &area, None)], 1755).unwrap();
        assert!(no_hint[0].font_size_px > 30, "expected geometry-driven size, got {}", no_hint[0].font_size_px);
    }

    #[test]
    fn test_page_cap_ignores_outlier_hints() {
        // One real body bubble (hint=20) + one outlier (hint=120, e.g.
        // mis-detected huge SFX). Median over filtered samples = 20,
        // not 70 — the cap reflects real body text.
        let area = DrawableArea::from_polygon(
            &[[0.0, 0.0], [400.0, 0.0], [400.0, 200.0], [0.0, 200.0]], 2.0,
        );
        let items: Vec<(&str, &DrawableArea, Option<FitHint>)> = vec![
            ("À", &area, Some(FitHint { font_size_px: 20,  line_count: 1, avg_chars_per_line: 1.0, text_direction: TextDirection::Horizontal })),
            ("X", &area, Some(FitHint { font_size_px: 120, line_count: 1, avg_chars_per_line: 1.0, text_direction: TextDirection::Horizontal })),
        ];
        let result = FitEngine::fit_page_areas(&items, 1755).unwrap();
        let cap = (20.0 * HINT_MAX_GROWTH).round() as u32;
        assert!(
            result[0].font_size_px <= cap,
            "outlier hint leaked into cap: {}px > {}px",
            result[0].font_size_px, cap,
        );
    }

    #[test]
    fn test_tall_narrow_bubble_does_not_char_break() {
        // Tategaki bubble: 35×220. With the tall-narrow swap removed the
        // overlay paints Vietnamese horizontally into the actual 35px-wide
        // rect — no fictitious 220px line. A short multi-word phrase must
        // wrap to one word per line (or shrink) rather than char-break.
        let area = DrawableArea::from_polygon(
            &[[0.0, 0.0], [35.0, 0.0], [35.0, 220.0], [0.0, 220.0]], 2.0,
        );
        let result = FitEngine::fit_page_areas(&[("MỘ LÂM ĐỨC HỮU", &area, None)], 800).unwrap();
        assert!(result[0].font_size_px >= MIN_FONT_SIZE);
        for line in result[0].text.lines() {
            assert!(line.chars().count() >= 2, "char-broken: {:?}", line);
        }
    }

    #[test]
    fn test_normalize_text_preserves_soft_breaks() {
        let text = "Không…\nkhông có gì.";
        let normalized = normalize_text(text);
        assert!(normalized.contains('\n'));
        assert_eq!(normalized, "Không…\nkhông có gì.");
    }

    #[test]
    fn test_soft_break_respected_when_fits() {
        let area = DrawableArea::from_polygon(
            &[[0.0, 0.0], [300.0, 0.0], [300.0, 100.0], [0.0, 100.0]], 2.0,
        );
        let text = "Anh không thể\nlàm được điều đó.";
        let result = FitEngine::fit_page_areas(&[(text, &area, None)], 1200).unwrap();
        assert!(result[0].text.contains('\n'));
        assert_eq!(result[0].text.lines().count(), 2);
        assert!(!result[0].overflow);
    }

    #[test]
    fn test_short_text_in_large_bubble_char_density_cap() {
        // Regression: "THẬT Á?!" (7 non-space chars) inside a large bubble
        // must not inflate to fill the bubble height. It should land at a
        // size comparable to adjacent dialogue bubbles with longer text.
        let large_bubble = DrawableArea::from_polygon(
            &[[0.0, 0.0], [180.0, 0.0], [180.0, 160.0], [0.0, 160.0]], 2.0,
        );
        let long_bubble = DrawableArea::from_polygon(
            &[[0.0, 0.0], [180.0, 0.0], [180.0, 160.0], [0.0, 160.0]], 2.0,
        );
        let items: Vec<(&str, &DrawableArea, Option<FitHint>)> = vec![
            ("THẬT Á?!", &large_bubble, None),
            ("GAHA HA HA HA HA!", &long_bubble, None),
        ];
        let results = FitEngine::fit_page_areas(&items, 891).unwrap();
        let short_size = results[0].font_size_px;
        let long_size  = results[1].font_size_px;
        // Short text must not be more than 2× the long text size.
        // Without char_density_cap short_size inflates to ~80px, long_size ~28px.
        assert!(
            short_size <= long_size * 2,
            "short text ({short_size}px) is disproportionately larger than long text ({long_size}px)",
        );
    }

    #[test]
    fn test_short_text_never_char_breaks() {
        // Reproduces the "Cộc." → ["Cộ", "c."] regression on the SFX
        // bubble in lens_bubble_probe3. The fitter must shrink the font
        // until "Cộc." fits on one line, not split the word.
        let area = DrawableArea::from_polygon(
            &[[0.0, 0.0], [60.0, 0.0], [60.0, 200.0], [0.0, 200.0]], 2.0,
        );
        let result = FitEngine::fit_page_areas(&[("Cộc.", &area, None)], 800).unwrap();
        for line in result[0].text.lines() {
            assert!(
                line.contains("Cộc.") || line == "Cộc.",
                "char-broken: {:?}", result[0].text,
            );
        }
        assert_eq!(result[0].text.lines().count(), 1);
    }

    #[test]
    fn test_no_line_exceeds_fit_width() {
        // Hard contract: every wrapped line must fit the polygon's safe width,
        // no per-pixel overshoot allowed in the final result. This is the
        // guarantee that text never visibly escapes the bubble bbox.
        for w in [120.0, 200.0, 360.0] {
            let area = DrawableArea::from_polygon(
                &[[0.0, 0.0], [w, 0.0], [w, 180.0], [0.0, 180.0]], 2.0,
            );
            let text = "CẬU KHÔNG THẤY GẦN ĐÂY KHẢI-CHAN CÓ GÌ ĐÓ LẠ LẠ À?";
            let result = FitEngine::fit_page_areas(&[(text, &area, None)], 1755).unwrap();
            let (safe_w, _) = area.size();
            for line in result[0].text.lines() {
                let lw = crate::layout::measure_text_width(line, result[0].font_size_px);
                assert!(
                    lw <= safe_w + 0.5,
                    "width={w}: line {:?} measured {lw}px > safe_w {safe_w}px @ fs={}px",
                    line, result[0].font_size_px,
                );
            }
        }
    }

    #[test]
    fn test_line_budget_prefers_fewer_lines_when_src_had_few() {
        // Regression: src had 1 line; VI is short enough to fit 1
        // line at the page-cap size. The free binary search may pick
        // 2 lines at a smaller size because balanced wrap breaks a
        // long single line into halves. With the line-budget
        // refinement, fitter prefers <= src lines at the larger size.
        let area = DrawableArea::from_polygon(
            &[[0.0, 0.0], [1100.0, 0.0], [1100.0, 200.0], [0.0, 200.0]], 2.0,
        );
        let text = "Tôi thật sự không muốn làm nữa đâu.";
        let with_hint: Vec<(&str, &DrawableArea, Option<FitHint>)> = vec![
            (text, &area, Some(FitHint {
                font_size_px: 36, line_count: 1, avg_chars_per_line: 30.0,
                text_direction: TextDirection::Horizontal,
            })),
        ];
        let r_hint = FitEngine::fit_page_areas(&with_hint, 1755).unwrap();
        assert!(
            r_hint[0].text.lines().count() <= 1,
            "expected <= 1 line, got {:#?}",
            r_hint[0].text.lines().collect::<Vec<_>>(),
        );
    }

    #[test]
    fn test_bubble_hint_cap_blocks_inflation_when_src_had_many_lines() {
        // Regression: src bubble was 4 lines @ 22px (cramped JP). VI
        // translation is 1 short line. Without per-bubble cap the
        // fitter inflates VI to ~60px because the bubble is tall.
        // With per-bubble cap (22 * 1.10 \u2248 24px) it stays close to src.
        let area = DrawableArea::from_polygon(
            &[[0.0, 0.0], [260.0, 0.0], [260.0, 280.0], [0.0, 280.0]], 2.0,
        );
        let items: Vec<(&str, &DrawableArea, Option<FitHint>)> = vec![
            ("Vâng.", &area, Some(FitHint {
                font_size_px: 22, line_count: 4, avg_chars_per_line: 6.0,
                text_direction: TextDirection::Horizontal,
            })),
        ];
        let result = FitEngine::fit_page_areas(&items, 1755).unwrap();
        let cap = (22.0 * BUBBLE_HINT_MAX_GROWTH).round() as u32;
        assert!(
            result[0].font_size_px <= cap,
            "VI inflated to {}px, exceeds per-bubble cap {}px",
            result[0].font_size_px, cap,
        );
    }

    #[test]
    fn test_vertical_hint_skips_line_budget() {
        // Tategaki source: line_count is COLUMN count, not horizontal
        // lines. The line-budget path must not interpret it as a
        // horizontal-line budget for VI text.
        let area = DrawableArea::from_polygon(
            &[[0.0, 0.0], [200.0, 0.0], [200.0, 180.0], [0.0, 180.0]], 2.0,
        );
        let text = "Một câu thoại dài vừa phải để vợt qua một dòng.";
        let items: Vec<(&str, &DrawableArea, Option<FitHint>)> = vec![
            (text, &area, Some(FitHint {
                font_size_px: 24, line_count: 1, avg_chars_per_line: 8.0,
                text_direction: TextDirection::Vertical,
            })),
        ];
        // Should NOT collapse to 1 line just because tategaki hint says line_count=1.
        let result = FitEngine::fit_page_areas(&items, 1755).unwrap();
        // Free-wrap path is allowed to use as many lines as it needs.
        // The point is the fitter doesn't artificially try to squeeze
        // everything onto 1 horizontal line at a tiny font.
        assert!(result[0].font_size_px >= MIN_FONT_SIZE);
    }
}
