use anyhow::Result;

use crate::text_layout;
use crate::text_layout::DrawableArea;

/// Minimum font size (px) before declaring overflow.
const MIN_FONT_SIZE: u32 = 8;
/// Absolute maximum font size (sanity bound).
const MAX_FONT_SIZE: u32 = 72;

#[derive(Clone)]
pub struct FitResult {
    /// Wrapped text with newlines
    pub text: String,
    pub font_size_px: u32,
    /// Line height multiplier
    pub line_height: f64,
    pub overflow: bool,
}

pub struct FitEngine;

impl FitEngine {
    /// Fit a page of bubbles. Each bubble gets the largest font that fits
    /// its own drawable area — no artificial caps, no cross-bubble normalization.
    /// The bbox from detection already encodes the original text's size.
    pub fn fit_page_areas(items: &[(&str, &DrawableArea)], _page_width: u32) -> Result<Vec<FitResult>> {
        items
            .iter()
            .map(|(text, area)| Self::fit_area(text, area))
            .collect()
    }

    /// Fit translated text into a drawable area.
    /// Binary search for the largest font where wrapped text fits within (w, h).
    fn fit_area(translated_text: &str, area: &DrawableArea) -> Result<FitResult> {
        let text = normalize_text(translated_text);
        if text.is_empty() {
            return Ok(FitResult {
                text,
                font_size_px: MIN_FONT_SIZE,
                line_height: text_layout::LINE_HEIGHT_MULTIPLIER,
                overflow: false,
            });
        }

        let (safe_w, safe_h) = area.size();
        if safe_w < 1.0 || safe_h < 1.0 {
            return Ok(FitResult {
                text,
                font_size_px: MIN_FONT_SIZE,
                line_height: text_layout::LINE_HEIGHT_MULTIPLIER,
                overflow: true,
            });
        }

        let font = text_layout::get_font();
        let hi_bound = (safe_h as u32).min(MAX_FONT_SIZE);

        let mut lo = MIN_FONT_SIZE;
        let mut hi = hi_bound;
        let mut best_size = MIN_FONT_SIZE;
        let mut best_wrapped = text_layout::wrap_text(&text, safe_w, MIN_FONT_SIZE, font);

        while lo <= hi {
            let mid = (lo + hi) / 2;
            let wrapped = text_layout::wrap_text(&text, safe_w, mid, font);
            let total_h = text_block_height(wrapped.len(), mid);

            if total_h <= safe_h {
                best_size = mid;
                best_wrapped = wrapped;
                lo = mid + 1;
            } else {
                if mid == 0 {
                    break;
                }
                hi = mid - 1;
            }
        }

        let total_h = text_block_height(best_wrapped.len(), best_size);
        let overflow = total_h > safe_h || best_size < MIN_FONT_SIZE;

        Ok(FitResult {
            text: best_wrapped.join("\n"),
            font_size_px: best_size,
            line_height: text_layout::LINE_HEIGHT_MULTIPLIER,
            overflow,
        })
    }

    /// Fit translated text into a bubble polygon (convenience for single-bubble use).
    pub fn fit(translated_text: &str, polygon: &[[f64; 2]]) -> Result<FitResult> {
        let area = DrawableArea::from_polygon(polygon, text_layout::DEFAULT_INSET);
        Self::fit_area(translated_text, &area)
    }
}

/// Total height of a text block.
/// Last line needs only font height (no trailing line spacing).
///   height = (n-1) × line_spacing + font_size
fn text_block_height(n_lines: usize, font_size_px: u32) -> f64 {
    if n_lines == 0 {
        return 0.0;
    }
    let spacing = font_size_px as f64 * text_layout::LINE_HEIGHT_MULTIPLIER;
    (n_lines - 1) as f64 * spacing + font_size_px as f64
}

/// Normalize text: trim, collapse whitespace, remove internal newlines.
fn normalize_text(text: &str) -> String {
    text.split_whitespace().collect::<Vec<_>>().join(" ")
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
    fn test_fit_single_line_fills_height() {
        // Short text in a 300×50 bbox (inset=2 → 296×46)
        // Single line should get font ≈ 46 (fills height)
        let polygon = vec![[0.0, 0.0], [300.0, 0.0], [300.0, 50.0], [0.0, 50.0]];
        let area = DrawableArea::from_polygon(&polygon, 2.0);
        let result = FitEngine::fit_area("Hello", &area).unwrap();
        assert!(!result.overflow);
        assert!(
            result.font_size_px >= 40,
            "Single line should fill height: got {}px for 46px safe_h",
            result.font_size_px
        );
    }

    #[test]
    fn test_fit_respects_bbox_height() {
        // Narration box: 400×80 (inset=2 → 396×76)
        // vs dialogue: 200×40 (inset=2 → 196×36)
        // Narration should get ~2× the font size of dialogue
        let narration = DrawableArea::from_polygon(
            &[[0.0, 0.0], [400.0, 0.0], [400.0, 80.0], [0.0, 80.0]], 2.0);
        let dialogue = DrawableArea::from_polygon(
            &[[0.0, 0.0], [200.0, 0.0], [200.0, 40.0], [0.0, 40.0]], 2.0);
        let items: Vec<(&str, &DrawableArea)> = vec![
            ("Big narration text", &narration),
            ("Hello world", &dialogue),
        ];
        let results = FitEngine::fit_page_areas(&items, 720).unwrap();
        assert!(
            results[0].font_size_px > results[1].font_size_px,
            "Narration {}px should be bigger than dialogue {}px",
            results[0].font_size_px, results[1].font_size_px
        );
    }

    #[test]
    fn test_fit_overflow() {
        let polygon = vec![[0.0, 0.0], [20.0, 0.0], [20.0, 10.0], [0.0, 10.0]];
        let result = FitEngine::fit("This is a very long text that should overflow", &polygon).unwrap();
        assert!(result.overflow);
    }

    #[test]
    fn test_fit_wrapping() {
        let polygon = vec![[0.0, 0.0], [150.0, 0.0], [150.0, 200.0], [0.0, 200.0]];
        let result = FitEngine::fit("Hello wonderful world of manga translation", &polygon).unwrap();
        assert!(!result.overflow);
        assert!(result.text.contains('\n'), "Expected wrapped text: {:?}", result.text);
    }

    #[test]
    fn test_normalize_text() {
        assert_eq!(normalize_text("  hello   world  "), "hello world");
        assert_eq!(normalize_text("line1\nline2"), "line1 line2");
    }

    #[test]
    fn test_drawable_area_size() {
        use crate::text_layout::DrawableArea;
        let area = DrawableArea::from_polygon(
            &[[10.0, 20.0], [210.0, 20.0], [210.0, 120.0], [10.0, 120.0]], 5.0);
        let (w, h) = area.size();
        assert!((w - 190.0).abs() < 0.1, "w={w}");
        assert!((h - 90.0).abs() < 0.1, "h={h}");
    }
}
