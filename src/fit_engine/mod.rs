use anyhow::Result;

use crate::text_layout;

/// Minimum font size (px) before declaring overflow
const MIN_FONT_SIZE: u32 = 8;
/// Absolute maximum font size to try
const MAX_FONT_SIZE: u32 = 48;
/// Font size cap as fraction of page width (~4.5%)
const MAX_FONT_RATIO: f64 = 0.045;

/// Horizontal padding as fraction of bubble width, clamped to [4, 24] px
const PAD_X_RATIO: f64 = 0.08;
/// Vertical padding as fraction of bubble height, clamped to [4, 24] px
const PAD_Y_RATIO: f64 = 0.10;
const PAD_MIN: f64 = 4.0;
const PAD_MAX: f64 = 24.0;

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
    /// Fit a page of bubbles with normalized font sizes.
    ///
    /// 1. Binary search max fitting size per bubble (capped by page width)
    /// 2. Compute page-level cap = median × 1.15
    /// 3. Re-wrap only outlier bubbles exceeding the cap
    pub fn fit_page(items: &[(&str, &[[f64; 2]])], page_width: u32) -> Result<Vec<FitResult>> {
        if items.is_empty() {
            return Ok(Vec::new());
        }

        // Hard cap: page_width × 2% (800px→16, 1200px→24, 1442px→28)
        let width_cap = ((page_width as f64) * MAX_FONT_RATIO) as u32;
        let abs_cap = width_cap.clamp(MIN_FONT_SIZE, MAX_FONT_SIZE);

        // Pass 1: compute max fitting font size per bubble
        let individual: Vec<FitResult> = items
            .iter()
            .map(|(text, poly)| Self::fit_capped(text, poly, abs_cap))
            .collect::<Result<Vec<_>>>()?;

        // Collect non-overflow sizes for normalization
        let mut sizes: Vec<u32> = individual
            .iter()
            .filter(|r| !r.overflow && !r.text.is_empty())
            .map(|r| r.font_size_px)
            .collect();

        if sizes.len() < 2 {
            return Ok(individual);
        }

        sizes.sort_unstable();
        // Median × 1.35 cap — preserves natural size variation while capping extreme outliers
        let median = sizes[sizes.len() / 2];
        let page_target = ((median as f64 * 1.35) as u32).min(abs_cap);

        // Pass 2: re-fit each bubble capped at page_target
        items
            .iter()
            .zip(individual.iter())
            .map(|((text, poly), orig)| {
                if orig.text.is_empty() || orig.overflow {
                    return Ok(orig.clone());
                }
                if orig.font_size_px <= page_target {
                    return Ok(orig.clone());
                }
                Self::fit_at(text, poly, page_target)
            })
            .collect()
    }

    /// Fit text at a specific (capped) font size.
    fn fit_at(translated_text: &str, polygon: &[[f64; 2]], max_size: u32) -> Result<FitResult> {
        let text = normalize_text(translated_text);
        let (safe_w, safe_h) = safe_rect(polygon);
        if safe_w < 1.0 || safe_h < 1.0 {
            return Ok(FitResult {
                text,
                font_size_px: MIN_FONT_SIZE,
                line_height: text_layout::LINE_HEIGHT_MULTIPLIER,
                overflow: true,
            });
        }

        let font = text_layout::get_font();
        let hi_cap = max_size.min(safe_h as u32).min(MAX_FONT_SIZE);

        let mut lo = MIN_FONT_SIZE;
        let mut hi = hi_cap;
        let mut best_size = MIN_FONT_SIZE;
        let mut best_wrapped = text_layout::wrap_text(&text, safe_w, MIN_FONT_SIZE, font);

        while lo <= hi {
            let mid = (lo + hi) / 2;
            let wrapped = text_layout::wrap_text(&text, safe_w, mid, font);
            let total_h = wrapped.len() as f64 * mid as f64 * text_layout::LINE_HEIGHT_MULTIPLIER;

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

        let total_h = best_wrapped.len() as f64 * best_size as f64 * text_layout::LINE_HEIGHT_MULTIPLIER;
        let overflow = total_h > safe_h || best_size < MIN_FONT_SIZE;

        Ok(FitResult {
            text: best_wrapped.join("\n"),
            font_size_px: best_size,
            line_height: text_layout::LINE_HEIGHT_MULTIPLIER,
            overflow,
        })
    }

    /// Fit with a custom max font size cap.
    fn fit_capped(translated_text: &str, polygon: &[[f64; 2]], max_size: u32) -> Result<FitResult> {
        let text = normalize_text(translated_text);
        if text.is_empty() {
            return Ok(FitResult {
                text,
                font_size_px: MIN_FONT_SIZE,
                line_height: text_layout::LINE_HEIGHT_MULTIPLIER,
                overflow: false,
            });
        }

        let (safe_w, safe_h) = safe_rect(polygon);
        if safe_w < 1.0 || safe_h < 1.0 {
            return Ok(FitResult {
                text,
                font_size_px: MIN_FONT_SIZE,
                line_height: text_layout::LINE_HEIGHT_MULTIPLIER,
                overflow: true,
            });
        }

        let font = text_layout::get_font();
        let max_font = (safe_h as u32).min(max_size);

        // Binary search for largest fitting font size
        let mut lo = MIN_FONT_SIZE;
        let mut hi = max_font;
        let mut best_size = MIN_FONT_SIZE;
        let mut best_wrapped = text_layout::wrap_text(&text, safe_w, MIN_FONT_SIZE, font);

        while lo <= hi {
            let mid = (lo + hi) / 2;
            let wrapped = text_layout::wrap_text(&text, safe_w, mid, font);
            let total_h = wrapped.len() as f64 * mid as f64 * text_layout::LINE_HEIGHT_MULTIPLIER;

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

        let total_h = best_wrapped.len() as f64 * best_size as f64 * text_layout::LINE_HEIGHT_MULTIPLIER;
        let overflow = total_h > safe_h || best_size < MIN_FONT_SIZE;

        Ok(FitResult {
            text: best_wrapped.join("\n"),
            font_size_px: best_size,
            line_height: text_layout::LINE_HEIGHT_MULTIPLIER,
            overflow,
        })
    }

    /// Fit translated text into a bubble polygon (single bubble, no normalization).
    pub fn fit(translated_text: &str, polygon: &[[f64; 2]]) -> Result<FitResult> {
        Self::fit_capped(translated_text, polygon, MAX_FONT_SIZE)
    }
}

/// Compute safe inner rect from a 4-point axis-aligned polygon.
fn safe_rect(polygon: &[[f64; 2]]) -> (f64, f64) {
    if polygon.len() < 2 {
        return (0.0, 0.0);
    }
    let (x1, y1, x2, y2) = text_layout::polygon_bbox(polygon);
    let w = x2 - x1;
    let h = y2 - y1;
    let pad_x = (w * PAD_X_RATIO).clamp(PAD_MIN, PAD_MAX);
    let pad_y = (h * PAD_Y_RATIO).clamp(PAD_MIN, PAD_MAX);
    ((w - 2.0 * pad_x).max(0.0), (h - 2.0 * pad_y).max(0.0))
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
    fn test_fit_overflow() {
        // Tiny bubble
        let polygon = vec![[0.0, 0.0], [20.0, 0.0], [20.0, 10.0], [0.0, 10.0]];
        let result = FitEngine::fit("This is a very long text that should overflow", &polygon).unwrap();
        assert!(result.overflow);
    }

    #[test]
    fn test_fit_wrapping() {
        // Medium bubble — text should wrap
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
    fn test_fit_page_normalizes() {
        let big = vec![[0.0, 0.0], [400.0, 0.0], [400.0, 300.0], [0.0, 300.0]];
        let med = vec![[0.0, 0.0], [200.0, 0.0], [200.0, 150.0], [0.0, 150.0]];
        let items: Vec<(&str, &[[f64; 2]])> = vec![
            ("Hi", &big),                              // short text, big bubble → big font
            ("Hello world", &med),                     // more text, smaller bubble → smaller font
            ("Some longer dialogue here", &med),       // longest text → smallest font
        ];
        let results = FitEngine::fit_page(&items, 800).unwrap();
        assert_eq!(results.len(), 3);
        // First bubble (short text, big bubble) should be capped down to page target
        // instead of being much larger than the others
        let sizes: Vec<u32> = results.iter().map(|r| r.font_size_px).collect();
        // With median×1.15 cap, the big bubble is capped but variation is preserved
        let max_size = *sizes.iter().max().unwrap();
        let median = {
            let mut s = sizes.clone();
            s.sort();
            s[s.len() / 2]
        };
        assert!(max_size <= ((median as f64 * 1.35) as u32) + 1,
            "Max size should be capped at ~median×1.35: sizes={sizes:?}, median={median}");
    }

    #[test]
    fn test_safe_rect() {
        let poly = vec![[10.0, 20.0], [210.0, 20.0], [210.0, 120.0], [10.0, 120.0]];
        let (w, h) = safe_rect(&poly);
        // 200px wide, 100px tall → pad_x = clamp(16, 4, 24) = 16, pad_y = clamp(10, 4, 24) = 10
        assert!((w - 168.0).abs() < 0.1, "w={w}");
        assert!((h - 80.0).abs() < 0.1, "h={h}");
    }
}
