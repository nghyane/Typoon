use std::sync::OnceLock;

use ab_glyph::{Font, FontRef, ScaleFont};
use anyhow::Result;

/// Embedded font for text measurement (Noto Sans — Latin + Vietnamese coverage)
static FONT: OnceLock<FontRef<'static>> = OnceLock::new();
const FONT_BYTES: &[u8] = include_bytes!("../../assets/NotoSans-Medium.ttf");

/// Line height multiplier (line spacing relative to font size)
const LINE_HEIGHT_MULTIPLIER: f64 = 1.18;

/// Minimum font size (px) before declaring overflow
const MIN_FONT_SIZE: u32 = 8;
/// Maximum font size to try
const MAX_FONT_SIZE: u32 = 72;

/// Horizontal padding as fraction of bubble width, clamped to [4, 24] px
const PAD_X_RATIO: f64 = 0.08;
/// Vertical padding as fraction of bubble height, clamped to [4, 24] px
const PAD_Y_RATIO: f64 = 0.10;
const PAD_MIN: f64 = 4.0;
const PAD_MAX: f64 = 24.0;

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
    /// Fit translated text into a bubble polygon.
    ///
    /// 1. Compute safe rect from polygon with adaptive padding
    /// 2. Binary search font size (largest that fits)
    /// 3. Greedy word wrap with font metrics
    pub fn fit(translated_text: &str, polygon: &[[f64; 2]]) -> Result<FitResult> {
        let text = normalize_text(translated_text);
        if text.is_empty() {
            return Ok(FitResult {
                text,
                font_size_px: MIN_FONT_SIZE,
                line_height: LINE_HEIGHT_MULTIPLIER,
                overflow: false,
            });
        }

        let (safe_w, safe_h) = safe_rect(polygon);
        if safe_w < 1.0 || safe_h < 1.0 {
            return Ok(FitResult {
                text,
                font_size_px: MIN_FONT_SIZE,
                line_height: LINE_HEIGHT_MULTIPLIER,
                overflow: true,
            });
        }

        let font = get_font();
        let max_font = (safe_h as u32).min(MAX_FONT_SIZE);

        // Binary search for largest fitting font size
        let mut lo = MIN_FONT_SIZE;
        let mut hi = max_font;
        let mut best_size = MIN_FONT_SIZE;
        let mut best_wrapped = wrap_text(&text, safe_w, MIN_FONT_SIZE, font);

        while lo <= hi {
            let mid = (lo + hi) / 2;
            let wrapped = wrap_text(&text, safe_w, mid, font);
            let total_h = wrapped.len() as f64 * mid as f64 * LINE_HEIGHT_MULTIPLIER;

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

        let total_h = best_wrapped.len() as f64 * best_size as f64 * LINE_HEIGHT_MULTIPLIER;
        let overflow = total_h > safe_h || best_size < MIN_FONT_SIZE;

        Ok(FitResult {
            text: best_wrapped.join("\n"),
            font_size_px: best_size,
            line_height: LINE_HEIGHT_MULTIPLIER,
            overflow,
        })
    }
}

/// Compute safe inner rect from a 4-point axis-aligned polygon.
fn safe_rect(polygon: &[[f64; 2]]) -> (f64, f64) {
    if polygon.len() < 2 {
        return (0.0, 0.0);
    }

    let (mut x1, mut y1) = (f64::INFINITY, f64::INFINITY);
    let (mut x2, mut y2) = (f64::NEG_INFINITY, f64::NEG_INFINITY);
    for p in polygon {
        x1 = x1.min(p[0]);
        y1 = y1.min(p[1]);
        x2 = x2.max(p[0]);
        y2 = y2.max(p[1]);
    }

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

/// Get or initialize the embedded font.
fn get_font() -> &'static FontRef<'static> {
    FONT.get_or_init(|| {
        FontRef::try_from_slice(FONT_BYTES).expect("Failed to parse embedded font")
    })
}

/// Measure the width of a text string at a given font size in pixels.
fn measure_text_width(text: &str, font_size_px: u32, font: &FontRef<'_>) -> f64 {
    let scaled = font.as_scaled(font_size_px as f32);
    let mut width = 0.0f32;
    let mut prev_glyph: Option<ab_glyph::GlyphId> = None;

    for ch in text.chars() {
        let glyph_id = font.glyph_id(ch);
        if let Some(prev) = prev_glyph {
            width += scaled.kern(prev, glyph_id);
        }
        width += scaled.h_advance(glyph_id);
        prev_glyph = Some(glyph_id);
    }

    width as f64
}

/// Greedy word wrap: split text into lines that fit within `max_width_px`.
fn wrap_text(text: &str, max_width_px: f64, font_size_px: u32, font: &FontRef<'_>) -> Vec<String> {
    let words: Vec<&str> = text.split_whitespace().collect();
    if words.is_empty() {
        return vec![String::new()];
    }

    let space_w = measure_text_width(" ", font_size_px, font);
    let mut lines = Vec::new();
    let mut current_line = String::new();
    let mut current_width = 0.0;

    for word in &words {
        let word_w = measure_text_width(word, font_size_px, font);

        if current_line.is_empty() {
            // First word on line — check if it needs character-level breaking
            if word_w > max_width_px {
                char_break_into(word, max_width_px, font_size_px, font, &mut lines);
                continue;
            }
            current_line.push_str(word);
            current_width = word_w;
        } else if current_width + space_w + word_w <= max_width_px {
            current_line.push(' ');
            current_line.push_str(word);
            current_width += space_w + word_w;
        } else {
            lines.push(current_line);
            // Start new line with this word
            if word_w > max_width_px {
                current_line = String::new();
                current_width = 0.0;
                char_break_into(word, max_width_px, font_size_px, font, &mut lines);
            } else {
                current_line = word.to_string();
                current_width = word_w;
            }
        }
    }

    if !current_line.is_empty() {
        lines.push(current_line);
    }

    if lines.is_empty() {
        lines.push(String::new());
    }

    lines
}

/// Break a single long word into multiple lines at character boundaries.
fn char_break_into(
    word: &str,
    max_width_px: f64,
    font_size_px: u32,
    font: &FontRef<'_>,
    lines: &mut Vec<String>,
) {
    let mut current = String::new();
    let mut current_w = 0.0;

    for ch in word.chars() {
        let ch_w = measure_text_width(&ch.to_string(), font_size_px, font);
        if !current.is_empty() && current_w + ch_w > max_width_px {
            lines.push(current);
            current = String::new();
            current_w = 0.0;
        }
        current.push(ch);
        current_w += ch_w;
    }

    if !current.is_empty() {
        lines.push(current);
    }
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
    fn test_safe_rect() {
        let poly = vec![[10.0, 20.0], [210.0, 20.0], [210.0, 120.0], [10.0, 120.0]];
        let (w, h) = safe_rect(&poly);
        // 200px wide, 100px tall → pad_x = clamp(16, 4, 24) = 16, pad_y = clamp(10, 4, 24) = 10
        assert!((w - 168.0).abs() < 0.1, "w={w}");
        assert!((h - 80.0).abs() < 0.1, "h={h}");
    }
}
