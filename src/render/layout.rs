use std::sync::OnceLock;

use ab_glyph::{Font, FontRef, ScaleFont};
use serde::{Deserialize, Serialize};

/// Embedded font for text measurement (SamaritanTall TB — comic style + Vietnamese coverage)
static FONT: OnceLock<FontRef<'static>> = OnceLock::new();
pub const FONT_BYTES: &[u8] = include_bytes!("../../assets/SamaritanTall-TB.ttf");

/// Line height multiplier (line spacing relative to font size).
/// 1.22 balances Vietnamese diacritics clearance with compact typesetting.
pub const LINE_HEIGHT_MULTIPLIER: f64 = 1.22;

/// Default inset from bbox edge when border detection is unavailable.
pub const DEFAULT_INSET: f64 = 2.0;

/// Per-side insets from bbox edge.
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct EdgeInsets {
    pub left: f64,
    pub right: f64,
    pub top: f64,
    pub bottom: f64,
}

impl EdgeInsets {
    pub fn uniform(v: f64) -> Self {
        Self {
            left: v,
            right: v,
            top: v,
            bottom: v,
        }
    }
}

/// Canonical drawable area inside a bubble, rotation-aware.
///
/// Stores the oriented quad (TL, TR, BR, BL) and computes width/height
/// along the quad's own axes — not the axis-aligned bounding box.
/// This means a 15° tilted text region gets the correct local dimensions
/// for text fitting, and the overlay can draw text at the matching angle.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DrawableArea {
    /// Polygon bounding box [x1, y1, x2, y2] (axis-aligned, for fast clipping)
    pub bbox: [f64; 4],
    /// Insets from each bbox edge
    pub insets: EdgeInsets,
    /// Rotation angle in radians (0 = horizontal text). Derived from the
    /// polygon's top edge (TL→TR).
    pub angle_rad: f64,
    /// Center of the oriented quad in page coordinates.
    pub center: [f64; 2],
    /// Full width along the quad's top/bottom edges (before insets).
    pub oriented_w: f64,
    /// Full height along the quad's left/right edges (before insets).
    pub oriented_h: f64,
}

impl DrawableArea {
    /// Create from an ordered polygon [TL, TR, BR, BL] with uniform inset.
    ///
    /// If the polygon has exactly 4 points (rotated quad from PP-OCR),
    /// width and height are measured along the quad's own edges.
    /// Otherwise falls back to axis-aligned bbox.
    pub fn from_polygon(polygon: &[[f64; 2]], inset: f64) -> Self {
        let (x1, y1, x2, y2) = polygon_bbox(polygon);
        let cx = polygon.iter().map(|p| p[0]).sum::<f64>() / polygon.len().max(1) as f64;
        let cy = polygon.iter().map(|p| p[1]).sum::<f64>() / polygon.len().max(1) as f64;

        let (angle_rad, ow, oh) = if polygon.len() == 4 {
            let [tl, tr, _br, bl] = [polygon[0], polygon[1], polygon[2], polygon[3]];
            let dx = tr[0] - tl[0];
            let dy = tr[1] - tl[1];
            let w = (dx * dx + dy * dy).sqrt();
            let dx2 = bl[0] - tl[0];
            let dy2 = bl[1] - tl[1];
            let h = (dx2 * dx2 + dy2 * dy2).sqrt();
            let angle = dy.atan2(dx);
            (angle, w, h)
        } else {
            (0.0, x2 - x1, y2 - y1)
        };

        Self {
            bbox: [x1, y1, x2, y2],
            insets: EdgeInsets::uniform(inset),
            angle_rad,
            center: [cx, cy],
            oriented_w: ow,
            oriented_h: oh,
        }
    }

    /// Derive a new area with per-side crop values clamped to at least the current insets.
    pub fn with_crop_min(&self, crop: [f64; 4]) -> Self {
        Self {
            bbox: self.bbox,
            insets: EdgeInsets {
                left: self.insets.left.max(crop[0]),
                right: self.insets.right.max(crop[1]),
                top: self.insets.top.max(crop[2]),
                bottom: self.insets.bottom.max(crop[3]),
            },
            angle_rad: self.angle_rad,
            center: self.center,
            oriented_w: self.oriented_w,
            oriented_h: self.oriented_h,
        }
    }

    /// Inner drawable rectangle in **page coordinates** (axis-aligned fallback).
    /// Used for erase region and backward-compatible callers.
    pub fn rect(&self) -> (f64, f64, f64, f64) {
        let [x1, y1, x2, y2] = self.bbox;
        let x = x1 + self.insets.left;
        let y = y1 + self.insets.top;
        let w = (x2 - x1 - self.insets.left - self.insets.right).max(0.0);
        let h = (y2 - y1 - self.insets.top - self.insets.bottom).max(0.0);
        (x, y, w, h)
    }

    /// Inner drawable size along the quad's own axes (rotation-aware).
    /// This is what FitEngine should use for text wrapping.
    pub fn size(&self) -> (f64, f64) {
        let w = (self.oriented_w - self.insets.left - self.insets.right).max(0.0);
        let h = (self.oriented_h - self.insets.top - self.insets.bottom).max(0.0);
        (w, h)
    }

    /// True if the text is meaningfully rotated (> ~2°).
    pub fn is_rotated(&self) -> bool {
        self.angle_rad.abs() > 0.035 // ~2 degrees
    }
}

/// Get or initialize the embedded font.
pub fn get_font() -> &'static FontRef<'static> {
    FONT.get_or_init(|| FontRef::try_from_slice(FONT_BYTES).expect("Failed to parse embedded font"))
}

/// Measure the width of a text string at a given font size in pixels.
pub fn measure_text_width(text: &str, font_size_px: u32, font: &FontRef<'_>) -> f64 {
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

/// Balanced word wrap: split text into lines that fit within `max_width_px`,
/// distributing words evenly so lines have similar widths.
pub fn wrap_text(
    text: &str,
    max_width_px: f64,
    font_size_px: u32,
    font: &FontRef<'_>,
) -> Vec<String> {
    let words: Vec<&str> = text.split_whitespace().collect();
    if words.is_empty() {
        return vec![String::new()];
    }

    let space_w = measure_text_width(" ", font_size_px, font);

    // Check if any word needs character-level breaking
    let has_long_word = words
        .iter()
        .any(|w| measure_text_width(w, font_size_px, font) > max_width_px);
    if has_long_word {
        return wrap_greedy(text, max_width_px, font_size_px, font);
    }

    // Measure all word widths
    let word_widths: Vec<f64> = words
        .iter()
        .map(|w| measure_text_width(w, font_size_px, font))
        .collect();

    // Greedy pass to find minimum number of lines
    let n_lines = count_greedy_lines(&word_widths, space_w, max_width_px);
    if n_lines <= 1 {
        return vec![words.join(" ")];
    }

    // Balanced wrap: try to equalize line widths using target width
    let total_w: f64 = word_widths.iter().sum::<f64>() + space_w * (words.len() as f64 - 1.0);
    let target_w = (total_w / n_lines as f64).min(max_width_px);

    let mut lines = Vec::new();
    let mut line_words: Vec<&str> = Vec::new();
    let mut current_w = 0.0;

    for (i, word) in words.iter().enumerate() {
        let ww = word_widths[i];
        if line_words.is_empty() {
            line_words.push(word);
            current_w = ww;
            continue;
        }

        let new_w = current_w + space_w + ww;
        // Break if adding this word exceeds both target and max,
        // or if we're past target and breaking still leaves enough words for remaining lines.
        let remaining_lines = n_lines.saturating_sub(lines.len() + 1);
        let remaining_words = words.len() - i;
        let past_target = current_w >= target_w * 0.85;
        let must_break = new_w > max_width_px;
        let should_break = past_target && remaining_words >= remaining_lines && new_w > target_w;

        if must_break || should_break {
            lines.push(line_words.join(" "));
            line_words = vec![word];
            current_w = ww;
        } else {
            line_words.push(word);
            current_w = new_w;
        }
    }

    if !line_words.is_empty() {
        lines.push(line_words.join(" "));
    }

    if lines.is_empty() {
        lines.push(String::new());
    }

    lines
}

/// Count how many lines greedy wrapping would produce.
fn count_greedy_lines(word_widths: &[f64], space_w: f64, max_width: f64) -> usize {
    let mut lines = 1usize;
    let mut current_w = 0.0;
    for (i, &ww) in word_widths.iter().enumerate() {
        if i == 0 {
            current_w = ww;
        } else if current_w + space_w + ww <= max_width {
            current_w += space_w + ww;
        } else {
            lines += 1;
            current_w = ww;
        }
    }
    lines
}

/// Greedy word wrap fallback (for texts with oversized words needing char-breaking).
pub fn wrap_greedy(
    text: &str,
    max_width_px: f64,
    font_size_px: u32,
    font: &FontRef<'_>,
) -> Vec<String> {
    let words: Vec<&str> = text.split_whitespace().collect();
    let space_w = measure_text_width(" ", font_size_px, font);
    let mut lines = Vec::new();
    let mut current_line = String::new();
    let mut current_width = 0.0;

    for word in &words {
        let word_w = measure_text_width(word, font_size_px, font);

        if current_line.is_empty() {
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
    fn test_wrap_vietnamese() {
        let font = get_font();
        let text = "ĐƯỢC CÔNG CHÚNG BIẾT ĐẾN LÀ NỮ PHẢN DIỆN TỒI TỆ NHẤT THỜI ĐẠI";
        let full_w = measure_text_width(text, 16, font);
        println!("full width at 16px: {full_w:.1}");

        // 200px wide bubble should force wrapping
        let lines = wrap_text(text, 200.0, 16, font);
        println!("wrap at 200px: {lines:?}");
        assert!(
            lines.len() > 1,
            "Should wrap: full_w={full_w:.0}, got {} line(s)",
            lines.len()
        );

        // 150px wide bubble
        let lines = wrap_text(text, 150.0, 16, font);
        println!("wrap at 150px: {lines:?}");
        assert!(lines.len() > 1);
    }

    #[test]
    fn test_measure_width_nonzero() {
        let font = get_font();
        let w = measure_text_width("Hello", 16, font);
        println!("'Hello' at 16px = {w:.1}px");
        assert!(w > 10.0, "Width should be significant: {w}");

        let w_vn = measure_text_width("Xin chào", 16, font);
        println!("'Xin chào' at 16px = {w_vn:.1}px");
        assert!(w_vn > 10.0, "VN width should be significant: {w_vn}");
    }
}

/// Compute axis-aligned bounding box from a polygon: (x1, y1, x2, y2).
pub fn polygon_bbox(polygon: &[[f64; 2]]) -> (f64, f64, f64, f64) {
    let mut x1 = f64::INFINITY;
    let mut y1 = f64::INFINITY;
    let mut x2 = f64::NEG_INFINITY;
    let mut y2 = f64::NEG_INFINITY;
    for p in polygon {
        x1 = x1.min(p[0]);
        y1 = y1.min(p[1]);
        x2 = x2.max(p[0]);
        y2 = y2.max(p[1]);
    }
    (x1, y1, x2, y2)
}
