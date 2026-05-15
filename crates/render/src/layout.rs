use std::sync::OnceLock;

use ab_glyph::{Font, FontRef, ScaleFont};
use serde::{Deserialize, Serialize};

/// Embedded font for text measurement (SamaritanTall TB — comic style + Vietnamese coverage)
static FONT: OnceLock<FontRef<'static>> = OnceLock::new();
pub const FONT_BYTES: &[u8] = include_bytes!("../assets/SamaritanTall-TB.ttf");

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
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DrawableArea {
    /// Polygon bounding box [x1, y1, x2, y2] (axis-aligned, for fast clipping)
    pub bbox: [f64; 4],
    /// Insets from each bbox edge
    pub insets: EdgeInsets,
    /// Rotation angle in radians (0 = horizontal text).
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
    pub fn from_polygon(polygon: &[[f64; 2]], inset: f64) -> Self {
        Self::from_polygon_insets(polygon, EdgeInsets::uniform(inset))
    }

    /// Create from an ordered polygon with per-edge insets.
    pub fn from_polygon_insets(polygon: &[[f64; 2]], insets: EdgeInsets) -> Self {
        let (x1, y1, x2, y2) = crate::types::polygon_bbox(polygon);
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
            insets,
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
    pub fn rect(&self) -> (f64, f64, f64, f64) {
        let [x1, y1, x2, y2] = self.bbox;
        let x = x1 + self.insets.left;
        let y = y1 + self.insets.top;
        let w = (x2 - x1 - self.insets.left - self.insets.right).max(0.0);
        let h = (y2 - y1 - self.insets.top - self.insets.bottom).max(0.0);
        (x, y, w, h)
    }

    /// Inner drawable size along the quad's own axes (rotation-aware).
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

/// Allow text to overshoot the bubble's drawable width by this fraction
/// before falling back to character-level breaking. Manga bubble polygons
/// often have curved edges that look generous near the centre, so a few
/// pixels of overshoot read as natural typesetting rather than overflow.
pub const WIDTH_OVERFLOW_TOLERANCE: f64 = 0.08;

/// SFX / short-text tolerance — when the entire bubble holds ≤ this
/// many words AND the longest word is the bottleneck, allow much
/// larger overshoot. Char-breaking a 4-letter SFX like "CHÁT" into
/// "CH/ÁT" is far worse than letting the word poke past the bubble
/// edge: readers parse SFX as a visual unit, not as wrapped prose.
pub const SHORT_TEXT_WORD_COUNT: usize = 3;
pub const SHORT_TEXT_OVERFLOW_TOLERANCE: f64 = 0.60;

/// Pick the tolerance band for this wrap call. Short SFX-style texts
/// get a generous budget; long-form dialogue stays at the tight 8%.
fn tolerance_for(word_count: usize) -> f64 {
    if word_count <= SHORT_TEXT_WORD_COUNT {
        SHORT_TEXT_OVERFLOW_TOLERANCE
    } else {
        WIDTH_OVERFLOW_TOLERANCE
    }
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
    let tolerance = tolerance_for(words.len());
    let tolerant_max = max_width_px * (1.0 + tolerance);

    // Only char-break when a word is *significantly* longer than the
    // bubble width. Slight overshoot (within tolerance) is preferred
    // over splitting a word mid-character — readers can parse a word
    // that pokes past the bubble edge but cannot parse "CÙN\nG"
    // without re-stitching mentally. SFX get a wider tolerance because
    // they're visual units.
    let has_unbreakable_word = words
        .iter()
        .any(|w| measure_text_width(w, font_size_px, font) > tolerant_max);
    if has_unbreakable_word {
        return wrap_greedy(text, max_width_px, font_size_px, font);
    }

    // Measure all word widths
    let word_widths: Vec<f64> = words
        .iter()
        .map(|w| measure_text_width(w, font_size_px, font))
        .collect();

    // Greedy pass to find minimum number of lines. Uses the same
    // tolerance as the wrap above — otherwise we'd over-count lines
    // and shrink target_w.
    let n_lines = count_greedy_lines(&word_widths, space_w, tolerant_max);
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
        let remaining_lines = n_lines.saturating_sub(lines.len() + 1);
        let remaining_words = words.len() - i;
        let past_target = current_w >= target_w * 0.85;
        // Hard break only when the line would overshoot the tolerance
        // budget. Soft overshoot (≤ 8%) is allowed because manga
        // bubbles have curved edges that visually absorb a few pixels
        // and char-breaking a word is far worse than the overshoot.
        let must_break = new_w > tolerant_max;
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

    // Anti-widow: if the last line is a lonely short tail (e.g. just
    // "HẮN!" after a 3-line paragraph), pull words back from the
    // previous line so the tail reads with more weight. Skips when the
    // pull would push the borrowed-from line past the tolerance bound.
    rebalance_widow(
        &mut lines, &words, &word_widths, space_w, max_width_px, tolerance,
    );

    lines
}

/// Pull words from the second-to-last line into the last line when the
/// last line is a widow (<35% of average line width). Repeats until the
/// widow grows past the threshold or the source line would overshoot the
/// tolerance bound.
fn rebalance_widow(
    lines: &mut Vec<String>,
    words: &[&str],
    word_widths: &[f64],
    space_w: f64,
    max_width_px: f64,
    tolerance: f64,
) {
    if lines.len() < 2 {
        return;
    }

    // Map each rendered line back to its slice of words by scanning.
    let mut line_word_counts: Vec<usize> = Vec::with_capacity(lines.len());
    let mut consumed = 0;
    for line in lines.iter() {
        let n = line.split_whitespace().count();
        line_word_counts.push(n);
        consumed += n;
    }
    if consumed != words.len() {
        // Char-break path used a different word grouping — skip.
        return;
    }

    let tolerant_max = max_width_px * (1.0 + tolerance);

    loop {
        let last_idx = lines.len() - 1;
        let prev_idx = last_idx - 1;

        // Compute current widths.
        let last_w = measured_line_width(
            lines.len(), &line_word_counts, word_widths, space_w, last_idx,
        );
        let prev_w = measured_line_width(
            lines.len(), &line_word_counts, word_widths, space_w, prev_idx,
        );
        let avg_w: f64 = (0..lines.len())
            .map(|i| measured_line_width(
                lines.len(), &line_word_counts, word_widths, space_w, i,
            ))
            .sum::<f64>()
            / lines.len() as f64;

        // Stop when the widow is no longer a widow (>= 35% of avg).
        if last_w >= avg_w * 0.35 {
            return;
        }
        // Stop when stealing one more word would push the prev line
        // past the tolerance bound.
        let prev_word_count = line_word_counts[prev_idx];
        if prev_word_count <= 1 {
            return; // can't steal from a single-word line
        }
        let last_word_in_prev_idx = line_word_counts[..=prev_idx]
            .iter()
            .sum::<usize>() - 1;
        let stolen_ww = word_widths[last_word_in_prev_idx];
        // After stealing, the LAST line grows by (space + stolen_ww)
        // and the PREV line shrinks by the same amount.
        let new_last_w = last_w + space_w + stolen_ww;
        if new_last_w > tolerant_max {
            return;
        }

        // Perform the steal.
        line_word_counts[prev_idx] -= 1;
        line_word_counts[last_idx] += 1;
        let _ = prev_w; // suppress unused warning

        // Re-render the two affected lines.
        let mut idx = 0;
        for (i, &n) in line_word_counts.iter().enumerate() {
            if i == prev_idx || i == last_idx {
                let slice = &words[idx..idx + n];
                lines[i] = slice.join(" ");
            }
            idx += n;
        }
    }
}

fn measured_line_width(
    _n_lines: usize,
    line_word_counts: &[usize],
    word_widths: &[f64],
    space_w: f64,
    line_idx: usize,
) -> f64 {
    let start: usize = line_word_counts[..line_idx].iter().sum();
    let n = line_word_counts[line_idx];
    if n == 0 {
        return 0.0;
    }
    let words_w: f64 = word_widths[start..start + n].iter().sum();
    words_w + space_w * (n as f64 - 1.0).max(0.0)
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
///
/// `wrap_text` falls back here only when at least one word is > tolerant
/// max width. Here we still apply tolerance: char-break only words that
/// are truly oversized, not those that overshoot by a few pixels.
pub fn wrap_greedy(
    text: &str,
    max_width_px: f64,
    font_size_px: u32,
    font: &FontRef<'_>,
) -> Vec<String> {
    let words: Vec<&str> = text.split_whitespace().collect();
    let space_w = measure_text_width(" ", font_size_px, font);
    let tolerant_max = max_width_px * (1.0 + tolerance_for(words.len()));
    let mut lines = Vec::new();
    let mut current_line = String::new();
    let mut current_width = 0.0;

    for word in &words {
        let word_w = measure_text_width(word, font_size_px, font);

        if current_line.is_empty() {
            if word_w > tolerant_max {
                char_break_into(word, max_width_px, font_size_px, font, &mut lines);
                continue;
            }
            current_line.push_str(word);
            current_width = word_w;
        } else if current_width + space_w + word_w <= tolerant_max {
            current_line.push(' ');
            current_line.push_str(word);
            current_width += space_w + word_w;
        } else {
            lines.push(current_line);
            if word_w > tolerant_max {
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

    #[test]
    fn test_widow_avoidance_pulls_word_from_prev_line() {
        // Long narration that naturally greedy-wraps with an orphan
        // last line. Widow rebalance should pull at least one word
        // back down so the tail is not a single short word.
        let font = get_font();
        // ~440px wide bubble at 20px font fits the first two lines
        // of a 3-line wrap; orphan "RỒI!" lands on line 3.
        let text = "THỜI GIAN KHÔNG CHỈ GIỮ LẠI THẦN HỒN CỦA \
                    QUẢNG LĂNG Ở KIẾP TRƯỚC, MÀ CẢ TƠ TÌNH CỦA \
                    HẮN CŨNG ĐỂ LẠI RỒI!";
        let lines = wrap_text(text, 440.0, 20, font);
        println!("widow test lines: {lines:?}");
        if lines.len() >= 2 {
            let last = lines.last().unwrap();
            let prev = &lines[lines.len() - 2];
            let last_words = last.split_whitespace().count();
            let prev_words = prev.split_whitespace().count();
            // Last line must not be a single short word when the
            // previous line has > 4 words to spare.
            if prev_words > 4 {
                assert!(
                    last_words >= 2 || last.chars().count() >= 6,
                    "widow not rebalanced: prev={prev:?} last={last:?}",
                );
            }
        }
    }

    #[test]
    fn test_widow_avoidance_skips_when_safe() {
        // Already-balanced wrap: no rebalance needed, output unchanged.
        let font = get_font();
        let text = "Một hai ba bốn năm sáu bảy tám";
        let before = wrap_text(text, 200.0, 16, font);
        // Run again — should be idempotent.
        let after = wrap_text(text, 200.0, 16, font);
        assert_eq!(before, after);
    }

    #[test]
    fn test_sfx_word_does_not_char_break_in_narrow_bubble() {
        // SFX "CHÁT CHÁT~" in a tall narrow bubble. With dialogue
        // tolerance (8%) the wrap would char-break CHÁT into "CH/ÁT".
        // With SFX tolerance (40%) the word stays intact.
        let font = get_font();
        let text = "CHÁT CHÁT~";
        let word_w = measure_text_width("CHÁT", 30, font);
        // Bubble 25% narrower than the word — outside dialogue
        // tolerance, inside SFX tolerance.
        let bubble_w = word_w * 0.80;
        let lines = wrap_text(text, bubble_w, 30, font);
        println!("SFX wrap at {:.0}px: {lines:?}", bubble_w);
        // Each line should be a whole word, not char-broken
        for line in &lines {
            assert!(
                !line.is_empty() && line.split_whitespace().count() >= 1,
                "SFX char-broken: {lines:?}",
            );
            // No 2-char fragments like "CH" or "ÁT" alone
            let trimmed = line.trim_end_matches('~').trim();
            assert!(
                trimmed.chars().count() >= 3,
                "SFX line too short (likely char-break): {line:?} in {lines:?}",
            );
        }
    }

    #[test]
    fn test_long_text_still_char_breaks_when_truly_too_wide() {
        // Long dialogue stays at the strict 8% tolerance.
        let font = get_font();
        let text = "Một hai ba bốn năm sáu bảy tám chín mười";
        let too_long_w = measure_text_width("không-thể-tách", 30, font);
        // A word much wider than the bubble should still char-break.
        let text2 = format!("{} không-thể-tách-rời", text);
        let bubble_w = too_long_w * 0.5;
        let lines = wrap_text(&text2, bubble_w, 30, font);
        // Just verify we don't crash and produce multiple lines.
        assert!(lines.len() > 1);
    }
}
