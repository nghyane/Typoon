/// Layout engine — text wrapping + drawable-area geometry.
///
/// Shaping is in `crate::shape` (HarfBuzz). Outline drawing is in
/// `crate::overlay` (skrifa → tiny-skia). Both stages use em-based pixel
/// scale so advances and glyph paths stay in lock-step. This module is the
/// pure-geometry layer that turns text + drawable area into wrapped lines.
///
/// Writing mode: Vietnamese target text is ALWAYS `Horizontal`.
/// `text_direction` from the source (Japanese vertical) is intentionally
/// ignored for layout — it only affects whether TypesettingHint is applied.

use serde::{Deserialize, Serialize};
use skrifa::{MetadataProvider, instance::{LocationRef, Size}};

use crate::font;

/// Extra leading between lines as a fraction of `font_size_px`. Added on top
/// of the font's natural ascent+|descent| so consecutive Vietnamese diacritics
/// (V̆ over a tone, e.g. Ặ / Ệ) never touch the descenders of the line above.
pub const LINE_LEADING_FRAC: f64 = 0.05;

/// Backwards-compatible "line height multiplier" used by `fit.rs` for cheap
/// budget math (`char_budget`, hint estimates). Calibrated to the embedded
/// SamaritanTall metrics: ascent (1.318 em) + |descent| (0.227 em) + leading.
///
/// For exact baseline-to-baseline use [`line_spacing_px`].
pub const LINE_HEIGHT_MULTIPLIER: f64 = 1.6;

/// Default inset from bbox edge when border detection is unavailable.
pub const DEFAULT_INSET: f64 = 2.0;

/// Allow text width to overshoot bubble by this fraction before char-breaking.
/// Set to 0 to match the inscribed-ellipse fit contract — any line wider
/// than `safe_w` is overflow, full stop.
pub const WIDTH_OVERFLOW_TOLERANCE: f64 = 0.0;

/// Short SFX use the same hard contract as body text. Earlier 60% allowance
/// was a workaround for too-tight Lens bboxes; now that fit area = bubble
/// inscribed rect, there's nothing to compensate for.
pub const SHORT_TEXT_WORD_COUNT: usize = 3;
pub const SHORT_TEXT_OVERFLOW_TOLERANCE: f64 = WIDTH_OVERFLOW_TOLERANCE;

// ─── Geometry types ────────────────────────────────────────────────────────

#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct EdgeInsets {
    pub left:   f64,
    pub right:  f64,
    pub top:    f64,
    pub bottom: f64,
}

impl EdgeInsets {
    pub fn uniform(v: f64) -> Self {
        Self { left: v, right: v, top: v, bottom: v }
    }
}

/// Canonical drawable area inside a bubble, rotation-aware.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DrawableArea {
    pub bbox:       [f64; 4],
    pub insets:     EdgeInsets,
    pub angle_rad:  f64,
    pub center:     [f64; 2],
    pub oriented_w: f64,
    pub oriented_h: f64,
    /// True when the polygon is a sampled curve (the grouper uses 24-vertex
    /// inscribed ellipses for sane-aspect dialogue balloons). Fit's
    /// `size()` then shrinks the wrap rect so a centred line doesn't extend
    /// past the ellipse curve at top/bottom — the AABB of an ellipse
    /// includes the four corners that lie OUTSIDE the curve.
    #[serde(default)]
    pub is_ellipse: bool,
}

/// Linear scale applied to `oriented_w/h` when `is_ellipse`. Picked
/// empirically: at 0.85 a centred text block in a 200×120 ellipse
/// touches but doesn't cross the curve at the widest line. √(π/4) ≈
/// 0.886 is the area-matched ratio (rect area == ellipse area); 0.85
/// trades a hair of usable area for a small visual safety margin.
const ELLIPSE_FIT_SCALE: f64 = 0.85;

impl DrawableArea {
    pub fn from_polygon(polygon: &[[f64; 2]], inset: f64) -> Self {
        Self::from_polygon_insets(polygon, EdgeInsets::uniform(inset))
    }

    pub fn from_polygon_insets(polygon: &[[f64; 2]], insets: EdgeInsets) -> Self {
        let (x1, y1, x2, y2) = crate::types::polygon_bbox(polygon);
        let cx = polygon.iter().map(|p| p[0]).sum::<f64>() / polygon.len().max(1) as f64;
        let cy = polygon.iter().map(|p| p[1]).sum::<f64>() / polygon.len().max(1) as f64;
        let (angle_rad, ow, oh) = if polygon.len() == 4 {
            let [tl, tr, _br, bl] = [polygon[0], polygon[1], polygon[2], polygon[3]];
            let dx = tr[0] - tl[0]; let dy = tr[1] - tl[1];
            let dx2 = bl[0] - tl[0]; let dy2 = bl[1] - tl[1];
            (dy.atan2(dx), (dx*dx+dy*dy).sqrt(), (dx2*dx2+dy2*dy2).sqrt())
        } else {
            (0.0, x2 - x1, y2 - y1)
        };
        // > 4 vertices → ellipse polygon (grouper emits 24 verts; safe
        // threshold for "this came from inscribed_ellipse, not OBB").
        let is_ellipse = polygon.len() > 4;
        Self {
            bbox: [x1,y1,x2,y2], insets, angle_rad, center: [cx,cy],
            oriented_w: ow, oriented_h: oh, is_ellipse,
        }
    }

    pub fn with_crop_min(&self, crop: [f64; 4]) -> Self {
        Self {
            bbox: self.bbox,
            insets: EdgeInsets {
                left:   self.insets.left.max(crop[0]),
                right:  self.insets.right.max(crop[1]),
                top:    self.insets.top.max(crop[2]),
                bottom: self.insets.bottom.max(crop[3]),
            },
            angle_rad:  self.angle_rad,
            center:     self.center,
            oriented_w: self.oriented_w,
            oriented_h: self.oriented_h,
            is_ellipse: self.is_ellipse,
        }
    }

    pub fn rect(&self) -> (f64, f64, f64, f64) {
        let [x1, y1, x2, y2] = self.bbox;
        let x = x1 + self.insets.left;
        let y = y1 + self.insets.top;
        let w = (x2 - x1 - self.insets.left - self.insets.right).max(0.0);
        let h = (y2 - y1 - self.insets.top  - self.insets.bottom).max(0.0);
        (x, y, w, h)
    }

    pub fn size(&self) -> (f64, f64) {
        let raw_w = (self.oriented_w - self.insets.left - self.insets.right).max(0.0);
        let raw_h = (self.oriented_h - self.insets.top  - self.insets.bottom).max(0.0);
        if self.is_ellipse {
            // The polygon's AABB equals the ellipse's bbox, which includes
            // the four corners that lie OUTSIDE the curve. Fitting text to
            // the AABB lets glyphs at the top/bottom of a centred line slip
            // past the visible balloon outline. Shrinking by
            // `ELLIPSE_FIT_SCALE` keeps the text block inside an inscribed
            // rectangle of the ellipse. Width and height scale together so
            // aspect-driven wrap decisions stay consistent.
            (raw_w * ELLIPSE_FIT_SCALE, raw_h * ELLIPSE_FIT_SCALE)
        } else {
            (raw_w, raw_h)
        }
    }

    pub fn is_rotated(&self) -> bool { self.angle_rad.abs() > 0.035 }
}

// ─── Shaping re-exports ────────────────────────────────────────────────────

/// Re-export so existing call sites (`layout::measure_text_width`) keep working.
pub use crate::shape::measure_width as measure_text_width;

// ─── ICU4X line breaker ────────────────────────────────────────────────────

/// Atomic wrap token: a chunk that cannot be broken further. Tokens from
/// the same source word stay glued (no inter-token space); tokens from
/// different source words get a single space when joined on the same line.
#[derive(Debug, Clone)]
struct Token {
    text:          String,
    leading_space: bool,   // true → space between this token and the previous on the same line
}

/// Hyphen / dash characters that introduce a wrap opportunity inside a
/// word. Includes hyphen-minus, hyphen, en/em dash, soft hyphen and
/// underscore. Slash is intentionally excluded — `kg/m` should not break.
fn is_break_char(c: char) -> bool {
    matches!(c, '-' | '\u{2010}' | '\u{2013}' | '\u{2014}' | '\u{00AD}' | '_')
}

/// Split one whitespace-delimited word at internal break chars, keeping
/// the break char attached to the preceding chunk so the hyphen stays
/// visible at line end ("28-" then "Shonan", not "28" then "-Shonan").
/// Returns at least one chunk; never an empty chunk.
fn split_at_break_chars(word: &str) -> Vec<String> {
    if word.is_empty() {
        return vec![String::new()];
    }
    let mut out: Vec<String> = Vec::new();
    let mut cur = String::new();
    for ch in word.chars() {
        cur.push(ch);
        if is_break_char(ch) {
            // Cut AFTER the break char so it sticks to the left chunk.
            out.push(std::mem::take(&mut cur));
        }
    }
    if !cur.is_empty() {
        out.push(cur);
    } else if out.is_empty() {
        out.push(String::new());
    }
    out
}

/// Tokenise `text` into atomic wrap units. Hyphens and dashes inside a
/// word become break opportunities; tokens from the same word are flagged
/// `leading_space=false` so the renderer joins them without a space.
fn tokenize_for_wrap(text: &str) -> Vec<Token> {
    let mut tokens: Vec<Token> = Vec::new();
    for (wi, word) in text.split_whitespace().enumerate() {
        let chunks = split_at_break_chars(word);
        for (ci, chunk) in chunks.into_iter().enumerate() {
            tokens.push(Token {
                text:          chunk,
                leading_space: ci == 0 && wi > 0,
            });
        }
    }
    tokens
}

/// Width of the longest atomic token (incl. hyphen tail). Used by the
/// fit binary search as the hard "can this size wrap at all?" predicate.
pub fn longest_atom_width(text: &str, font_size_px: u32) -> f64 {
    tokenize_for_wrap(text).iter()
        .map(|t| measure_text_width(&t.text, font_size_px))
        .fold(0.0_f64, f64::max)
}

/// Render a list of tokens as a single line: tokens flagged
/// `leading_space=true` get a space prefix, others glue to the previous.
fn join_tokens(tokens: &[&Token]) -> String {
    let mut s = String::new();
    for t in tokens {
        if t.leading_space && !s.is_empty() {
            s.push(' ');
        }
        s.push_str(&t.text);
    }
    s
}

/// Measure a line built from `tokens`: sum of token widths plus a single
/// space-width for each `leading_space=true` token after the first.
fn line_width(tokens: &[&Token], space_w: f64, font_size_px: u32) -> f64 {
    let mut w = 0.0_f64;
    for (i, t) in tokens.iter().enumerate() {
        if i > 0 && t.leading_space {
            w += space_w;
        }
        w += measure_text_width(&t.text, font_size_px);
    }
    w
}

/// Wrap `text` into lines that fit within `max_width_px`. Words split at
/// internal hyphens / dashes so a long hyphenated compound ("28-Shonan")
/// can wrap mid-word rather than forcing the fitter to a smaller font.
pub fn wrap_text(text: &str, max_width_px: f64, font_size_px: u32) -> Vec<String> {
    if text.is_empty() {
        return vec![String::new()];
    }

    let tokens = tokenize_for_wrap(text);
    if tokens.is_empty() {
        return vec![String::new()];
    }

    // Determine overshoot tolerance based on word count (whitespace-delimited
    // — hyphen splits don't reduce the "is this a short label" signal).
    let word_count = text.split_whitespace().count();
    let tolerance = if word_count <= SHORT_TEXT_WORD_COUNT {
        SHORT_TEXT_OVERFLOW_TOLERANCE
    } else {
        WIDTH_OVERFLOW_TOLERANCE
    };
    let tolerant_max = max_width_px * (1.0 + tolerance);

    let space_w = measure_text_width(" ", font_size_px);
    let token_ws: Vec<f64> = tokens.iter()
        .map(|t| measure_text_width(&t.text, font_size_px))
        .collect();

    // If any atomic token exceeds tolerant_max → greedy char-break fallback
    let any_oversize = token_ws.iter().any(|&w| w > tolerant_max);
    if any_oversize {
        return wrap_greedy(text, max_width_px, font_size_px);
    }

    // Greedy line count using tokens
    let n_lines = {
        let mut lines = 1usize;
        let mut cur = 0.0f64;
        for (i, t) in tokens.iter().enumerate() {
            let tw = token_ws[i];
            if i == 0 { cur = tw; continue; }
            let gap = if t.leading_space { space_w } else { 0.0 };
            if cur + gap + tw <= tolerant_max {
                cur += gap + tw;
            } else {
                lines += 1;
                cur = tw;
            }
        }
        lines
    };

    if n_lines <= 1 {
        return vec![join_tokens(&tokens.iter().collect::<Vec<_>>())];
    }

    // Balanced wrap
    let total_w: f64 = token_ws.iter().sum::<f64>()
        + space_w * tokens.iter().filter(|t| t.leading_space).count() as f64;
    let target_w = (total_w / n_lines as f64).min(max_width_px);

    let mut lines: Vec<Vec<&Token>> = Vec::new();
    let mut line_toks: Vec<&Token> = Vec::new();
    let mut cur_w = 0.0f64;

    for (i, tok) in tokens.iter().enumerate() {
        let tw  = token_ws[i];
        let gap = if tok.leading_space && !line_toks.is_empty() { space_w } else { 0.0 };
        if line_toks.is_empty() {
            line_toks.push(tok); cur_w = tw; continue;
        }
        let new_w = cur_w + gap + tw;
        let remaining_lines = n_lines.saturating_sub(lines.len() + 1);
        let remaining_toks  = tokens.len() - i;
        let past_target = cur_w >= target_w * 0.85;
        let must_break   = new_w > tolerant_max;
        let should_break = past_target && remaining_toks >= remaining_lines && new_w > target_w;
        if must_break || should_break {
            lines.push(std::mem::take(&mut line_toks));
            line_toks.push(tok); cur_w = tw;
        } else {
            line_toks.push(tok); cur_w = new_w;
        }
    }
    if !line_toks.is_empty() { lines.push(line_toks); }
    if lines.is_empty() { return vec![String::new()]; }

    let mut result: Vec<String> = lines.iter().map(|toks| join_tokens(toks)).collect();

    // Widow pull-back: a last line that is a single short token (e.g. "À?")
    // looks like a layout glitch. Demote one whitespace-delimited word from
    // the previous line down — but only if the merged last line still fits
    // the tolerant max and the donor line keeps at least one word. Operates
    // on the joined strings (whitespace boundaries) so we don't accidentally
    // split a hyphenated compound mid-token.
    if result.len() >= 2 {
        let last_idx = result.len() - 1;
        let last_w = measure_text_width(&result[last_idx], font_size_px);
        let is_short_widow = result[last_idx].split_whitespace().count() == 1
            && last_w < max_width_px * 0.30;
        if is_short_widow {
            let donor_words: Vec<&str> = result[last_idx - 1].split_whitespace().collect();
            if donor_words.len() >= 2 {
                let tail = donor_words[donor_words.len() - 1];
                let head = donor_words[..donor_words.len() - 1].join(" ");
                let merged = format!("{tail} {}", result[last_idx]);
                if measure_text_width(&merged, font_size_px) <= tolerant_max {
                    result[last_idx - 1] = head;
                    result[last_idx]     = merged;
                }
            }
        }
    }

    result
}

fn wrap_greedy(text: &str, max_width_px: f64, font_size_px: u32) -> Vec<String> {
    let words: Vec<&str> = text.split_whitespace().collect();
    let space_w   = measure_text_width(" ", font_size_px);
    let tolerance = if words.len() <= SHORT_TEXT_WORD_COUNT {
        SHORT_TEXT_OVERFLOW_TOLERANCE
    } else {
        WIDTH_OVERFLOW_TOLERANCE
    };
    let tolerant_max = max_width_px * (1.0 + tolerance);

    let mut lines: Vec<String> = Vec::new();
    let mut cur = String::new();
    let mut cur_w = 0.0f64;

    for word in &words {
        let ww = measure_text_width(word, font_size_px);
        if cur.is_empty() {
            if ww > tolerant_max {
                char_break_into(word, max_width_px, font_size_px, &mut lines);
            } else {
                cur.push_str(word); cur_w = ww;
            }
        } else if cur_w + space_w + ww <= tolerant_max {
            cur.push(' '); cur.push_str(word); cur_w += space_w + ww;
        } else {
            lines.push(cur.clone());
            cur.clear(); cur_w = 0.0;
            if ww > tolerant_max {
                char_break_into(word, max_width_px, font_size_px, &mut lines);
            } else {
                cur.push_str(word); cur_w = ww;
            }
        }
    }
    if !cur.is_empty() { lines.push(cur); }
    if lines.is_empty() { lines.push(String::new()); }
    lines
}

fn char_break_into(word: &str, max_width_px: f64, font_size_px: u32, lines: &mut Vec<String>) {
    let mut cur = String::new();
    let mut cur_w = 0.0f64;
    for ch in word.chars() {
        let cw = measure_text_width(&ch.to_string(), font_size_px);
        if !cur.is_empty() && cur_w + cw > max_width_px {
            lines.push(cur.clone()); cur.clear(); cur_w = 0.0;
        }
        cur.push(ch); cur_w += cw;
    }
    if !cur.is_empty() { lines.push(cur); }
}

// ─── Block height ──────────────────────────────────────────────────────────

/// Baseline-to-baseline distance in pixels at `font_size_px`. Pulls
/// ascent/descent from the embedded font's hhea so it adapts if the font is
/// ever swapped, plus a small leading so stacked diacritics don't kiss.
pub fn line_spacing_px(font_size_px: u32) -> f64 {
    let m = font::skrifa_font().metrics(
        Size::new(font_size_px as f32),
        LocationRef::default(),
    );
    let ascent_descent = m.ascent as f64 + (-m.descent) as f64;
    ascent_descent + font_size_px as f64 * LINE_LEADING_FRAC
}

/// Total text-block height in pixels: from the topmost ink (first line ascent)
/// to the bottommost ink (last line descent), with leading between lines.
pub fn text_block_height(n_lines: usize, font_size_px: u32) -> f64 {
    if n_lines == 0 { return 0.0; }
    let m = font::skrifa_font().metrics(
        Size::new(font_size_px as f32),
        LocationRef::default(),
    );
    let asc = m.ascent as f64;
    let dsc = (-m.descent) as f64;
    let spacing = line_spacing_px(font_size_px);
    (n_lines - 1) as f64 * spacing + asc + dsc
}

// ─── Tests ─────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn measure_width_nonzero() {
        let w = measure_text_width("Hello", 16);
        assert!(w > 5.0, "expected >5px, got {w}");
    }

    #[test]
    fn wrap_produces_lines() {
        let text = "ĐƯỢC CÔNG CHÚNG BIẾT ĐẾN LÀ NỮ PHẢN DIỆN TỒI TỆ NHẤT";
        let lines = wrap_text(text, 200.0, 16);
        assert!(lines.len() > 1);
    }

    #[test]
    fn wrap_avoids_short_last_word_widow() {
        // Reproduces the "À?" widow seen in lens_bubble_probe3:
        // last line was a single 2-char token while line above had 6 words.
        let text = "CẬU KHÔNG THẤY GẦN ĐÂY KHẢI-CHAN CÓ GÌ ĐÓ LẠ LẠ À?";
        let lines = wrap_text(text, 360.0, 28);
        let last = lines.last().expect("at least one line");
        let last_words = last.split_whitespace().count();
        assert!(
            last_words >= 2 || last == &text,
            "widow not pulled back: lines = {:#?}",
            lines,
        );
    }

    #[test]
    fn wrap_splits_hyphenated_compound() {
        // Tall-narrow tategaki bubble (~80px wide). "28-Shonan" as one
        // atom forced the fitter to a tiny font; splitting at the hyphen
        // lets it wrap "28-" + "Shonan" and use available height.
        // Hyphen must stay attached to the LEFT chunk.
        let text = "NĂM CHIÊU HÒA 28-SHONAN";
        let lines = wrap_text(text, 80.0, 16);
        assert!(lines.len() >= 3, "expected ≥3 lines, got {lines:#?}");

        // All characters preserved in order (modulo whitespace).
        let joined: String = lines.iter()
            .flat_map(|l| l.split_whitespace())
            .collect::<Vec<_>>()
            .join(" ");
        assert_eq!(joined, "NĂM CHIÊU HÒA 28- SHONAN",
            "tokens reshuffled or hyphen detached: {lines:#?}");

        // Hyphen never starts a line — it sticks to the left chunk.
        for l in &lines {
            assert!(!l.trim_start().starts_with('-'),
                "hyphen leaked to next line: {lines:#?}");
        }
    }

    #[test]
    fn longest_atom_width_respects_hyphen_split() {
        // "28-Shonan" as a whole word is wider than its longest atom.
        // The fit gate uses longest_atom_width so a narrow bubble can
        // still fit a reasonable font size.
        let fs = 24;
        let whole = measure_text_width("28-Shonan", fs);
        let atom  = longest_atom_width("28-Shonan", fs);
        assert!(atom < whole, "atom {atom} should be < whole {whole}");
    }
}
