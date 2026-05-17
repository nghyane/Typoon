//! HarfBuzz text shaping wrapper.
//!
//! Returns positioned glyphs in pixel units, em-based scale (matches skrifa
//! outline drawing at `Size::new(font_size_px)`). Single source of truth
//! for advance/offset/glyph-id used by both `layout::measure_text_width`
//! and `overlay` rasterisation.

use harfrust::{Direction, ShaperData, UnicodeBuffer};

use crate::font;

/// One shaped glyph in pixel coordinates (em-based, matches skrifa
/// `Size::new(font_size_px)`).
#[derive(Debug, Clone, Copy)]
pub struct PositionedGlyph {
    pub glyph_id: u32,
    pub x_advance: f64,
    pub y_advance: f64,
    pub x_offset: f64,
    pub y_offset: f64,
}

/// Shape `text` with HarfBuzz at `font_size_px`. Direction LTR (Vietnamese
/// target is always horizontal — vertical source scripts are normalised
/// upstream by the typesetting hint).
pub fn shape(text: &str, font_size_px: u32) -> Vec<PositionedGlyph> {
    if text.is_empty() {
        return Vec::new();
    }

    let font_ref = font::harfrust_font();
    let data     = ShaperData::new(&font_ref);
    let shaper   = data.shaper(&font_ref)
        .point_size(Some(font_size_px as f32))
        .build();
    let upem     = shaper.units_per_em() as f64;
    let scale    = font_size_px as f64 / upem;

    let mut buf = UnicodeBuffer::new();
    buf.push_str(text);
    buf.set_direction(Direction::LeftToRight);

    let glyphs    = shaper.shape(buf, &[]);
    let positions = glyphs.glyph_positions();
    let infos     = glyphs.glyph_infos();

    infos.iter().zip(positions).map(|(info, pos)| PositionedGlyph {
        glyph_id:  info.glyph_id,
        x_advance: pos.x_advance as f64 * scale,
        y_advance: pos.y_advance as f64 * scale,
        x_offset:  pos.x_offset  as f64 * scale,
        y_offset:  pos.y_offset  as f64 * scale,
    }).collect()
}

/// Sum of x-advances after shaping. Cheap-ish — one shape pass.
pub fn measure_width(text: &str, font_size_px: u32) -> f64 {
    shape(text, font_size_px).iter().map(|g| g.x_advance).sum()
}
