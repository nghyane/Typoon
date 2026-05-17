//! Text rendering — skrifa outlines + tiny-skia rasterisation.
//!
//! Pipeline per bubble:
//!   1. HarfBuzz shape via `crate::shape::shape` → `PositionedGlyph`s
//!   2. Skrifa `OutlineGlyph::draw` → tiny-skia `Path` (cached per glyph_id)
//!   3. tiny-skia `fill_path` (text) + `stroke_path` (halo) on a Pixmap
//!   4. Composite Pixmap → image::RgbaImage canvas, with rotation when needed
//!
//! Why this stack: HarfBuzz, skrifa and tiny-skia all interpret font size as
//! "pixels-per-em" (em-based). Mixing them keeps advances and outlines on the
//! same scale, fixing the wide-spacing bug that came from the old
//! ab_glyph rasteriser using a height-based scale (`px / height_unscaled`).
//!
//! Stroke is a vector outline pass (tiny-skia `stroke_path`) — one expand of
//! the actual glyph path, instead of the previous "draw text 9× at offsets"
//! halo trick that produced visible offset artefacts and ran the rasteriser
//! 10× per line.

use std::collections::HashMap;
use std::sync::{Mutex, OnceLock};

use image::{Rgba, RgbaImage};
use skrifa::{
    GlyphId, MetadataProvider,
    instance::{LocationRef, Size},
    outline::{DrawSettings, OutlinePen},
};
use tiny_skia::{
    Color, FillRule, LineCap, LineJoin, Paint, Path, PathBuilder, Pixmap, Stroke, Transform,
};

use crate::font;
use crate::layout::{self, DrawableArea};
use crate::shape;

/// Per-bubble input for text rendering. `translated_text` may contain
/// embedded `\n` (LLM soft breaks) — each line is shaped independently.
pub struct RenderBubble {
    pub translated_text: String,
    pub area:            DrawableArea,
    pub font_size_px:    u32,
    pub line_height:     f64,
}

/// Render translated text onto a clean page (text already erased by Python).
///
/// Takes ownership of `canvas` and mutates it in place to avoid the
/// 15 MB `to_rgba8()` clone that used to happen on every page. The
/// returned `RgbaImage` is the same allocation.
pub fn render(mut canvas: RgbaImage, bubbles: &[RenderBubble]) -> RgbaImage {
    for bubble in bubbles {
        if bubble.translated_text.is_empty() { continue; }

        let bg = sample_bg_color(
            &canvas,
            bubble.area.bbox[0] as u32,
            bubble.area.bbox[1] as u32,
            (bubble.area.bbox[2] - bubble.area.bbox[0]) as u32,
            (bubble.area.bbox[3] - bubble.area.bbox[1]) as u32,
        );
        let dark_bg     = luminance(&bg) < 128;
        let text_color  = if dark_bg { [255, 255, 255, 255] } else { [0,   0,   0,   255] };
        let halo_color  = if dark_bg { [0,   0,   0,   230] } else { [255, 255, 255, 230] };

        if bubble.area.is_rotated() {
            draw_rotated(&mut canvas, bubble, text_color, halo_color);
        } else {
            draw_horizontal(&mut canvas, bubble, text_color, halo_color);
        }
    }
    canvas
}

// ─── Horizontal bubble ────────────────────────────────────────────────────

fn draw_horizontal(
    canvas:     &mut RgbaImage,
    bubble:     &RenderBubble,
    text_color: [u8; 4],
    halo_color: [u8; 4],
) {
    let (draw_x1, draw_y1, draw_w, draw_h) = bubble.area.rect();
    let (cw, ch) = canvas.dimensions();
    let line_spacing = layout::line_spacing_px(bubble.font_size_px);
    let asc = ascent_px(bubble.font_size_px);
    let dsc = descent_px(bubble.font_size_px);

    let lines: Vec<&str> = bubble.translated_text.lines().collect();
    let total_h = layout::text_block_height(lines.len(), bubble.font_size_px);
    let start_y_baseline = draw_y1 + ((draw_h - total_h) / 2.0).max(0.0) + asc;
    let center_x = draw_x1 + draw_w / 2.0;

    // Render each line into its own pixmap then composite — keeps stroke
    // halos inside an RGBA buffer so they alpha-blend properly with the
    // already-erased page (instead of stamping halo + fill onto live pixels
    // and bleeding into glyph corners).
    for (i, line) in lines.iter().enumerate() {
        let line_w = shape::measure_width(line, bubble.font_size_px);
        let pen_x  = center_x - line_w / 2.0;
        let pen_y  = start_y_baseline + i as f64 * line_spacing;

        let pad   = stroke_width_px(bubble.font_size_px).ceil() as u32 + 2;
        let buf_w = line_w.ceil() as u32 + pad * 2;
        let buf_h = (asc + dsc).ceil() as u32 + pad * 2;
        if buf_w == 0 || buf_h == 0 { continue; }

        let Some(mut pm) = Pixmap::new(buf_w, buf_h) else { continue; };
        let baseline_in_buf_y = pad as f64 + asc;
        rasterise_line(
            &mut pm,
            line,
            bubble.font_size_px,
            pad as f64,
            baseline_in_buf_y,
            text_color,
            halo_color,
        );

        let dst_x = (pen_x - pad as f64).floor() as i32;
        let dst_y = (pen_y - baseline_in_buf_y).floor() as i32;
        composite_pixmap(canvas, &pm, dst_x, dst_y, cw, ch);
    }
}

// ─── Rotated bubble ───────────────────────────────────────────────────────

fn draw_rotated(
    canvas:     &mut RgbaImage,
    bubble:     &RenderBubble,
    text_color: [u8; 4],
    halo_color: [u8; 4],
) {
    let (safe_w, safe_h) = bubble.area.size();
    let line_spacing = layout::line_spacing_px(bubble.font_size_px);
    let asc = ascent_px(bubble.font_size_px);
    let lines: Vec<&str> = bubble.translated_text.lines().collect();
    let total_h = layout::text_block_height(lines.len(), bubble.font_size_px);

    let max_line_w = lines.iter()
        .map(|l| shape::measure_width(l, bubble.font_size_px))
        .fold(0.0_f64, f64::max);

    let pad   = stroke_width_px(bubble.font_size_px).ceil() as u32 + 2;
    let buf_w = (max_line_w.max(safe_w).ceil() as u32).max(1) + pad * 2;
    let buf_h = (safe_h.ceil() as u32).max(1) + pad * 2;
    let Some(mut pm) = Pixmap::new(buf_w, buf_h) else { return; };

    let start_y_baseline = pad as f64
        + ((safe_h - total_h) / 2.0).max(0.0)
        + asc;

    for (i, line) in lines.iter().enumerate() {
        let line_w = shape::measure_width(line, bubble.font_size_px);
        let pen_x  = pad as f64 + (max_line_w.max(safe_w) - line_w) / 2.0;
        let pen_y  = start_y_baseline + i as f64 * line_spacing;
        rasterise_line(
            &mut pm,
            line,
            bubble.font_size_px,
            pen_x,
            pen_y,
            text_color,
            halo_color,
        );
    }
    composite_rotated(canvas, &pm, &bubble.area, max_line_w, safe_w, pad);
}

// ─── Glyph rasterisation (skrifa → tiny-skia) ─────────────────────────────

/// Stamp one shaped line of text into `pm`, baseline at `(pen_x, pen_y)`.
/// Halo (stroke) is painted first so the fill ends up on top.
fn rasterise_line(
    pm:           &mut Pixmap,
    text:         &str,
    font_size_px: u32,
    pen_x:        f64,
    pen_y:        f64,
    text_color:   [u8; 4],
    halo_color:   [u8; 4],
) {
    let glyphs = shape::shape(text, font_size_px);
    if glyphs.is_empty() { return; }

    let stroke_w = stroke_width_px(font_size_px);

    // Build per-glyph paths once; we draw them twice (stroke + fill).
    struct GlyphDraw { x: f64, y: f64, path: Path }
    let mut draws: Vec<GlyphDraw> = Vec::with_capacity(glyphs.len());

    let mut cur_x = pen_x;
    let cur_y = pen_y;
    for g in &glyphs {
        let gx = cur_x + g.x_offset;
        let gy = cur_y - g.y_offset;  // font y is up, screen y is down
        if let Some(path) = glyph_path(g.glyph_id, font_size_px) {
            draws.push(GlyphDraw { x: gx, y: gy, path });
        }
        cur_x += g.x_advance;
    }

    // Pass 1 — stroke (halo). One stroke_path per glyph, vector.
    if halo_color[3] > 0 && stroke_w > 0.0 {
        let halo_paint = solid_paint(halo_color);
        let stroke = Stroke {
            width:     stroke_w as f32,
            line_cap:  LineCap::Round,
            line_join: LineJoin::Round,
            ..Stroke::default()
        };
        for d in &draws {
            let xform = Transform::from_translate(d.x as f32, d.y as f32);
            pm.stroke_path(&d.path, &halo_paint, &stroke, xform, None);
        }
    }

    // Pass 2 — fill (glyph body). Drawn after stroke so the halo only
    // shows where it extends past the glyph outline.
    let fill_paint = solid_paint(text_color);
    for d in &draws {
        let xform = Transform::from_translate(d.x as f32, d.y as f32);
        pm.fill_path(&d.path, &fill_paint, FillRule::Winding, xform, None);
    }
}

/// Cached glyph outline path at a given font size. Outline paths scale
/// linearly so we cache by `(glyph_id, font_size_px)`.
fn glyph_path(glyph_id: u32, font_size_px: u32) -> Option<Path> {
    let key = (glyph_id, font_size_px);
    {
        let cache = path_cache().lock().unwrap();
        if let Some(slot) = cache.get(&key) {
            return slot.clone();
        }
    }
    let computed = build_glyph_path(glyph_id, font_size_px);
    let mut cache = path_cache().lock().unwrap();
    cache.entry(key).or_insert_with(|| computed.clone());
    computed
}

fn build_glyph_path(glyph_id: u32, font_size_px: u32) -> Option<Path> {
    let outlines = font::skrifa_font().outline_glyphs();
    let glyph    = outlines.get(GlyphId::new(glyph_id))?;
    let mut pen  = SkiaPen::default();
    glyph.draw(
        DrawSettings::unhinted(Size::new(font_size_px as f32), LocationRef::default()),
        &mut pen,
    ).ok()?;
    pen.finish()
}

/// Convert skrifa's `OutlinePen` callbacks to a `tiny_skia::PathBuilder`.
/// y is flipped here (font y-up → image y-down) so callers can translate
/// the path by the baseline `(pen_x, pen_y)` directly.
#[derive(Default)]
struct SkiaPen { builder: PathBuilder }

impl SkiaPen {
    fn finish(self) -> Option<Path> { self.builder.finish() }
}

impl OutlinePen for SkiaPen {
    fn move_to(&mut self, x: f32, y: f32) { self.builder.move_to(x, -y); }
    fn line_to(&mut self, x: f32, y: f32) { self.builder.line_to(x, -y); }
    fn quad_to(&mut self, cx0: f32, cy0: f32, x: f32, y: f32) {
        self.builder.quad_to(cx0, -cy0, x, -y);
    }
    fn curve_to(&mut self, cx0: f32, cy0: f32, cx1: f32, cy1: f32, x: f32, y: f32) {
        self.builder.cubic_to(cx0, -cy0, cx1, -cy1, x, -y);
    }
    fn close(&mut self) { self.builder.close(); }
}

fn path_cache() -> &'static Mutex<HashMap<(u32, u32), Option<Path>>> {
    static CACHE: OnceLock<Mutex<HashMap<(u32, u32), Option<Path>>>> = OnceLock::new();
    CACHE.get_or_init(|| Mutex::new(HashMap::new()))
}

// ─── Pixmap → RgbaImage compositing ───────────────────────────────────────

fn composite_pixmap(
    canvas: &mut RgbaImage,
    pm:     &Pixmap,
    dst_x:  i32,
    dst_y:  i32,
    cw:     u32,
    ch:     u32,
) {
    let pw = pm.width();
    let ph = pm.height();
    let src = pm.data();
    for sy in 0..ph {
        let py = dst_y + sy as i32;
        if py < 0 || py >= ch as i32 { continue; }
        for sx in 0..pw {
            let px = dst_x + sx as i32;
            if px < 0 || px >= cw as i32 { continue; }
            let i = ((sy * pw + sx) * 4) as usize;
            let a = src[i + 3];
            if a == 0 { continue; }
            // tiny-skia is premultiplied RGBA; image::Rgba is straight.
            // Recover straight RGB before blending into the canvas.
            let (sr, sg, sb) = unpremul(src[i], src[i + 1], src[i + 2], a);
            blend_pixel(canvas, px as u32, py as u32, sr, sg, sb, a);
        }
    }
}

fn composite_rotated(
    canvas:     &mut RgbaImage,
    pm:         &Pixmap,
    area:       &DrawableArea,
    max_line_w: f64,
    safe_w:     f64,
    pad:        u32,
) {
    let [cx, cy] = area.center;
    let cos_a = area.angle_rad.cos();
    let sin_a = area.angle_rad.sin();
    let pw = pm.width();
    let ph = pm.height();
    let local_cx = pw as f64 / 2.0;
    let local_cy = ph as f64 / 2.0;
    let (cw, ch) = canvas.dimensions();
    let src = pm.data();

    let extra = ((max_line_w - safe_w) / 2.0).max(0.0).ceil() as i32 + pad as i32;
    let [bx1, by1, bx2, by2] = area.bbox;
    let scan_x1 = ((bx1.floor() as i32 - extra).max(0)) as u32;
    let scan_y1 = ((by1.floor() as i32 - extra).max(0)) as u32;
    let scan_x2 = ((bx2.ceil()  as i32 + extra) as u32).min(cw);
    let scan_y2 = ((by2.ceil()  as i32 + extra) as u32).min(ch);

    for py in scan_y1..scan_y2 {
        for px in scan_x1..scan_x2 {
            let dx = px as f64 - cx;
            let dy = py as f64 - cy;
            let lx = ( dx * cos_a + dy * sin_a + local_cx).floor() as i32;
            let ly = (-dx * sin_a + dy * cos_a + local_cy).floor() as i32;
            if lx < 0 || ly < 0 || lx >= pw as i32 || ly >= ph as i32 { continue; }
            let i = ((ly as u32 * pw + lx as u32) * 4) as usize;
            let a = src[i + 3];
            if a == 0 { continue; }
            let (sr, sg, sb) = unpremul(src[i], src[i + 1], src[i + 2], a);
            blend_pixel(canvas, px, py, sr, sg, sb, a);
        }
    }
}

#[inline]
fn unpremul(r: u8, g: u8, b: u8, a: u8) -> (u8, u8, u8) {
    if a == 255 { return (r, g, b); }
    let af = a as u32;
    if af == 0 { return (0, 0, 0); }
    (
        ((r as u32 * 255 + af / 2) / af).min(255) as u8,
        ((g as u32 * 255 + af / 2) / af).min(255) as u8,
        ((b as u32 * 255 + af / 2) / af).min(255) as u8,
    )
}

#[inline]
fn blend_pixel(canvas: &mut RgbaImage, x: u32, y: u32, sr: u8, sg: u8, sb: u8, sa: u8) {
    let dst = canvas.get_pixel_mut(x, y);
    let sa_u = sa as u32;
    let da   = 255 - sa_u;
    let blend = |s: u8, d: u8| ((s as u32 * sa_u + d as u32 * da) / 255) as u8;
    dst.0 = [
        blend(sr, dst.0[0]),
        blend(sg, dst.0[1]),
        blend(sb, dst.0[2]),
        (sa_u + dst.0[3] as u32 * da / 255).min(255) as u8,
    ];
}

// ─── Paint / metric helpers ───────────────────────────────────────────────

fn solid_paint(rgba: [u8; 4]) -> Paint<'static> {
    let mut p = Paint::default();
    p.set_color(Color::from_rgba8(rgba[0], rgba[1], rgba[2], rgba[3]));
    p.anti_alias = true;
    p
}

/// Halo (stroke) width in px. Empirically `0.07 × font_size_px`, floored at
/// 2px so small fonts still get a visible outline.
fn stroke_width_px(font_size_px: u32) -> f64 {
    (font_size_px as f64 * 0.07).round().max(2.0)
}

/// Pixel ascent at `font_size_px` from the embedded font's hhea table.
fn ascent_px(font_size_px: u32) -> f64 {
    let f = font::skrifa_font();
    let metrics = f.metrics(Size::new(font_size_px as f32), LocationRef::default());
    metrics.ascent as f64
}

/// Pixel descent (positive value) at `font_size_px`.
fn descent_px(font_size_px: u32) -> f64 {
    let f = font::skrifa_font();
    let metrics = f.metrics(Size::new(font_size_px as f32), LocationRef::default());
    (-metrics.descent) as f64
}

// ─── Background sampling (unchanged from previous overlay) ────────────────

fn sample_bg_color(img: &RgbaImage, x: u32, y: u32, w: u32, h: u32) -> Rgba<u8> {
    let mut rs = Vec::new(); let mut gs = Vec::new(); let mut bs = Vec::new();
    for sy in (y..y + h).step_by(4) {
        for sx in (x..x + w).step_by(4) {
            if sx < img.width() && sy < img.height() {
                let p = img.get_pixel(sx, sy);
                rs.push(p.0[0]); gs.push(p.0[1]); bs.push(p.0[2]);
            }
        }
    }
    if rs.is_empty() { return Rgba([255, 255, 255, 255]); }
    rs.sort_unstable(); gs.sort_unstable(); bs.sort_unstable();
    let mid = rs.len() / 2;
    Rgba([rs[mid], gs[mid], bs[mid], 255])
}

fn luminance(p: &Rgba<u8>) -> u32 {
    (p.0[0] as u32 * 299 + p.0[1] as u32 * 587 + p.0[2] as u32 * 114) / 1000
}

pub fn encode_png(img: &RgbaImage) -> Vec<u8> {
    let mut buf = std::io::Cursor::new(Vec::new());
    img.write_to(&mut buf, image::ImageFormat::Png).expect("PNG encoding failed");
    buf.into_inner()
}

// ─── Tests ────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use image::DynamicImage;

    #[test]
    fn render_empty_bubbles_no_crash() {
        let img = DynamicImage::new_rgb8(200, 100);
        let result = render(img.to_rgba8(), &[]);
        assert_eq!(result.width(), 200);
        assert_eq!(result.height(), 100);
    }

    #[test]
    fn render_single_bubble_has_dark_pixels() {
        let canvas = RgbaImage::from_pixel(400, 300, Rgba([255, 255, 255, 255]));
        let polygon = vec![[50.0,50.0],[350.0,50.0],[350.0,250.0],[50.0,250.0]];
        let bubble = RenderBubble {
            translated_text: "Xin chào".into(),
            area: DrawableArea::from_polygon(&polygon, 3.0),
            font_size_px: 24,
            line_height: crate::layout::LINE_HEIGHT_MULTIPLIER,
        };
        let result = render(canvas, &[bubble]);
        let has_dark = (100u32..200).any(|y|
            (150u32..250).any(|x| result.get_pixel(x, y).0[0] < 128));
        assert!(has_dark, "should have dark text pixels");
    }
}
