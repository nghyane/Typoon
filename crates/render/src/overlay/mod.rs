use ab_glyph::PxScale;
use image::{DynamicImage, Rgba, RgbaImage};
use imageproc::drawing::draw_text_mut;

use crate::layout;

/// Per-bubble input for text rendering.
pub struct RenderBubble {
    pub translated_text: String,
    pub area: crate::layout::DrawableArea,
    pub font_size_px: u32,
    pub line_height: f64,
}

/// Render translated text onto a clean page (text already erased by Python).
pub fn render(img: &DynamicImage, bubbles: &[RenderBubble]) -> RgbaImage {
    let mut canvas = img.to_rgba8();
    draw_translated_text(&mut canvas, bubbles);
    canvas
}

fn draw_translated_text(canvas: &mut RgbaImage, bubbles: &[RenderBubble]) {
    let font = layout::get_font();

    for bubble in bubbles {
        if bubble.translated_text.is_empty() {
            continue;
        }

        let (draw_x1, draw_y1, draw_w, draw_h) = bubble.area.rect();

        let bg = sample_bg_color(canvas, draw_x1 as u32, draw_y1 as u32, draw_w as u32, draw_h as u32);
        let dark_bg = luminance(&bg) < 128;
        // Text vs stroke: opaque text + nearly-opaque stroke. The stroke
        // serves dual purpose: legibility on busy backgrounds AND a
        // visible halo separating text from the cleaned bubble fill.
        // Alpha 230 (was 180) makes the halo readable on screentone /
        // gradient backgrounds without looking hard-edged.
        let text_color = if dark_bg {
            Rgba([255, 255, 255, 255])
        } else {
            Rgba([0, 0, 0, 255])
        };
        let stroke_color = if dark_bg {
            Rgba([0, 0, 0, 230])
        } else {
            Rgba([255, 255, 255, 230])
        };

        if bubble.area.is_rotated() {
            draw_rotated_bubble(canvas, bubble, font, text_color, stroke_color);
        } else {
            draw_horizontal_bubble(canvas, bubble, font, text_color, stroke_color);
        }
    }
}

fn draw_horizontal_bubble(
    canvas: &mut RgbaImage,
    bubble: &RenderBubble,
    font: &ab_glyph::FontRef<'static>,
    text_color: Rgba<u8>,
    stroke_color: Rgba<u8>,
) {
    let (draw_x1, draw_y1, draw_w, draw_h) = bubble.area.rect();
    let scale = PxScale::from(bubble.font_size_px as f32);
    let line_spacing = bubble.font_size_px as f64 * bubble.line_height;

    let lines: Vec<&str> = bubble.translated_text.lines().collect();
    let total_text_h = if lines.is_empty() {
        0.0
    } else {
        (lines.len() - 1) as f64 * line_spacing + bubble.font_size_px as f64
    };
    let start_y = (draw_y1 + (draw_h - total_text_h) / 2.0).max(draw_y1);

    for (i, line) in lines.iter().enumerate() {
        let line_w = layout::measure_text_width(line, bubble.font_size_px, font);
        let x = draw_x1 + (draw_w - line_w) / 2.0;
        let y = start_y + i as f64 * line_spacing;
        draw_stroked_text(canvas, font, line, x as i32, y as i32, scale, text_color, stroke_color, bubble.font_size_px);
    }
}

fn draw_rotated_bubble(
    canvas: &mut RgbaImage,
    bubble: &RenderBubble,
    font: &ab_glyph::FontRef<'static>,
    text_color: Rgba<u8>,
    stroke_color: Rgba<u8>,
) {
    let (safe_w, safe_h) = bubble.area.size();
    let buf_w = (safe_w.ceil() as u32).max(1);
    let buf_h = (safe_h.ceil() as u32).max(1);

    let mut buf = RgbaImage::from_pixel(buf_w, buf_h, Rgba([0, 0, 0, 0]));
    let scale = PxScale::from(bubble.font_size_px as f32);
    let line_spacing = bubble.font_size_px as f64 * bubble.line_height;

    let lines: Vec<&str> = bubble.translated_text.lines().collect();
    let total_text_h = if lines.is_empty() {
        0.0
    } else {
        (lines.len() - 1) as f64 * line_spacing + bubble.font_size_px as f64
    };
    let start_y = ((safe_h - total_text_h) / 2.0).max(0.0);

    for (i, line) in lines.iter().enumerate() {
        let line_w = layout::measure_text_width(line, bubble.font_size_px, font);
        let x = ((safe_w - line_w) / 2.0).max(0.0);
        let y = start_y + i as f64 * line_spacing;
        draw_stroked_text(&mut buf, font, line, x as i32, y as i32, scale, text_color, stroke_color, bubble.font_size_px);
    }

    let [cx, cy] = bubble.area.center;
    let cos_a = bubble.area.angle_rad.cos();
    let sin_a = bubble.area.angle_rad.sin();
    let local_cx = buf_w as f64 / 2.0;
    let local_cy = buf_h as f64 / 2.0;
    let (cw, ch) = canvas.dimensions();

    let [bx1, by1, bx2, by2] = bubble.area.bbox;
    let scan_x1 = (bx1.floor() as i32).max(0) as u32;
    let scan_y1 = (by1.floor() as i32).max(0) as u32;
    let scan_x2 = (bx2.ceil() as u32).min(cw);
    let scan_y2 = (by2.ceil() as u32).min(ch);

    for py in scan_y1..scan_y2 {
        for px in scan_x1..scan_x2 {
            let dx = px as f64 - cx;
            let dy = py as f64 - cy;
            let lx = dx * cos_a + dy * sin_a + local_cx;
            let ly = -dx * sin_a + dy * cos_a + local_cy;

            let lxi = lx.floor() as i32;
            let lyi = ly.floor() as i32;
            if lxi < 0 || lyi < 0 || lxi >= buf_w as i32 || lyi >= buf_h as i32 {
                continue;
            }

            let src = buf.get_pixel(lxi as u32, lyi as u32);
            if src.0[3] == 0 { continue; }

            let dst = canvas.get_pixel(px, py);
            let sa = src.0[3] as u32;
            let da = 255 - sa;
            let blend = |s: u8, d: u8| ((s as u32 * sa + d as u32 * da) / 255) as u8;
            canvas.put_pixel(px, py, Rgba([
                blend(src.0[0], dst.0[0]),
                blend(src.0[1], dst.0[1]),
                blend(src.0[2], dst.0[2]),
                (sa + dst.0[3] as u32 * da / 255).min(255) as u8,
            ]));
        }
    }
}

fn draw_stroked_text(
    img: &mut RgbaImage,
    font: &ab_glyph::FontRef<'static>,
    text: &str,
    ix: i32, iy: i32,
    scale: PxScale,
    text_color: Rgba<u8>,
    stroke_color: Rgba<u8>,
    font_size_px: u32,
) {
    // Stroke width as a fraction of font height. 0.07 was empirically
    // chosen on the fixture chapters: legible on solid white bubbles
    // (problem case at 0.04) and on screentone overlap, without
    // becoming chunky at large SFX font sizes. Floor of 2px ensures
    // small UI font (16-20px in webtoons) gets a 2px halo, not 1px
    // that disappears against grey.
    let sw = ((font_size_px as f64 * 0.07).round() as i32).max(2);
    for dy in -sw..=sw {
        for dx in -sw..=sw {
            if (dx == 0 && dy == 0) || dx * dx + dy * dy > sw * sw {
                continue;
            }
            draw_text_mut(img, stroke_color, ix + dx, iy + dy, scale, font, text);
        }
    }
    draw_text_mut(img, text_color, ix, iy, scale, font, text);
}

fn sample_bg_color(img: &RgbaImage, x: u32, y: u32, w: u32, h: u32) -> Rgba<u8> {
    let mut rs = Vec::new();
    let mut gs = Vec::new();
    let mut bs = Vec::new();
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::layout::DrawableArea;

    #[test]
    fn test_render_empty_bubbles() {
        let img = DynamicImage::new_rgb8(200, 100);
        let result = render(&img, &[]);
        assert_eq!(result.width(), 200);
        assert_eq!(result.height(), 100);
    }

    #[test]
    fn test_render_single_bubble() {
        let img = DynamicImage::ImageRgba8(RgbaImage::from_pixel(400, 300, Rgba([255, 255, 255, 255])));
        let polygon = vec![[50.0, 50.0], [350.0, 50.0], [350.0, 250.0], [50.0, 250.0]];
        let bubble = RenderBubble {
            translated_text: "Xin chào".into(),
            area: DrawableArea::from_polygon(&polygon, 3.0),
            font_size_px: 24,
            line_height: 1.18,
        };
        let result = render(&img, &[bubble]);
        let mut has_dark = false;
        for y in 100..200 {
            for x in 150..250 {
                let px = result.get_pixel(x, y);
                if px.0[0] < 128 { has_dark = true; break; }
            }
            if has_dark { break; }
        }
        assert!(has_dark, "should have dark text pixels");
    }

    #[test]
    fn test_encode_png() {
        let img = RgbaImage::new(10, 10);
        let bytes = encode_png(&img);
        assert!(bytes.len() > 8);
        assert_eq!(&bytes[..4], &[0x89, 0x50, 0x4E, 0x47]);
    }
}
