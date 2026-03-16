pub mod erase;

use ab_glyph::PxScale;
use image::{DynamicImage, Rgba, RgbaImage};
use imageproc::drawing::draw_text_mut;

use crate::pipeline::types::TranslatedBubble;
use crate::vision::inpaint::LamaInpainter;
use crate::render::layout;

// Re-export erasure utilities for use by examples and tests.
use crate::vision::detection::LocalTextMask;
pub use erase::{
    apply_mask_pixels, erase_masks, erase_with_median, median_bg_color, page_mask_from_local,
    pixel_luminance,
};

/// Render translated text overlay on a manga/manhwa page.
///
/// For each bubble:
/// 1. Erase original text (dual-path: median fill or LaMa inpainting)
/// 2. Draw translated text centered in the drawable area
///
/// Returns the composited RGBA image.
pub fn render(
    img: &DynamicImage,
    bubbles: &[TranslatedBubble],
    inpainter: Option<&LamaInpainter>,
) -> RgbaImage {
    let mut canvas = img.to_rgba8();

    let masks: Vec<&LocalTextMask> = bubbles
        .iter()
        .filter(|b| !b.translated_text.is_empty())
        .filter_map(|b| b.mask.as_ref())
        .collect();

    if !masks.is_empty() {
        erase_masks(&mut canvas, &masks, inpainter);
    }

    draw_translated_text(&mut canvas, bubbles);

    canvas
}

/// Draw translated text for all bubbles onto the canvas.
/// Text color is auto-detected: white on dark backgrounds, black on light.
/// Rotated bubbles are rendered into a local horizontal buffer then
/// composited at the correct angle.
fn draw_translated_text(canvas: &mut RgbaImage, bubbles: &[TranslatedBubble]) {
    let font = layout::get_font();

    for bubble in bubbles {
        if bubble.translated_text.is_empty() {
            continue;
        }

        let (draw_x1, draw_y1, draw_w, draw_h) = bubble.area.rect();

        let bg = median_bg_color(
            canvas,
            draw_x1 as u32,
            draw_y1 as u32,
            draw_w as u32,
            draw_h as u32,
        );
        let dark_bg = pixel_luminance(&bg) < 128;
        let text_color = if dark_bg {
            Rgba([255u8, 255, 255, 255])
        } else {
            Rgba([0u8, 0, 0, 255])
        };
        let stroke_color = if dark_bg {
            Rgba([0u8, 0, 0, 180])
        } else {
            Rgba([255u8, 255, 255, 180])
        };

        if bubble.area.is_rotated() {
            draw_rotated_bubble(canvas, bubble, font, text_color, stroke_color);
        } else {
            draw_horizontal_bubble(canvas, bubble, font, text_color, stroke_color);
        }
    }
}

/// Draw text horizontally (the common case for most bubbles).
fn draw_horizontal_bubble(
    canvas: &mut RgbaImage,
    bubble: &TranslatedBubble,
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

/// Draw text into a horizontal buffer, then composite it rotated onto the canvas.
fn draw_rotated_bubble(
    canvas: &mut RgbaImage,
    bubble: &TranslatedBubble,
    font: &ab_glyph::FontRef<'static>,
    text_color: Rgba<u8>,
    stroke_color: Rgba<u8>,
) {
    let (safe_w, safe_h) = bubble.area.size();
    let buf_w = (safe_w.ceil() as u32).max(1);
    let buf_h = (safe_h.ceil() as u32).max(1);

    // Draw text horizontally into a transparent local buffer
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

    // Composite rotated buffer onto canvas using affine transform.
    // For each pixel in the destination region, sample from the local buffer.
    let [cx, cy] = bubble.area.center;
    let angle = bubble.area.angle_rad;
    let cos_a = angle.cos();
    let sin_a = angle.sin();
    let local_cx = buf_w as f64 / 2.0;
    let local_cy = buf_h as f64 / 2.0;

    let (cw, ch) = canvas.dimensions();

    // Scan the axis-aligned bounding box of the rotated buffer
    let [bx1, by1, bx2, by2] = bubble.area.bbox;
    let scan_x1 = (bx1.floor() as i32).max(0) as u32;
    let scan_y1 = (by1.floor() as i32).max(0) as u32;
    let scan_x2 = (bx2.ceil() as u32).min(cw);
    let scan_y2 = (by2.ceil() as u32).min(ch);

    for py in scan_y1..scan_y2 {
        for px in scan_x1..scan_x2 {
            // Transform page coord → local buffer coord (inverse rotation around center)
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
            if src.0[3] == 0 {
                continue;
            }

            // Alpha composite
            let dst = canvas.get_pixel(px, py);
            let sa = src.0[3] as u32;
            let da = 255 - sa;
            let blend = |s: u8, d: u8| ((s as u32 * sa + d as u32 * da) / 255) as u8;
            canvas.put_pixel(
                px,
                py,
                Rgba([
                    blend(src.0[0], dst.0[0]),
                    blend(src.0[1], dst.0[1]),
                    blend(src.0[2], dst.0[2]),
                    (sa + dst.0[3] as u32 * da / 255).min(255) as u8,
                ]),
            );
        }
    }
}

/// Draw text with stroke outline at (ix, iy) on the given image.
fn draw_stroked_text(
    img: &mut RgbaImage,
    font: &ab_glyph::FontRef<'static>,
    text: &str,
    ix: i32,
    iy: i32,
    scale: PxScale,
    text_color: Rgba<u8>,
    stroke_color: Rgba<u8>,
    font_size_px: u32,
) {
    let sw = ((font_size_px as f64 * 0.04).round() as i32).max(1);
    for dy in -sw..=sw {
        for dx in -sw..=sw {
            if dx == 0 && dy == 0 {
                continue;
            }
            if dx * dx + dy * dy > sw * sw {
                continue;
            }
            draw_text_mut(img, stroke_color, ix + dx, iy + dy, scale, font, text);
        }
    }
    draw_text_mut(img, text_color, ix, iy, scale, font, text);
}

/// Encode an RGBA image as PNG bytes.
pub fn encode_png(img: &RgbaImage) -> Vec<u8> {
    let mut buf = std::io::Cursor::new(Vec::new());
    img.write_to(&mut buf, image::ImageFormat::Png)
        .expect("PNG encoding failed");
    buf.into_inner()
}

/// Encode a DynamicImage as a JPEG data URI (base64) at the given quality (0–100).
pub fn encode_jpeg_data_uri(img: &DynamicImage, quality: u8) -> String {
    use base64::{Engine, engine::general_purpose::STANDARD};
    let mut buf = std::io::Cursor::new(Vec::new());
    let encoder = image::codecs::jpeg::JpegEncoder::new_with_quality(&mut buf, quality);
    img.write_with_encoder(encoder)
        .expect("JPEG encoding failed");
    let b64 = STANDARD.encode(buf.into_inner());
    format!("data:image/jpeg;base64,{b64}")
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::render::layout::DrawableArea;

    #[test]
    fn test_render_empty_bubbles() {
        let img = DynamicImage::new_rgb8(200, 100);
        let result = render(&img, &[], None);
        assert_eq!(result.width(), 200);
        assert_eq!(result.height(), 100);
    }

    #[test]
    fn test_render_single_bubble() {
        let img =
            DynamicImage::ImageRgba8(RgbaImage::from_pixel(400, 300, Rgba([255, 255, 255, 255])));
        let polygon = vec![[50.0, 50.0], [350.0, 50.0], [350.0, 250.0], [50.0, 250.0]];
        let bubble = TranslatedBubble {
            idx: 0,
            source_text: "Hello".into(),
            translated_text: "Xin chào".into(),
            polygon: polygon.clone(),
            area: DrawableArea::from_polygon(&polygon, 3.0),
            mask: None,
            font_size_px: 24,
            line_height: 1.18,
            overflow: false,
        };
        let result = render(&img, &[bubble], None);
        assert_eq!(result.width(), 400);
        let mut has_dark = false;
        for y in 100..200 {
            for x in 150..250 {
                let px = result.get_pixel(x, y);
                if px.0[0] < 128 || px.0[1] < 128 || px.0[2] < 128 {
                    has_dark = true;
                    break;
                }
            }
            if has_dark {
                break;
            }
        }
        assert!(has_dark, "Center region should have dark text pixels");
    }

    #[test]
    fn test_encode_png() {
        let img = RgbaImage::new(10, 10);
        let bytes = encode_png(&img);
        assert!(bytes.len() > 8);
        assert_eq!(&bytes[..4], &[0x89, 0x50, 0x4E, 0x47]);
    }
}
