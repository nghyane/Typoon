pub mod erase;

use ab_glyph::PxScale;
use image::{DynamicImage, Rgba, RgbaImage};
use imageproc::drawing::draw_text_mut;

use crate::api::BubbleResult;
use crate::inpaint::LamaInpainter;
use crate::text_layout;

// Re-export erasure utilities for use by examples and tests.
pub use erase::{
    apply_mask_pixels, erase_masks, erase_with_median, is_flat_background, median_bg_color,
    page_mask_from_local, pixel_luminance,
};
use crate::detection::LocalTextMask;

/// Render translated text overlay on a manga/manhwa page.
///
/// For each bubble:
/// 1. Erase original text (dual-path: median fill or LaMa inpainting)
/// 2. Draw translated text centered in the drawable area
///
/// Returns the composited RGBA image.
pub fn render(
    img: &DynamicImage,
    bubbles: &[BubbleResult],
    inpainter: Option<&LamaInpainter>,
) -> RgbaImage {
    let mut canvas = img.to_rgba8();

    let masks: Vec<&LocalTextMask> = bubbles
        .iter()
        .filter(|b| !b.translated_text.is_empty())
        .filter_map(|b| b.text_mask.as_ref())
        .collect();

    if !masks.is_empty() {
        erase_masks(&mut canvas, &masks, inpainter);
    }

    draw_translated_text(&mut canvas, bubbles);

    canvas
}

/// Draw translated text for all bubbles onto the canvas.
/// Text color is auto-detected: white on dark backgrounds, black on light.
fn draw_translated_text(canvas: &mut RgbaImage, bubbles: &[BubbleResult]) {
    let font = text_layout::get_font();

    for bubble in bubbles {
        if bubble.translated_text.is_empty() {
            continue;
        }

        let area = bubble.drawable_area.as_ref().unwrap_or_else(|| {
            panic!("BubbleResult.drawable_area must be set by pipeline")
        });
        let (draw_x1, draw_y1, draw_w, draw_h) = area.rect();

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
            let line_w = text_layout::measure_text_width(line, bubble.font_size_px, font);
            let x = match bubble.align.as_str() {
                "left" => draw_x1,
                "right" => draw_x1 + draw_w - line_w,
                _ => draw_x1 + (draw_w - line_w) / 2.0,
            };
            let y = start_y + i as f64 * line_spacing;
            let ix = x as i32;
            let iy = y as i32;

            // Stroke: draw at offsets scaled by font size (~4% of font size, min 1px)
            let sw = ((bubble.font_size_px as f64 * 0.04).round() as i32).max(1);
            for dy in -sw..=sw {
                for dx in -sw..=sw {
                    if dx == 0 && dy == 0 { continue; }
                    if dx * dx + dy * dy > sw * sw { continue; }
                    draw_text_mut(canvas, stroke_color, ix + dx, iy + dy, scale, font, line);
                }
            }
            draw_text_mut(canvas, text_color, ix, iy, scale, font, line);
        }
    }
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
    img.write_with_encoder(encoder).expect("JPEG encoding failed");
    let b64 = STANDARD.encode(buf.into_inner());
    format!("data:image/jpeg;base64,{b64}")
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::text_layout::DrawableArea;

    #[test]
    fn test_render_empty_bubbles() {
        let img = DynamicImage::new_rgb8(200, 100);
        let result = render(&img, &[], None);
        assert_eq!(result.width(), 200);
        assert_eq!(result.height(), 100);
    }

    #[test]
    fn test_render_single_bubble() {
        let img = DynamicImage::ImageRgba8(
            RgbaImage::from_pixel(400, 300, Rgba([255, 255, 255, 255])),
        );
        let bubble = BubbleResult {
            bubble_id: "b0".into(),
            polygon: vec![[50.0, 50.0], [350.0, 50.0], [350.0, 250.0], [50.0, 250.0]],
            source_text: "Hello".into(),
            translated_text: "Xin chào".into(),
            font_size_px: 24,
            line_height: 1.18,
            overflow: false,
            align: "center".into(),
            drawable_area: Some(DrawableArea::from_polygon(
                &[[50.0, 50.0], [350.0, 50.0], [350.0, 250.0], [50.0, 250.0]],
                3.0,
            )),
            text_mask: None,
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
            if has_dark { break; }
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
