use ab_glyph::PxScale;
use image::{DynamicImage, Rgba, RgbaImage};
use imageproc::drawing::{draw_filled_rect_mut, draw_text_mut};
use imageproc::rect::Rect;

use crate::api::BubbleResult;
use crate::text_layout;

/// Render translated text overlay on a manga/manhwa page.
///
/// For each bubble:
/// 1. Fill polygon bounding box with white (erase original text)
/// 2. Draw translated text centered in the bbox
///
/// Returns the composited RGBA image.
pub fn render(img: &DynamicImage, bubbles: &[BubbleResult]) -> RgbaImage {
    let mut canvas = img.to_rgba8();
    let font = text_layout::get_font();

    let white = Rgba([255u8, 255, 255, 255]);
    let black = Rgba([0u8, 0, 0, 255]);
    let img_w = canvas.width() as i32;
    let img_h = canvas.height() as i32;

    for bubble in bubbles {
        if bubble.translated_text.is_empty() {
            continue;
        }

        let (bx1, by1, bx2, by2) = text_layout::polygon_bbox(&bubble.polygon);
        let bbox_w = bx2 - bx1;
        let bbox_h = by2 - by1;
        let inset = bubble.inset;

        // 1. Erase: fill inset bbox with white (preserves bubble border)
        let rx = (bx1 as i32 + inset as i32).max(0);
        let ry = (by1 as i32 + inset as i32).max(0);
        let rw = ((bbox_w - 2.0 * inset) as i32).max(0).min(img_w - rx) as u32;
        let rh = ((bbox_h - 2.0 * inset) as i32).max(0).min(img_h - ry) as u32;
        if rw > 0 && rh > 0 {
            draw_filled_rect_mut(&mut canvas, Rect::at(rx, ry).of_size(rw, rh), white);
        }

        // 2. Draw translated text centered in bbox
        let scale = PxScale::from(bubble.font_size_px as f32);
        let line_spacing = bubble.font_size_px as f64 * bubble.line_height;

        let lines: Vec<&str> = bubble.translated_text.lines().collect();
        let total_text_h = lines.len() as f64 * line_spacing;

        let start_y = (by1 + (bbox_h - total_text_h) / 2.0).max(by1);

        for (i, line) in lines.iter().enumerate() {
            let line_w = text_layout::measure_text_width(line, bubble.font_size_px, font);
            let x = match bubble.align.as_str() {
                "left" => bx1 + inset,
                "right" => bx1 + bbox_w - inset - line_w,
                _ => bx1 + (bbox_w - line_w) / 2.0,
            };
            let y = start_y + i as f64 * line_spacing;
            if bubble.font_size_px <= 16 {
                draw_text_mut(&mut canvas, black, x as i32 + 1, y as i32, scale, font, line);
                draw_text_mut(&mut canvas, black, x as i32, y as i32 + 1, scale, font, line);
            }
            draw_text_mut(&mut canvas, black, x as i32, y as i32, scale, font, line);
        }
    }

    canvas
}

/// Encode an RGBA image as PNG bytes.
pub fn encode_png(img: &RgbaImage) -> Vec<u8> {
    let mut buf = std::io::Cursor::new(Vec::new());
    img.write_to(&mut buf, image::ImageFormat::Png)
        .expect("PNG encoding failed");
    buf.into_inner()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_render_empty_bubbles() {
        let img = DynamicImage::new_rgb8(200, 100);
        let result = render(&img, &[]);
        assert_eq!(result.width(), 200);
        assert_eq!(result.height(), 100);
    }

    #[test]
    fn test_render_single_bubble() {
        let img = DynamicImage::new_rgb8(400, 300);
        let bubble = BubbleResult {
            bubble_id: "b0".into(),
            polygon: vec![[50.0, 50.0], [350.0, 50.0], [350.0, 250.0], [50.0, 250.0]],
            source_text: "Hello".into(),
            translated_text: "Xin chào".into(),
            font_size_px: 24,
            line_height: 1.18,
            overflow: false,
            align: "center".into(),
            inset: 3.0,
        };
        let result = render(&img, &[bubble]);
        assert_eq!(result.width(), 400);
        // Text block center should have been painted (white bg or black text)
        let px = result.get_pixel(200, 150);
        assert!(px.0[0] != 0 || px.0[1] != 0 || px.0[2] != 0, "Center should have content, got {px:?}");
    }

    #[test]
    fn test_encode_png() {
        let img = RgbaImage::new(10, 10);
        let bytes = encode_png(&img);
        assert!(bytes.len() > 8);
        // PNG magic bytes
        assert_eq!(&bytes[..4], &[0x89, 0x50, 0x4E, 0x47]);
    }
}
