use ab_glyph::PxScale;
use image::{DynamicImage, Rgba, RgbaImage};
use imageproc::drawing::{draw_filled_rect_mut, draw_polygon_mut, draw_text_mut};
use imageproc::point::Point;
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

        // Use canonical DrawableArea if available, otherwise fall back to polygon + inset
        let (draw_x1, draw_y1, draw_w, draw_h) = if let Some(area) = &bubble.drawable_area {
            area.rect()
        } else {
            let (bx1, by1, bx2, by2) = text_layout::polygon_bbox(&bubble.polygon);
            let inset = bubble.inset;
            (bx1 + inset, by1 + inset,
             (bx2 - bx1 - 2.0 * inset).max(0.0),
             (by2 - by1 - 2.0 * inset).max(0.0))
        };

        // 1. Erase: fill the actual polygon with white (tight fill for rotated quads)
        if bubble.polygon.len() >= 3 {
            let inset = bubble.inset;
            let poly_points = shrink_polygon(&bubble.polygon, inset);
            if poly_points.len() >= 3 {
                draw_polygon_mut(&mut canvas, &poly_points, white);
            }
        } else {
            // Fallback: fill drawable rect
            let rx = (draw_x1 as i32).max(0);
            let ry = (draw_y1 as i32).max(0);
            let rw = (draw_w as i32).max(0).min(img_w - rx) as u32;
            let rh = (draw_h as i32).max(0).min(img_h - ry) as u32;
            if rw > 0 && rh > 0 {
                draw_filled_rect_mut(&mut canvas, Rect::at(rx, ry).of_size(rw, rh), white);
            }
        }

        // 2. Draw translated text within the erased area
        let scale = PxScale::from(bubble.font_size_px as f32);
        let line_spacing = bubble.font_size_px as f64 * bubble.line_height;

        let lines: Vec<&str> = bubble.translated_text.lines().collect();
        let total_text_h = lines.len() as f64 * line_spacing;

        let start_y = (draw_y1 + (draw_h - total_text_h) / 2.0).max(draw_y1);

        for (i, line) in lines.iter().enumerate() {
            let line_w = text_layout::measure_text_width(line, bubble.font_size_px, font);
            let x = match bubble.align.as_str() {
                "left" => draw_x1,
                "right" => draw_x1 + draw_w - line_w,
                _ => draw_x1 + (draw_w - line_w) / 2.0,
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

/// Shrink a polygon inward by `inset` pixels (move each vertex toward the centroid).
/// Returns points suitable for `draw_polygon_mut`.
fn shrink_polygon(polygon: &[[f64; 2]], inset: f64) -> Vec<Point<i32>> {
    if polygon.is_empty() || inset <= 0.0 {
        return polygon.iter()
            .map(|p| Point::new(p[0] as i32, p[1] as i32))
            .collect();
    }

    // Compute centroid
    let n = polygon.len() as f64;
    let cx: f64 = polygon.iter().map(|p| p[0]).sum::<f64>() / n;
    let cy: f64 = polygon.iter().map(|p| p[1]).sum::<f64>() / n;

    // Move each vertex toward centroid by `inset` pixels
    polygon.iter().map(|p| {
        let dx = p[0] - cx;
        let dy = p[1] - cy;
        let dist = (dx * dx + dy * dy).sqrt();
        if dist < 1e-6 {
            Point::new(p[0] as i32, p[1] as i32)
        } else {
            let shrink = (inset / dist).min(0.9); // never collapse past 90%
            Point::new(
                (p[0] - dx * shrink) as i32,
                (p[1] - dy * shrink) as i32,
            )
        }
    }).collect()
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
            drawable_area: None,
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
