use ab_glyph::{Font, FontRef, PxScale, ScaleFont};
use image::{DynamicImage, Rgba, RgbaImage};
use imageproc::drawing::{draw_polygon_mut, draw_text_mut};
use imageproc::point::Point;

use crate::api::BubbleResult;

const FONT_BYTES: &[u8] = include_bytes!("../../assets/NotoSans-Medium.ttf");

/// Render translated text overlay on a manga/manhwa page.
///
/// For each bubble:
/// 1. Fill polygon with white (erase original text)
/// 2. Draw translated text centered in the bubble
///
/// Returns the composited RGBA image.
pub fn render(img: &DynamicImage, bubbles: &[BubbleResult]) -> RgbaImage {
    let mut canvas = img.to_rgba8();
    let font = FontRef::try_from_slice(FONT_BYTES).expect("Failed to parse embedded font");

    let white = Rgba([255u8, 255, 255, 255]);
    let black = Rgba([0u8, 0, 0, 255]);

    for bubble in bubbles {
        if bubble.translated_text.is_empty() {
            continue;
        }

        // 1. Erase: fill polygon with white
        let points: Vec<Point<i32>> = bubble.polygon.iter()
            .map(|p| Point::new(p[0] as i32, p[1] as i32))
            .collect();
        if points.len() >= 3 {
            draw_polygon_mut(&mut canvas, &points, white);
        }

        // 2. Draw translated text centered in bubble bbox
        let bbox = bounding_box(&bubble.polygon);
        draw_centered_text(
            &mut canvas,
            &bubble.translated_text,
            bubble.font_size_px,
            bubble.line_height,
            &bbox,
            &font,
            black,
        );
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

struct BBox {
    x: f64,
    y: f64,
    w: f64,
    h: f64,
}

fn bounding_box(polygon: &[[f64; 2]]) -> BBox {
    let (mut x1, mut y1) = (f64::INFINITY, f64::INFINITY);
    let (mut x2, mut y2) = (f64::NEG_INFINITY, f64::NEG_INFINITY);
    for p in polygon {
        x1 = x1.min(p[0]);
        y1 = y1.min(p[1]);
        x2 = x2.max(p[0]);
        y2 = y2.max(p[1]);
    }
    BBox { x: x1, y: y1, w: x2 - x1, h: y2 - y1 }
}

/// Draw multi-line text centered both horizontally and vertically within a bbox.
fn draw_centered_text(
    canvas: &mut RgbaImage,
    text: &str,
    font_size_px: u32,
    line_height: f64,
    bbox: &BBox,
    font: &FontRef<'_>,
    color: Rgba<u8>,
) {
    let scale = PxScale::from(font_size_px as f32);
    let scaled = font.as_scaled(scale);
    let line_spacing = font_size_px as f64 * line_height;

    let lines: Vec<&str> = text.lines().collect();
    let total_text_h = lines.len() as f64 * line_spacing;

    // Vertical centering: start Y so text block is centered in bbox
    let start_y = bbox.y + (bbox.h - total_text_h) / 2.0;

    for (i, line) in lines.iter().enumerate() {
        // Measure line width using font metrics
        let line_w = measure_width(line, &scaled, font);

        // Horizontal centering
        let x = bbox.x + (bbox.w - line_w as f64) / 2.0;
        let y = start_y + i as f64 * line_spacing;

        draw_text_mut(canvas, color, x as i32, y as i32, scale, font, line);
    }
}

fn measure_width(text: &str, scaled: &ab_glyph::PxScaleFont<&FontRef<'_>>, font: &FontRef<'_>) -> f32 {
    let mut width = 0.0f32;
    let mut prev: Option<ab_glyph::GlyphId> = None;
    for ch in text.chars() {
        let glyph_id = font.glyph_id(ch);
        if let Some(p) = prev {
            width += scaled.kern(p, glyph_id);
        }
        width += scaled.h_advance(glyph_id);
        prev = Some(glyph_id);
    }
    width
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
        };
        let result = render(&img, &[bubble]);
        assert_eq!(result.width(), 400);

        // Check that the polygon area was filled white
        let px = result.get_pixel(200, 150);
        assert_eq!(px, &Rgba([255, 255, 255, 255]), "Center should be white (erased)");
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
