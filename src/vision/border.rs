use image::{DynamicImage, GenericImageView};

use crate::render::layout;

/// Maximum scan depth from bbox edge (px)
const MAX_SCAN_DEPTH: u32 = 20;
/// Minimum number of edge samples to consider valid
const MIN_SAMPLES: usize = 4;
/// Luminance threshold: below this = "dark" (border pixel)
const DARK_THRESHOLD: u8 = 140;
/// Center region must be brighter than this to qualify as a bordered bubble
const CENTER_LIGHT_THRESHOLD: u8 = 180;
/// Fraction of samples that must agree for a confident border detection
const CONSENSUS_RATIO: f64 = 0.4;
/// Minimum inset (always leave at least this much padding)
const MIN_INSET: f64 = 2.0;
/// Maximum realistic border width (manga/manhwa borders are 1-8px)
const MAX_BORDER: f64 = 8.0;
/// Extra padding beyond detected border
const BORDER_PAD: f64 = 1.0;

/// Detect the border thickness of a bubble by scanning inward from its bbox edges.
///
/// Returns the inset in pixels: the distance from bbox edge to the inner content area.
/// Scans all 4 edges, takes the median of detected border widths.
pub fn detect_inset(img: &DynamicImage, polygon: &[[f64; 2]]) -> f64 {
    let (x1, y1, x2, y2) = layout::polygon_bbox(polygon);
    let (img_w, img_h) = img.dimensions();

    // Clamp bbox to image bounds
    let bx1 = (x1 as u32).min(img_w.saturating_sub(1));
    let by1 = (y1 as u32).min(img_h.saturating_sub(1));
    let bx2 = (x2 as u32).min(img_w.saturating_sub(1));
    let by2 = (y2 as u32).min(img_h.saturating_sub(1));

    if bx2 <= bx1 + 4 || by2 <= by1 + 4 {
        return MIN_INSET;
    }

    // Guard: only run on light-interior regions (speech bubbles).
    // If the center is dark, this is likely a narration box on dark bg, SFX, etc.
    if !is_center_light(img, bx1, by1, bx2, by2) {
        return MIN_INSET;
    }

    let bbox_w = bx2 - bx1;
    let bbox_h = by2 - by1;
    let scan_depth = MAX_SCAN_DEPTH.min(bbox_w / 4).min(bbox_h / 4);
    if scan_depth < 2 {
        return MIN_INSET;
    }

    // Sample along each edge at ~8px intervals
    let step = 8u32;
    let mut edge_widths: Vec<f64> = Vec::new();

    // Top edge: scan downward
    for sx in (bx1 + 4..bx2.saturating_sub(4)).step_by(step as usize) {
        if let Some(w) = scan_edge(img, sx, by1, 0, 1, scan_depth) {
            edge_widths.push(w as f64);
        }
    }

    // Bottom edge: scan upward
    for sx in (bx1 + 4..bx2.saturating_sub(4)).step_by(step as usize) {
        if let Some(w) = scan_edge(img, sx, by2, 0, -1, scan_depth) {
            edge_widths.push(w as f64);
        }
    }

    // Left edge: scan rightward
    for sy in (by1 + 4..by2.saturating_sub(4)).step_by(step as usize) {
        if let Some(w) = scan_edge(img, bx1, sy, 1, 0, scan_depth) {
            edge_widths.push(w as f64);
        }
    }

    // Right edge: scan leftward
    for sy in (by1 + 4..by2.saturating_sub(4)).step_by(step as usize) {
        if let Some(w) = scan_edge(img, bx2, sy, -1, 0, scan_depth) {
            edge_widths.push(w as f64);
        }
    }

    if edge_widths.len() < MIN_SAMPLES {
        return MIN_INSET;
    }

    // Use the value at the CONSENSUS_RATIO percentile (robust against outliers)
    edge_widths.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let idx = ((edge_widths.len() as f64 * CONSENSUS_RATIO) as usize).min(edge_widths.len() - 1);
    let border_width = edge_widths[idx];

    (border_width + BORDER_PAD).clamp(MIN_INSET, MAX_BORDER + BORDER_PAD)
}

/// Check if the center region of a bbox is predominantly light (= white bubble interior).
/// Samples a 3×3 grid in the center 50% of the bbox.
fn is_center_light(img: &DynamicImage, bx1: u32, by1: u32, bx2: u32, by2: u32) -> bool {
    let cx = (bx1 + bx2) / 2;
    let cy = (by1 + by2) / 2;
    let qw = (bx2 - bx1) / 4;
    let qh = (by2 - by1) / 4;

    let mut light_count = 0u32;
    let mut total = 0u32;
    for dx in [-(qw as i32), 0, qw as i32] {
        for dy in [-(qh as i32), 0, qh as i32] {
            let sx = (cx as i32 + dx).max(0) as u32;
            let sy = (cy as i32 + dy).max(0) as u32;
            if sx < img.width() && sy < img.height() {
                total += 1;
                if pixel_luminance(img, sx, sy) >= CENTER_LIGHT_THRESHOLD {
                    light_count += 1;
                }
            }
        }
    }

    // At least half of center samples must be light
    total > 0 && light_count * 2 >= total
}

/// Scan from (start_x, start_y) in direction (dx, dy) for up to `depth` pixels.
/// Returns the number of consecutive dark pixels from the start (= border width),
/// or None if the edge pixel isn't dark (no border detected at this sample point).
fn scan_edge(
    img: &DynamicImage,
    start_x: u32,
    start_y: u32,
    dx: i32,
    dy: i32,
    depth: u32,
) -> Option<u32> {
    let (img_w, img_h) = img.dimensions();

    // First pixel should be dark (part of border)
    let first_lum = pixel_luminance(img, start_x, start_y);
    if first_lum > DARK_THRESHOLD {
        return Some(0); // No border at this sample — edge pixel is already light
    }

    let mut dark_count = 0u32;
    for step in 0..depth {
        let px = start_x as i32 + dx * step as i32;
        let py = start_y as i32 + dy * step as i32;
        if px < 0 || py < 0 || px >= img_w as i32 || py >= img_h as i32 {
            break;
        }
        let lum = pixel_luminance(img, px as u32, py as u32);
        if lum > DARK_THRESHOLD {
            break;
        }
        dark_count += 1;
    }

    Some(dark_count)
}

/// Get luminance (0-255) of a pixel using standard weights.
fn pixel_luminance(img: &DynamicImage, x: u32, y: u32) -> u8 {
    let [r, g, b, _] = img.get_pixel(x, y).0;
    // ITU-R BT.601 luma
    ((r as u32 * 299 + g as u32 * 587 + b as u32 * 114) / 1000) as u8
}

#[cfg(test)]
mod tests {
    use super::*;
    use image::{Rgba, RgbaImage};

    fn make_bordered_bubble(border_width: u32) -> DynamicImage {
        let w = 200u32;
        let h = 150u32;
        let mut img = RgbaImage::from_pixel(w, h, Rgba([255, 255, 255, 255]));

        let black = Rgba([0, 0, 0, 255]);
        // Draw border around the image edges
        for bw in 0..border_width {
            for x in 0..w {
                img.put_pixel(x, bw, black);
                img.put_pixel(x, h - 1 - bw, black);
            }
            for y in 0..h {
                img.put_pixel(bw, y, black);
                img.put_pixel(w - 1 - bw, y, black);
            }
        }

        DynamicImage::ImageRgba8(img)
    }

    #[test]
    fn test_no_border() {
        let img =
            DynamicImage::ImageRgba8(RgbaImage::from_pixel(200, 150, Rgba([255, 255, 255, 255])));
        let polygon = vec![[0.0, 0.0], [200.0, 0.0], [200.0, 150.0], [0.0, 150.0]];
        let inset = detect_inset(&img, &polygon);
        assert!(
            inset <= MIN_INSET + BORDER_PAD + 1.0,
            "No border: inset={inset}"
        );
    }

    #[test]
    fn test_thin_border() {
        let img = make_bordered_bubble(2);
        let polygon = vec![[0.0, 0.0], [200.0, 0.0], [200.0, 150.0], [0.0, 150.0]];
        let inset = detect_inset(&img, &polygon);
        assert!(inset >= 2.0, "2px border: inset={inset}");
        assert!(
            inset <= 6.0,
            "2px border should not produce huge inset: {inset}"
        );
    }

    #[test]
    fn test_thick_border() {
        let img = make_bordered_bubble(5);
        let polygon = vec![[0.0, 0.0], [200.0, 0.0], [200.0, 150.0], [0.0, 150.0]];
        let inset = detect_inset(&img, &polygon);
        assert!(inset >= 4.0, "5px border: inset={inset}");
        assert!(
            inset <= 10.0,
            "5px border should not produce huge inset: {inset}"
        );
    }

    #[test]
    fn test_tiny_bbox() {
        let img = DynamicImage::new_rgb8(10, 10);
        let polygon = vec![[0.0, 0.0], [5.0, 0.0], [5.0, 5.0], [0.0, 5.0]];
        let inset = detect_inset(&img, &polygon);
        assert_eq!(inset, MIN_INSET);
    }

    #[test]
    fn test_dark_background_skipped() {
        // Dark bg (narration box, SFX on dark area) → should NOT detect border
        let img =
            DynamicImage::ImageRgba8(RgbaImage::from_pixel(200, 150, Rgba([30, 30, 30, 255])));
        let polygon = vec![[0.0, 0.0], [200.0, 0.0], [200.0, 150.0], [0.0, 150.0]];
        let inset = detect_inset(&img, &polygon);
        assert_eq!(inset, MIN_INSET, "Dark bg should skip border detection");
    }

    #[test]
    fn test_colored_background_skipped() {
        // Mid-tone colored bg → center not light enough → skip
        let img =
            DynamicImage::ImageRgba8(RgbaImage::from_pixel(200, 150, Rgba([100, 80, 60, 255])));
        let polygon = vec![[0.0, 0.0], [200.0, 0.0], [200.0, 150.0], [0.0, 150.0]];
        let inset = detect_inset(&img, &polygon);
        assert_eq!(inset, MIN_INSET, "Colored bg should skip border detection");
    }

    #[test]
    fn test_border_capped() {
        // Artificial 15px border — should be capped to MAX_BORDER + BORDER_PAD
        let img = make_bordered_bubble(15);
        let polygon = vec![[0.0, 0.0], [200.0, 0.0], [200.0, 150.0], [0.0, 150.0]];
        let inset = detect_inset(&img, &polygon);
        assert!(
            inset <= MAX_BORDER + BORDER_PAD + 0.1,
            "Should be capped: inset={inset}, max={}",
            MAX_BORDER + BORDER_PAD
        );
    }
}
