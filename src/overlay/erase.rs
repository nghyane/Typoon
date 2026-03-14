use image::{DynamicImage, GrayImage, Luma, Rgba, RgbaImage};

use crate::detection::LocalTextMask;
use crate::inpaint::LamaInpainter;

/// Per-channel standard deviation threshold for flat background detection.
/// Below this → median fill (fast). Above → LaMa inpainting (neural).
const FLAT_BG_STDDEV_THRESHOLD: f64 = 12.0;

/// Minimum number of background samples required for variance analysis.
/// Below this we assume flat (not enough data to judge).
const MIN_BG_SAMPLES: usize = 4;

/// Erase original text from masked regions of a page image.
///
/// For each mask:
/// - Flat background (low variance) → fast median-color fill
/// - Complex background (screentone, gradients) → LaMa neural inpainting if available
/// - LaMa failure/unavailable → falls back to median fill
///
/// Returns the erased RGBA canvas.
pub fn erase_masks(
    canvas: &mut RgbaImage,
    masks: &[&LocalTextMask],
    inpainter: Option<&LamaInpainter>,
) {
    let mut lama_masks: Vec<&LocalTextMask> = Vec::new();

    for mask in masks {
        if is_flat_background(canvas, mask) {
            erase_with_median(canvas, mask);
        } else {
            lama_masks.push(mask);
        }
    }

    if lama_masks.is_empty() {
        return;
    }

    if let Some(lama) = inpainter {
        // Clone canvas once — each per-mask LaMa call uses a small ROI internally
        let base = DynamicImage::ImageRgba8(canvas.clone());
        for mask in &lama_masks {
            let page_mask = page_mask_from_local(mask, canvas.width(), canvas.height());
            match lama.inpaint(&base, &page_mask) {
                Ok(inpainted) => {
                    let inpainted_rgba = inpainted.to_rgba8();
                    apply_mask_pixels(canvas, mask, |_, (px, py)| {
                        *inpainted_rgba.get_pixel(px, py)
                    });
                }
                Err(e) => {
                    tracing::warn!("LaMa inpaint failed for mask, falling back to median: {e}");
                    erase_with_median(canvas, mask);
                }
            }
        }
    } else {
        for mask in &lama_masks {
            erase_with_median(canvas, mask);
        }
    }
}

/// Erase text pixels using median background color (fast path for flat backgrounds).
pub fn erase_with_median(canvas: &mut RgbaImage, text_mask: &LocalTextMask) {
    let mw = text_mask.image.width();
    let mh = text_mask.image.height();
    let bg = median_bg_color(canvas, text_mask.x, text_mask.y, mw, mh);
    apply_mask_pixels(canvas, text_mask, |_, _| bg);
}

/// Check if the background around a mask region is flat (low color variance).
///
/// Samples all non-masked pixels (mask == 0) at every 3rd pixel.
/// Returns true if:
/// - Fewer than MIN_BG_SAMPLES non-masked samples (not enough data → assume flat)
/// - Per-channel stddev of sampled pixels is below threshold
pub fn is_flat_background(img: &RgbaImage, text_mask: &LocalTextMask) -> bool {
    let (x, y) = (text_mask.x, text_mask.y);
    let (w, h) = (text_mask.image.width(), text_mask.image.height());
    let mut rs = Vec::new();
    let mut gs = Vec::new();
    let mut bs = Vec::new();

    for ly in (0..h).step_by(3) {
        for lx in (0..w).step_by(3) {
            if text_mask.image.get_pixel(lx, ly).0[0] == 255 {
                continue;
            }
            let sx = x + lx;
            let sy = y + ly;
            if sx < img.width() && sy < img.height() {
                let p = img.get_pixel(sx, sy);
                rs.push(p.0[0]);
                gs.push(p.0[1]);
                bs.push(p.0[2]);
            }
        }
    }

    if rs.len() < MIN_BG_SAMPLES {
        return true;
    }

    channel_stddev(&rs).max(channel_stddev(&gs)).max(channel_stddev(&bs)) < FLAT_BG_STDDEV_THRESHOLD
}

/// Sample the median background color inside a bbox region.
pub fn median_bg_color(img: &RgbaImage, x: u32, y: u32, w: u32, h: u32) -> Rgba<u8> {
    let mut rs = Vec::new();
    let mut gs = Vec::new();
    let mut bs = Vec::new();
    for sy in (y..y + h).step_by(4) {
        for sx in (x..x + w).step_by(4) {
            if sx < img.width() && sy < img.height() {
                let p = img.get_pixel(sx, sy);
                rs.push(p.0[0]);
                gs.push(p.0[1]);
                bs.push(p.0[2]);
            }
        }
    }
    if rs.is_empty() {
        return Rgba([255, 255, 255, 255]);
    }
    rs.sort_unstable();
    gs.sort_unstable();
    bs.sort_unstable();
    let mid = rs.len() / 2;
    Rgba([rs[mid], gs[mid], bs[mid], 255])
}

/// Convert a LocalTextMask to a full-page GrayImage for LaMa inpainting.
pub fn page_mask_from_local(text_mask: &LocalTextMask, page_w: u32, page_h: u32) -> GrayImage {
    let mut full = GrayImage::new(page_w, page_h);
    let mw = text_mask.image.width();
    let mh = text_mask.image.height();
    for my in 0..mh {
        for mx in 0..mw {
            if text_mask.image.get_pixel(mx, my).0[0] == 255 {
                let px = text_mask.x + mx;
                let py = text_mask.y + my;
                if px < page_w && py < page_h {
                    full.put_pixel(px, py, Luma([255]));
                }
            }
        }
    }
    full
}

/// Iterate over mask pixels and apply a transform function.
pub fn apply_mask_pixels(
    canvas: &mut RgbaImage,
    mask: &LocalTextMask,
    f: impl Fn(Rgba<u8>, (u32, u32)) -> Rgba<u8>,
) {
    let (cw, ch) = (canvas.width(), canvas.height());
    for ly in 0..mask.image.height() {
        for lx in 0..mask.image.width() {
            if mask.image.get_pixel(lx, ly).0[0] == 255 {
                let px = mask.x + lx;
                let py = mask.y + ly;
                if px < cw && py < ch {
                    let orig = *canvas.get_pixel(px, py);
                    canvas.put_pixel(px, py, f(orig, (px, py)));
                }
            }
        }
    }
}

fn channel_stddev(vals: &[u8]) -> f64 {
    let mean = vals.iter().map(|&v| v as f64).sum::<f64>() / vals.len() as f64;
    let var = vals.iter().map(|&v| (v as f64 - mean).powi(2)).sum::<f64>() / vals.len() as f64;
    var.sqrt()
}

/// Weighted luminance of an RGBA pixel (ITU-R BT.601).
pub fn pixel_luminance(p: &Rgba<u8>) -> u32 {
    (p.0[0] as u32 * 299 + p.0[1] as u32 * 587 + p.0[2] as u32 * 114) / 1000
}

#[cfg(test)]
mod tests {
    use super::*;
    use image::RgbaImage;

    #[test]
    fn test_is_flat_white_background() {
        let img = RgbaImage::from_pixel(100, 80, Rgba([255, 255, 255, 255]));
        let mask = LocalTextMask {
            x: 10,
            y: 10,
            image: GrayImage::new(50, 40),
        };
        assert!(is_flat_background(&img, &mask));
    }

    #[test]
    fn test_is_flat_noisy_background() {
        let mut img = RgbaImage::from_pixel(100, 80, Rgba([255, 255, 255, 255]));
        // Alternate between colors that differ enough to produce high per-channel stddev (>12)
        for y in 0..80 {
            for x in 0..100 {
                let c = if (x + y) % 2 == 0 {
                    Rgba([255, 200, 200, 255])
                } else {
                    Rgba([200, 255, 200, 255])
                };
                img.put_pixel(x, y, c);
            }
        }
        let mask = LocalTextMask {
            x: 10,
            y: 10,
            image: GrayImage::new(50, 40),
        };
        assert!(!is_flat_background(&img, &mask));
    }

    #[test]
    fn test_median_bg_color_white() {
        let img = RgbaImage::from_pixel(100, 80, Rgba([255, 255, 255, 255]));
        let bg = median_bg_color(&img, 0, 0, 100, 80);
        assert_eq!(bg, Rgba([255, 255, 255, 255]));
    }

    #[test]
    fn test_median_bg_color_dark() {
        // All-black image → median is black
        let img = RgbaImage::from_pixel(100, 80, Rgba([0, 0, 0, 255]));
        let bg = median_bg_color(&img, 0, 0, 100, 80);
        assert_eq!(bg, Rgba([0, 0, 0, 255]));
    }

    #[test]
    fn test_is_flat_dark_background() {
        // Uniform dark background → flat
        let img = RgbaImage::from_pixel(100, 80, Rgba([30, 30, 40, 255]));
        let mask = LocalTextMask {
            x: 10,
            y: 10,
            image: GrayImage::new(50, 40),
        };
        assert!(is_flat_background(&img, &mask));
    }

    #[test]
    fn test_is_not_flat_dark_gradient() {
        // Dark gradient (top dark, bottom lighter) → not flat
        let mut img = RgbaImage::from_pixel(100, 80, Rgba([0, 0, 0, 255]));
        for y in 0..80 {
            let v = (y as f64 / 79.0 * 120.0) as u8;
            for x in 0..100 {
                img.put_pixel(x, y, Rgba([v, v, v, 255]));
            }
        }
        let mask = LocalTextMask {
            x: 10,
            y: 10,
            image: GrayImage::new(50, 40),
        };
        assert!(!is_flat_background(&img, &mask));
    }

    #[test]
    fn test_page_mask_from_local() {
        let mut mask_img = GrayImage::new(10, 10);
        mask_img.put_pixel(5, 5, Luma([255]));
        let mask = LocalTextMask { x: 20, y: 30, image: mask_img };
        let full = page_mask_from_local(&mask, 100, 100);
        assert_eq!(full.get_pixel(25, 35).0[0], 255);
        assert_eq!(full.get_pixel(0, 0).0[0], 0);
    }

}
