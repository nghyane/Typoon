use image::{DynamicImage, GrayImage, Luma, Rgba, RgbaImage};

use crate::detection::LocalTextMask;
use crate::inpaint::LamaInpainter;

/// Erase original text from masked regions of a page image.
///
/// All masks are merged into one page-level mask, then passed to LaMa
/// which splits sparse regions into independent tight ROIs internally.
/// Falls back to median-color fill when LaMa is unavailable or fails.
pub fn erase_masks(
    canvas: &mut RgbaImage,
    masks: &[&LocalTextMask],
    inpainter: Option<&LamaInpainter>,
) {
    let t0 = std::time::Instant::now();

    if let Some(lama) = inpainter {
        let (pw, ph) = (canvas.width(), canvas.height());
        let merged = merge_masks_to_page(masks, pw, ph);
        let base = DynamicImage::ImageRgba8(canvas.clone());
        match lama.inpaint(&base, &merged) {
            Ok(inpainted) => {
                let inpainted_rgba = inpainted.to_rgba8();
                for mask in masks {
                    apply_mask_pixels(canvas, mask, |_, (px, py)| {
                        *inpainted_rgba.get_pixel(px, py)
                    });
                }
                tracing::info!(
                    "Erase: {} masks via LaMa in {:.0?}",
                    masks.len(),
                    t0.elapsed()
                );
                return;
            }
            Err(e) => {
                tracing::warn!("LaMa inpaint failed, falling back to median: {e}");
            }
        }
    }

    for mask in masks {
        erase_with_median(canvas, mask);
    }
    tracing::info!(
        "Erase: {} masks via median in {:.0?}",
        masks.len(),
        t0.elapsed()
    );
}

/// Merge multiple local text masks into a single page-level binary mask.
fn merge_masks_to_page(masks: &[&LocalTextMask], page_w: u32, page_h: u32) -> GrayImage {
    let mut full = GrayImage::new(page_w, page_h);
    for mask in masks {
        for my in 0..mask.image.height() {
            for mx in 0..mask.image.width() {
                if mask.image.get_pixel(mx, my).0[0] == 255 {
                    let px = mask.x + mx;
                    let py = mask.y + my;
                    if px < page_w && py < page_h {
                        full.put_pixel(px, py, Luma([255]));
                    }
                }
            }
        }
    }
    full
}

/// Erase text pixels using median background color (fallback when LaMa unavailable).
pub fn erase_with_median(canvas: &mut RgbaImage, text_mask: &LocalTextMask) {
    let bg = median_bg_color(
        canvas,
        text_mask.x,
        text_mask.y,
        text_mask.image.width(),
        text_mask.image.height(),
    );
    apply_mask_pixels(canvas, text_mask, |_, _| bg);
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

/// Weighted luminance of an RGBA pixel (ITU-R BT.601).
pub fn pixel_luminance(p: &Rgba<u8>) -> u32 {
    (p.0[0] as u32 * 299 + p.0[1] as u32 * 587 + p.0[2] as u32 * 114) / 1000
}

#[cfg(test)]
mod tests {
    use super::*;
    use image::RgbaImage;

    #[test]
    fn test_median_bg_color_white() {
        let img = RgbaImage::from_pixel(100, 80, Rgba([255, 255, 255, 255]));
        assert_eq!(
            median_bg_color(&img, 0, 0, 100, 80),
            Rgba([255, 255, 255, 255])
        );
    }

    #[test]
    fn test_median_bg_color_dark() {
        let img = RgbaImage::from_pixel(100, 80, Rgba([0, 0, 0, 255]));
        assert_eq!(median_bg_color(&img, 0, 0, 100, 80), Rgba([0, 0, 0, 255]));
    }

    #[test]
    fn test_page_mask_from_local() {
        let mut mask_img = GrayImage::new(10, 10);
        mask_img.put_pixel(5, 5, Luma([255]));
        let mask = LocalTextMask {
            x: 20,
            y: 30,
            image: mask_img,
        };
        let full = page_mask_from_local(&mask, 100, 100);
        assert_eq!(full.get_pixel(25, 35).0[0], 255);
        assert_eq!(full.get_pixel(0, 0).0[0], 0);
    }
}
