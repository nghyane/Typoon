use anyhow::Result;
use image::{DynamicImage, GenericImageView, GrayImage, Rgb, RgbImage};
use ndarray::Array4;
use ort::value::TensorRef;

use crate::model_hub::lazy::LazySession;

const MODEL_SIZE: usize = 512;
const MODEL_SIZE_U32: u32 = MODEL_SIZE as u32;

/// Padding added around the mask bounding box to provide context for inpainting.
const CONTEXT_PAD: u32 = 32;

/// Overlap between adjacent tiles when the region exceeds 512×512.
const TILE_OVERLAP: u32 = 64;

pub struct LamaInpainter {
    session: LazySession,
}

impl LamaInpainter {
    pub fn new(session: LazySession) -> Self {
        Self { session }
    }

    pub fn is_loaded(&self) -> bool {
        self.session.is_loaded()
    }

    /// Inpaint masked regions of an image using LaMa FFC-ResNet.
    ///
    /// - `img`: original image
    /// - `mask`: binary mask in page coordinates (255 = inpaint, 0 = keep)
    ///
    /// Returns the full image with masked regions replaced by inpainted content.
    pub fn inpaint(&self, img: &DynamicImage, mask: &GrayImage) -> Result<DynamicImage> {
        let (img_w, img_h) = img.dimensions();
        let (mask_w, mask_h) = mask.dimensions();
        anyhow::ensure!(
            img_w == mask_w && img_h == mask_h,
            "Image ({img_w}x{img_h}) and mask ({mask_w}x{mask_h}) dimensions must match"
        );

        // Find bounding box of non-zero mask pixels
        let (bbox_x1, bbox_y1, bbox_x2, bbox_y2) = match mask_bbox(mask) {
            Some(bb) => bb,
            None => {
                tracing::debug!("Mask is empty, returning original image");
                return Ok(img.clone());
            }
        };

        // Expand bbox with context padding, clamped to image bounds
        let roi_x1 = bbox_x1.saturating_sub(CONTEXT_PAD);
        let roi_y1 = bbox_y1.saturating_sub(CONTEXT_PAD);
        let roi_x2 = (bbox_x2 + CONTEXT_PAD).min(img_w);
        let roi_y2 = (bbox_y2 + CONTEXT_PAD).min(img_h);
        let roi_w = roi_x2 - roi_x1;
        let roi_h = roi_y2 - roi_y1;

        tracing::debug!(
            "Inpaint ROI: ({roi_x1},{roi_y1})-({roi_x2},{roi_y2}) = {roi_w}x{roi_h}"
        );

        // Crop ROI from image and mask
        let roi_img = img.crop_imm(roi_x1, roi_y1, roi_w, roi_h).to_rgb8();
        let roi_mask = crop_gray(mask, roi_x1, roi_y1, roi_w, roi_h);

        // Inpaint the ROI (handles tiling if needed)
        let inpainted_roi = if roi_w <= MODEL_SIZE_U32 && roi_h <= MODEL_SIZE_U32 {
            self.inpaint_tile(&roi_img, &roi_mask)?
        } else {
            self.inpaint_tiled(&roi_img, &roi_mask)?
        };

        // Composite: replace only masked pixels in the original image
        let mut result = img.to_rgb8();
        for y in 0..roi_h {
            for x in 0..roi_w {
                if roi_mask.get_pixel(x, y).0[0] > 0 {
                    let px = roi_x1 + x;
                    let py = roi_y1 + y;
                    result.put_pixel(px, py, *inpainted_roi.get_pixel(x, y));
                }
            }
        }

        Ok(DynamicImage::ImageRgb8(result))
    }

    /// Inpaint a single tile that fits within 512×512.
    /// The input is padded to 512×512 if smaller.
    fn inpaint_tile(&self, img: &RgbImage, mask: &GrayImage) -> Result<RgbImage> {
        let (w, h) = img.dimensions();
        debug_assert!(w <= MODEL_SIZE_U32 && h <= MODEL_SIZE_U32);

        // Build padded 512×512 image tensor [1, 3, 512, 512] normalized to [0, 1]
        let mut img_arr = Array4::<f32>::zeros((1, 3, MODEL_SIZE, MODEL_SIZE));
        for y in 0..h as usize {
            for x in 0..w as usize {
                let pixel = img.get_pixel(x as u32, y as u32);
                img_arr[[0, 0, y, x]] = pixel[0] as f32 / 255.0;
                img_arr[[0, 1, y, x]] = pixel[1] as f32 / 255.0;
                img_arr[[0, 2, y, x]] = pixel[2] as f32 / 255.0;
            }
        }

        // Build padded 512×512 mask tensor [1, 1, 512, 512] with 0/1 values
        let mut mask_arr = Array4::<f32>::zeros((1, 1, MODEL_SIZE, MODEL_SIZE));
        for y in 0..h as usize {
            for x in 0..w as usize {
                if mask.get_pixel(x as u32, y as u32).0[0] > 0 {
                    mask_arr[[0, 0, y, x]] = 1.0;
                }
            }
        }

        // Run ort inference
        let img_ref = TensorRef::from_array_view(&img_arr)?;
        let mask_ref = TensorRef::from_array_view(&mask_arr)?;
        let session_mutex = self.session.get()
            .ok_or_else(|| anyhow::anyhow!("LaMa inpainter session not loaded"))?;
        let mut session = session_mutex.lock().unwrap();
        let outputs = session.run(ort::inputs![
            "image" => img_ref,
            "mask" => mask_ref,
        ])?;

        // Output is [1, 3, 512, 512] in [0, 1]
        let (out_shape, out_data) = outputs["output"].try_extract_tensor::<f32>()?;
        let plane = out_shape[2] as usize * out_shape[3] as usize;

        let mut result = RgbImage::new(w, h);
        for y in 0..h as usize {
            for x in 0..w as usize {
                let idx = y * MODEL_SIZE + x;
                let r = (out_data[idx] * 255.0).round().clamp(0.0, 255.0) as u8;
                let g = (out_data[plane + idx] * 255.0).round().clamp(0.0, 255.0) as u8;
                let b = (out_data[2 * plane + idx] * 255.0).round().clamp(0.0, 255.0) as u8;
                result.put_pixel(x as u32, y as u32, Rgb([r, g, b]));
            }
        }

        Ok(result)
    }

    /// Inpaint a region larger than 512×512 by tiling with overlap.
    /// Each tile is 512×512 with `TILE_OVERLAP` pixel overlap.
    /// Overlapping regions are blended with linear weights.
    fn inpaint_tiled(&self, img: &RgbImage, mask: &GrayImage) -> Result<RgbImage> {
        let (w, h) = img.dimensions();
        let step = MODEL_SIZE_U32 - TILE_OVERLAP;

        // Accumulate weighted RGB + weight map for blending
        let mut accum_r = vec![0.0f64; (w * h) as usize];
        let mut accum_g = vec![0.0f64; (w * h) as usize];
        let mut accum_b = vec![0.0f64; (w * h) as usize];
        let mut weight_map = vec![0.0f64; (w * h) as usize];

        let mut ty = 0u32;
        while ty < h {
            let tile_y = ty.min(h.saturating_sub(MODEL_SIZE_U32));
            let mut tx = 0u32;
            while tx < w {
                let tile_x = tx.min(w.saturating_sub(MODEL_SIZE_U32));
                let tile_w = MODEL_SIZE_U32.min(w - tile_x);
                let tile_h = MODEL_SIZE_U32.min(h - tile_y);

                // Check if this tile has any masked pixels — skip if not
                if !has_mask_pixels(mask, tile_x, tile_y, tile_w, tile_h) {
                    tx += step;
                    continue;
                }

                let tile_img = crop_rgb(img, tile_x, tile_y, tile_w, tile_h);
                let tile_mask = crop_gray(mask, tile_x, tile_y, tile_w, tile_h);

                let inpainted = self.inpaint_tile(&tile_img, &tile_mask)?;

                // Blend into accumulator with linear ramp weights at edges
                for ly in 0..tile_h {
                    for lx in 0..tile_w {
                        let gx = tile_x + lx;
                        let gy = tile_y + ly;
                        let idx = (gy * w + gx) as usize;

                        let wt = edge_weight(lx, ly, tile_w, tile_h, TILE_OVERLAP);
                        let px = inpainted.get_pixel(lx, ly);
                        accum_r[idx] += px[0] as f64 * wt;
                        accum_g[idx] += px[1] as f64 * wt;
                        accum_b[idx] += px[2] as f64 * wt;
                        weight_map[idx] += wt;
                    }
                }

                tx += step;
            }
            ty += step;
        }

        // Build final image: use blended result where weight > 0, original elsewhere
        let mut result = img.clone();
        for y in 0..h {
            for x in 0..w {
                let idx = (y * w + x) as usize;
                if weight_map[idx] > 0.0 {
                    let r = (accum_r[idx] / weight_map[idx]).round().clamp(0.0, 255.0) as u8;
                    let g = (accum_g[idx] / weight_map[idx]).round().clamp(0.0, 255.0) as u8;
                    let b = (accum_b[idx] / weight_map[idx]).round().clamp(0.0, 255.0) as u8;
                    result.put_pixel(x, y, Rgb([r, g, b]));
                }
            }
        }

        Ok(result)
    }
}

/// Find the axis-aligned bounding box of non-zero pixels in a grayscale mask.
/// Returns `(x1, y1, x2, y2)` where x2/y2 are exclusive.
fn mask_bbox(mask: &GrayImage) -> Option<(u32, u32, u32, u32)> {
    let (w, h) = mask.dimensions();
    let mut x1 = w;
    let mut y1 = h;
    let mut x2 = 0u32;
    let mut y2 = 0u32;

    for y in 0..h {
        for x in 0..w {
            if mask.get_pixel(x, y).0[0] > 0 {
                x1 = x1.min(x);
                y1 = y1.min(y);
                x2 = x2.max(x + 1);
                y2 = y2.max(y + 1);
            }
        }
    }

    if x2 > x1 && y2 > y1 {
        Some((x1, y1, x2, y2))
    } else {
        None
    }
}

/// Crop a sub-region from a GrayImage.
fn crop_gray(img: &GrayImage, x: u32, y: u32, w: u32, h: u32) -> GrayImage {
    let mut out = GrayImage::new(w, h);
    for ly in 0..h {
        for lx in 0..w {
            let sx = (x + lx).min(img.width() - 1);
            let sy = (y + ly).min(img.height() - 1);
            out.put_pixel(lx, ly, *img.get_pixel(sx, sy));
        }
    }
    out
}

/// Crop a sub-region from an RgbImage.
fn crop_rgb(img: &RgbImage, x: u32, y: u32, w: u32, h: u32) -> RgbImage {
    let mut out = RgbImage::new(w, h);
    for ly in 0..h {
        for lx in 0..w {
            let sx = (x + lx).min(img.width() - 1);
            let sy = (y + ly).min(img.height() - 1);
            out.put_pixel(lx, ly, *img.get_pixel(sx, sy));
        }
    }
    out
}

/// Check if any mask pixels in the given region are non-zero.
fn has_mask_pixels(mask: &GrayImage, x: u32, y: u32, w: u32, h: u32) -> bool {
    for ly in 0..h {
        for lx in 0..w {
            let sx = (x + lx).min(mask.width() - 1);
            let sy = (y + ly).min(mask.height() - 1);
            if mask.get_pixel(sx, sy).0[0] > 0 {
                return true;
            }
        }
    }
    false
}

/// Linear ramp blending weight: 1.0 in the interior, ramps down to 0.0
/// over `overlap` pixels at each edge. Prevents seam artifacts between tiles.
fn edge_weight(x: u32, y: u32, w: u32, h: u32, overlap: u32) -> f64 {
    let ramp = |pos: u32, size: u32| -> f64 {
        let d_start = pos as f64;
        let d_end = (size - 1 - pos) as f64;
        let d = d_start.min(d_end);
        (d / overlap as f64).min(1.0)
    };
    ramp(x, w) * ramp(y, h)
}

#[cfg(test)]
mod tests {
    use super::*;
    use image::Luma;

    #[test]
    fn test_mask_bbox_empty() {
        let mask = GrayImage::new(100, 100);
        assert!(mask_bbox(&mask).is_none());
    }

    #[test]
    fn test_mask_bbox_single_pixel() {
        let mut mask = GrayImage::new(100, 100);
        mask.put_pixel(50, 60, Luma([255]));
        assert_eq!(mask_bbox(&mask), Some((50, 60, 51, 61)));
    }

    #[test]
    fn test_mask_bbox_region() {
        let mut mask = GrayImage::new(200, 200);
        for y in 30..80 {
            for x in 40..120 {
                mask.put_pixel(x, y, Luma([255]));
            }
        }
        assert_eq!(mask_bbox(&mask), Some((40, 30, 120, 80)));
    }

    #[test]
    fn test_edge_weight_center() {
        let w = edge_weight(256, 256, 512, 512, 64);
        assert!((w - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_edge_weight_corner() {
        let w = edge_weight(0, 0, 512, 512, 64);
        assert!(w < 0.01);
    }

    #[test]
    fn test_edge_weight_edge() {
        let w = edge_weight(32, 256, 512, 512, 64);
        assert!((w - 0.5).abs() < 0.02);
    }

    #[test]
    fn test_crop_gray() {
        let mut img = GrayImage::new(100, 100);
        img.put_pixel(50, 50, Luma([200]));
        let cropped = crop_gray(&img, 40, 40, 20, 20);
        assert_eq!(cropped.dimensions(), (20, 20));
        assert_eq!(cropped.get_pixel(10, 10).0[0], 200);
    }

    #[test]
    fn test_has_mask_pixels_empty() {
        let mask = GrayImage::new(100, 100);
        assert!(!has_mask_pixels(&mask, 0, 0, 100, 100));
    }

    #[test]
    fn test_has_mask_pixels_present() {
        let mut mask = GrayImage::new(100, 100);
        mask.put_pixel(50, 50, Luma([255]));
        assert!(has_mask_pixels(&mask, 40, 40, 20, 20));
        assert!(!has_mask_pixels(&mask, 0, 0, 10, 10));
    }
}
