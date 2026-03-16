use anyhow::Result;
use image::{DynamicImage, GenericImageView, GrayImage, Rgb, RgbImage};
use ndarray::Array4;
use ort::value::TensorRef;

use crate::model_hub::lazy::LazySession;

const MODEL_SIZE: usize = 512;
const MODEL_SIZE_U32: u32 = MODEL_SIZE as u32;

/// Overlap between adjacent tiles when the region exceeds 512×512.
const TILE_OVERLAP: u32 = 64;

/// Background luminance range (P1–P99) below which a tile is "flat"
/// and can be filled with median color instead of running LaMa (~280ms saved).
const FLAT_TILE_RANGE: u8 = 8;
const INV_255: f32 = 1.0 / 255.0;

/// Single LaMa inpainter backed by one ONNX session.
///
/// `Session::run()` requires `&mut self`, so each instance is single-threaded.
/// For parallel page rendering, the pipeline should create a pool of inpainters
/// and distribute pages across them.
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
    /// Tiles the full page with 512×512 windows (stride 448), skipping tiles
    /// without mask pixels. Nearby bubbles naturally share tiles, minimizing
    /// the total number of inference calls.
    pub fn inpaint(&self, img: &DynamicImage, mask: &GrayImage) -> Result<DynamicImage> {
        let (img_w, img_h) = img.dimensions();
        let (mask_w, mask_h) = mask.dimensions();
        anyhow::ensure!(
            img_w == mask_w && img_h == mask_h,
            "Image ({img_w}x{img_h}) and mask ({mask_w}x{mask_h}) dimensions must match"
        );

        if !has_mask_pixels(mask, 0, 0, mask_w, mask_h) {
            return Ok(img.clone());
        }

        let rgb = img.to_rgb8();
        let mut result = rgb.clone();
        let mut tile_count = 0u32;
        let x_starts = tile_starts(img_w);
        let y_starts = tile_starts(img_h);

        for &tile_y in &y_starts {
            for &tile_x in &x_starts {
                let tile_w = MODEL_SIZE_U32.min(img_w - tile_x);
                let tile_h = MODEL_SIZE_U32.min(img_h - tile_y);

                if !has_mask_pixels(mask, tile_x, tile_y, tile_w, tile_h) {
                    continue;
                }

                let tile_mask = crop_gray(mask, tile_x, tile_y, tile_w, tile_h);

                if is_flat_tile(&rgb, &tile_mask, tile_x, tile_y) {
                    // Flat background — median fill, skip LaMa
                    let bg = median_bg(&rgb, &tile_mask, tile_x, tile_y);
                    for ly in 0..tile_h {
                        for lx in 0..tile_w {
                            if tile_mask.get_pixel(lx, ly).0[0] > 0 {
                                result.put_pixel(tile_x + lx, tile_y + ly, bg);
                            }
                        }
                    }
                } else {
                    let tile_img = crop_rgb(&rgb, tile_x, tile_y, tile_w, tile_h);
                    let inpainted = self.inpaint_tile(&tile_img, &tile_mask)?;
                    for ly in 0..tile_h {
                        for lx in 0..tile_w {
                            if tile_mask.get_pixel(lx, ly).0[0] > 0 {
                                result.put_pixel(
                                    tile_x + lx,
                                    tile_y + ly,
                                    *inpainted.get_pixel(lx, ly),
                                );
                            }
                        }
                    }
                    tile_count += 1;
                }
            }
        }

        tracing::debug!("Inpaint: {tile_count} tiles on {img_w}x{img_h} page");
        Ok(DynamicImage::ImageRgb8(result))
    }

    /// Inpaint a single tile that fits within 512×512.
    /// The input is padded to 512×512 if smaller.
    fn inpaint_tile(&self, img: &RgbImage, mask: &GrayImage) -> Result<RgbImage> {
        let (w, h) = img.dimensions();
        debug_assert!(w <= MODEL_SIZE_U32 && h <= MODEL_SIZE_U32);
        let w = w as usize;
        let h = h as usize;
        let model_plane = MODEL_SIZE * MODEL_SIZE;

        // Build padded 512×512 image tensor [1, 3, 512, 512] normalized to [0, 1]
        let mut img_arr = Array4::<f32>::zeros((1, 3, MODEL_SIZE, MODEL_SIZE));
        {
            let src = img.as_raw();
            let buf = img_arr
                .as_slice_mut()
                .ok_or_else(|| anyhow::anyhow!("LaMa image tensor must be contiguous"))?;
            let (r_plane, gb_plane) = buf.split_at_mut(model_plane);
            let (g_plane, b_plane) = gb_plane.split_at_mut(model_plane);
            for y in 0..h {
                let src_row = y * w * 3;
                let dst_row = y * MODEL_SIZE;
                for x in 0..w {
                    let src_idx = src_row + x * 3;
                    let dst_idx = dst_row + x;
                    r_plane[dst_idx] = src[src_idx] as f32 * INV_255;
                    g_plane[dst_idx] = src[src_idx + 1] as f32 * INV_255;
                    b_plane[dst_idx] = src[src_idx + 2] as f32 * INV_255;
                }
            }
        }

        // Build padded 512×512 mask tensor [1, 1, 512, 512] with 0/1 values
        let mut mask_arr = Array4::<f32>::zeros((1, 1, MODEL_SIZE, MODEL_SIZE));
        {
            let src = mask.as_raw();
            let plane = &mut mask_arr
                .as_slice_mut()
                .ok_or_else(|| anyhow::anyhow!("LaMa mask tensor must be contiguous"))?
                [..model_plane];
            for y in 0..h {
                let src_row = y * w;
                let dst_row = y * MODEL_SIZE;
                let src_line = &src[src_row..src_row + w];
                let dst_line = &mut plane[dst_row..dst_row + w];
                for (dst, &value) in dst_line.iter_mut().zip(src_line.iter()) {
                    *dst = if value > 0 { 1.0 } else { 0.0 };
                }
            }
        }

        // Run ort inference
        let img_ref = TensorRef::from_array_view(&img_arr)?;
        let mask_ref = TensorRef::from_array_view(&mask_arr)?;
        let session = self
            .session
            .get()
            .ok_or_else(|| anyhow::anyhow!("LaMa inpainter session not loaded"))?;
        let outputs = session.run(ort::inputs![
            "image" => img_ref,
            "mask" => mask_ref,
        ])?;

        // Output is [1, 3, 512, 512] in [0, 1]
        let (out_shape, out_data) = outputs["output"].try_extract_tensor::<f32>()?;
        anyhow::ensure!(
            out_shape.len() == 4,
            "Unexpected LaMa output rank: {:?}",
            out_shape
        );
        let out_h = out_shape[2] as usize;
        let out_w = out_shape[3] as usize;
        anyhow::ensure!(
            out_h >= h && out_w >= w,
            "Unexpected LaMa output size: {:?}, expected at least {}x{}",
            out_shape,
            w,
            h
        );
        let out_plane = out_h * out_w;

        let mut result_raw = vec![0u8; w * h * 3];
        for y in 0..h {
            let src_row = y * out_w;
            let dst_row = y * w * 3;
            for x in 0..w {
                let src_idx = src_row + x;
                let dst_idx = dst_row + x * 3;
                result_raw[dst_idx] = unit_f32_to_u8(out_data[src_idx]);
                result_raw[dst_idx + 1] = unit_f32_to_u8(out_data[out_plane + src_idx]);
                result_raw[dst_idx + 2] = unit_f32_to_u8(out_data[2 * out_plane + src_idx]);
            }
        }

        let result = RgbImage::from_raw(w as u32, h as u32, result_raw)
            .ok_or_else(|| anyhow::anyhow!("Failed to build LaMa output image"))?;

        Ok(result)
    }
}

#[inline]
fn unit_f32_to_u8(value: f32) -> u8 {
    (value * 255.0).round().clamp(0.0, 255.0) as u8
}

/// Generate non-duplicated tile origins that fully cover a 1D axis.
fn tile_starts(length: u32) -> Vec<u32> {
    if length <= MODEL_SIZE_U32 {
        return vec![0];
    }

    let step = MODEL_SIZE_U32 - TILE_OVERLAP;
    let last = length - MODEL_SIZE_U32;
    let mut starts = Vec::new();
    let mut pos = 0u32;
    while pos < last {
        starts.push(pos);
        pos = pos.saturating_add(step);
    }
    starts.push(last);
    starts
}

/// Crop a sub-region from a GrayImage.
fn crop_gray(img: &GrayImage, x: u32, y: u32, w: u32, h: u32) -> GrayImage {
    let mut out = GrayImage::new(w, h);
    let src_w = img.width() as usize;
    let x = x as usize;
    let y = y as usize;
    let w = w as usize;
    let h = h as usize;
    debug_assert!(x + w <= src_w);
    debug_assert!(y + h <= img.height() as usize);

    let src = img.as_raw();
    let dst = out.as_mut();
    for row in 0..h {
        let src_start = (y + row) * src_w + x;
        let dst_start = row * w;
        dst[dst_start..dst_start + w].copy_from_slice(&src[src_start..src_start + w]);
    }
    out
}

/// Crop a sub-region from an RgbImage.
fn crop_rgb(img: &RgbImage, x: u32, y: u32, w: u32, h: u32) -> RgbImage {
    let mut out = RgbImage::new(w, h);
    let src_w = img.width() as usize;
    let x = x as usize;
    let y = y as usize;
    let w = w as usize;
    let h = h as usize;
    let src_stride = src_w * 3;
    let dst_stride = w * 3;
    debug_assert!(x + w <= src_w);
    debug_assert!(y + h <= img.height() as usize);

    let src = img.as_raw();
    let dst = out.as_mut();
    for row in 0..h {
        let src_start = (y + row) * src_stride + x * 3;
        let dst_start = row * dst_stride;
        dst[dst_start..dst_start + dst_stride]
            .copy_from_slice(&src[src_start..src_start + dst_stride]);
    }
    out
}

/// Check if a tile's background is flat (uniform color) by sampling non-masked
/// pixels and checking their luminance range.
fn is_flat_tile(img: &RgbImage, tile_mask: &GrayImage, ox: u32, oy: u32) -> bool {
    let (tw, th) = tile_mask.dimensions();
    let (iw, ih) = img.dimensions();
    let mut lums: Vec<u8> = Vec::new();

    for ly in (0..th).step_by(4) {
        for lx in (0..tw).step_by(4) {
            if tile_mask.get_pixel(lx, ly).0[0] > 0 {
                continue;
            }
            let sx = (ox + lx).min(iw - 1);
            let sy = (oy + ly).min(ih - 1);
            let p = img.get_pixel(sx, sy);
            lums.push(((p[0] as u32 * 299 + p[1] as u32 * 587 + p[2] as u32 * 114) / 1000) as u8);
        }
    }
    if lums.len() < 16 {
        return true;
    }
    lums.sort_unstable();
    let n = lums.len();
    (lums[n * 99 / 100] - lums[n / 100]) <= FLAT_TILE_RANGE
}

/// Compute median background color from non-masked pixels in a tile.
fn median_bg(img: &RgbImage, tile_mask: &GrayImage, ox: u32, oy: u32) -> Rgb<u8> {
    let (tw, th) = tile_mask.dimensions();
    let (iw, ih) = img.dimensions();
    let mut rs = Vec::new();
    let mut gs = Vec::new();
    let mut bs = Vec::new();

    for ly in (0..th).step_by(4) {
        for lx in (0..tw).step_by(4) {
            if tile_mask.get_pixel(lx, ly).0[0] > 0 {
                continue;
            }
            let sx = (ox + lx).min(iw - 1);
            let sy = (oy + ly).min(ih - 1);
            let p = img.get_pixel(sx, sy);
            rs.push(p[0]);
            gs.push(p[1]);
            bs.push(p[2]);
        }
    }
    if rs.is_empty() {
        return Rgb([255, 255, 255]);
    }
    rs.sort_unstable();
    gs.sort_unstable();
    bs.sort_unstable();
    let m = rs.len() / 2;
    Rgb([rs[m], gs[m], bs[m]])
}

/// Check if any mask pixels in the given region are non-zero.
fn has_mask_pixels(mask: &GrayImage, x: u32, y: u32, w: u32, h: u32) -> bool {
    let mask_w = mask.width() as usize;
    let x = x as usize;
    let y = y as usize;
    let w = w as usize;
    let h = h as usize;
    debug_assert!(x + w <= mask_w);
    debug_assert!(y + h <= mask.height() as usize);

    let raw = mask.as_raw();
    for row in 0..h {
        let start = (y + row) * mask_w + x;
        if raw[start..start + w].iter().any(|&v| v > 0) {
            return true;
        }
    }
    false
}

#[cfg(test)]
mod tests {
    use super::*;
    use image::Luma;

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

    #[test]
    fn test_tile_starts_no_duplicates() {
        assert_eq!(tile_starts(512), vec![0]);
        assert_eq!(tile_starts(513), vec![0, 1]);
        assert_eq!(tile_starts(1024), vec![0, 448, 512]);
    }
}
