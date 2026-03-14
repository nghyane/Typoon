use std::path::Path;

use anyhow::{Context, Result};
use geo::{ConvexHull, Coord, LineString, Polygon};
use image::{DynamicImage, Rgb, RgbImage};
use ndarray::Array4;
use ort::value::TensorRef;

use super::OcrResult;
use crate::detection::{LocalTextMask, TextRegion, dilate_mask};
use crate::model_hub::lazy::LazySession;

// ── Recognition constants ──
/// Fixed input height for PP-OCR recognition models (v5 uses 48)
const REC_IMG_HEIGHT: u32 = 48;
/// Minimum padded width (PaddleOCR default)
const REC_IMG_MIN_WIDTH: u32 = 320;
/// Crops shorter than this are upscaled 2× before recognition to avoid blur
const REC_UPSCALE_THRESHOLD: u32 = 32;

// ── Detection constants ──
/// Resize longest side to this value for detection
const DET_RESIZE_LONG: u32 = 960;
/// For small images, use a larger resize target so text lines are big enough
const DET_RESIZE_LONG_SMALL: u32 = 1280;
/// Images with longest side below this threshold use DET_RESIZE_LONG_SMALL
const DET_SMALL_IMAGE_THRESHOLD: u32 = 960;
/// Minimum width after resize — prevents text becoming unreadable on tall/narrow images
const DET_MIN_WIDTH: u32 = 768;
/// Maximum total pixels in detection input to prevent OOM
const DET_MAX_PIXELS: u32 = 1_500_000;
/// DB binarization threshold
const DET_THRESH: f32 = 0.3;
/// Minimum mean score inside a detected box to keep it
const DET_BOX_THRESH: f32 = 0.6;
/// Polygon expansion ratio (unclip)
const DET_UNCLIP_RATIO: f64 = 1.5;
/// Minimum side length of a detected box (in model space)
const DET_MIN_SIZE: f64 = 3.0;
/// Maximum number of contour candidates to process (like PaddleOCR's max_candidates)
const DET_MAX_CANDIDATES: usize = 1000;
/// ImageNet normalization for detection
const DET_MEAN: [f32; 3] = [0.485, 0.456, 0.406];
const DET_STD: [f32; 3] = [0.229, 0.224, 0.225];

pub struct PpOcrAdapter {
    det_session: Option<LazySession>,
    rec_session: LazySession,
    dictionary: Vec<String>,
}

impl PpOcrAdapter {
    pub fn new(
        rec_session: LazySession,
        dict_path: &Path,
        det_session: Option<LazySession>,
    ) -> Result<Self> {
        let dict_text = std::fs::read_to_string(dict_path)
            .with_context(|| format!("Failed to read dictionary: {}", dict_path.display()))?;
        let mut dictionary = vec!["".to_string()]; // CTC blank at index 0
        for line in dict_text.lines() {
            if !line.is_empty() {
                dictionary.push(line.to_string());
            }
        }
        dictionary.push(" ".to_string()); // use_space_char=true

        tracing::debug!("PpOcrAdapter initialized (lazy): det={}", det_session.is_some());

        Ok(Self {
            det_session,
            rec_session,
            dictionary,
        })
    }


    pub fn can_detect(&self) -> bool {
        self.det_session.is_some()
    }

    /// Detect text regions using PP-OCR DB detection model.
    pub fn detect(&self, img: &DynamicImage) -> Result<Vec<TextRegion>> {
        let det_session = self.det_session.as_ref()
            .ok_or_else(|| anyhow::anyhow!("PP-OCR detection model not loaded"))?
            .get()
            .ok_or_else(|| anyhow::anyhow!("PP-OCR det session failed to load"))?;

        let (orig_w, orig_h) = (img.width(), img.height());

        // 1. Preprocess: resize longest side to DET_RESIZE_LONG, pad to multiple of 32
        let (input_tensor, content_w, content_h, pad_w, pad_h) = self.det_preprocess(img);

        // 2. Run inference
        let input_ref = TensorRef::from_array_view(&input_tensor)?;
        let mut session = det_session.lock().unwrap();
        let outputs = session.run(ort::inputs![input_ref])?;

        // 3. Extract probability map [1, 1, H, W]
        let output_key = outputs.keys().next()
            .ok_or_else(|| anyhow::anyhow!("PP-OCR det model produced no outputs"))?
            .to_string();
        let (_shape, prob_data) = outputs[&*output_key].try_extract_tensor::<f32>()?;

        // 4. DB post-processing: threshold → contours → unclip → crop
        let regions = self.db_postprocess(
            prob_data, pad_w, pad_h, content_w, content_h,
            orig_w, orig_h, img,
        );

        tracing::debug!("PP-OCR detected {} text regions", regions.len());
        Ok(regions)
    }

    /// Preprocess for detection: resize, normalize with ImageNet stats, NCHW
    fn det_preprocess(&self, img: &DynamicImage) -> (Array4<f32>, usize, usize, usize, usize) {
        let (orig_w, orig_h) = (img.width(), img.height());

        // Use larger resize target for small images so text lines remain readable
        let det_long = if orig_w.max(orig_h) < DET_SMALL_IMAGE_THRESHOLD {
            DET_RESIZE_LONG_SMALL
        } else {
            DET_RESIZE_LONG
        };
        // Ensure minimum width for readability, cap total pixels to prevent OOM
        let ratio_long = det_long as f32 / orig_w.max(orig_h) as f32;
        let ratio_min_w = DET_MIN_WIDTH as f32 / orig_w as f32;
        let ratio = ratio_long.max(ratio_min_w);
        let mut new_w = (orig_w as f32 * ratio) as u32;
        let mut new_h = (orig_h as f32 * ratio) as u32;

        // Cap to max pixels to prevent OOM on very tall images
        let total_pixels = new_w as u64 * new_h as u64;
        if total_pixels > DET_MAX_PIXELS as u64 {
            let scale_down = (DET_MAX_PIXELS as f32 / total_pixels as f32).sqrt();
            new_w = (new_w as f32 * scale_down) as u32;
            new_h = (new_h as f32 * scale_down) as u32;
        }

        // Pad to multiple of 32
        let pad_w = new_w.div_ceil(32) as usize * 32;
        let pad_h = new_h.div_ceil(32) as usize * 32;

        let resized = img.resize_exact(new_w, new_h, image::imageops::FilterType::Lanczos3);
        let rgb = resized.to_rgb8();

        let mut arr = Array4::<f32>::zeros((1, 3, pad_h, pad_w));
        for y in 0..new_h as usize {
            for x in 0..new_w as usize {
                let pixel = rgb.get_pixel(x as u32, y as u32);
                for c in 0..3 {
                    arr[[0, c, y, x]] = (pixel[c] as f32 / 255.0 - DET_MEAN[c]) / DET_STD[c];
                }
            }
        }

        (arr, new_w as usize, new_h as usize, pad_w, pad_h)
    }

    /// DB post-processing: binarize probability map, find connected components,
    /// compute convex hulls + min-area rotated rects, unclip, and perspective-warp
    /// crop from original image (matching PaddleOCR's official approach).
    #[allow(clippy::too_many_arguments)]
    fn db_postprocess(
        &self,
        prob_data: &[f32],
        map_w: usize,
        map_h: usize,
        content_w: usize,
        content_h: usize,
        orig_w: u32,
        orig_h: u32,
        img: &DynamicImage,
    ) -> Vec<TextRegion> {
        // Binarize
        let mut binary = vec![0u8; map_h * map_w];
        for i in 0..map_h * map_w {
            if prob_data[i] > DET_THRESH {
                binary[i] = 255;
            }
        }

        // Find rotated quads via contour → convex hull → min-area rotated rect
        let quads = find_rotated_quads(&binary, prob_data, map_w, map_h);

        // Scale using actual content size, not padded size, to avoid shrinking crops
        let scale_x = orig_w as f64 / content_w as f64;
        let scale_y = orig_h as f64 / content_h as f64;

        let mut regions = Vec::new();
        for RotatedQuad { corners, width, height, score } in &quads {
            if *score < DET_BOX_THRESH as f64 {
                continue;
            }
            if *width < DET_MIN_SIZE || *height < DET_MIN_SIZE {
                continue;
            }

            // Unclip: expand rotated rect by distance = area * unclip_ratio / perimeter
            let area = width * height;
            let perimeter = 2.0 * (width + height);
            let d = area * DET_UNCLIP_RATIO / perimeter;

            let expanded = unclip_rotated_rect(corners, *width, *height, d);

            // Scale corners to original image coordinates
            let orig_corners: Vec<[f64; 2]> = expanded
                .iter()
                .map(|[x, y]| [
                    (x * scale_x).clamp(0.0, orig_w as f64 - 1.0),
                    (y * scale_y).clamp(0.0, orig_h as f64 - 1.0),
                ])
                .collect();

            // Perspective warp crop from original image
            let crop = match warp_quad(img, &orig_corners) {
                Some(c) => c,
                None => continue,
            };

            if crop.width() < 5 || crop.height() < 5 {
                continue;
            }

            // Build text mask from prob_data in original page coordinates.
            // The DB prob map represents shrunken text cores (similar to DBNet shrink map).
            // We treat it as a soft probability mask: scale to [0..255], then threshold
            // at a low value and dilate to recover full stroke width.
            let mask = {
                // Axis-aligned bbox of the expanded quad in original image space
                let mut ax1 = f64::INFINITY;
                let mut ay1 = f64::INFINITY;
                let mut ax2 = f64::NEG_INFINITY;
                let mut ay2 = f64::NEG_INFINITY;
                for [px, py] in &orig_corners {
                    ax1 = ax1.min(*px);
                    ay1 = ay1.min(*py);
                    ax2 = ax2.max(*px);
                    ay2 = ay2.max(*py);
                }
                let mask_x = ax1.floor().max(0.0) as u32;
                let mask_y = ay1.floor().max(0.0) as u32;
                let mask_w = ((ax2.ceil() as u32).saturating_sub(mask_x)).min(orig_w - mask_x).max(1);
                let mask_h = ((ay2.ceil() as u32).saturating_sub(mask_y)).min(orig_h - mask_y).max(1);

                // Sample prob_data at detection map resolution as soft grayscale,
                // then bilinear resize to original image resolution. This avoids
                // nearest-neighbor staircase artifacts from the resolution mismatch.
                let inv_sx = content_w as f64 / orig_w as f64;
                let inv_sy = content_h as f64 / orig_h as f64;

                // Compute detection-map-space bbox for this region
                let dm_x1 = (mask_x as f64 * inv_sx).floor() as usize;
                let dm_y1 = (mask_y as f64 * inv_sy).floor() as usize;
                let dm_x2 = (((mask_x + mask_w) as f64 * inv_sx).ceil() as usize).min(map_w);
                let dm_y2 = (((mask_y + mask_h) as f64 * inv_sy).ceil() as usize).min(map_h);
                let dm_w = (dm_x2 - dm_x1).max(1) as u32;
                let dm_h = (dm_y2 - dm_y1).max(1) as u32;

                // Extract soft grayscale crop at detection map resolution
                let mut dm_crop = image::GrayImage::new(dm_w, dm_h);
                for ly in 0..dm_h {
                    for lx in 0..dm_w {
                        let mx = dm_x1 + lx as usize;
                        let my = dm_y1 + ly as usize;
                        if mx < map_w && my < map_h {
                            let v = (prob_data[my * map_w + mx] * 255.0).round().clamp(0.0, 255.0) as u8;
                            dm_crop.put_pixel(lx, ly, image::Luma([v]));
                        }
                    }
                }

                // Bilinear resize to original image mask dimensions (smooth edges)
                let resized = image::imageops::resize(
                    &dm_crop, mask_w, mask_h, image::imageops::FilterType::Triangle,
                );

                // Threshold at ~0.10 (25/255) to capture soft edges from
                // bilinear interpolation, then dilate to cover anti-aliased
                // stroke boundaries. Dilation scales with line height: larger
                // text has thicker strokes needing more coverage.
                let mut mask_img = image::GrayImage::new(mask_w, mask_h);
                for (px, py, pixel) in resized.enumerate_pixels() {
                    if pixel.0[0] > 25 {
                        mask_img.put_pixel(px, py, image::Luma([255]));
                    }
                }
                let dilate_r = (mask_h.min(mask_w) / 4).max(12);
                let mask_img = dilate_mask(&mask_img, dilate_r);

                Some(LocalTextMask { x: mask_x, y: mask_y, image: mask_img })
            };

            regions.push(TextRegion {
                polygon: orig_corners,
                crop,
                confidence: *score,
                mask,
            });
        }

        regions
    }

    // ── Recognition methods ──

    /// Preprocess for recognition: resize to height=48, dynamic width based on
    /// aspect ratio (matching PaddleOCR's resize_norm_img logic), normalize.
    fn rec_preprocess(&self, img: &DynamicImage) -> Array4<f32> {
        // Upscale tiny crops so downscale to height=48 doesn't lose detail
        let img = if img.height() < REC_UPSCALE_THRESHOLD {
            std::borrow::Cow::Owned(img.resize(
                img.width() * 2,
                img.height() * 2,
                image::imageops::FilterType::Lanczos3,
            ))
        } else {
            std::borrow::Cow::Borrowed(img)
        };

        let rgb = img.to_rgb8();
        let (orig_w, orig_h) = (rgb.width(), rgb.height());

        let wh_ratio = orig_w as f32 / orig_h as f32;
        // Dynamic width: height * aspect_ratio, at least REC_IMG_MIN_WIDTH
        let target_w = (REC_IMG_HEIGHT as f32 * wh_ratio).ceil() as u32;
        let pad_w = target_w.max(REC_IMG_MIN_WIDTH);
        let resized_w = target_w.min(pad_w);

        let resized = image::imageops::resize(
            &rgb,
            resized_w,
            REC_IMG_HEIGHT,
            image::imageops::FilterType::Triangle,
        );

        let h = REC_IMG_HEIGHT as usize;
        let w = pad_w as usize;
        let mut arr = Array4::<f32>::zeros((1, 3, h, w));

        for y in 0..h {
            for x in 0..resized_w as usize {
                let pixel = resized.get_pixel(x as u32, y as u32);
                for c in 0..3 {
                    arr[[0, c, y, x]] = (pixel[c] as f32 / 255.0 - 0.5) / 0.5;
                }
            }
        }

        arr
    }

    /// CTC greedy decode
    fn ctc_decode(&self, logits: &[f32], seq_len: usize, vocab_size: usize) -> (String, f64) {
        let mut text = String::new();
        let mut prev_idx: Option<usize> = None;
        let mut total_prob: f64 = 0.0;
        let mut char_count: usize = 0;

        for t in 0..seq_len {
            let offset = t * vocab_size;
            let step_logits = &logits[offset..offset + vocab_size];

            let (best_idx, best_logit) = step_logits
                .iter()
                .enumerate()
                .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
                .unwrap();

            if Some(best_idx) == prev_idx {
                prev_idx = Some(best_idx);
                continue;
            }
            prev_idx = Some(best_idx);

            if best_idx == 0 {
                continue;
            }

            if let Some(ch) = self.dictionary.get(best_idx) {
                text.push_str(ch);
                // Model output is already softmax probabilities, use directly
                total_prob += *best_logit as f64;
                char_count += 1;
            }
        }

        let confidence = if char_count > 0 {
            total_prob / char_count as f64
        } else {
            0.0
        };

        (text, confidence)
    }
}

impl PpOcrAdapter {
    pub fn recognize(&self, image: &DynamicImage) -> Result<OcrResult> {
        let input_tensor = self.rec_preprocess(image);

        let input_ref = TensorRef::from_array_view(&input_tensor)?;
        let rec_session = self.rec_session.get()
            .ok_or_else(|| anyhow::anyhow!("PP-OCR rec session failed to load"))?;
        let mut session = rec_session.lock().unwrap();
        let outputs = session.run(ort::inputs![input_ref])?;

        let output_key = outputs.keys().next()
            .ok_or_else(|| anyhow::anyhow!("PP-OCR rec model produced no outputs"))?
            .to_string();
        let (shape, data) = outputs[&*output_key].try_extract_tensor::<f32>()?;

        let seq_len = shape[1] as usize;
        let vocab_size = shape[2] as usize;

        let (text, confidence) = self.ctc_decode(data, seq_len, vocab_size);

        tracing::debug!("ppocr: text={:?}, conf={:.3}", text, confidence);

        Ok(OcrResult { text, confidence })
    }

}

// ── DB post-processing helpers ──

/// Result of contour analysis: a rotated quad with metadata.
struct RotatedQuad {
    /// Four corner points ordered [TL, TR, BR, BL].
    corners: [[f64; 2]; 4],
    /// Width of the min-area rotated rect.
    width: f64,
    /// Height of the min-area rotated rect.
    height: f64,
    /// Mean probability score inside the contour.
    score: f64,
}

/// Find rotated quads from a binary mask using connected-component labeling,
/// convex hull, and minimum-area rotated rectangle.
fn find_rotated_quads(
    binary: &[u8],
    prob_map: &[f32],
    width: usize,
    height: usize,
) -> Vec<RotatedQuad> {
    // 1. Connected-component labeling via flood fill
    let mut labels = vec![0u32; height * width];
    let mut label_id: u32 = 0;
    let mut components: Vec<Vec<(usize, usize)>> = Vec::new();

    for start_y in 0..height {
        for start_x in 0..width {
            let idx = start_y * width + start_x;
            if labels[idx] != 0 || binary[idx] == 0 {
                continue;
            }

            label_id += 1;
            if components.len() >= DET_MAX_CANDIDATES {
                break;
            }

            let mut stack = vec![(start_x, start_y)];
            let mut component = Vec::new();
            while let Some((x, y)) = stack.pop() {
                let i = y * width + x;
                if labels[i] != 0 || binary[i] == 0 {
                    continue;
                }
                labels[i] = label_id;
                component.push((x, y));

                if x > 0 { stack.push((x - 1, y)); }
                if x + 1 < width { stack.push((x + 1, y)); }
                if y > 0 { stack.push((x, y - 1)); }
                if y + 1 < height { stack.push((x, y + 1)); }
            }

            if !component.is_empty() {
                components.push(component);
            }
        }
        if components.len() >= DET_MAX_CANDIDATES {
            break;
        }
    }

    // 2. For each component: compute mean score, extract boundary, convex hull, min-area rect
    let mut results = Vec::new();
    for component in &components {
        if component.len() < 3 {
            continue;
        }

        // Mean probability score inside contour
        let score_sum: f64 = component.iter()
            .map(|&(x, y)| prob_map[y * width + x] as f64)
            .sum();
        let score = score_sum / component.len() as f64;

        // Extract boundary pixels (pixels with at least one 4-neighbor outside the component)
        let boundary: Vec<Coord<f64>> = component
            .iter()
            .filter(|&&(x, y)| {
                (x == 0 || binary[y * width + (x - 1)] == 0)
                    || (x + 1 >= width || binary[y * width + (x + 1)] == 0)
                    || (y == 0 || binary[(y - 1) * width + x] == 0)
                    || (y + 1 >= height || binary[(y + 1) * width + x] == 0)
            })
            .map(|&(x, y)| Coord { x: x as f64, y: y as f64 })
            .collect();

        if boundary.len() < 3 {
            continue;
        }

        // Convex hull via geo crate
        let line_string = LineString::new(boundary);
        let poly = Polygon::new(line_string, vec![]);
        let hull = poly.convex_hull();
        let hull_points: Vec<[f64; 2]> = hull
            .exterior()
            .points()
            .map(|p| [p.x(), p.y()])
            .collect();

        if hull_points.len() < 3 {
            continue;
        }

        // Min-area rotated rectangle
        let (corners, w, h) = min_area_rect(&hull_points);
        if w < DET_MIN_SIZE || h < DET_MIN_SIZE {
            continue;
        }

        results.push(RotatedQuad { corners, width: w, height: h, score });
    }

    results
}

/// Compute the minimum-area rotated bounding rectangle for a convex hull.
/// Returns (corners [TL, TR, BR, BL], width, height).
fn min_area_rect(hull: &[[f64; 2]]) -> ([[f64; 2]; 4], f64, f64) {
    let n = hull.len();
    let mut best_area = f64::MAX;
    let mut best_corners = [[0.0; 2]; 4];
    let mut best_w = 0.0;
    let mut best_h = 0.0;

    // Test each edge angle of the hull
    for i in 0..n {
        let j = (i + 1) % n;
        let dx = hull[j][0] - hull[i][0];
        let dy = hull[j][1] - hull[i][1];
        let angle = dy.atan2(dx);
        let cos_a = angle.cos();
        let sin_a = angle.sin();

        // Rotate all hull points by -angle
        let mut min_rx = f64::MAX;
        let mut max_rx = f64::MIN;
        let mut min_ry = f64::MAX;
        let mut max_ry = f64::MIN;

        for pt in hull {
            let rx = pt[0] * cos_a + pt[1] * sin_a;
            let ry = -pt[0] * sin_a + pt[1] * cos_a;
            min_rx = min_rx.min(rx);
            max_rx = max_rx.max(rx);
            min_ry = min_ry.min(ry);
            max_ry = max_ry.max(ry);
        }

        let w = max_rx - min_rx;
        let h = max_ry - min_ry;
        let area = w * h;

        if area < best_area {
            best_area = area;
            best_w = w;
            best_h = h;

            // Corners in rotated space: TL, TR, BR, BL
            let rotated_corners = [
                [min_rx, min_ry],
                [max_rx, min_ry],
                [max_rx, max_ry],
                [min_rx, max_ry],
            ];
            // Rotate back to original space
            for (k, rc) in rotated_corners.iter().enumerate() {
                best_corners[k] = [
                    rc[0] * cos_a - rc[1] * sin_a,
                    rc[0] * sin_a + rc[1] * cos_a,
                ];
            }
        }
    }

    (order_quad_tl_tr_br_bl(best_corners), best_w, best_h)
}

/// Reorder 4 corners to canonical [TL, TR, BR, BL] in image space.
/// Uses centroid-angle sorting + TL anchor + clockwise winding enforcement.
fn order_quad_tl_tr_br_bl(corners: [[f64; 2]; 4]) -> [[f64; 2]; 4] {
    let cx = corners.iter().map(|p| p[0]).sum::<f64>() / 4.0;
    let cy = corners.iter().map(|p| p[1]).sum::<f64>() / 4.0;

    // Sort around centroid so adjacent points stay adjacent
    let mut pts = corners.to_vec();
    pts.sort_by(|a, b| {
        let aa = (a[1] - cy).atan2(a[0] - cx);
        let bb = (b[1] - cy).atan2(b[0] - cx);
        aa.partial_cmp(&bb).unwrap_or(std::cmp::Ordering::Equal)
    });

    // Rotate so smallest x+y (top-left-ish) comes first
    let start = (0..4)
        .min_by(|&i, &j| {
            let si = pts[i][0] + pts[i][1];
            let sj = pts[j][0] + pts[j][1];
            si.partial_cmp(&sj).unwrap_or(std::cmp::Ordering::Equal)
        })
        .unwrap();
    pts.rotate_left(start);

    // Ensure clockwise winding: TL→TR→BR→BL (not TL→BL→BR→TR)
    let cross = (pts[1][0] - pts[0][0]) * (pts[2][1] - pts[0][1])
              - (pts[1][1] - pts[0][1]) * (pts[2][0] - pts[0][0]);
    if cross < 0.0 {
        pts = vec![pts[0], pts[3], pts[2], pts[1]];
    }

    [pts[0], pts[1], pts[2], pts[3]]
}

/// Expand a rotated rect by distance `d` along both width and height directions.
/// Returns 4 expanded corners in the same order [TL, TR, BR, BL].
fn unclip_rotated_rect(
    corners: &[[f64; 2]; 4],
    _width: f64,
    _height: f64,
    d: f64,
) -> [[f64; 2]; 4] {
    // Compute edge unit normals pointing outward, then offset each corner
    // along the two adjacent edge normals by distance d.

    // Edge vectors: TL→TR (top), TR→BR (right), BR→BL (bottom), BL→TL (left)
    let edge_vec = |a: usize, b: usize| -> [f64; 2] {
        [corners[b][0] - corners[a][0], corners[b][1] - corners[a][1]]
    };

    // Outward normal for edge a→b (rotate 90° CW for a clockwise polygon)
    let outward_normal = |a: usize, b: usize| -> [f64; 2] {
        let [dx, dy] = edge_vec(a, b);
        let len = (dx * dx + dy * dy).sqrt();
        if len < 1e-10 {
            return [0.0, 0.0];
        }
        [dy / len, -dx / len]
    };

    // For each corner, move it along the two outward normals of its adjacent edges
    // TL: top edge (0→1) and left edge (3→0)
    // TR: top edge (0→1) and right edge (1→2)
    // BR: right edge (1→2) and bottom edge (2→3)
    // BL: bottom edge (2→3) and left edge (3→0)
    let n_top = outward_normal(0, 1);
    let n_right = outward_normal(1, 2);
    let n_bottom = outward_normal(2, 3);
    let n_left = outward_normal(3, 0);

    let offset = |corner: [f64; 2], n1: [f64; 2], n2: [f64; 2]| -> [f64; 2] {
        [
            corner[0] + d * (n1[0] + n2[0]),
            corner[1] + d * (n1[1] + n2[1]),
        ]
    };

    [
        offset(corners[0], n_top, n_left),   // TL
        offset(corners[1], n_top, n_right),  // TR
        offset(corners[2], n_bottom, n_right), // BR
        offset(corners[3], n_bottom, n_left),  // BL
    ]
}

/// Perspective warp: extract a deskewed crop from `img` given 4 source corners [TL, TR, BR, BL].
/// Returns None if output dimensions are too small.
fn warp_quad(img: &DynamicImage, corners: &[[f64; 2]]) -> Option<DynamicImage> {
    if corners.len() != 4 {
        return None;
    }

    let [tl, tr, br, bl] = [corners[0], corners[1], corners[2], corners[3]];

    // Output size
    let dist = |a: [f64; 2], b: [f64; 2]| ((a[0] - b[0]).powi(2) + (a[1] - b[1]).powi(2)).sqrt();
    let out_w = dist(tl, tr).max(dist(bl, br)).round() as u32;
    let out_h = dist(tl, bl).max(dist(tr, br)).round() as u32;

    if out_w < 2 || out_h < 2 {
        return None;
    }

    // Destination corners
    let dst: [[f64; 2]; 4] = [
        [0.0, 0.0],
        [out_w as f64 - 1.0, 0.0],
        [out_w as f64 - 1.0, out_h as f64 - 1.0],
        [0.0, out_h as f64 - 1.0],
    ];

    // Compute inverse perspective transform (dst → src) so we can sample src for each dst pixel
    let inv_matrix = match perspective_transform_matrix(&dst, &[tl, tr, br, bl]) {
        Some(m) => m,
        None => return None,
    };

    let rgb = img.to_rgb8();
    let (img_w, img_h) = (rgb.width(), rgb.height());
    let mut output = RgbImage::new(out_w, out_h);

    for oy in 0..out_h {
        for ox in 0..out_w {
            let (sx, sy) = apply_perspective(&inv_matrix, ox as f64, oy as f64);
            let pixel = bilinear_sample(&rgb, sx, sy, img_w, img_h);
            output.put_pixel(ox, oy, pixel);
        }
    }

    let result = DynamicImage::ImageRgb8(output);

    // If height/width ratio >= 1.5, rotate 90° (vertical text)
    if out_h as f64 / out_w as f64 >= 1.5 {
        Some(result.rotate270())
    } else {
        Some(result)
    }
}

/// Solve for the 3×3 perspective transform matrix that maps src[i] → dst[i].
/// Returns the 8 coefficients [a,b,c, d,e,f, g,h] of:
///   x' = (a*x + b*y + c) / (g*x + h*y + 1)
///   y' = (d*x + e*y + f) / (g*x + h*y + 1)
fn perspective_transform_matrix(
    src: &[[f64; 2]; 4],
    dst: &[[f64; 2]; 4],
) -> Option<[f64; 8]> {
    // Set up 8×8 linear system: A * coeffs = b
    // For each point pair (x,y) → (x',y'):
    //   a*x + b*y + c - g*x*x' - h*y*x' = x'
    //   d*x + e*y + f - g*x*y' - h*y*y' = y'
    let mut a_mat = [[0.0f64; 8]; 8];
    let mut b_vec = [0.0f64; 8];

    for i in 0..4 {
        let (x, y) = (src[i][0], src[i][1]);
        let (xp, yp) = (dst[i][0], dst[i][1]);
        let row1 = i * 2;
        let row2 = row1 + 1;

        a_mat[row1] = [x, y, 1.0, 0.0, 0.0, 0.0, -x * xp, -y * xp];
        b_vec[row1] = xp;

        a_mat[row2] = [0.0, 0.0, 0.0, x, y, 1.0, -x * yp, -y * yp];
        b_vec[row2] = yp;
    }

    // Gaussian elimination with partial pivoting
    let n = 8;
    let mut aug = [[0.0f64; 9]; 8];
    for i in 0..n {
        for j in 0..n {
            aug[i][j] = a_mat[i][j];
        }
        aug[i][n] = b_vec[i];
    }

    for col in 0..n {
        // Find pivot
        let mut max_row = col;
        let mut max_val = aug[col][col].abs();
        for row in (col + 1)..n {
            let v = aug[row][col].abs();
            if v > max_val {
                max_val = v;
                max_row = row;
            }
        }
        if max_val < 1e-12 {
            return None; // Singular
        }
        aug.swap(col, max_row);

        let pivot = aug[col][col];
        for j in col..=n {
            aug[col][j] /= pivot;
        }
        for row in 0..n {
            if row == col {
                continue;
            }
            let factor = aug[row][col];
            for j in col..=n {
                aug[row][j] -= factor * aug[col][j];
            }
        }
    }

    Some([
        aug[0][n], aug[1][n], aug[2][n], aug[3][n],
        aug[4][n], aug[5][n], aug[6][n], aug[7][n],
    ])
}

/// Apply perspective transform to a point.
fn apply_perspective(coeffs: &[f64; 8], x: f64, y: f64) -> (f64, f64) {
    let denom = coeffs[6] * x + coeffs[7] * y + 1.0;
    let sx = (coeffs[0] * x + coeffs[1] * y + coeffs[2]) / denom;
    let sy = (coeffs[3] * x + coeffs[4] * y + coeffs[5]) / denom;
    (sx, sy)
}

/// Bilinear interpolation sampling from an RGB image.
fn bilinear_sample(img: &RgbImage, x: f64, y: f64, w: u32, h: u32) -> Rgb<u8> {
    let x0 = x.floor() as i64;
    let y0 = y.floor() as i64;
    let x1 = x0 + 1;
    let y1 = y0 + 1;
    let fx = x - x0 as f64;
    let fy = y - y0 as f64;

    let get = |px: i64, py: i64| -> [f64; 3] {
        let cx = px.clamp(0, w as i64 - 1) as u32;
        let cy = py.clamp(0, h as i64 - 1) as u32;
        let p = img.get_pixel(cx, cy);
        [p[0] as f64, p[1] as f64, p[2] as f64]
    };

    let p00 = get(x0, y0);
    let p10 = get(x1, y0);
    let p01 = get(x0, y1);
    let p11 = get(x1, y1);

    let mut out = [0u8; 3];
    for c in 0..3 {
        let v = p00[c] * (1.0 - fx) * (1.0 - fy)
            + p10[c] * fx * (1.0 - fy)
            + p01[c] * (1.0 - fx) * fy
            + p11[c] * fx * fy;
        out[c] = v.round().clamp(0.0, 255.0) as u8;
    }
    Rgb(out)
}
