use anyhow::Result;
use image::{DynamicImage, GenericImageView, GrayImage, Luma};
use ort::value::TensorRef;

use crate::model_hub::lazy::LazySession;

const MODEL_INPUT_SIZE: u32 = 1024;
const MODEL_SIZE: usize = MODEL_INPUT_SIZE as usize;

/// Minimum confidence (objectness × class score) to keep a detection
const CONF_THRESHOLD: f32 = 0.5;
/// IoU threshold for Non-Maximum Suppression
const NMS_THRESHOLD: f32 = 0.45;
/// Minimum area (in original image px²) of a final region to output
const MIN_REGION_AREA: u32 = 500;
/// blk output columns: [cx, cy, w, h, objectness, cls0, cls1]
/// cls1 = speech bubble class
const BLK_COLS: usize = 7;

/// Binary mask of text pixels in original page coordinates.
/// Used by overlay to erase original text before drawing translation.
#[derive(Debug, Clone)]
pub struct LocalTextMask {
    /// Page-space X of the mask's top-left corner
    pub x: u32,
    /// Page-space Y of the mask's top-left corner
    pub y: u32,
    /// Binary mask (255 = text pixel, 0 = background)
    pub image: GrayImage,
}

/// Text region detected by comic-text-detector
pub struct TextRegion {
    /// Polygon vertices defining the bubble boundary
    pub polygon: Vec<[f64; 2]>,
    /// Cropped image of the text region
    pub crop: DynamicImage,
    /// Detection confidence
    pub confidence: f64,
    /// Per-pixel text mask in page coordinates (from UNet segmentation or PP-OCR prob map)
    pub mask: Option<LocalTextMask>,
}

pub struct TextDetector {
    session: LazySession,
}

impl TextDetector {
    pub fn new(session: LazySession) -> Self {
        Self { session }
    }

    pub fn is_loaded(&self) -> bool {
        self.session.is_loaded()
    }

    /// Detect text regions in a comic image.
    /// Returns bubble polygons + cropped regions for OCR.
    pub fn detect(&self, img: &DynamicImage) -> Result<Vec<TextRegion>> {
        let (orig_w, orig_h) = img.dimensions();

        // 1. Preprocess: resize to 1024x1024, normalize to [0,1], NCHW layout
        let input_tensor = self.preprocess(img);

        // 2. Run ONNX inference
        let input_value = TensorRef::from_array_view(&input_tensor)?;
        let session_mutex = self.session.get()
            .ok_or_else(|| anyhow::anyhow!("TextDetector session not loaded"))?;
        let mut session = session_mutex.lock().unwrap();
        let outputs = session.run(ort::inputs![input_value])?;

        // 3. Extract blk output [1, N, 7]: [cx, cy, w, h, obj, cls0, cls1]
        let (blk_shape, blk_data) = outputs["blk"].try_extract_tensor::<f32>()?;
        let n_boxes = blk_shape[1] as usize;

        // Extract text masks from model outputs:
        //   "det" [1, 2, 1024, 1024] — DBNet shrink map (channel 0), tight per-character
        //   "seg" [1, 1, 1024, 1024] — UNet segmentation, covers whole text region
        // Strategy: dilate "det" for stroke coverage, intersect with "seg" for boundary
        let det_map: Option<Vec<f32>> = outputs
            .get("det")
            .and_then(|v| v.try_extract_tensor::<f32>().ok())
            .map(|(shape, data)| {
                let plane = MODEL_SIZE * MODEL_SIZE;
                if shape.len() == 4 && shape[1] >= 2 {
                    data[..plane].to_vec()
                } else {
                    data.to_vec()
                }
            });
        let seg_map: Option<Vec<f32>> = outputs
            .get("seg")
            .and_then(|v| v.try_extract_tensor::<f32>().ok())
            .map(|(_, data)| data.to_vec());

        // 4. Filter by confidence and convert to [x1, y1, x2, y2, conf]
        //    Use max(cls0, cls1) so both free text (cls0) and speech bubbles (cls1) are detected
        let mut detections: Vec<[f64; 5]> = Vec::new();
        for i in 0..n_boxes {
            let base = i * BLK_COLS;
            let obj = blk_data[base + 4];
            let cls0 = blk_data[base + 5]; // text class
            let cls1 = blk_data[base + 6]; // bubble class
            let conf = obj * cls0.max(cls1);
            if conf < CONF_THRESHOLD {
                continue;
            }
            let cx = blk_data[base] as f64;
            let cy = blk_data[base + 1] as f64;
            let w = blk_data[base + 2] as f64;
            let h = blk_data[base + 3] as f64;
            detections.push([cx - w / 2.0, cy - h / 2.0, cx + w / 2.0, cy + h / 2.0, conf as f64]);
        }

        // 5. Sort by confidence descending, then NMS
        detections.sort_by(|a, b| b[4].partial_cmp(&a[4]).unwrap());
        let kept = nms(&detections, NMS_THRESHOLD as f64);

        // 6. Scale factors from model space → original image
        let scale_x = orig_w as f64 / MODEL_INPUT_SIZE as f64;
        let scale_y = orig_h as f64 / MODEL_INPUT_SIZE as f64;

        // 7. Build TextRegion for each kept detection
        let mut regions = Vec::new();
        for [x1, y1, x2, y2, conf] in &kept {
            let polygon = vec![
                [x1 * scale_x, y1 * scale_y],
                [x2 * scale_x, y1 * scale_y],
                [x2 * scale_x, y2 * scale_y],
                [x1 * scale_x, y2 * scale_y],
            ];

            let crop_x = (x1 * scale_x).floor().max(0.0) as u32;
            let crop_y = (y1 * scale_y).floor().max(0.0) as u32;
            let crop_x = crop_x.min(orig_w.saturating_sub(1));
            let crop_y = crop_y.min(orig_h.saturating_sub(1));
            let crop_w = ((x2 * scale_x).ceil() as u32).saturating_sub(crop_x).min(orig_w - crop_x).max(1);
            let crop_h = ((y2 * scale_y).ceil() as u32).saturating_sub(crop_y).min(orig_h - crop_y).max(1);

            if crop_w * crop_h < MIN_REGION_AREA {
                continue;
            }

            let crop = img.crop_imm(crop_x, crop_y, crop_w, crop_h);

            // Build text mask: combine det (tight strokes) + seg (region boundary)
            let mask = build_combined_mask(
                det_map.as_deref(), seg_map.as_deref(),
                *x1, *y1, *x2, *y2,
                crop_x, crop_y, crop_w, crop_h,
            );

            regions.push(TextRegion {
                polygon,
                crop,
                confidence: *conf,
                mask,
            });
        }

        tracing::debug!("Detected {} text regions", regions.len());
        Ok(regions)
    }

    /// Resize to 1024x1024, normalize to [0,1], return NCHW f32 array
    fn preprocess(&self, img: &DynamicImage) -> ndarray::Array4<f32> {
        let resized = img.resize_exact(
            MODEL_INPUT_SIZE,
            MODEL_INPUT_SIZE,
            image::imageops::FilterType::Lanczos3,
        );
        let rgb = resized.to_rgb8();

        let mut arr = ndarray::Array4::<f32>::zeros((1, 3, MODEL_SIZE, MODEL_SIZE));
        for y in 0..MODEL_SIZE {
            for x in 0..MODEL_SIZE {
                let pixel = rgb.get_pixel(x as u32, y as u32);
                arr[[0, 0, y, x]] = pixel[0] as f32 / 255.0;
                arr[[0, 1, y, x]] = pixel[1] as f32 / 255.0;
                arr[[0, 2, y, x]] = pixel[2] as f32 / 255.0;
            }
        }
        arr
    }
}

/// Build a per-region text mask from the UNet segmentation map.
///
/// The seg map (UNet output) is a soft probability mask [0.0, 1.0] where higher values
/// indicate text pixels. Sigmoid is already baked into the model. We scale to [0..255],
/// resize to crop dimensions via bilinear interpolation, then threshold.
///
/// The det map (DBNet shrink map) represents *shrunken* text cores for line detection,
/// NOT pixel-accurate text masks. It is intentionally undersized and meant for contour
/// extraction → unclip expansion. We do NOT use it for mask building.
///
/// Coordinates: (bx1,by1,bx2,by2) are model-space bbox, (crop_x,crop_y,crop_w,crop_h) are
/// original-image crop rect.
///
/// Reference: dmMaze/comic-text-detector postprocess_mask() scales seg to uint8 without
/// binary thresholding. Koharu refines with Otsu + colour analysis. We use seg with a
/// moderate threshold for clean erasure.
#[allow(clippy::too_many_arguments)]
fn build_combined_mask(
    _det_map: Option<&[f32]>,
    seg_map: Option<&[f32]>,
    bx1: f64, by1: f64, bx2: f64, by2: f64,
    crop_x: u32, crop_y: u32, crop_w: u32, crop_h: u32,
) -> Option<LocalTextMask> {
    let seg = seg_map?;

    let mx1 = (bx1.max(0.0) as u32).min(MODEL_INPUT_SIZE - 1);
    let my1 = (by1.max(0.0) as u32).min(MODEL_INPUT_SIZE - 1);
    let mx2 = (bx2.ceil() as u32).min(MODEL_INPUT_SIZE);
    let my2 = (by2.ceil() as u32).min(MODEL_INPUT_SIZE);
    let mw = (mx2 - mx1).max(1);
    let mh = (my2 - my1).max(1);

    // Extract seg crop in model space as soft grayscale (scale [0,1] → [0,255])
    let mut seg_crop = GrayImage::new(mw, mh);
    for ly in 0..mh {
        for lx in 0..mw {
            let idx = (my1 + ly) as usize * MODEL_SIZE + (mx1 + lx) as usize;
            if idx < seg.len() {
                let v = (seg[idx] * 255.0).round().clamp(0.0, 255.0) as u8;
                seg_crop.put_pixel(lx, ly, Luma([v]));
            }
        }
    }

    // Bilinear resize to original crop dimensions (preserves soft gradients)
    let resized = image::imageops::resize(
        &seg_crop, crop_w, crop_h, image::imageops::FilterType::Triangle,
    );

    // Threshold: seg values > ~0.30 (77 in uint8) are text pixels.
    // The UNet seg output is sigmoid-activated and tends to produce high-confidence
    // values (>0.7) on text cores and softer values (0.3–0.5) on stroke edges.
    // A lower threshold captures these edges; the 2px dilation below fills remaining gaps.
    let mut mask = GrayImage::new(crop_w, crop_h);
    for (px, py, pixel) in resized.enumerate_pixels() {
        if pixel.0[0] > 77 {
            mask.put_pixel(px, py, Luma([255]));
        }
    }

    // Dilate by 2px to ensure full stroke coverage after resize interpolation
    let mask = dilate_mask(&mask, 3);

    Some(LocalTextMask { x: crop_x, y: crop_y, image: mask })
}

/// Non-Maximum Suppression: keep boxes that don't overlap too much with higher-confidence ones.
fn nms(detections: &[[f64; 5]], iou_threshold: f64) -> Vec<[f64; 5]> {
    let mut keep = Vec::new();
    for det in detections {
        let dominated = keep.iter().any(|k: &[f64; 5]| {
            let ix1 = det[0].max(k[0]);
            let iy1 = det[1].max(k[1]);
            let ix2 = det[2].min(k[2]);
            let iy2 = det[3].min(k[3]);
            let inter = (ix2 - ix1).max(0.0) * (iy2 - iy1).max(0.0);
            let a1 = (det[2] - det[0]) * (det[3] - det[1]);
            let a2 = (k[2] - k[0]) * (k[3] - k[1]);
            let iou = inter / (a1 + a2 - inter + 1e-6);
            iou > iou_threshold
        });
        if !dominated {
            keep.push(*det);
        }
    }
    keep
}

/// Fast L1-norm dilation of a binary mask by `radius` pixels.
pub(crate) fn dilate_mask(img: &GrayImage, radius: u32) -> GrayImage {
    let (w, h) = img.dimensions();
    let r = radius as i32;

    // Horizontal pass
    let mut horiz = img.clone();
    for y in 0..h {
        for x in 0..w {
            if img.get_pixel(x, y).0[0] == 255 {
                continue;
            }
            let x0 = (x as i32 - r).max(0) as u32;
            let x1 = (x as i32 + r).min(w as i32 - 1) as u32;
            for nx in x0..=x1 {
                if img.get_pixel(nx, y).0[0] == 255 {
                    horiz.put_pixel(x, y, Luma([255]));
                    break;
                }
            }
        }
    }

    // Vertical pass
    let mut out = horiz.clone();
    for y in 0..h {
        for x in 0..w {
            if horiz.get_pixel(x, y).0[0] == 255 {
                continue;
            }
            let y0 = (y as i32 - r).max(0) as u32;
            let y1 = (y as i32 + r).min(h as i32 - 1) as u32;
            for ny in y0..=y1 {
                if horiz.get_pixel(x, ny).0[0] == 255 {
                    out.put_pixel(x, y, Luma([255]));
                    break;
                }
            }
        }
    }
    out
}
