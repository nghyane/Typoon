use std::path::Path;

use anyhow::{Context, Result};
use image::{DynamicImage, GenericImageView};
use ort::session::Session;
use ort::value::TensorRef;

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

/// Text region detected by comic-text-detector
pub struct TextRegion {
    /// Polygon vertices defining the bubble boundary
    pub polygon: Vec<[f64; 2]>,
    /// Cropped image of the text region
    pub crop: DynamicImage,
    /// Detection confidence
    pub confidence: f64,
}

pub struct TextDetector {
    session: Session,
}

impl TextDetector {
    pub fn new(model_path: &Path) -> Result<Self> {
        let session = Session::builder()?
            .commit_from_file(model_path)
            .with_context(|| format!("Failed to load ONNX model: {}", model_path.display()))?;

        tracing::info!("TextDetector loaded from {}", model_path.display());
        Ok(Self { session })
    }

    pub fn is_loaded(&self) -> bool {
        true
    }

    /// Detect text regions in a comic image.
    /// Returns bubble polygons + cropped regions for OCR.
    pub fn detect(&mut self, img: &DynamicImage) -> Result<Vec<TextRegion>> {
        let (orig_w, orig_h) = img.dimensions();

        // 1. Preprocess: resize to 1024x1024, normalize to [0,1], NCHW layout
        let input_tensor = self.preprocess(img);

        // 2. Run ONNX inference
        let input_value = TensorRef::from_array_view(&input_tensor)?;
        let outputs = self.session.run(ort::inputs![input_value])?;

        // 3. Extract blk output [1, N, 7]: [cx, cy, w, h, obj, cls0, cls1]
        let (blk_shape, blk_data) = outputs["blk"].try_extract_tensor::<f32>()?;
        let n_boxes = blk_shape[1] as usize;

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
            regions.push(TextRegion {
                polygon,
                crop,
                confidence: *conf,
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
