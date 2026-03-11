use std::path::Path;
use std::sync::Mutex;

use anyhow::{Context, Result};
use image::DynamicImage;
use ndarray::Array4;
use ort::session::Session;
use ort::value::TensorRef;

use super::{OcrProvider, OcrResult};
use crate::detection::TextRegion;

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
/// ImageNet normalization for detection
const DET_MEAN: [f32; 3] = [0.485, 0.456, 0.406];
const DET_STD: [f32; 3] = [0.229, 0.224, 0.225];

pub struct PpOcrAdapter {
    det_session: Option<Mutex<Session>>,
    rec_session: Mutex<Session>,
    dictionary: Vec<String>,
}

impl PpOcrAdapter {
    pub fn new(models_dir: &str) -> Result<Self> {
        let base = Path::new(models_dir);
        let rec_path = base.join("ppocr_rec.onnx");
        let dict_path = base.join("ppocr_keys.txt");

        anyhow::ensure!(rec_path.exists(), "ppocr_rec.onnx not found at {}", rec_path.display());
        anyhow::ensure!(dict_path.exists(), "ppocr_keys.txt not found at {}", dict_path.display());

        let rec_session = Session::builder()?
            .commit_from_file(&rec_path)
            .with_context(|| format!("Failed to load PP-OCR rec model: {}", rec_path.display()))?;

        let dict_text = std::fs::read_to_string(&dict_path)
            .with_context(|| format!("Failed to read dictionary: {}", dict_path.display()))?;
        // CTC blank at index 0, then dictionary characters starting at index 1
        let mut dictionary = vec!["".to_string()]; // blank token
        for line in dict_text.lines() {
            if !line.is_empty() {
                dictionary.push(line.to_string());
            }
        }
        // Add space character at the end (use_space_char=true in PaddleOCR)
        dictionary.push(" ".to_string());

        // Detection model is optional
        let det_path = base.join("ppocr_det.onnx");
        let det_session = if det_path.exists() {
            let session = Session::builder()?
                .commit_from_file(&det_path)
                .with_context(|| format!("Failed to load PP-OCR det model: {}", det_path.display()))?;
            tracing::info!("PP-OCR det loaded: {}", det_path.display());
            Some(Mutex::new(session))
        } else {
            tracing::warn!("ppocr_det.onnx not found, PP-OCR detection disabled");
            None
        };

        tracing::info!(
            "PpOcrAdapter loaded: rec={}, dict={} tokens, det={}",
            rec_path.display(),
            dictionary.len(),
            det_session.is_some(),
        );

        Ok(Self {
            det_session,
            rec_session: Mutex::new(rec_session),
            dictionary,
        })
    }

    pub fn can_detect(&self) -> bool {
        self.det_session.is_some()
    }

    /// Detect text regions using PP-OCR DB detection model.
    pub fn detect(&self, img: &DynamicImage) -> Result<Vec<TextRegion>> {
        let det_session = self.det_session.as_ref()
            .ok_or_else(|| anyhow::anyhow!("PP-OCR detection model not loaded"))?;

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
    /// compute bounding boxes, filter, unclip, and crop from original image.
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

        // Find connected components using simple flood-fill
        let boxes = find_boxes(&binary, prob_data, map_w, map_h);

        // Scale using actual content size, not padded size, to avoid shrinking crops
        let scale_x = orig_w as f64 / content_w as f64;
        let scale_y = orig_h as f64 / content_h as f64;

        let mut regions = Vec::new();
        for (bbox, score) in &boxes {
            if *score < DET_BOX_THRESH as f64 {
                continue;
            }

            let (bx, by, bw, bh) = *bbox;
            if (bw as f64) < DET_MIN_SIZE || (bh as f64) < DET_MIN_SIZE {
                continue;
            }

            // Unclip: expand box by unclip_ratio
            let area = bw as f64 * bh as f64;
            let perimeter = 2.0 * (bw as f64 + bh as f64);
            let distance = area * DET_UNCLIP_RATIO / perimeter;
            let ex = distance.ceil() as u32;
            let ey = distance.ceil() as u32;

            let ux = bx.saturating_sub(ex);
            let uy = by.saturating_sub(ey);
            let uw = (bw + 2 * ex).min(map_w as u32 - ux);
            let uh = (bh + 2 * ey).min(map_h as u32 - uy);

            if (uw as f64) < DET_MIN_SIZE + 2.0 || (uh as f64) < DET_MIN_SIZE + 2.0 {
                continue;
            }

            // Scale to original image coordinates
            let crop_x = (ux as f64 * scale_x).floor().max(0.0) as u32;
            let crop_y = (uy as f64 * scale_y).floor().max(0.0) as u32;
            let crop_x = crop_x.min(orig_w.saturating_sub(1));
            let crop_y = crop_y.min(orig_h.saturating_sub(1));
            let crop_w = ((uw as f64 * scale_x).ceil() as u32)
                .min(orig_w - crop_x)
                .max(1);
            let crop_h = ((uh as f64 * scale_y).ceil() as u32)
                .min(orig_h - crop_y)
                .max(1);

            if crop_w < 5 || crop_h < 5 {
                continue;
            }

            let polygon = vec![
                [crop_x as f64, crop_y as f64],
                [(crop_x + crop_w) as f64, crop_y as f64],
                [(crop_x + crop_w) as f64, (crop_y + crop_h) as f64],
                [crop_x as f64, (crop_y + crop_h) as f64],
            ];

            let crop = img.crop_imm(crop_x, crop_y, crop_w, crop_h);
            regions.push(TextRegion {
                polygon,
                crop,
                confidence: *score,
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
                let max_val = *best_logit;
                let sum_exp: f32 = step_logits.iter().map(|&v| (v - max_val).exp()).sum();
                let prob = 1.0 / sum_exp;
                total_prob += prob as f64;
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

impl OcrProvider for PpOcrAdapter {
    fn recognize(&self, image: &DynamicImage) -> Result<OcrResult> {
        let input_tensor = self.rec_preprocess(image);

        let input_ref = TensorRef::from_array_view(&input_tensor)?;
        let mut session = self.rec_session.lock().unwrap();
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

    fn name(&self) -> &str {
        "ppocr_v5"
    }
}

// ── DB post-processing helpers ──

/// Find bounding boxes from a binary mask using connected-component flood fill.
/// Returns Vec<((x, y, w, h), mean_score)>.
fn find_boxes(
    binary: &[u8],
    prob_map: &[f32],
    width: usize,
    height: usize,
) -> Vec<((u32, u32, u32, u32), f64)> {
    let mut visited = vec![false; height * width];
    let mut results = Vec::new();

    for start_y in 0..height {
        for start_x in 0..width {
            let idx = start_y * width + start_x;
            if visited[idx] || binary[idx] == 0 {
                continue;
            }

            // Flood fill to find connected component
            let mut stack = vec![(start_x, start_y)];
            let mut min_x = start_x;
            let mut min_y = start_y;
            let mut max_x = start_x;
            let mut max_y = start_y;
            let mut score_sum: f64 = 0.0;
            let mut pixel_count: usize = 0;

            while let Some((x, y)) = stack.pop() {
                let i = y * width + x;
                if visited[i] || binary[i] == 0 {
                    continue;
                }
                visited[i] = true;

                min_x = min_x.min(x);
                min_y = min_y.min(y);
                max_x = max_x.max(x);
                max_y = max_y.max(y);
                score_sum += prob_map[i] as f64;
                pixel_count += 1;

                if x > 0 { stack.push((x - 1, y)); }
                if x + 1 < width { stack.push((x + 1, y)); }
                if y > 0 { stack.push((x, y - 1)); }
                if y + 1 < height { stack.push((x, y + 1)); }
            }

            if pixel_count == 0 {
                continue;
            }

            let bx = min_x as u32;
            let by = min_y as u32;
            let bw = (max_x - min_x + 1) as u32;
            let bh = (max_y - min_y + 1) as u32;
            let mean_score = score_sum / pixel_count as f64;

            results.push(((bx, by, bw, bh), mean_score));
        }
    }

    results
}
