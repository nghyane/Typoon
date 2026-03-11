use std::path::Path;
use std::sync::Mutex;

use anyhow::{Context, Result};
use image::DynamicImage;
use ndarray::{Array2, Array4};
use ort::session::Session;
use ort::value::TensorRef;

use super::{OcrProvider, OcrResult};

const INPUT_SIZE: usize = 224;
const BOS_TOKEN: i64 = 2;
const EOS_TOKEN: i64 = 3;
const SPECIAL_TOKEN_CUTOFF: i64 = 5;
const MAX_LENGTH: usize = 300;

pub struct MangaOcrAdapter {
    encoder: Mutex<Session>,
    decoder: Mutex<Session>,
    vocab: Vec<String>,
}

impl MangaOcrAdapter {
    pub fn new(models_dir: &str) -> Result<Self> {
        let base = Path::new(models_dir);
        let encoder_path = base.join("encoder_model.onnx");
        let decoder_path = base.join("decoder_model.onnx");
        let vocab_path = base.join("vocab.txt");

        anyhow::ensure!(encoder_path.exists(), "encoder_model.onnx not found at {}", encoder_path.display());
        anyhow::ensure!(decoder_path.exists(), "decoder_model.onnx not found at {}", decoder_path.display());
        anyhow::ensure!(vocab_path.exists(), "vocab.txt not found at {}", vocab_path.display());

        let encoder = Session::builder()?
            .commit_from_file(&encoder_path)
            .with_context(|| format!("Failed to load encoder: {}", encoder_path.display()))?;

        let decoder = Session::builder()?
            .commit_from_file(&decoder_path)
            .with_context(|| format!("Failed to load decoder: {}", decoder_path.display()))?;

        let vocab_text = std::fs::read_to_string(&vocab_path)
            .with_context(|| format!("Failed to read vocab: {}", vocab_path.display()))?;
        let vocab: Vec<String> = vocab_text.lines().map(|l| l.to_string()).collect();

        tracing::info!(
            "MangaOcrAdapter loaded: encoder={}, decoder={}, vocab={} tokens",
            encoder_path.display(),
            decoder_path.display(),
            vocab.len(),
        );

        Ok(Self {
            encoder: Mutex::new(encoder),
            decoder: Mutex::new(decoder),
            vocab,
        })
    }

    /// Preprocess: grayscale → RGB → resize 224×224 → rescale/normalize → NCHW
    fn preprocess(&self, img: &DynamicImage) -> Array4<f32> {
        let gray = img.to_luma8();
        let gray_rgb = DynamicImage::ImageLuma8(gray).to_rgb8();
        let resized = image::imageops::resize(
            &gray_rgb,
            INPUT_SIZE as u32,
            INPUT_SIZE as u32,
            image::imageops::FilterType::Triangle, // bilinear
        );

        let mut arr = Array4::<f32>::zeros((1, 3, INPUT_SIZE, INPUT_SIZE));
        for y in 0..INPUT_SIZE {
            for x in 0..INPUT_SIZE {
                let pixel = resized.get_pixel(x as u32, y as u32);
                for c in 0..3 {
                    let v = pixel[c] as f32 / 255.0;
                    arr[[0, c, y, x]] = (v - 0.5) / 0.5;
                }
            }
        }
        arr
    }

    /// Autoregressive decode: BOS → argmax loop → EOS
    fn generate(&self, pixel_values: &Array4<f32>) -> Result<Vec<i64>> {
        // Encode
        let encoder_input = TensorRef::from_array_view(pixel_values)?;
        let mut encoder_session = self.encoder.lock().unwrap();
        let encoder_outputs = encoder_session.run(ort::inputs![encoder_input])?;

        // Extract encoder_hidden_states — first output, copy data to own it
        let hidden_key = encoder_outputs.keys().next()
            .ok_or_else(|| anyhow::anyhow!("Encoder produced no outputs"))?
            .to_string();
        let (shape, data) = encoder_outputs[&*hidden_key].try_extract_tensor::<f32>()?;
        let hidden_vec: Vec<f32> = data.to_vec();
        let hidden_shape: Vec<usize> = shape.iter().map(|&d| d as usize).collect();

        // Release encoder session and outputs before decoder loop
        drop(encoder_outputs);
        drop(encoder_session);

        // Decoder loop
        let mut token_ids: Vec<i64> = vec![BOS_TOKEN];
        let mut decoder_session = self.decoder.lock().unwrap();

        let hidden_arr = ndarray::ArrayD::from_shape_vec(
            ndarray::IxDyn(&hidden_shape),
            hidden_vec,
        )?;

        for _ in 0..MAX_LENGTH {
            let seq_len = token_ids.len();

            // Build input_ids [1, seq_len]
            let ids_arr = Array2::from_shape_vec((1, seq_len), token_ids.clone())?;

            let hidden_ref = TensorRef::from_array_view(&hidden_arr)?;
            let ids_ref = TensorRef::from_array_view(&ids_arr)?;
            let decoder_outputs = decoder_session.run(
                ort::inputs!["encoder_hidden_states" => hidden_ref, "input_ids" => ids_ref],
            )?;

            // logits: [1, seq_len, vocab_size]
            let logits_key = decoder_outputs.keys().next()
                .ok_or_else(|| anyhow::anyhow!("Decoder produced no outputs"))?
                .to_string();
            let (_logits_shape, logits_data) = decoder_outputs[&*logits_key].try_extract_tensor::<f32>()?;

            // Take argmax of logits[0, -1, :] (last position)
            let vocab_size = self.vocab.len();
            let last_pos_offset = (seq_len - 1) * vocab_size;
            let last_logits = &logits_data[last_pos_offset..last_pos_offset + vocab_size];

            let next_token = last_logits
                .iter()
                .enumerate()
                .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
                .map(|(idx, _)| idx as i64)
                .unwrap_or(EOS_TOKEN);

            token_ids.push(next_token);

            if next_token == EOS_TOKEN {
                break;
            }
        }

        Ok(token_ids)
    }

    /// Decode token IDs to text using vocab, handling wordpiece ## prefixes
    fn decode_tokens(&self, token_ids: &[i64]) -> String {
        let mut text = String::new();
        for &id in token_ids {
            if id < SPECIAL_TOKEN_CUTOFF {
                continue;
            }
            if let Some(token) = self.vocab.get(id as usize) {
                text.push_str(token);
            }
        }
        text
    }

    /// Postprocess: strip whitespace, collapse dots, halfwidth → fullwidth
    fn postprocess(text: &str) -> String {
        // Join and strip all whitespace (including wordpiece ## markers)
        let text: String = text.split_whitespace().collect();
        // Remove ## wordpiece continuation markers
        let text = text.replace("##", "");
        // Normalize ellipsis
        let text = text.replace('…', "...");
        // Halfwidth ASCII → fullwidth (jaconv h2z equivalent for ASCII + digits)
        half_to_full(&text)
    }
}

impl OcrProvider for MangaOcrAdapter {
    fn recognize(&self, image: &DynamicImage) -> Result<OcrResult> {
        let pixel_values = self.preprocess(image);
        let token_ids = self.generate(&pixel_values)?;
        let raw_text = self.decode_tokens(&token_ids);
        let text = Self::postprocess(&raw_text);

        tracing::debug!("manga-ocr: {:?} → {:?}", raw_text, text);

        Ok(OcrResult {
            text,
            confidence: 1.0,
        })
    }

    fn name(&self) -> &str {
        "manga_ocr"
    }
}

/// Convert halfwidth ASCII letters, digits, and common punctuation to fullwidth equivalents.
/// Equivalent to jaconv.h2z(text, ascii=True, digit=True).
fn half_to_full(s: &str) -> String {
    s.chars()
        .map(|c| match c {
            '!' ..= '~' => {
                // ASCII printable range 0x21..=0x7E maps to fullwidth 0xFF01..=0xFF5E
                char::from_u32(c as u32 - 0x21 + 0xFF01).unwrap_or(c)
            }
            ' ' => '\u{3000}', // fullwidth space
            _ => c,
        })
        .collect()
}
