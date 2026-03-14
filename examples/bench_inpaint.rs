//! Benchmark LaMa inpainting (model load + inference).
//!
//! Usage:
//!   cargo run --release --example bench_inpaint
//!
//! Requires: lama_fp32.onnx in models/ dir.

use std::time::Instant;

use anyhow::Result;
use image::{GrayImage, Luma, RgbImage, Rgb, DynamicImage};

#[tokio::main]
async fn main() -> Result<()> {
    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| "info,comic_scan=debug".parse().unwrap()),
        )
        .init();

    let config = comic_scan::config::AppConfig::load()?;

    // Resolve model path
    let model_path = comic_scan::model_hub::resolve(
        &config.models_dir,
        comic_scan::model_hub::Model::Lama,
    )
    .await?;

    // Benchmark model load
    let t = Instant::now();
    let inpainter = comic_scan::inpaint::LamaInpainter::new(model_path);
    // Trigger lazy load with a warmup call below
    println!("Model init: {:.1}ms (lazy, session loads on first use)", t.elapsed().as_secs_f64() * 1000.0);

    // Create test inputs: 256×256 white image with a 64×64 black square masked for inpaint
    let size = 256u32;
    let img = DynamicImage::ImageRgb8(RgbImage::from_pixel(size, size, Rgb([255, 255, 255])));
    let mut mask = GrayImage::new(size, size);
    for y in 96..160 {
        for x in 96..160 {
            mask.put_pixel(x, y, Luma([255]));
        }
    }

    // Warmup
    let _ = inpainter.inpaint(&img, &mask)?;

    // Benchmark inference (5 runs)
    let runs = 5;
    let t = Instant::now();
    for _ in 0..runs {
        let _ = inpainter.inpaint(&img, &mask)?;
    }
    let avg_ms = t.elapsed().as_secs_f64() * 1000.0 / runs as f64;
    println!("Inference ({size}×{size}): {avg_ms:.1}ms avg over {runs} runs");

    // 512×512 tile
    let size_lg = 512u32;
    let img_lg = DynamicImage::ImageRgb8(RgbImage::from_pixel(size_lg, size_lg, Rgb([255, 255, 255])));
    let mut mask_lg = GrayImage::new(size_lg, size_lg);
    for y in 128..384 {
        for x in 128..384 {
            mask_lg.put_pixel(x, y, Luma([255]));
        }
    }
    let _ = inpainter.inpaint(&img_lg, &mask_lg)?;
    let t = Instant::now();
    for _ in 0..runs {
        let _ = inpainter.inpaint(&img_lg, &mask_lg)?;
    }
    let avg_ms_lg = t.elapsed().as_secs_f64() * 1000.0 / runs as f64;
    println!("Inference ({size_lg}×{size_lg}): {avg_ms_lg:.1}ms avg over {runs} runs");

    Ok(())
}
