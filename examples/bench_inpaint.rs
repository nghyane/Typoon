//! Benchmark LaMa inpainting: sequential vs concurrent.
//!
//! Usage:
//!   cargo run --release --example bench_inpaint

use std::sync::Arc;
use std::time::Instant;

use anyhow::Result;
use image::{DynamicImage, GrayImage, Luma, Rgb, RgbImage};

fn make_test_input(size: u32) -> (DynamicImage, GrayImage) {
    // Non-flat image: gradient + noise so is_flat_tile() returns false
    // and LaMa inference actually runs.
    let mut img = RgbImage::new(size, size);
    let mut rng_state: u32 = 42;
    for y in 0..size {
        for x in 0..size {
            // Simple xorshift pseudo-random
            rng_state ^= rng_state << 13;
            rng_state ^= rng_state >> 17;
            rng_state ^= rng_state << 5;
            let noise = (rng_state % 60) as u8;
            let base = ((x + y) % 256) as u8;
            img.put_pixel(
                x,
                y,
                Rgb([base.wrapping_add(noise), base, base.wrapping_sub(noise)]),
            );
        }
    }
    let mut mask = GrayImage::new(size, size);
    let quarter = size / 4;
    for y in quarter..quarter * 3 {
        for x in quarter..quarter * 3 {
            mask.put_pixel(x, y, Luma([255]));
        }
    }
    (DynamicImage::ImageRgb8(img), mask)
}

#[tokio::main]
async fn main() -> Result<()> {
    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| "info,comic_scan=warn".parse().unwrap()),
        )
        .init();

    let config = comic_scan::config::AppConfig::load()?;
    let model_path =
        comic_scan::model_hub::resolve(&config.models_dir, comic_scan::model_hub::Model::Lama)
            .await?;

    let inpainter = Arc::new(comic_scan::inpaint::LamaInpainter::new(
        comic_scan::model_hub::lazy::LazySession::gpu(model_path),
    ));

    let (img, mask) = make_test_input(512);

    // Warmup
    let _ = inpainter.inpaint(&img, &mask)?;
    println!("Warmup done\n");

    // Sequential benchmark
    let runs = 8;
    let t = Instant::now();
    for _ in 0..runs {
        let _ = inpainter.inpaint(&img, &mask)?;
    }
    let seq_ms = t.elapsed().as_secs_f64() * 1000.0;
    println!(
        "Sequential: {runs} tiles in {seq_ms:.0}ms ({:.0}ms/tile)",
        seq_ms / runs as f64
    );

    // Concurrent benchmark (simulate N pages running in parallel)
    for threads in [2, 4, 8] {
        let t = Instant::now();
        let handles: Vec<_> = (0..threads)
            .map(|_| {
                let inp = Arc::clone(&inpainter);
                let img = img.clone();
                let mask = mask.clone();
                std::thread::spawn(move || {
                    for _ in 0..runs / threads {
                        let _ = inp.inpaint(&img, &mask).unwrap();
                    }
                })
            })
            .collect();
        for h in handles {
            h.join().unwrap();
        }
        let par_ms = t.elapsed().as_secs_f64() * 1000.0;
        let speedup = seq_ms / par_ms;
        println!(
            "Concurrent ({threads} threads): {runs} tiles in {par_ms:.0}ms ({:.0}ms/tile, {speedup:.2}x)",
            par_ms / runs as f64
        );
    }

    Ok(())
}
