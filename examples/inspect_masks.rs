//! Inspect ML text masks: runs detection pipeline and saves visual outputs.
//!
//! Outputs (saved to .amp/in/artifacts/):
//!   1. `{name}_masks.png`    — red overlay on detected text pixels
//!   2. `{name}_erase.png`    — median-fill erasure
//!   3. `{name}_dual.png`     — dual-path erasure (flat → median, complex → LaMa)
//!   4. `{name}_raw_mask.png` — binary mask (white = text, black = bg)
//!
//! Usage:
//!   cargo run --example inspect_masks [image_path] [--lang ja|en|ko]
//!
//! Defaults to tests/fixtures/ja/sample_manga.webp with --lang ja

use std::path::Path;
use std::time::Instant;

use image::{Rgba, RgbaImage};

use comic_scan::detection::LocalTextMask;
use comic_scan::overlay::{apply_mask_pixels, erase_masks, erase_with_median};

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| "warn,comic_scan=debug".parse().unwrap()),
        )
        .init();

    let (fixture, lang) = parse_args();
    if !Path::new(&fixture).exists() {
        anyhow::bail!("Image not found: {fixture}");
    }

    let out_dir = Path::new(".amp/in/artifacts");
    std::fs::create_dir_all(out_dir)?;

    let config = comic_scan::config::AppConfig::load()?;
    let img = image::load_from_memory(&std::fs::read(&fixture)?)?;

    let image_name = Path::new(&fixture)
        .file_stem()
        .and_then(|s| s.to_str())
        .unwrap_or("unknown");

    println!("═══ ComicScan Mask Inspector ═══");
    println!("Input:  {fixture}");
    println!("Lang:   {lang}");
    println!("Size:   {}x{}px", img.width(), img.height());
    println!();

    // ── Detect ──
    let t = Instant::now();
    let masks = detect_masks(&config, &img, &lang).await?;
    println!("Detection: {}ms — {} masks", t.elapsed().as_millis(), masks.len());
    println!();

    for (i, m) in masks.iter().enumerate() {
        let pixel_count = m.image.pixels().filter(|p| p.0[0] == 255).count();
        println!(
            "  [b{i:>2}] {}x{} at ({},{}) pixels={}",
            m.image.width(),
            m.image.height(),
            m.x,
            m.y,
            pixel_count,
        );
    }

    if masks.is_empty() {
        println!("⚠ No masks produced — nothing to render.");
        return Ok(());
    }

    let (img_w, img_h) = (img.width(), img.height());
    let rgba = img.to_rgba8();

    // 1. Red overlay
    let overlay = render_overlay(&rgba, &masks);
    let path = out_dir.join(format!("{image_name}_masks.png"));
    overlay.save(&path)?;
    println!("\nSaved: {}", path.display());

    // 2. Median-fill erasure
    let mut erased = rgba.clone();
    for m in &masks {
        erase_with_median(&mut erased, m);
    }
    let path = out_dir.join(format!("{image_name}_erase.png"));
    erased.save(&path)?;
    println!("Saved: {}", path.display());

    // 3. Dual-path erasure (uses production overlay::erase_masks)
    let mut inpainter = {
        let lama_path = comic_scan::model_hub::resolve_optional(
            &config.models_dir,
            comic_scan::model_hub::Model::Lama,
        )
        .await;
        lama_path.and_then(|p| {
            comic_scan::inpaint::LamaInpainter::new(&p)
                .inspect_err(|e| println!("LaMa model load failed: {e}"))
                .ok()
        })
    };
    {
        let t = Instant::now();
        let mut dual = rgba.clone();
        let mask_refs: Vec<&LocalTextMask> = masks.iter().collect();
        erase_masks(&mut dual, &mask_refs, inpainter.as_mut());
        let path = out_dir.join(format!("{image_name}_dual.png"));
        dual.save(&path)?;
        println!(
            "Saved (dual {}ms): {}",
            t.elapsed().as_millis(),
            path.display()
        );
    }

    // 4. Raw binary mask
    let mut raw = RgbaImage::from_pixel(img_w, img_h, Rgba([0, 0, 0, 255]));
    for m in &masks {
        apply_mask_pixels(&mut raw, m, |_, _| Rgba([255, 255, 255, 255]));
    }
    let path = out_dir.join(format!("{image_name}_raw_mask.png"));
    raw.save(&path)?;
    println!("Saved: {}", path.display());

    Ok(())
}

// ── Detection ──

async fn detect_masks(
    config: &comic_scan::config::AppConfig,
    img: &image::DynamicImage,
    lang: &str,
) -> anyhow::Result<Vec<LocalTextMask>> {
    match lang {
        "ja" => {
            let ctd_path = comic_scan::model_hub::resolve(
                &config.models_dir,
                comic_scan::model_hub::Model::ComicTextDetector,
            )
            .await?;
            let mut detector = comic_scan::detection::TextDetector::new(&ctd_path)?;
            let regions = detector.detect(img)?;
            println!("ComicTextDetector: {} regions", regions.len());
            Ok(regions.into_iter().filter_map(|r| r.mask).collect())
        }
        _ => {
            let ocr = comic_scan::ocr::OcrEngine::new(&config.models_dir).await?;
            if !ocr.can_detect() {
                anyhow::bail!("PP-OCR detection model not loaded");
            }
            let lines = ocr.detect(img)?;
            println!("PP-OCR det: {} raw lines", lines.len());
            let merged = comic_scan::pipeline::merge::group_lines(lines);
            println!("Merged: {} bubbles", merged.len());
            Ok(merged.into_iter().filter_map(|b| b.mask).collect())
        }
    }
}

// ── Rendering helpers ──

fn render_overlay(rgba: &RgbaImage, masks: &[LocalTextMask]) -> RgbaImage {
    let mut out = rgba.clone();
    for m in masks {
        apply_mask_pixels(&mut out, m, |orig, _| {
            let r = ((orig.0[0] as u16 * 4 + 255 * 6) / 10) as u8;
            let g = (orig.0[1] as u16 * 4 / 10) as u8;
            let b = (orig.0[2] as u16 * 4 / 10) as u8;
            Rgba([r, g, b, 255])
        });
    }
    out
}

// ── CLI args ──

fn parse_args() -> (String, String) {
    let args: Vec<String> = std::env::args().collect();
    let mut fixture = "tests/fixtures/ja/sample_manga.webp".to_string();
    let mut lang = "ja".to_string();
    let mut i = 1;
    while i < args.len() {
        if args[i] == "--lang" && i + 1 < args.len() {
            lang = args[i + 1].clone();
            i += 2;
        } else {
            fixture = args[i].clone();
            i += 1;
        }
    }
    (fixture, lang)
}
