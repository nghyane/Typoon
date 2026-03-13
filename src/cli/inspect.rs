use std::path::PathBuf;
use std::time::Instant;

use anyhow::Result;
use clap::Args;
use image::{Rgba, RgbaImage};
use imageproc::drawing::{draw_hollow_rect_mut, draw_text_mut};
use imageproc::rect::Rect;
use ab_glyph::PxScale;

use crate::config;
use crate::detection::LocalTextMask;
use crate::overlay::{apply_mask_pixels, erase_masks, erase_with_median};

const BBOX_COLORS: [Rgba<u8>; 6] = [
    Rgba([255, 0, 0, 255]),
    Rgba([0, 255, 0, 255]),
    Rgba([0, 0, 255, 255]),
    Rgba([255, 0, 255, 255]),
    Rgba([0, 255, 255, 255]),
    Rgba([255, 255, 0, 255]),
];

#[derive(Args)]
pub struct InspectArgs {
    /// Path to image file
    pub image: PathBuf,

    /// Source language (ja → comic-text-detector, other → PP-OCR)
    #[arg(short, long, default_value = "en")]
    pub lang: String,

    /// Show detection bounding boxes
    #[arg(long)]
    pub detect: bool,

    /// Show text masks and erasure
    #[arg(long)]
    pub masks: bool,

    /// Output directory
    #[arg(short, long, default_value = "./output")]
    pub output: PathBuf,
}

pub async fn run(mut args: InspectArgs) -> Result<()> {
    if !args.image.exists() {
        anyhow::bail!("Image not found: {}", args.image.display());
    }

    // Default: show everything
    if !args.detect && !args.masks {
        args.detect = true;
        args.masks = true;
    }

    std::fs::create_dir_all(&args.output)?;

    let config = config::AppConfig::load()?;
    let img = image::open(&args.image)?;
    let name = args.image.file_stem().and_then(|s| s.to_str()).unwrap_or("unknown");

    println!("═══ ComicScan Inspect ═══");
    println!("Input:  {}", args.image.display());
    println!("Lang:   {}", args.lang);
    println!("Size:   {}x{}px", img.width(), img.height());
    println!();

    if args.detect {
        run_detect(&config, &img, &args.lang, name, &args.output).await?;
    }

    if args.masks {
        run_masks(&config, &img, &args.lang, name, &args.output).await?;
    }

    Ok(())
}

// ── Detection visualization ──

async fn run_detect(
    config: &config::AppConfig,
    img: &image::DynamicImage,
    lang: &str,
    name: &str,
    output: &PathBuf,
) -> Result<()> {
    let t = Instant::now();
    let bubbles = match lang {
        "ja" => {
            let ctd_path = crate::model_hub::resolve(
                &config.models_dir, crate::model_hub::Model::ComicTextDetector,
            ).await?;
            let mut detector = crate::detection::TextDetector::new(&ctd_path)?;
            let regions = detector.detect(img)?;
            println!("comic-text-detector: {} raw regions", regions.len());
            regions.into_iter().map(|r| (r.polygon, r.confidence, 1usize)).collect::<Vec<_>>()
        }
        _ => {
            let ocr = crate::ocr::OcrEngine::new(&config.models_dir).await?;
            if !ocr.can_detect() {
                anyhow::bail!("PP-OCR detection model not loaded");
            }
            let lines = ocr.detect(img)?;
            println!("PP-OCR det: {} raw lines", lines.len());
            let merged = crate::pipeline::merge::group_lines(lines);
            merged.into_iter().map(|b| (b.polygon, b.confidence, b.lines.len())).collect::<Vec<_>>()
        }
    };

    println!("Merged:  {} bubbles in {}ms\n", bubbles.len(), t.elapsed().as_millis());

    let font = crate::text_layout::get_font();
    let label_scale = PxScale::from(18.0);
    let mut canvas = img.to_rgba8();

    for (i, (polygon, confidence, line_count)) in bubbles.iter().enumerate() {
        let color = BBOX_COLORS[i % BBOX_COLORS.len()];
        let (x1, y1, x2, y2) = poly_bbox(polygon);

        let rx = x1.max(0.0) as i32;
        let ry = y1.max(0.0) as i32;
        let rw = ((x2 - x1).max(0.0) as u32).min(canvas.width().saturating_sub(rx as u32));
        let rh = ((y2 - y1).max(0.0) as u32).min(canvas.height().saturating_sub(ry as u32));

        if rw > 2 && rh > 2 {
            let rect = Rect::at(rx, ry).of_size(rw, rh);
            draw_hollow_rect_mut(&mut canvas, rect, color);
            if rw > 4 && rh > 4 {
                draw_hollow_rect_mut(&mut canvas, Rect::at(rx + 1, ry + 1).of_size(rw - 2, rh - 2), color);
            }
        }

        let label = format!("[b{i}] {line_count}L {:.0}%", confidence * 100.0);
        draw_text_mut(&mut canvas, color, rx + 4, ry.saturating_sub(20), label_scale, font, &label);

        println!(
            "  [b{i}] bbox=({:.0},{:.0})-({:.0},{:.0}) lines={line_count} conf={:.1}%",
            x1, y1, x2, y2, confidence * 100.0,
        );
    }

    let path = output.join(format!("{name}_detect.png"));
    canvas.save(&path)?;
    println!("\nSaved: {}", path.display());

    Ok(())
}

// ── Mask inspection ──

async fn run_masks(
    config: &config::AppConfig,
    img: &image::DynamicImage,
    lang: &str,
    name: &str,
    output: &PathBuf,
) -> Result<()> {
    let t = Instant::now();
    let masks = detect_masks(config, img, lang).await?;
    println!("\nMasks: {} in {}ms", masks.len(), t.elapsed().as_millis());

    for (i, m) in masks.iter().enumerate() {
        let pixel_count = m.image.pixels().filter(|p| p.0[0] == 255).count();
        println!("  [b{i:>2}] {}x{} at ({},{}) pixels={pixel_count}", m.image.width(), m.image.height(), m.x, m.y);
    }

    if masks.is_empty() {
        println!("⚠ No masks produced.");
        return Ok(());
    }

    let (img_w, img_h) = (img.width(), img.height());
    let rgba = img.to_rgba8();

    // Red overlay
    let mut overlay = rgba.clone();
    for m in &masks {
        apply_mask_pixels(&mut overlay, m, |orig, _| {
            let r = ((orig.0[0] as u16 * 4 + 255 * 6) / 10) as u8;
            let g = (orig.0[1] as u16 * 4 / 10) as u8;
            let b = (orig.0[2] as u16 * 4 / 10) as u8;
            Rgba([r, g, b, 255])
        });
    }
    let path = output.join(format!("{name}_masks.png"));
    overlay.save(&path)?;
    println!("\nSaved: {}", path.display());

    // Median-fill erasure
    let mut erased = rgba.clone();
    for m in &masks {
        erase_with_median(&mut erased, m);
    }
    let path = output.join(format!("{name}_erase.png"));
    erased.save(&path)?;
    println!("Saved: {}", path.display());

    // Dual-path erasure (median + LaMa)
    let mut inpainter = {
        let lama_path = crate::model_hub::resolve_optional(&config.models_dir, crate::model_hub::Model::Lama).await;
        lama_path.and_then(|p| {
            crate::inpaint::LamaInpainter::new(&p)
                .inspect_err(|e| println!("LaMa load failed: {e}"))
                .ok()
        })
    };
    {
        let t = Instant::now();
        let mut dual = rgba.clone();
        let mask_refs: Vec<&LocalTextMask> = masks.iter().collect();
        erase_masks(&mut dual, &mask_refs, inpainter.as_mut());
        let path = output.join(format!("{name}_dual.png"));
        dual.save(&path)?;
        println!("Saved (dual {}ms): {}", t.elapsed().as_millis(), path.display());
    }

    // Raw binary mask
    let mut raw = RgbaImage::from_pixel(img_w, img_h, Rgba([0, 0, 0, 255]));
    for m in &masks {
        apply_mask_pixels(&mut raw, m, |_, _| Rgba([255, 255, 255, 255]));
    }
    let path = output.join(format!("{name}_raw_mask.png"));
    raw.save(&path)?;
    println!("Saved: {}", path.display());

    Ok(())
}

async fn detect_masks(
    config: &config::AppConfig,
    img: &image::DynamicImage,
    lang: &str,
) -> Result<Vec<LocalTextMask>> {
    match lang {
        "ja" => {
            let ctd_path = crate::model_hub::resolve(
                &config.models_dir, crate::model_hub::Model::ComicTextDetector,
            ).await?;
            let mut detector = crate::detection::TextDetector::new(&ctd_path)?;
            let regions = detector.detect(img)?;
            Ok(regions.into_iter().filter_map(|r| r.mask).collect())
        }
        _ => {
            let ocr = crate::ocr::OcrEngine::new(&config.models_dir).await?;
            if !ocr.can_detect() {
                anyhow::bail!("PP-OCR detection model not loaded");
            }
            let lines = ocr.detect(img)?;
            let merged = crate::pipeline::merge::group_lines(lines);
            Ok(merged.into_iter().filter_map(|b| b.mask).collect())
        }
    }
}

fn poly_bbox(polygon: &[[f64; 2]]) -> (f64, f64, f64, f64) {
    let (mut x1, mut y1) = (f64::INFINITY, f64::INFINITY);
    let (mut x2, mut y2) = (f64::NEG_INFINITY, f64::NEG_INFINITY);
    for p in polygon {
        x1 = x1.min(p[0]);
        y1 = y1.min(p[1]);
        x2 = x2.max(p[0]);
        y2 = y2.max(p[1]);
    }
    (x1, y1, x2, y2)
}
