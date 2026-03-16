use std::path::PathBuf;
use std::time::Instant;

use ab_glyph::PxScale;
use anyhow::Result;
use clap::Args;
use image::{Rgba, RgbaImage};
use imageproc::drawing::{draw_hollow_rect_mut, draw_text_mut};
use imageproc::rect::Rect;

use crate::config;
use crate::vision::detection::LocalTextMask;
use crate::render::overlay::{apply_mask_pixels, erase_masks, erase_with_median};

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

/// Pipeline result shared between detect and mask visualization.
struct PipelineResult {
    bubbles: Vec<crate::pipeline::types::RawBubble>,
}

pub async fn run(mut args: InspectArgs) -> Result<()> {
    if !args.image.exists() {
        anyhow::bail!("Image not found: {}", args.image.display());
    }

    if !args.detect && !args.masks {
        args.detect = true;
        args.masks = true;
    }

    std::fs::create_dir_all(&args.output)?;

    let config = config::AppConfig::load()?;
    let img = image::open(&args.image)?;
    let name = args
        .image
        .file_stem()
        .and_then(|s| s.to_str())
        .unwrap_or("unknown");

    println!("═══ ComicScan Inspect ═══");
    println!("Input:  {}", args.image.display());
    println!("Lang:   {}", args.lang);
    println!("Size:   {}x{}px", img.width(), img.height());
    println!();

    // Run pipeline once, share results
    let t = Instant::now();
    let result = run_pipeline(&config, &img, &args.lang).await?;
    println!(
        "Pipeline: {} bubbles in {}ms\n",
        result.bubbles.len(),
        t.elapsed().as_millis()
    );

    for (i, b) in result.bubbles.iter().enumerate() {
        let (x1, y1, x2, y2) = poly_bbox(&b.polygon);
        println!(
            "  [b{i:>3}] bbox=({:.0},{:.0})-({:.0},{:.0}) det={:.2} ocr={:.2} {:?}",
            x1, y1, x2, y2, b.det_confidence, b.ocr_confidence, b.source_text
        );
    }

    if args.detect {
        render_detect(&img, &result, name, &args.output)?;
    }

    if args.masks {
        render_masks(&config, &img, &result, name, &args.output).await?;
    }

    Ok(())
}

// ── Pipeline ──

async fn run_pipeline(
    config: &config::AppConfig,
    img: &image::DynamicImage,
    lang: &str,
) -> Result<PipelineResult> {
    let ctd_path = crate::model_hub::resolve(
        &config.models_dir,
        crate::model_hub::Model::ComicTextDetector,
    )
    .await?;
    let detector =
        crate::vision::detection::TextDetector::new(crate::model_hub::lazy::LazySession::new(ctd_path));
    let ocr = crate::vision::ocr::OcrEngine::new(&config.models_dir).await?;

    let bubbles = crate::pipeline::detect_and_ocr(&detector, &ocr, img, lang)?;

    Ok(PipelineResult { bubbles })
}

// ── Detection visualization ──

fn render_detect(
    img: &image::DynamicImage,
    result: &PipelineResult,
    name: &str,
    output: &PathBuf,
) -> Result<()> {
    let font = crate::render::layout::get_font();
    let label_scale = PxScale::from(18.0);
    let mut canvas = img.to_rgba8();

    for (i, b) in result.bubbles.iter().enumerate() {
        let color = BBOX_COLORS[i % BBOX_COLORS.len()];
        let (x1, y1, x2, y2) = poly_bbox(&b.polygon);

        let rx = x1.max(0.0) as i32;
        let ry = y1.max(0.0) as i32;
        let rw = ((x2 - x1).max(0.0) as u32).min(canvas.width().saturating_sub(rx as u32));
        let rh = ((y2 - y1).max(0.0) as u32).min(canvas.height().saturating_sub(ry as u32));

        if rw > 2 && rh > 2 {
            let rect = Rect::at(rx, ry).of_size(rw, rh);
            draw_hollow_rect_mut(&mut canvas, rect, color);
            if rw > 4 && rh > 4 {
                draw_hollow_rect_mut(
                    &mut canvas,
                    Rect::at(rx + 1, ry + 1).of_size(rw - 2, rh - 2),
                    color,
                );
            }
        }

        let label = format!(
            "[b{i}] d={:.0}% o={:.0}%",
            b.det_confidence * 100.0,
            b.ocr_confidence * 100.0
        );
        draw_text_mut(
            &mut canvas,
            color,
            rx + 4,
            ry.saturating_sub(20),
            label_scale,
            font,
            &label,
        );
    }

    let path = output.join(format!("{name}_detect.png"));
    canvas.save(&path)?;
    println!("Saved: {}", path.display());
    Ok(())
}

// ── Mask visualization ──

async fn render_masks(
    config: &config::AppConfig,
    img: &image::DynamicImage,
    result: &PipelineResult,
    name: &str,
    output: &PathBuf,
) -> Result<()> {
    let masks: Vec<&LocalTextMask> = result.bubbles.iter().filter_map(|b| b.mask.as_ref()).collect();
    println!("\nMasks: {}", masks.len());

    for (i, m) in masks.iter().enumerate() {
        let pixel_count = m.image.pixels().filter(|p| p.0[0] == 255).count();
        println!(
            "  [b{i:>2}] {}x{} at ({},{}) pixels={pixel_count}",
            m.image.width(),
            m.image.height(),
            m.x,
            m.y
        );
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
    let inpainter = {
        let lama_path =
            crate::model_hub::resolve_optional(&config.models_dir, crate::model_hub::Model::Lama)
                .await;
        lama_path.map(|p| {
            crate::vision::inpaint::LamaInpainter::new(crate::model_hub::lazy::LazySession::gpu(p))
        })
    };
    {
        let t = Instant::now();
        let mut dual = rgba.clone();
        erase_masks(&mut dual, &masks, inpainter.as_ref());
        let path = output.join(format!("{name}_dual.png"));
        dual.save(&path)?;
        println!(
            "Saved (dual {}ms): {}",
            t.elapsed().as_millis(),
            path.display()
        );
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
