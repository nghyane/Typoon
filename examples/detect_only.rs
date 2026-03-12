//! Detect-only example: runs text detection and draws bounding boxes on the image.
//! No OCR or translation — useful for visually inspecting detection quality.
//!
//! Usage:
//!   cargo run --example detect_only [image_path] [--lang ja|en|ko|...]
//!
//! --lang ja  → comic-text-detector (manga)
//! --lang *   → PP-OCR det + line merge (manhwa/webtoon, default)
//!
//! Defaults to tests/fixtures/en/manhwa_test.webp if no arg given.
//! Output: .amp/in/artifacts/{name}_detect.png

use std::path::Path;
use std::time::Instant;

use image::Rgba;
use imageproc::drawing::{draw_hollow_rect_mut, draw_text_mut};
use imageproc::rect::Rect;
use ab_glyph::PxScale;

const BBOX_COLORS: [Rgba<u8>; 6] = [
    Rgba([255, 0, 0, 255]),
    Rgba([0, 255, 0, 255]),
    Rgba([0, 0, 255, 255]),
    Rgba([255, 0, 255, 255]),
    Rgba([0, 255, 255, 255]),
    Rgba([255, 255, 0, 255]),
];

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| "warn,comic_scan=debug".parse().unwrap()),
        )
        .init();

    let args: Vec<String> = std::env::args().collect();

    let mut fixture = "tests/fixtures/en/manhwa_test.webp".to_string();
    let mut lang = "en".to_string();

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

    if !Path::new(&fixture).exists() {
        anyhow::bail!("Image not found: {fixture}");
    }

    let out_dir = Path::new(".amp/in/artifacts");
    std::fs::create_dir_all(out_dir)?;

    let config = comic_scan::config::AppConfig::load()?;
    let image_bytes = std::fs::read(&fixture)?;
    let img = image::load_from_memory(&image_bytes)?;

    let image_name = Path::new(&fixture)
        .file_stem()
        .and_then(|s| s.to_str())
        .unwrap_or("unknown");

    println!("═══ ComicScan Detect Only ═══");
    println!("Input:  {fixture}");
    println!("Lang:   {lang}");
    println!("Size:   {}x{}px", img.width(), img.height());
    println!();

    // ── 1. Detect ──
    let t = Instant::now();

    let bubbles = match lang.as_str() {
        "ja" => {
            // Manga: comic-text-detector (whole bubble polygons)
            let ctd_path = comic_scan::model_hub::resolve(
                &config.models_dir, comic_scan::model_hub::Model::ComicTextDetector,
            ).await?;
            let mut detector = comic_scan::detection::TextDetector::new(&ctd_path)?;
            let regions = detector.detect(&img)?;
            println!("comic-text-detector: {} raw regions", regions.len());
            regions
                .into_iter()
                .map(|r| (r.polygon, r.confidence, 1usize))
                .collect::<Vec<_>>()
        }
        _ => {
            // Manhwa/webtoon: PP-OCR det → line merge
            let ocr = comic_scan::ocr::OcrEngine::new(&config.models_dir).await?;
            if !ocr.can_detect() {
                anyhow::bail!("PP-OCR detection model not loaded");
            }
            let lines = ocr.detect(&img)?;
            println!("PP-OCR det: {} raw lines", lines.len());
            let merged = comic_scan::pipeline::merge::group_lines(lines);
            merged
                .into_iter()
                .map(|b| (b.polygon, b.confidence, b.lines.len()))
                .collect::<Vec<_>>()
        }
    };

    let detect_ms = t.elapsed().as_millis();
    println!("Merged:  {} bubbles in {}ms", bubbles.len(), detect_ms);
    println!();

    // ── 2. Draw bboxes on image ──
    let font = comic_scan::text_layout::get_font();
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

    // ── 3. Save ──
    let out_path = out_dir.join(format!("{image_name}_detect.png"));
    canvas.save(&out_path)?;
    println!();
    println!("Saved: {}", out_path.display());

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
