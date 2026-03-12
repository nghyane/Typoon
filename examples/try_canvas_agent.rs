//! Quick smoke test for the canvas agent.
//!
//! Usage:
//!   cargo run --example try_canvas_agent [image_path]
//!
//! Defaults to tests/fixtures/en/manhwa_test.webp if no arg given.
//! Runs: detect → OCR → translate → canvas agent → save PNG.
//!
//! Output in .amp/in/artifacts/:
//!   {name}_annotated.png  — what the LLM sees (bubbles with colored bboxes)
//!   {name}_fit.png        — fit engine fallback rendering
//!   {name}.png            — canvas agent final rendering

use std::path::Path;
use std::time::Instant;

use base64::Engine;
use base64::engine::general_purpose::STANDARD;

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| "warn,comic_scan=debug".parse().unwrap()),
        )
        .init();

    let fixture = std::env::args()
        .nth(1)
        .unwrap_or_else(|| "tests/fixtures/en/manhwa_test.webp".into());

    if !Path::new(&fixture).exists() {
        anyhow::bail!("Image not found: {fixture}");
    }

    let out_dir = Path::new(".amp/in/artifacts");
    std::fs::create_dir_all(out_dir)?;

    let config = comic_scan::config::AppConfig::load()?;
    let state = comic_scan::api::AppState::new(&config).await?;

    let image_bytes = std::fs::read(&fixture)?;
    let img = image::load_from_memory(&image_bytes)?;

    let image_name = Path::new(&fixture)
        .file_stem()
        .and_then(|s| s.to_str())
        .unwrap_or("unknown");

    println!("═══ ComicScan Canvas Agent ═══");
    println!("Input:  {fixture}");
    println!("Size:   {}x{}px", img.width(), img.height());
    println!();

    // ── 1. Detect + OCR ──
    let t = Instant::now();
    let req = comic_scan::api::TranslateImageRequest {
        image_id: image_name.to_string(),
        image_blob_b64: STANDARD.encode(&image_bytes),
        source_lang: Some("en".into()),
        target_lang: "vi".into(),
        ocr_provider: None,
        translation_provider: None,
        provider_config: None,
        context_hint: None,
    };

    let response = comic_scan::pipeline::process_image(&state, &req).await?;
    let elapsed = t.elapsed();

    // ── 2. Display results ──
    println!("── Bubbles ({}) ──", response.bubbles.len());
    for b in &response.bubbles {
        println!(
            "  [{}] font={}px inset={:.1} overflow={} align={}",
            b.bubble_id, b.font_size_px, b.inset, b.overflow, b.align
        );
        println!("        src: {:?}", b.source_text);
        println!("        out: {:?}", b.translated_text);
    }
    println!();

    // ── 3. Save outputs ──
    // Always save fit engine fallback for comparison
    let fit_path = out_dir.join(format!("{image_name}_fit.png"));
    let overlay = comic_scan::overlay::render(&img, &response.bubbles);
    overlay.save(&fit_path)?;
    println!("Saved fit fallback:  {}", fit_path.display());

    // Canvas agent output
    let out_path = out_dir.join(format!("{image_name}.png"));
    if let Some(rendered_b64) = &response.rendered_image_png_b64 {
        let png_bytes = STANDARD.decode(rendered_b64)?;
        std::fs::write(&out_path, &png_bytes)?;
        println!("Saved canvas agent:  {}", out_path.display());
    } else {
        overlay.save(&out_path)?;
        println!("Saved (no canvas):   {}", out_path.display());
    }

    println!();
    println!("Total: {:.1}s", elapsed.as_secs_f64());

    Ok(())
}
