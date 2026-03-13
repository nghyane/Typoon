//! End-to-end translation: detect → OCR → translate → fit → erase → render.
//!
//! Usage:
//!   cargo run --example translate [image_path] [--lang en|ko|ja] [--target vi]
//!
//! Defaults to tests/fixtures/en/manhwa_test.webp → Vietnamese.
//! Output saved to .amp/in/artifacts/{name}_translated.png

use std::path::Path;
use std::time::Instant;

use base64::engine::general_purpose::STANDARD;
use base64::Engine;

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| "warn,comic_scan=debug".parse().unwrap()),
        )
        .init();

    let args = parse_args();
    if !Path::new(&args.input).exists() {
        anyhow::bail!("Image not found: {}", args.input);
    }

    let out_dir = Path::new(".amp/in/artifacts");
    std::fs::create_dir_all(out_dir)?;

    let config = comic_scan::config::AppConfig::load()?;
    let state = comic_scan::api::AppState::new(&config).await?;

    let image_bytes = std::fs::read(&args.input)?;
    let image_name = Path::new(&args.input)
        .file_stem()
        .and_then(|s| s.to_str())
        .unwrap_or("unknown");

    println!("═══ ComicScan Translate ═══");
    println!("Input:   {}", args.input);
    println!("Lang:    {} → {}", args.source_lang, args.target_lang);
    println!();

    let t = Instant::now();
    let req = comic_scan::api::TranslateImageRequest {
        image_id: image_name.to_string(),
        image_blob_b64: STANDARD.encode(&image_bytes),
        source_lang: Some(args.source_lang.clone()),
        target_lang: args.target_lang.clone(),
        ocr_provider: None,
        translation_provider: None,
        provider_config: None,
        context_hint: None,
    };

    let response = comic_scan::pipeline::process_image(&state, &req).await?;

    println!("── Bubbles ({}) ──", response.bubbles.len());
    for b in &response.bubbles {
        println!("  [{}] {}px \"{}\" → \"{}\"",
            b.bubble_id, b.font_size_px,
            b.source_text.chars().take(40).collect::<String>(),
            b.translated_text.replace('\n', " ").chars().take(50).collect::<String>(),
        );
    }

    if let Some(rendered_b64) = &response.rendered_image_png_b64 {
        let path = out_dir.join(format!("{image_name}_translated.png"));
        std::fs::write(&path, STANDARD.decode(rendered_b64)?)?;
        println!("\nSaved: {}", path.display());
    }

    println!("Total: {:.1}s", t.elapsed().as_secs_f64());
    Ok(())
}

struct Args {
    input: String,
    source_lang: String,
    target_lang: String,
}

fn parse_args() -> Args {
    let args: Vec<String> = std::env::args().collect();
    let mut input = "tests/fixtures/en/manhwa_test.webp".to_string();
    let mut source_lang = "en".to_string();
    let mut target_lang = "vi".to_string();
    let mut i = 1;
    while i < args.len() {
        match args[i].as_str() {
            "--lang" if i + 1 < args.len() => { source_lang = args[i + 1].clone(); i += 2; }
            "--target" if i + 1 < args.len() => { target_lang = args[i + 1].clone(); i += 2; }
            _ => { input = args[i].clone(); i += 1; }
        }
    }
    Args { input, source_lang, target_lang }
}
