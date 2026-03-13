use std::path::Path;

use base64::Engine;
use base64::engine::general_purpose::STANDARD;

const MODELS: &str = "models";

const FIXTURES: &[(&str, &str)] = &[
    ("sample_multi_bubble", "tests/fixtures/en/sample_multi_bubble.webp"),
    ("manhwa_test", "tests/fixtures/en/manhwa_test.webp"),
    ("manhwa_test2", "tests/fixtures/en/manhwa_test2.webp"),
    ("manhwa_test3", "tests/fixtures/en/manhwa_test3.webp"),
];

fn has_ppocr() -> bool {
    Path::new(MODELS).join("ppocr_det.onnx").exists()
        && Path::new(MODELS).join("ppocr_rec.onnx").exists()
}

/// Full pipeline → canvas agent render → save PNG for visual inspection.
///
/// Uses the actual webtoon/manga pipeline (including Canvas Agent) via
/// `pipeline::process_image`. If canvas agent produced a rendered image,
/// saves that directly; otherwise falls back to overlay::render.
///
/// Output: .amp/in/artifacts/overlay_{name}.png
#[tokio::test]
async fn test_overlay_render() {
    if !has_ppocr() {
        eprintln!("Skipping: PP-OCR models not found");
        return;
    }

    let out_dir = Path::new(".amp/in/artifacts");
    std::fs::create_dir_all(out_dir).unwrap();

    let config = comic_scan::config::AppConfig::load().unwrap();
    let state = comic_scan::api::AppState::new(&config).await.unwrap();

    for (name, fixture) in FIXTURES {
        if !Path::new(fixture).exists() {
            eprintln!("Skipping {name}: fixture not found");
            continue;
        }

        println!("\n=== {name} ===");
        let image_bytes = std::fs::read(fixture).expect("Failed to read fixture");
        let img = image::load_from_memory(&image_bytes).expect("Failed to decode image");
        println!("  Image: {}x{}", img.width(), img.height());

        let req = comic_scan::api::TranslateImageRequest {
            image_id: name.to_string(),
            image_blob_b64: STANDARD.encode(&image_bytes),
            source_lang: Some("en".into()),
            target_lang: "vi".into(),
            ocr_provider: None,
            translation_provider: None,
            provider_config: None,
            context_hint: None,
        };

        let response = match comic_scan::pipeline::process_image(&state, &req).await {
            Ok(r) => r,
            Err(e) => {
                eprintln!("  Pipeline failed (LLM may be unavailable): {e:#}");
                continue;
            }
        };

        println!("  {} bubbles", response.bubbles.len());
        for b in &response.bubbles {
            println!(
                "  [{}] font={}px overflow={} text={:?}",
                b.bubble_id, b.font_size_px, b.overflow, b.translated_text
            );
        }

        // Save rendered image: prefer canvas agent output, fall back to overlay::render
        let out_path = out_dir.join(format!("overlay_{name}.png"));
        if let Some(rendered_b64) = &response.rendered_image_png_b64 {
            let png_bytes = STANDARD.decode(rendered_b64).expect("Failed to decode rendered PNG");
            std::fs::write(&out_path, &png_bytes).expect("Failed to write rendered PNG");
            println!("  Saved (canvas agent): {}", out_path.display());
        } else {
            let overlay = comic_scan::overlay::render(&img, &response.bubbles, None);
            overlay.save(&out_path).expect("Failed to save overlay");
            println!("  Saved (fit fallback): {}", out_path.display());
        }
    }
}
