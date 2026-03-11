use std::path::Path;

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

/// Full pipeline → overlay render → save PNG for visual inspection.
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

    let ocr = comic_scan::ocr::OcrEngine::new(MODELS).unwrap();
    assert!(ocr.can_detect());

    let config = comic_scan::config::AppConfig::load().unwrap();
    let translation = comic_scan::translation::TranslationEngine::new(&config.translation).unwrap();

    for (name, fixture) in FIXTURES {
        if !Path::new(fixture).exists() {
            eprintln!("Skipping {name}: fixture not found");
            continue;
        }

        println!("\n=== {name} ===");
        let img = image::open(fixture).expect("Failed to load");
        println!("  Image: {}x{}", img.width(), img.height());

        // Detect + OCR
        let lines = ocr.detect(&img).expect("Detection failed");
        let merged = comic_scan::pipeline::merge::group_lines(lines);
        let (inputs, polygons) = comic_scan::pipeline::ocr_merged_bubbles(&ocr, &merged, "en")
            .expect("OCR failed");
        println!("  {} bubbles detected", inputs.len());

        if inputs.is_empty() {
            println!("  No bubbles, skipping");
            continue;
        }

        // Translate
        let translate_req = comic_scan::translation::TranslateRequest {
            bubbles: inputs,
            target_lang: "vi".into(),
            context: vec![],
        };
        let translated = match translation.translate(&translate_req).await {
            Ok(t) => t,
            Err(e) => {
                eprintln!("  Translation failed (LLM may be unavailable): {e:#}");
                continue;
            }
        };
        println!("  {} bubbles translated", translated.len());

        // Fit
        let id_to_polygon: std::collections::HashMap<String, usize> = translate_req.bubbles
            .iter()
            .enumerate()
            .map(|(i, b)| (b.id.clone(), i))
            .collect();

        let mut bubble_results = Vec::new();
        for t in &translated {
            let poly_idx = match id_to_polygon.get(&t.id) {
                Some(&idx) => idx,
                None => continue,
            };
            let polygon = &polygons[poly_idx];
            let fit = comic_scan::fit_engine::FitEngine::fit(&t.translated_text, polygon).unwrap();
            println!(
                "  [{}] font={}px overflow={} text={:?}",
                t.id, fit.font_size_px, fit.overflow, fit.text
            );
            bubble_results.push(comic_scan::api::BubbleResult {
                bubble_id: t.id.clone(),
                polygon: polygon.clone(),
                source_text: t.source_text.clone(),
                translated_text: fit.text,
                font_size_px: fit.font_size_px,
                line_height: fit.line_height,
                overflow: fit.overflow,
            });
        }

        // Render overlay
        let overlay = comic_scan::overlay::render(&img, &bubble_results);

        // Save
        let out_path = out_dir.join(format!("overlay_{name}.png"));
        overlay.save(&out_path).expect("Failed to save overlay");
        println!("  Saved: {}", out_path.display());
    }
}
