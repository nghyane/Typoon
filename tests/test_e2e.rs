use std::path::Path;

const MODELS: &str = "models";
const FIXTURE: &str = "tests/fixtures/en/sample_multi_bubble.webp";
const FIXTURE_MANHWA: &str = "tests/fixtures/en/manhwa_test.webp";
const FIXTURE_MANHWA2: &str = "tests/fixtures/en/manhwa_test2.webp";
const FIXTURE_MANHWA3: &str = "tests/fixtures/en/manhwa_test3.webp";

fn has_ppocr() -> bool {
    Path::new(MODELS).join("ppocr_det.onnx").exists()
        && Path::new(MODELS).join("ppocr_rec.onnx").exists()
}

/// End-to-end: detect → OCR → translate (live LLM) → fit
#[tokio::test]
async fn test_e2e_translate_en() {
    if !has_ppocr() || !Path::new(FIXTURE).exists() {
        eprintln!("Skipping: models or fixture not found");
        return;
    }

    // 1. Load image
    let img = image::open(FIXTURE).expect("Failed to load");
    println!("Image: {} ({}x{})", FIXTURE, img.width(), img.height());

    // 2. Detect + OCR
    let ocr = comic_scan::ocr::OcrEngine::new(MODELS).unwrap();
    assert!(ocr.can_detect(), "PP-OCR detection model not loaded");

    let lines = ocr.detect(&img).expect("Detection failed");
    println!("Detected {} text lines", lines.len());
    assert!(!lines.is_empty());

    let merged = comic_scan::pipeline::merge::group_lines(lines);
    println!("Merged into {} bubbles", merged.len());

    let (inputs, polygons) = comic_scan::pipeline::ocr_merged_bubbles(&ocr, &merged, "en")
        .expect("OCR failed");
    println!("OCR produced {} bubbles", inputs.len());
    assert!(!inputs.is_empty());

    for (i, input) in inputs.iter().enumerate() {
        println!("  [{}] {:?}", input.id, input.source_text);
        let _ = i;
    }

    // 3. Translate via live LLM
    let config = comic_scan::config::AppConfig::load().unwrap();
    let translation = comic_scan::translation::TranslationEngine::new(&config.translation).unwrap();

    let translate_req = comic_scan::translation::TranslateRequest {
        bubbles: inputs,
        target_lang: "vi".into(),
        context: vec![],
    };

    let translated = translation.translate(&translate_req).await;
    match &translated {
        Ok(results) => {
            println!("\n=== Translation Results ({} bubbles) ===", results.len());
            for t in results {
                println!("  [{}] {} → {}", t.id, t.source_text, t.translated_text);
            }
        }
        Err(e) => {
            eprintln!("Translation failed (LLM may be unavailable): {e:#}");
            return;
        }
    }
    let translated = translated.unwrap();
    assert!(!translated.is_empty(), "Expected at least one translation");

    // 4. Fit text into bubbles
    let id_to_polygon: std::collections::HashMap<String, usize> = translate_req.bubbles
        .iter()
        .enumerate()
        .map(|(i, b)| (b.id.clone(), i))
        .collect();

    println!("\n=== Fit Results ===");
    for t in &translated {
        let poly_idx = match id_to_polygon.get(&t.id) {
            Some(&idx) => idx,
            None => {
                println!("  [{}] SKIP (unknown id)", t.id);
                continue;
            }
        };
        let polygon = &polygons[poly_idx];
        let fit = comic_scan::fit_engine::FitEngine::fit(&t.translated_text, polygon).unwrap();
        let (w, h) = (polygon[1][0] - polygon[0][0], polygon[2][1] - polygon[0][1]);
        println!(
            "  [{}] {}x{:.0} → font={}px lines={} overflow={}",
            t.id, w as u32, h, fit.font_size_px,
            fit.text.lines().count(), fit.overflow,
        );
        println!("    text: {:?}", fit.text);
    }
}

/// End-to-end: detect → OCR → translate (live LLM) → fit (manhwa)
#[tokio::test]
async fn test_e2e_translate_manhwa() {
    if !has_ppocr() || !Path::new(FIXTURE_MANHWA).exists() {
        eprintln!("Skipping: models or fixture not found");
        return;
    }

    // 1. Load image
    let img = image::open(FIXTURE_MANHWA).expect("Failed to load");
    println!("Image: {} ({}x{})", FIXTURE_MANHWA, img.width(), img.height());

    // 2. Detect + OCR
    let ocr = comic_scan::ocr::OcrEngine::new(MODELS).unwrap();
    assert!(ocr.can_detect(), "PP-OCR detection model not loaded");

    let lines = ocr.detect(&img).expect("Detection failed");
    println!("Detected {} text lines", lines.len());
    assert!(!lines.is_empty());

    let merged = comic_scan::pipeline::merge::group_lines(lines);
    println!("Merged into {} bubbles", merged.len());

    let (inputs, polygons) = comic_scan::pipeline::ocr_merged_bubbles(&ocr, &merged, "en")
        .expect("OCR failed");
    println!("OCR produced {} bubbles", inputs.len());
    assert!(!inputs.is_empty());

    for (i, input) in inputs.iter().enumerate() {
        println!("  [{}] {:?}", input.id, input.source_text);
        let _ = i;
    }

    // 3. Translate via live LLM
    let config = comic_scan::config::AppConfig::load().unwrap();
    let translation = comic_scan::translation::TranslationEngine::new(&config.translation).unwrap();

    let translate_req = comic_scan::translation::TranslateRequest {
        bubbles: inputs,
        target_lang: "vi".into(),
        context: vec![],
    };

    let translated = translation.translate(&translate_req).await;
    match &translated {
        Ok(results) => {
            println!("\n=== Translation Results ({} bubbles) ===", results.len());
            for t in results {
                println!("  [{}] {} → {}", t.id, t.source_text, t.translated_text);
            }
        }
        Err(e) => {
            eprintln!("Translation failed (LLM may be unavailable): {e:#}");
            return;
        }
    }
    let translated = translated.unwrap();
    assert!(!translated.is_empty(), "Expected at least one translation");

    // 4. Fit text into bubbles
    let id_to_polygon: std::collections::HashMap<String, usize> = translate_req.bubbles
        .iter()
        .enumerate()
        .map(|(i, b)| (b.id.clone(), i))
        .collect();

    println!("\n=== Fit Results ===");
    for t in &translated {
        let poly_idx = match id_to_polygon.get(&t.id) {
            Some(&idx) => idx,
            None => {
                println!("  [{}] SKIP (unknown id)", t.id);
                continue;
            }
        };
        let polygon = &polygons[poly_idx];
        let fit = comic_scan::fit_engine::FitEngine::fit(&t.translated_text, polygon).unwrap();
        let (w, h) = (polygon[1][0] - polygon[0][0], polygon[2][1] - polygon[0][1]);
        println!(
            "  [{}] {}x{:.0} → font={}px lines={} overflow={}",
            t.id, w as u32, h, fit.font_size_px,
            fit.text.lines().count(), fit.overflow,
        );
        println!("    text: {:?}", fit.text);
    }
}

/// End-to-end: detect → OCR → translate (live LLM) → fit (manhwa2)
#[tokio::test]
async fn test_e2e_translate_manhwa2() {
    if !has_ppocr() || !Path::new(FIXTURE_MANHWA2).exists() {
        eprintln!("Skipping: models or fixture not found");
        return;
    }

    // 1. Load image
    let img = image::open(FIXTURE_MANHWA2).expect("Failed to load");
    println!("Image: {} ({}x{})", FIXTURE_MANHWA2, img.width(), img.height());

    // 2. Detect + OCR
    let ocr = comic_scan::ocr::OcrEngine::new(MODELS).unwrap();
    assert!(ocr.can_detect(), "PP-OCR detection model not loaded");

    let lines = ocr.detect(&img).expect("Detection failed");
    println!("Detected {} text lines", lines.len());
    assert!(!lines.is_empty());

    let merged = comic_scan::pipeline::merge::group_lines(lines);
    println!("Merged into {} bubbles", merged.len());

    let (inputs, polygons) = comic_scan::pipeline::ocr_merged_bubbles(&ocr, &merged, "en")
        .expect("OCR failed");
    println!("OCR produced {} bubbles", inputs.len());
    assert!(!inputs.is_empty());

    for (i, input) in inputs.iter().enumerate() {
        println!("  [{}] {:?}", input.id, input.source_text);
        let _ = i;
    }

    // 3. Translate via live LLM
    let config = comic_scan::config::AppConfig::load().unwrap();
    let translation = comic_scan::translation::TranslationEngine::new(&config.translation).unwrap();

    let translate_req = comic_scan::translation::TranslateRequest {
        bubbles: inputs,
        target_lang: "vi".into(),
        context: vec![],
    };

    let translated = translation.translate(&translate_req).await;
    match &translated {
        Ok(results) => {
            println!("\n=== Translation Results ({} bubbles) ===", results.len());
            for t in results {
                println!("  [{}] {} → {}", t.id, t.source_text, t.translated_text);
            }
        }
        Err(e) => {
            eprintln!("Translation failed (LLM may be unavailable): {e:#}");
            return;
        }
    }
    let translated = translated.unwrap();
    assert!(!translated.is_empty(), "Expected at least one translation");

    // 4. Fit text into bubbles
    let id_to_polygon: std::collections::HashMap<String, usize> = translate_req.bubbles
        .iter()
        .enumerate()
        .map(|(i, b)| (b.id.clone(), i))
        .collect();

    println!("\n=== Fit Results ===");
    for t in &translated {
        let poly_idx = match id_to_polygon.get(&t.id) {
            Some(&idx) => idx,
            None => {
                println!("  [{}] SKIP (unknown id)", t.id);
                continue;
            }
        };
        let polygon = &polygons[poly_idx];
        let fit = comic_scan::fit_engine::FitEngine::fit(&t.translated_text, polygon).unwrap();
        let (w, h) = (polygon[1][0] - polygon[0][0], polygon[2][1] - polygon[0][1]);
        println!(
            "  [{}] {}x{:.0} → font={}px lines={} overflow={}",
            t.id, w as u32, h, fit.font_size_px,
            fit.text.lines().count(), fit.overflow,
        );
        println!("    text: {:?}", fit.text);
    }
}

/// End-to-end: detect → OCR → translate (live LLM) → fit (manhwa3)
#[tokio::test]
async fn test_e2e_translate_manhwa3() {
    if !has_ppocr() || !Path::new(FIXTURE_MANHWA3).exists() {
        eprintln!("Skipping: models or fixture not found");
        return;
    }

    // 1. Load image
    let img = image::open(FIXTURE_MANHWA3).expect("Failed to load");
    println!("Image: {} ({}x{})", FIXTURE_MANHWA3, img.width(), img.height());

    // 2. Detect + OCR
    let ocr = comic_scan::ocr::OcrEngine::new(MODELS).unwrap();
    assert!(ocr.can_detect(), "PP-OCR detection model not loaded");

    let lines = ocr.detect(&img).expect("Detection failed");
    println!("Detected {} text lines", lines.len());
    assert!(!lines.is_empty());

    let merged = comic_scan::pipeline::merge::group_lines(lines);
    println!("Merged into {} bubbles", merged.len());

    let (inputs, polygons) = comic_scan::pipeline::ocr_merged_bubbles(&ocr, &merged, "en")
        .expect("OCR failed");
    println!("OCR produced {} bubbles", inputs.len());
    assert!(!inputs.is_empty());

    for (i, input) in inputs.iter().enumerate() {
        println!("  [{}] {:?}", input.id, input.source_text);
        let _ = i;
    }

    // 3. Translate via live LLM
    let config = comic_scan::config::AppConfig::load().unwrap();
    let translation = comic_scan::translation::TranslationEngine::new(&config.translation).unwrap();

    let translate_req = comic_scan::translation::TranslateRequest {
        bubbles: inputs,
        target_lang: "vi".into(),
        context: vec![],
    };

    let translated = translation.translate(&translate_req).await;
    match &translated {
        Ok(results) => {
            println!("\n=== Translation Results ({} bubbles) ===", results.len());
            for t in results {
                println!("  [{}] {} → {}", t.id, t.source_text, t.translated_text);
            }
        }
        Err(e) => {
            eprintln!("Translation failed (LLM may be unavailable): {e:#}");
            return;
        }
    }
    let translated = translated.unwrap();
    assert!(!translated.is_empty(), "Expected at least one translation");

    // 4. Fit text into bubbles
    let id_to_polygon: std::collections::HashMap<String, usize> = translate_req.bubbles
        .iter()
        .enumerate()
        .map(|(i, b)| (b.id.clone(), i))
        .collect();

    println!("\n=== Fit Results ===");
    for t in &translated {
        let poly_idx = match id_to_polygon.get(&t.id) {
            Some(&idx) => idx,
            None => {
                println!("  [{}] SKIP (unknown id)", t.id);
                continue;
            }
        };
        let polygon = &polygons[poly_idx];
        let fit = comic_scan::fit_engine::FitEngine::fit(&t.translated_text, polygon).unwrap();
        let (w, h) = (polygon[1][0] - polygon[0][0], polygon[2][1] - polygon[0][1]);
        println!(
            "  [{}] {}x{:.0} → font={}px lines={} overflow={}",
            t.id, w as u32, h, fit.font_size_px,
            fit.text.lines().count(), fit.overflow,
        );
        println!("    text: {:?}", fit.text);
    }
}
