use std::path::Path;

const MODELS: &str = "models";
const FIXTURE_MULTI: &str = "tests/fixtures/en/sample_multi_bubble.webp";
const FIXTURE_MANHWA: &str = "tests/fixtures/en/manhwa_test.webp";
const FIXTURE_MANHWA2: &str = "tests/fixtures/en/manhwa_test2.webp";
const FIXTURE_MANHWA3: &str = "tests/fixtures/en/manhwa_test3.webp";
const FIXTURE_TEXT: &str = "tests/fixtures/en/test_text_eng.jpg";

fn has_ppocr() -> bool {
    Path::new(MODELS).join("ppocr_det.onnx").exists()
        && Path::new(MODELS).join("ppocr_rec.onnx").exists()
}

#[test]
fn test_ppocr_loads() {
    if !Path::new(MODELS).join("ppocr_rec.onnx").exists() {
        eprintln!("Skipping: ppocr_rec.onnx not found");
        return;
    }
    let engine = comic_scan::ocr::OcrEngine::new(MODELS);
    assert!(engine.is_ok(), "Failed to load: {:?}", engine.err());
}

/// PP-OCR recognition on a clean text image
#[test]
fn test_ppocr_rec_direct() {
    if !Path::new(MODELS).join("ppocr_rec.onnx").exists()
        || !Path::new(FIXTURE_TEXT).exists()
    {
        eprintln!("Skipping: model or fixture not found");
        return;
    }

    let img = image::open(FIXTURE_TEXT).expect("Failed to load");
    let ocr = comic_scan::ocr::OcrEngine::new(MODELS).unwrap();
    let result = ocr.recognize(&img, "en").expect("OCR failed");
    println!("PP-OCR direct: text={:?}, conf={:.3}", result.text, result.confidence);
    assert!(!result.text.is_empty(), "Expected non-empty text");
}

/// Full EN pipeline: PP-OCR det → proximity merge → PP-OCR rec
#[test]
fn test_en_pipeline() {
    if !has_ppocr() || !Path::new(FIXTURE_MULTI).exists() {
        eprintln!("Skipping: models or fixture not found");
        return;
    }

    let img = image::open(FIXTURE_MULTI).expect("Failed to load");
    println!("Image: {} ({}x{})", FIXTURE_MULTI, img.width(), img.height());

    let ocr = comic_scan::ocr::OcrEngine::new(MODELS).unwrap();
    assert!(ocr.can_detect(), "PP-OCR detection model not loaded");

    // Detect text lines
    let lines = ocr.detect(&img).expect("PP-OCR detection failed");
    println!("PP-OCR lines: {}", lines.len());
    assert!(!lines.is_empty(), "Expected at least one text line");

    // Proximity merge → OCR with filters (production path)
    let merged = comic_scan::pipeline::merge::group_lines(lines);
    println!("Grouped into {} bubbles", merged.len());
    assert!(!merged.is_empty(), "Expected at least one bubble group");

    let (inputs, polygons) = comic_scan::pipeline::ocr_merged_bubbles(&ocr, &merged, "en")
        .expect("OCR failed");

    for (i, (input, poly)) in inputs.iter().zip(polygons.iter()).enumerate() {
        let (x, y) = (poly[0][0], poly[0][1]);
        let (w, h) = (poly[1][0] - poly[0][0], poly[2][1] - poly[0][1]);
        println!("  Bubble {i}: pos=({:.0},{:.0}) size={:.0}x{:.0} text={:?}",
            x, y, w, h, input.source_text);
    }

    assert!(!inputs.is_empty(), "Expected at least one bubble with text");
}

/// Manhwa (webtoon) EN pipeline test — long vertical strip
#[test]
fn test_manhwa_pipeline() {
    if !has_ppocr() || !Path::new(FIXTURE_MANHWA).exists() {
        eprintln!("Skipping: models or fixture not found");
        return;
    }

    let img = image::open(FIXTURE_MANHWA).expect("Failed to load");
    println!("Image: {} ({}x{})", FIXTURE_MANHWA, img.width(), img.height());

    let ocr = comic_scan::ocr::OcrEngine::new(MODELS).unwrap();
    assert!(ocr.can_detect(), "PP-OCR detection model not loaded");

    let lines = ocr.detect(&img).expect("PP-OCR detection failed");
    println!("PP-OCR lines: {}", lines.len());

    let merged = comic_scan::pipeline::merge::group_lines(lines);
    println!("Grouped into {} bubbles", merged.len());

    let (inputs, polygons) = comic_scan::pipeline::ocr_merged_bubbles(&ocr, &merged, "en")
        .expect("OCR failed");

    for (i, (input, poly)) in inputs.iter().zip(polygons.iter()).enumerate() {
        let (x, y) = (poly[0][0], poly[0][1]);
        let (w, h) = (poly[1][0] - poly[0][0], poly[2][1] - poly[0][1]);
        println!("  Bubble {i}: pos=({:.0},{:.0}) size={:.0}x{:.0} text={:?}",
            x, y, w, h, input.source_text);
    }

    println!("\n{}/{} bubbles output", inputs.len(), merged.len());
    assert!(!inputs.is_empty(), "Expected at least one bubble with text");
}

/// Manhwa (webtoon) EN pipeline test 2 — narration boxes
#[test]
fn test_manhwa2_pipeline() {
    if !has_ppocr() || !Path::new(FIXTURE_MANHWA2).exists() {
        eprintln!("Skipping: models or fixture not found");
        return;
    }

    let img = image::open(FIXTURE_MANHWA2).expect("Failed to load");
    println!("Image: {} ({}x{})", FIXTURE_MANHWA2, img.width(), img.height());

    let ocr = comic_scan::ocr::OcrEngine::new(MODELS).unwrap();
    assert!(ocr.can_detect(), "PP-OCR detection model not loaded");

    let lines = ocr.detect(&img).expect("PP-OCR detection failed");
    println!("PP-OCR lines: {}", lines.len());

    let merged = comic_scan::pipeline::merge::group_lines(lines);
    println!("Grouped into {} bubbles", merged.len());

    let (inputs, polygons) = comic_scan::pipeline::ocr_merged_bubbles(&ocr, &merged, "en")
        .expect("OCR failed");

    for (i, (input, poly)) in inputs.iter().zip(polygons.iter()).enumerate() {
        let (x, y) = (poly[0][0], poly[0][1]);
        let (w, h) = (poly[1][0] - poly[0][0], poly[2][1] - poly[0][1]);
        println!("  Bubble {i}: pos=({:.0},{:.0}) size={:.0}x{:.0} text={:?}",
            x, y, w, h, input.source_text);
    }

    println!("\n{}/{} bubbles output", inputs.len(), merged.len());
    assert!(!inputs.is_empty(), "Expected at least one bubble with text");
}

/// Manhwa (webtoon) EN pipeline test 3 — decorative fonts + many watermarks
#[test]
fn test_manhwa3_pipeline() {
    if !has_ppocr() || !Path::new(FIXTURE_MANHWA3).exists() {
        eprintln!("Skipping: models or fixture not found");
        return;
    }

    let img = image::open(FIXTURE_MANHWA3).expect("Failed to load");
    println!("Image: {} ({}x{})", FIXTURE_MANHWA3, img.width(), img.height());

    let ocr = comic_scan::ocr::OcrEngine::new(MODELS).unwrap();
    assert!(ocr.can_detect(), "PP-OCR detection model not loaded");

    let lines = ocr.detect(&img).expect("PP-OCR detection failed");
    println!("PP-OCR lines: {}", lines.len());

    let merged = comic_scan::pipeline::merge::group_lines(lines);
    println!("Grouped into {} bubbles", merged.len());

    let (inputs, polygons) = comic_scan::pipeline::ocr_merged_bubbles(&ocr, &merged, "en")
        .expect("OCR failed");

    for (i, (input, poly)) in inputs.iter().zip(polygons.iter()).enumerate() {
        let (x, y) = (poly[0][0], poly[0][1]);
        let (w, h) = (poly[1][0] - poly[0][0], poly[2][1] - poly[0][1]);
        println!("  Bubble {i}: pos=({:.0},{:.0}) size={:.0}x{:.0} text={:?}",
            x, y, w, h, input.source_text);
    }

    println!("\n{}/{} bubbles output", inputs.len(), merged.len());
    assert!(!inputs.is_empty(), "Expected at least one bubble with text");
}
