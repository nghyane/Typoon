use std::path::Path;

const MODELS: &str = "models";
const FIXTURE: &str = "tests/fixtures/ja/sample_manga.webp";

fn has_models() -> bool {
    Path::new(MODELS).join("comic-text-detector.onnx").exists()
        && Path::new(MODELS).join("encoder_model.onnx").exists()
}

#[test]
fn test_ctd_loads() {
    if !Path::new(MODELS).join("comic-text-detector.onnx").exists() {
        eprintln!("Skipping: comic-text-detector.onnx not found");
        return;
    }
    let detector = comic_scan::detection::TextDetector::new(MODELS);
    assert!(detector.is_ok(), "Failed to load: {:?}", detector.err());
}

#[test]
fn test_manga_ocr_loads() {
    if !Path::new(MODELS).join("encoder_model.onnx").exists() {
        eprintln!("Skipping: manga-ocr models not found");
        return;
    }
    let adapter = comic_scan::ocr::manga_ocr_adapter(MODELS);
    assert!(adapter.is_ok(), "Failed to load: {:?}", adapter.err());
}

/// Full JA pipeline: comic-text-detector → manga-ocr
#[test]
fn test_ja_pipeline() {
    if !has_models() || !Path::new(FIXTURE).exists() {
        eprintln!("Skipping: models or fixture not found");
        return;
    }

    let img = image::open(FIXTURE).expect("Failed to load");
    println!("Image: {} ({}x{})", FIXTURE, img.width(), img.height());

    // Detect
    let mut detector = comic_scan::detection::TextDetector::new(MODELS).unwrap();
    let regions = detector.detect(&img).expect("Detection failed");
    println!("Detected {} regions", regions.len());
    assert!(!regions.is_empty(), "Expected at least one region");

    // OCR
    let ocr = comic_scan::ocr::manga_ocr_adapter(MODELS).unwrap();
    use comic_scan::ocr::OcrProvider;

    println!("=== JA Pipeline: CTD → manga-ocr ===");
    let mut recognized = 0;
    for (i, region) in regions.iter().enumerate() {
        match ocr.recognize(&region.crop) {
            Ok(r) => {
                println!("  Region {i}: crop={}x{}, conf={:.3}, text={:?}",
                    region.crop.width(), region.crop.height(), region.confidence, r.text);
                if !r.text.trim().is_empty() {
                    recognized += 1;
                }
            }
            Err(e) => println!("  Region {i}: ERROR: {e}"),
        }
    }
    println!("=== {recognized}/{} recognized ===", regions.len());
    assert!(recognized > 0, "Expected at least one recognized region");
}
