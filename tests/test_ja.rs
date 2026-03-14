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
    let model_path = Path::new(MODELS).join("comic-text-detector.onnx");
    let detector = comic_scan::detection::TextDetector::new(model_path);
    // Trigger lazy load by detecting on a small test image
    let test_img = image::DynamicImage::new_rgb8(100, 100);
    let result = detector.detect(&test_img);
    assert!(result.is_ok(), "Failed to load/detect: {:?}", result.err());
}

#[tokio::test]
async fn test_manga_ocr_loads() {
    if !Path::new(MODELS).join("encoder_model.onnx").exists() {
        eprintln!("Skipping: manga-ocr models not found");
        return;
    }
    let engine = comic_scan::ocr::OcrEngine::new(MODELS).await;
    assert!(engine.is_ok(), "Failed to load: {:?}", engine.err());
}

/// Full JA pipeline: comic-text-detector → manga-ocr
#[tokio::test]
async fn test_ja_pipeline() {
    if !has_models() || !Path::new(FIXTURE).exists() {
        eprintln!("Skipping: models or fixture not found");
        return;
    }

    let img = image::open(FIXTURE).expect("Failed to load");
    println!("Image: {} ({}x{})", FIXTURE, img.width(), img.height());

    // Detect
    let model_path = Path::new(MODELS).join("comic-text-detector.onnx");
    let detector = comic_scan::detection::TextDetector::new(model_path);
    let regions = detector.detect(&img).expect("Detection failed");
    println!("Detected {} regions", regions.len());
    assert!(!regions.is_empty(), "Expected at least one region");

    // OCR
    let ocr = comic_scan::ocr::OcrEngine::new(MODELS).await.unwrap();

    println!("=== JA Pipeline: CTD → manga-ocr ===");
    let mut recognized = 0;
    for (i, region) in regions.iter().enumerate() {
        match ocr.recognize(&region.crop, "ja") {
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
