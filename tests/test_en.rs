use std::path::Path;

const MODELS: &str = "models";

#[tokio::test]
async fn test_ppocr_loads() {
    if !Path::new(MODELS).join("ppocr_rec.onnx").exists() {
        eprintln!("Skipping: ppocr_rec.onnx not found");
        return;
    }
    let engine = comic_scan::ocr::OcrEngine::new(MODELS).await;
    assert!(engine.is_ok(), "Failed to load: {:?}", engine.err());
}

/// PP-OCR recognition on a clean text image
#[tokio::test]
async fn test_ppocr_rec_direct() {
    let fixture = "tests/fixtures/en/test_text_eng.jpg";
    if !Path::new(MODELS).join("ppocr_rec.onnx").exists() || !Path::new(fixture).exists() {
        eprintln!("Skipping: model or fixture not found");
        return;
    }

    let img = image::open(fixture).expect("Failed to load");
    let ocr = comic_scan::ocr::OcrEngine::new(MODELS).await.unwrap();
    let result = ocr.recognize(&img, "en").expect("OCR failed");
    println!(
        "PP-OCR direct: text={:?}, conf={:.3}",
        result.text, result.confidence
    );
    assert!(!result.text.is_empty(), "Expected non-empty text");
}
