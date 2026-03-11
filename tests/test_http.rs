use std::path::Path;

use base64::Engine;
use base64::engine::general_purpose::STANDARD;
use http::StatusCode;
use http_body_util::BodyExt;
use tower::ServiceExt;

const MODELS: &str = "models";
const FIXTURE: &str = "tests/fixtures/en/sample_multi_bubble.webp";

fn has_models() -> bool {
    Path::new(MODELS).join("ppocr_det.onnx").exists()
        && Path::new(MODELS).join("ppocr_rec.onnx").exists()
}

/// HTTP integration: POST /translate-image with a real image → get translated bubbles
#[tokio::test]
async fn test_http_translate_image() {
    if !has_models() || !Path::new(FIXTURE).exists() {
        eprintln!("Skipping: models or fixture not found");
        return;
    }

    let config = comic_scan::config::AppConfig::load().unwrap();
    let state = comic_scan::api::AppState::new(&config).await.unwrap();
    let app = comic_scan::api::router(state);

    let image_bytes = std::fs::read(FIXTURE).unwrap();
    let image_b64 = STANDARD.encode(&image_bytes);

    let body = serde_json::json!({
        "image_id": "test-http-001",
        "image_blob_b64": image_b64,
        "source_lang": "en",
        "target_lang": "vi"
    });

    let req = http::Request::builder()
        .method("POST")
        .uri("/translate-image")
        .header("content-type", "application/json")
        .body(axum::body::Body::from(serde_json::to_vec(&body).unwrap()))
        .unwrap();

    let resp = app.oneshot(req).await.unwrap();

    if resp.status() == StatusCode::INTERNAL_SERVER_ERROR {
        let body_bytes = resp.into_body().collect().await.unwrap().to_bytes();
        let body: serde_json::Value = serde_json::from_slice(&body_bytes).unwrap();
        eprintln!("Server error (LLM may be unavailable): {}", body["status"]);
        return;
    }

    assert_eq!(resp.status(), StatusCode::OK);

    let body_bytes = resp.into_body().collect().await.unwrap().to_bytes();
    let resp: comic_scan::api::TranslateImageResponse = serde_json::from_slice(&body_bytes).unwrap();

    println!("Response: image_id={}, status={}", resp.image_id, resp.status);
    assert_eq!(resp.image_id, "test-http-001");
    assert!(resp.status.starts_with("ok"), "Unexpected status: {}", resp.status);

    println!("{} bubbles returned", resp.bubbles.len());
    for b in &resp.bubbles {
        println!(
            "  [{}] font={}px overflow={} text={:?}",
            b.bubble_id, b.font_size_px, b.overflow, b.translated_text
        );
        assert!(!b.translated_text.is_empty());
        assert!(b.font_size_px > 0);
        assert!(!b.polygon.is_empty());
    }
    assert!(!resp.bubbles.is_empty(), "Expected at least one bubble");
}

/// HTTP: GET /health returns ready status
#[tokio::test]
async fn test_http_health() {
    if !has_models() {
        eprintln!("Skipping: models not found");
        return;
    }

    let config = comic_scan::config::AppConfig::load().unwrap();
    let state = comic_scan::api::AppState::new(&config).await.unwrap();
    let app = comic_scan::api::router(state);

    let req = http::Request::builder()
        .method("GET")
        .uri("/health")
        .body(axum::body::Body::empty())
        .unwrap();

    let resp = app.oneshot(req).await.unwrap();
    assert_eq!(resp.status(), StatusCode::OK);

    let body_bytes = resp.into_body().collect().await.unwrap().to_bytes();
    let health: comic_scan::api::HealthResponse = serde_json::from_slice(&body_bytes).unwrap();

    println!("Health: ready={}, det={}, ocr={}, translation={}",
        health.ready, health.detection_model_loaded,
        health.ocr_model_loaded, health.translation_configured);
    assert!(health.ready);
}

/// HTTP: POST /translate-image with invalid JSON returns 4xx
#[tokio::test]
async fn test_http_bad_request() {
    if !has_models() {
        eprintln!("Skipping: models not found");
        return;
    }

    let config = comic_scan::config::AppConfig::load().unwrap();
    let state = comic_scan::api::AppState::new(&config).await.unwrap();
    let app = comic_scan::api::router(state);

    let req = http::Request::builder()
        .method("POST")
        .uri("/translate-image")
        .header("content-type", "application/json")
        .body(axum::body::Body::from(b"{}".to_vec()))
        .unwrap();

    let resp = app.oneshot(req).await.unwrap();
    // axum returns 422 for deserialization errors
    assert!(
        resp.status().is_client_error(),
        "Expected 4xx, got {}",
        resp.status()
    );
}
