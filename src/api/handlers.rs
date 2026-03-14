use std::sync::Arc;

use axum::Json;
use axum::extract::State;
use axum::http::StatusCode;

use super::{AppState, HealthResponse, TranslateImageRequest, TranslateImageResponse};
use crate::pipeline;

pub async fn health(
    State(state): State<Arc<AppState>>,
) -> Json<HealthResponse> {
    Json(HealthResponse {
        ready: true,
        detection_model_loaded: state.runner.detector.is_loaded(),
        ocr_model_loaded: state.runner.ocr.is_loaded(),
        translation_configured: true,
    })
}

pub async fn translate_image(
    State(state): State<Arc<AppState>>,
    Json(req): Json<TranslateImageRequest>,
) -> (StatusCode, Json<TranslateImageResponse>) {
    match pipeline::process_image(&state, &req).await {
        Ok(response) => (StatusCode::OK, Json(response)),
        Err(e) => {
            tracing::error!("translate-image failed: {e:#}");
            (StatusCode::INTERNAL_SERVER_ERROR, Json(TranslateImageResponse {
                image_id: req.image_id,
                status: format!("error: {e}"),
                bubbles: vec![],
                rendered_image_png_b64: None,
            }))
        }
    }
}
