use std::sync::Arc;

use anyhow::Result;
use axum::Json;
use axum::extract::State;
use axum::http::StatusCode;
use base64::Engine;
use base64::engine::general_purpose::STANDARD;

use super::{AppState, HealthResponse, TranslateImageRequest, TranslateImageResponse};
use crate::pipeline;
use crate::pipeline::types::BubbleResult;
use crate::translation::BubbleTranslated;

pub async fn health(State(state): State<Arc<AppState>>) -> Json<HealthResponse> {
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
    match process_image(&state, &req).await {
        Ok(response) => (StatusCode::OK, Json(response)),
        Err(e) => {
            tracing::error!("translate-image failed: {e:#}");
            (
                StatusCode::INTERNAL_SERVER_ERROR,
                Json(TranslateImageResponse {
                    image_id: req.image_id,
                    status: format!("error: {e}"),
                    bubbles: vec![],
                    rendered_image_png_b64: None,
                }),
            )
        }
    }
}

/// HTTP single-image pipeline: decode → detect → OCR → translate → fit → render.
async fn process_image(
    state: &Arc<AppState>,
    req: &TranslateImageRequest,
) -> Result<TranslateImageResponse> {
    let image_bytes = STANDARD.decode(&req.image_blob_b64)?;
    let img = image::load_from_memory(&image_bytes)?;

    let source_lang = pipeline::detect_source_lang(req.source_lang.as_deref(), &req.target_lang);
    let engine = super::resolve_engine(state, req)?;

    let runner = &state.runner;
    let det = Arc::clone(&runner.detector);
    let ocr = Arc::clone(&runner.ocr);
    let images = vec![img];

    let detections = tokio::task::spawn_blocking({
        let images = images.clone();
        let lang = source_lang.to_string();
        move || pipeline::chapter::detect_chapter(&det, &ocr, &images, &lang)
    })
    .await??;

    let context = build_context(req);

    let output = pipeline::chapter::translate_and_render_with_engine(
        runner,
        engine.as_ref(),
        detections,
        &images,
        &req.target_lang,
        source_lang,
        context,
    )
    .await?;

    let mut bubbles = Vec::new();
    let mut rendered_image_png_b64 = None;

    for page in &output.pages {
        for b in &page.bubbles {
            bubbles.push(BubbleResult::from_translated(b, page.page_index));
        }
        let png_bytes = crate::render::overlay::encode_png(&page.image);
        rendered_image_png_b64 = Some(STANDARD.encode(&png_bytes));
    }

    Ok(TranslateImageResponse {
        image_id: req.image_id.clone(),
        status: "ok".into(),
        bubbles,
        rendered_image_png_b64,
    })
}

fn build_context(req: &TranslateImageRequest) -> Vec<BubbleTranslated> {
    let Some(hint) = &req.context_hint else {
        return vec![];
    };

    hint.previous_translations
        .iter()
        .flat_map(|pt| {
            pt.bubbles.iter().map(|b| BubbleTranslated {
                id: String::new(),
                source_text: b.source_text.clone(),
                translated_text: b.translated_text.clone(),
            })
        })
        .collect()
}
