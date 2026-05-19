// SPDX-License-Identifier: GPL-3.0-or-later
//! HTTP server for the inpaint container.
//!
//! POST /inpaint?w=<width>&h=<height>
//!   Body: RGB bytes (W×H×3) ++ mask bytes (W×H)
//!   Response: RGB bytes (W×H×3) inpainted
//!
//! GET /health → {"ok":true}

use std::{path::PathBuf, sync::Arc};

use anyhow::Result;
use axum::{
    Router,
    body::Bytes,
    extract::{Query, State},
    http::StatusCode,
    response::{IntoResponse, Response},
    routing::{get, post},
};
use serde::Deserialize;
use tokio::net::TcpListener;
use tracing::info;
use typoon_inpaint::Inpainter;

// ──────────────────────────────────────────────────────────────────────────────

#[derive(Clone)]
struct AppState {
    inpainter: Arc<Inpainter>,
}

#[derive(Deserialize)]
struct InpaintQuery {
    w: u32,
    h: u32,
}

async fn health() -> impl IntoResponse {
    (StatusCode::OK, r#"{"ok":true}"#)
}

async fn inpaint_handler(
    Query(params): Query<InpaintQuery>,
    State(state): State<AppState>,
    body: Bytes,
) -> Response {
    let w = params.w as usize;
    let h = params.h as usize;
    let expected = w * h * 3 + w * h;

    if body.len() != expected {
        return (
            StatusCode::BAD_REQUEST,
            format!("body {len} != expected {expected}", len = body.len()),
        ).into_response();
    }

    let rgb_len  = w * h * 3;
    let image_rgb = body[..rgb_len].to_vec();
    let mask      = body[rgb_len..].to_vec();
    let inp       = Arc::clone(&state.inpainter);

    let result = tokio::task::spawn_blocking(move || {
        inp.inpaint(&image_rgb, &mask, params.w, params.h)
    }).await;

    match result {
        Ok(Ok(out)) => (StatusCode::OK, out).into_response(),
        Ok(Err(e))  => (StatusCode::INTERNAL_SERVER_ERROR, e.to_string()).into_response(),
        Err(e)      => (StatusCode::INTERNAL_SERVER_ERROR, e.to_string()).into_response(),
    }
}

// ──────────────────────────────────────────────────────────────────────────────

#[tokio::main]
async fn main() -> Result<()> {
    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::from_default_env()
                .add_directive("typoon_inpaint=info".parse()?)
                .add_directive("serve=info".parse()?),
        )
        .init();

    let weights_path = PathBuf::from(
        std::env::var("MODEL_PATH").unwrap_or_else(|_| "/app/model.safetensors".to_string()),
    );
    let fp16 = std::env::var("FP16").map(|v| v == "1" || v.eq_ignore_ascii_case("true")).unwrap_or(false);
    let port: u16 = std::env::var("PORT").ok().and_then(|p| p.parse().ok()).unwrap_or(8000);

    info!(path = %weights_path.display(), fp16, port, "loading model...");
    let inpainter = Arc::new(Inpainter::load(&weights_path, fp16)?);
    info!("model ready");

    let state = AppState { inpainter };
    let app = Router::new()
        .route("/health", get(health))
        .route("/inpaint", post(inpaint_handler))
        .with_state(state);

    let listener = TcpListener::bind(format!("0.0.0.0:{port}")).await?;
    info!(addr = %listener.local_addr()?, "listening");
    axum::serve(listener, app).await?;
    Ok(())
}
