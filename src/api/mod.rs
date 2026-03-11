mod handlers;
mod models;

pub use models::*;

use std::sync::{Arc, Mutex};

use anyhow::Result;
use axum::{Router, routing::{get, post}};
use tower_http::cors::{CorsLayer, AllowOrigin};

use crate::cache::DiskCache;
use crate::config::AppConfig;
use crate::detection::TextDetector;
use crate::ocr::OcrEngine;
use crate::translation::TranslationEngine;

pub struct AppState {
    pub detector: Mutex<TextDetector>,
    pub ocr: OcrEngine,
    pub translation: TranslationEngine,
    pub cache: DiskCache,
    pub config: AppConfig,
}

impl AppState {
    pub async fn new(config: &AppConfig) -> Result<Arc<Self>> {
        let detector = TextDetector::new(&config.models_dir)?;
        let ocr = OcrEngine::new(&config.models_dir)?;
        let translation = TranslationEngine::new(&config.translation)?;
        let cache = DiskCache::new(&config.cache_dir)?;

        Ok(Arc::new(Self {
            detector: Mutex::new(detector),
            ocr,
            translation,
            cache,
            config: config.clone(),
        }))
    }
}

pub fn router(state: Arc<AppState>) -> Router {
    let cors = CorsLayer::new()
        .allow_origin(AllowOrigin::predicate(|origin, _| {
            // Allow browser extensions and localhost
            let origin = origin.as_bytes();
            origin.starts_with(b"chrome-extension://")
                || origin.starts_with(b"moz-extension://")
                || origin.starts_with(b"http://127.0.0.1")
                || origin.starts_with(b"http://localhost")
        }))
        .allow_methods(tower_http::cors::Any)
        .allow_headers(tower_http::cors::Any);

    Router::new()
        .route("/health", get(handlers::health))
        .route("/translate-image", post(handlers::translate_image))
        .layer(cors)
        .with_state(state)
}
