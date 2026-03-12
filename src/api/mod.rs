mod handlers;
mod models;

pub use models::*;

use std::sync::{Arc, Mutex};

use anyhow::Result;
use axum::{Router, routing::{get, post}};
use tower_http::cors::{CorsLayer, AllowOrigin};

use crate::config::AppConfig;
use crate::detection::TextDetector;
use crate::glossary::Glossary;
use crate::model_hub::{self, Model};
use crate::ocr::OcrEngine;
use crate::canvas_agent::CanvasAgent;
use crate::translation::TranslationEngine;

pub struct AppState {
    pub detector: Mutex<TextDetector>,
    pub ocr: OcrEngine,
    pub translation: TranslationEngine,
    pub canvas_agent: Option<CanvasAgent>,
    pub glossary: Option<Glossary>,
    pub config: AppConfig,
}

impl AppState {
    pub async fn new(config: &AppConfig) -> Result<Arc<Self>> {
        let ctd_path = model_hub::resolve(&config.models_dir, Model::ComicTextDetector).await?;
        let detector = TextDetector::new(&ctd_path)?;
        let ocr = OcrEngine::new(&config.models_dir).await?;
        let translation = TranslationEngine::new(&config.translation)?;

        let canvas_agent = if config.canvas_agent.enabled {
            let agent_translation = config.canvas_agent.resolved_translation(&config.translation);
            match CanvasAgent::new(&agent_translation) {
                Ok(agent) => {
                    tracing::info!("CanvasAgent enabled");
                    Some(agent)
                }
                Err(e) => {
                    tracing::warn!("CanvasAgent init failed, disabled: {e}");
                    None
                }
            }
        } else {
            None
        };

        let glossary = if let Some(db_path) = &config.glossary.db_path {
            match Glossary::open(std::path::Path::new(db_path)) {
                Ok(g) => {
                    if let Some(toml_path) = &config.glossary.import_toml {
                        if let Err(e) = g.import_toml(std::path::Path::new(toml_path)) {
                            tracing::warn!("Glossary TOML import failed: {e}");
                        }
                    }
                    tracing::info!("Glossary loaded from {db_path}");
                    Some(g)
                }
                Err(e) => {
                    tracing::warn!("Glossary init failed: {e}");
                    None
                }
            }
        } else {
            None
        };

        Ok(Arc::new(Self {
            detector: Mutex::new(detector),
            ocr,
            translation,
            canvas_agent,
            glossary,
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
