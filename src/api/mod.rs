mod handlers;
mod models;

pub use models::*;

use std::sync::Arc;

use anyhow::Result;
use axum::{
    Router,
    routing::{get, post},
};
use tower_http::cors::{AllowOrigin, CorsLayer};

use crate::config::AppConfig;
use crate::runner::TranslationRunner;
use crate::translation::TranslationEngine;

pub struct AppState {
    pub runner: TranslationRunner,
    pub config: AppConfig,
}

impl AppState {
    pub async fn new(config: &AppConfig) -> Result<Arc<Self>> {
        let runner = TranslationRunner::new(config).await?;
        Ok(Arc::new(Self {
            runner,
            config: config.clone(),
        }))
    }
}

/// Resolve the translation engine: use provider_config override if present, else default.
pub fn resolve_engine<'a>(
    state: &'a AppState,
    req: &TranslateImageRequest,
) -> Result<ResolvedEngine<'a>> {
    let has_override = req
        .provider_config
        .as_ref()
        .is_some_and(|c| c.endpoint.is_some() || c.api_key.is_some() || c.model.is_some());

    if has_override {
        let pc = req.provider_config.as_ref().unwrap();
        let mut base = state.config.resolve_provider(&state.config.translation)?;
        if let Some(endpoint) = &pc.endpoint {
            base.endpoint = endpoint.clone();
        }
        if let Some(api_key) = &pc.api_key {
            base.api_key = Some(api_key.clone());
        }
        if let Some(model) = &pc.model {
            base.model = model.clone();
        }
        Ok(ResolvedEngine::Owned(
            crate::runner::build_translation_engine(&base)?,
        ))
    } else {
        Ok(ResolvedEngine::Borrowed(&state.runner.translation))
    }
}

pub enum ResolvedEngine<'a> {
    Borrowed(&'a TranslationEngine),
    Owned(TranslationEngine),
}

impl ResolvedEngine<'_> {
    pub fn as_ref(&self) -> &TranslationEngine {
        match self {
            ResolvedEngine::Borrowed(e) => e,
            ResolvedEngine::Owned(e) => e,
        }
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
