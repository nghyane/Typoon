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
use crate::storage::project::ProjectStore;
use crate::translation::TranslationEngine;

pub struct AppState {
    pub runner: Arc<TranslationRunner>,
    pub project: Option<Arc<ProjectStore>>,
    pub config: AppConfig,
}

impl AppState {
    pub async fn new(config: &AppConfig) -> Result<Arc<Self>> {
        let runner = Arc::new(TranslationRunner::new(config).await?);

        let project = config
            .context
            .project_dir
            .as_ref()
            .and_then(|dir| {
                ProjectStore::open(std::path::Path::new(dir))
                    .map(|store| {
                        if let Some(toml_path) = &config.glossary.import_toml {
                            if let Err(e) =
                                store.glossary_import_toml(std::path::Path::new(toml_path))
                            {
                                tracing::warn!("Glossary TOML import failed: {e}");
                            }
                        }
                        tracing::info!("HTTP project store opened: {dir}");
                        Arc::new(store)
                    })
                    .ok()
            });

        Ok(Arc::new(Self {
            runner,
            project,
            config: config.clone(),
        }))
    }
}

/// Build an overridden translation engine if `provider_config` is present.
/// Returns `Err` if no override is needed (caller uses default).
pub fn resolve_engine_override(
    state: &AppState,
    req: &TranslateImageRequest,
) -> Result<TranslationEngine> {
    let pc = req
        .provider_config
        .as_ref()
        .filter(|c| c.endpoint.is_some() || c.api_key.is_some() || c.model.is_some())
        .ok_or_else(|| anyhow::anyhow!("no override"))?;

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
    crate::runner::build_translation_engine(&base)
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
