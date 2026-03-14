use std::sync::Arc;

use anyhow::Result;

use crate::agent::Provider;
use crate::config::{AppConfig, ProviderType};
use crate::context::ContextStore;
use crate::detection::TextDetector;
use crate::glossary::Glossary;
use crate::inpaint::LamaInpainter;
use crate::model_hub::lazy::LazySession;
use crate::model_hub::{self, Model};
use crate::ocr::OcrEngine;
use crate::translation::TranslationEngine;

/// Headless translation runner — holds all pipeline components without Axum/HTTP.
///
/// All ONNX models use `LazySession` internally — sessions are created on first
/// use, not at startup. This avoids CoreML context overhead for unused models.
///
/// Detection components (`detector`, `ocr`) are wrapped in `Arc` so they can be
/// cloned and used in a separate task for pipeline parallelism.
pub struct TranslationRunner {
    pub detector: Arc<TextDetector>,
    pub ocr: Arc<OcrEngine>,
    pub translation: TranslationEngine,
    pub inpainter: Option<LamaInpainter>,
    pub glossary: Option<Glossary>,
    pub context_store: Option<Arc<ContextStore>>,
    pub context_agent: Option<Box<dyn Provider>>,
}

impl TranslationRunner {
    pub async fn new(config: &AppConfig) -> Result<Self> {
        let cache_dir = std::path::PathBuf::from(&config.models_dir).join("ep_cache");

        let ctd_path = model_hub::resolve(&config.models_dir, Model::ComicTextDetector).await?;
        let detector = TextDetector::new(
            LazySession::new_accelerated(ctd_path, Some(cache_dir.clone())),
        );
        let ocr = OcrEngine::new(&config.models_dir).await?;

        let resolved = config.resolve_provider(&config.translation)?;
        let translation = build_translation_engine(&resolved)?;

        let inpainter = match model_hub::resolve_optional(&config.models_dir, Model::Lama).await {
            Some(path) => {
                tracing::info!("LaMa model path resolved (lazy load): {}", path.display());
                Some(LamaInpainter::new(
                    LazySession::new(path),
                ))
            }
            None => {
                tracing::info!("LaMa model not available, using median fill only");
                None
            }
        };

        let context_store = if let Some(db_path) = &config.context.db_path {
            match ContextStore::open(std::path::Path::new(db_path)) {
                Ok(store) => {
                    tracing::info!("Context store opened: {db_path}");
                    Some(store)
                }
                Err(e) => {
                    tracing::warn!("Context store init failed: {e}");
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

        let context_agent: Option<Box<dyn Provider>> = if let (Some(agent_config), true) =
            (&config.context_agent, context_store.is_some())
        {
            match config.resolve_provider(agent_config) {
                Ok(resolved) => {
                    let api_key = resolved.api_key.as_deref().unwrap_or("not-needed");
                    let provider: Result<Box<dyn Provider>> = match resolved.provider_type {
                        ProviderType::Anthropic => {
                            crate::agent::anthropic::AnthropicProvider::new(
                                &resolved.endpoint, api_key, &resolved.model,
                            )
                            .map(|p| Box::new(p) as Box<dyn Provider>)
                        }
                        ProviderType::OpenAI => {
                            crate::agent::openai::OpenAIProvider::new(
                                &resolved.endpoint, Some(api_key), &resolved.model,
                            )
                            .map(|p| Box::new(p) as Box<dyn Provider>)
                        }
                    };
                    match provider {
                        Ok(p) => {
                            tracing::info!("Context agent ready ({})", resolved.model);
                            Some(p)
                        }
                        Err(e) => {
                            tracing::warn!("Context agent init failed: {e}");
                            None
                        }
                    }
                }
                Err(e) => {
                    tracing::warn!("Context agent provider resolution failed: {e}");
                    None
                }
            }
        } else {
            None
        };

        Ok(Self {
            detector: Arc::new(detector),
            ocr: Arc::new(ocr),
            translation,
            inpainter,
            glossary,
            context_store: context_store.map(Arc::new),
            context_agent,
        })
    }
}

/// Build a `TranslationEngine` from resolved provider config.
pub fn build_translation_engine(
    resolved: &crate::config::ResolvedProvider,
) -> Result<TranslationEngine> {
    let api_key = resolved.api_key.as_deref().unwrap_or("not-needed");
    let provider: Box<dyn Provider> = match resolved.provider_type {
        ProviderType::OpenAI => {
            let p = crate::agent::openai::OpenAIProvider::new(
                &resolved.endpoint,
                Some(api_key),
                &resolved.model,
            )?
            .with_reasoning_effort(resolved.reasoning_effort.clone());
            Box::new(p)
        }
        ProviderType::Anthropic => {
            Box::new(crate::agent::anthropic::AnthropicProvider::new(
                &resolved.endpoint,
                api_key,
                &resolved.model,
            )?)
        }
    };
    Ok(TranslationEngine::new(provider))
}
