use std::sync::Arc;

use anyhow::Result;

use crate::llm::Provider;
use crate::config::{AppConfig, ProviderType};
use crate::storage::project::ProjectStore;
use crate::vision::detection::TextDetector;
use crate::vision::inpaint::LamaInpainter;
use crate::model_hub::lazy::LazySession;
use crate::model_hub::{self, Model};
use crate::vision::ocr::OcrEngine;
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
    render_executor: RenderExecutor,
    max_pending_render_jobs: usize,
    pub default_project: Option<Arc<ProjectStore>>,
    pub context_agent: Option<Box<dyn Provider>>,
}

struct RenderExecutor {
    workers: usize,
    pool: rayon::ThreadPool,
}

impl RenderExecutor {
    fn new(workers: usize) -> Result<Self> {
        let workers = workers.max(1);
        let pool = rayon::ThreadPoolBuilder::new()
            .num_threads(workers)
            .thread_name(|idx| format!("render-{idx}"))
            .build()
            .map_err(|e| anyhow::anyhow!("Failed to create render worker pool: {e}"))?;
        Ok(Self { workers, pool })
    }

    fn workers(&self) -> usize {
        self.workers
    }

    fn install<R: Send>(&self, op: impl FnOnce() -> R + Send) -> R {
        self.pool.install(op)
    }
}

impl TranslationRunner {
    pub async fn new(config: &AppConfig) -> Result<Self> {
        let ctd_path = model_hub::resolve(&config.models_dir, Model::ComicTextDetector).await?;
        let detector = TextDetector::new(LazySession::new(ctd_path));
        let ocr = OcrEngine::new(&config.models_dir).await?;

        let resolved = config.resolve_provider(&config.translation)?;
        let translation = build_translation_engine(&resolved)?;

        let inpainter = match model_hub::resolve_optional(&config.models_dir, Model::Lama).await {
            Some(path) => {
                tracing::info!("LaMa model path resolved (lazy load): {}", path.display());
                Some(LamaInpainter::new(LazySession::gpu(path)))
            }
            None => {
                tracing::info!("LaMa model not available, using median fill only");
                None
            }
        };

        let render_workers = config.runtime.effective_render_workers(inpainter.is_some());
        let render_executor = RenderExecutor::new(render_workers)?;
        let max_pending_render_jobs = config.runtime.max_pending_render_jobs.max(1);
        tracing::info!(
            "Render executor initialized with {} worker(s), pending queue {} (LaMa: {})",
            render_executor.workers(),
            max_pending_render_jobs,
            inpainter.is_some()
        );

        let default_project = if let Some(project_dir) = &config.context.project_dir {
            match ProjectStore::open(std::path::Path::new(project_dir)) {
                Ok(store) => {
                    if let Some(toml_path) = &config.glossary.import_toml {
                        if let Err(e) =
                            store.glossary_import_toml(std::path::Path::new(toml_path))
                        {
                            tracing::warn!("Glossary TOML import failed: {e}");
                        }
                    }
                    tracing::info!("Project store opened: {project_dir}");
                    Some(store)
                }
                Err(e) => {
                    tracing::warn!("Project store init failed: {e}");
                    None
                }
            }
        } else {
            None
        };

        let context_agent: Option<Box<dyn Provider>> = if let (Some(agent_config), true) =
            (&config.context_agent, default_project.is_some())
        {
            match config.resolve_provider(agent_config) {
                Ok(resolved) => {
                    let api_key = resolved.api_key.as_deref().unwrap_or("not-needed");
                    let provider: Result<Box<dyn Provider>> = match resolved.provider_type {
                        ProviderType::Anthropic => crate::llm::anthropic::AnthropicProvider::new(
                            &resolved.endpoint,
                            api_key,
                            &resolved.model,
                        )
                        .map(|p| Box::new(p) as Box<dyn Provider>),
                        ProviderType::OpenAI => crate::llm::openai::OpenAIProvider::new(
                            &resolved.endpoint,
                            Some(api_key),
                            &resolved.model,
                        )
                        .map(|p| Box::new(p) as Box<dyn Provider>),
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
            render_executor,
            max_pending_render_jobs,
            default_project: default_project.map(Arc::new),
            context_agent,
        })
    }

    pub fn install_render<R: Send>(&self, op: impl FnOnce() -> R + Send) -> R {
        self.render_executor.install(op)
    }

    pub fn render_workers(&self) -> usize {
        self.render_executor.workers()
    }

    pub fn max_pending_render_jobs(&self) -> usize {
        self.max_pending_render_jobs
    }
}

/// Build a `TranslationEngine` from resolved provider config.
pub fn build_translation_engine(
    resolved: &crate::config::ResolvedProvider,
) -> Result<TranslationEngine> {
    let api_key = resolved.api_key.as_deref().unwrap_or("not-needed");
    let provider: Box<dyn Provider> = match resolved.provider_type {
        ProviderType::OpenAI => {
            let p = crate::llm::openai::OpenAIProvider::new(
                &resolved.endpoint,
                Some(api_key),
                &resolved.model,
            )?
            .with_reasoning_effort(resolved.reasoning_effort.clone());
            Box::new(p)
        }
        ProviderType::Anthropic => Box::new(crate::llm::anthropic::AnthropicProvider::new(
            &resolved.endpoint,
            api_key,
            &resolved.model,
        )?),
    };
    Ok(TranslationEngine::new(provider))
}
