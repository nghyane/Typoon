use std::sync::Arc;

use anyhow::Result;

use crate::config::{AppConfig, ProviderType};
use crate::llm::Provider;
use crate::storage::project::ProjectStore;
use crate::model_hub::lazy::LazySession;
use crate::model_hub::{self, Model};
use crate::translation::TranslationEngine;
use crate::vision::detection::TextDetector;
use crate::vision::inpaint::LamaInpainter;
use crate::vision::ocr::OcrEngine;

/// Per-series translation session — bundles runner + project + context provider.
///
/// Created once per series/request. Pipeline functions take `&Session`.
/// Runner is `Arc` so spawned tasks (render, knowledge) can share it.
pub struct Session {
    pub runner: Arc<TranslationRunner>,
    pub project: Option<Arc<ProjectStore>>,
    context_provider: Option<Box<dyn Provider>>,
    engine_override: Option<TranslationEngine>,
}

impl Session {
    pub fn new(runner: Arc<TranslationRunner>, project: Option<Arc<ProjectStore>>) -> Self {
        let context_provider = runner.build_context_agent_provider().ok().flatten();
        Self {
            runner,
            project,
            context_provider,
            engine_override: None,
        }
    }

    /// Use a custom translation engine (e.g., HTTP API provider override).
    pub fn with_engine(mut self, engine: TranslationEngine) -> Self {
        self.engine_override = Some(engine);
        self
    }

    pub fn engine(&self) -> &TranslationEngine {
        self.engine_override
            .as_ref()
            .unwrap_or(&self.runner.translation)
    }

    pub fn context_provider(&self) -> Option<&dyn Provider> {
        self.context_provider
            .as_ref()
            .map(|p| &**p as &dyn Provider)
    }
}

/// Pipeline infrastructure — models, providers, render pool.
///
/// Does NOT hold per-series data (project store). That's in `Session`.
pub struct TranslationRunner {
    pub detector: Arc<TextDetector>,
    pub ocr: Arc<OcrEngine>,
    pub translation: TranslationEngine,
    pub inpainter: Option<LamaInpainter>,
    render_executor: RenderExecutor,
    max_pending_render_jobs: usize,
    /// Context/knowledge agent provider config (for rebuilding per-spawn).
    context_agent_config: Option<crate::config::ResolvedProvider>,
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

        let context_agent_config = config
            .context_agent
            .as_ref()
            .and_then(|agent_config| config.resolve_provider(agent_config).ok());

        if let Some(ref cfg) = context_agent_config {
            tracing::info!("Context/knowledge agent configured ({})", cfg.model);
        }

        Ok(Self {
            detector: Arc::new(detector),
            ocr: Arc::new(ocr),
            translation,
            inpainter,
            render_executor,
            max_pending_render_jobs,
            context_agent_config,
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

    /// Build a context/knowledge agent provider instance.
    /// Each call creates a fresh provider (needed for spawned tasks).
    pub fn build_context_agent_provider(&self) -> Result<Option<Box<dyn Provider>>> {
        match &self.context_agent_config {
            Some(resolved) => build_provider(resolved).map(Some),
            None => Ok(None),
        }
    }
}

fn build_provider(resolved: &crate::config::ResolvedProvider) -> Result<Box<dyn Provider>> {
    let api_key = resolved.api_key.as_deref().unwrap_or("not-needed");
    match resolved.provider_type {
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
