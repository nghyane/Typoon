use std::collections::HashMap;
use std::path::Path;

use anyhow::Result;
use serde::Deserialize;

#[derive(Debug, Clone, Deserialize)]
pub struct AppConfig {
    #[serde(default = "default_port")]
    pub port: u16,
    #[serde(default = "default_models_dir")]
    pub models_dir: String,
    #[serde(default = "default_target_lang")]
    pub default_target_lang: String,

    /// Named provider configurations (e.g., "openai", "anthropic", "local")
    #[serde(default)]
    pub providers: HashMap<String, ProviderConfig>,

    /// Main translation agent config
    #[serde(default)]
    pub translation: RoleConfig,

    /// Context sub-agent config (cheap model for DB search + summarize)
    pub context_agent: Option<RoleConfig>,

    /// Context DB settings
    #[serde(default)]
    pub context: ContextConfig,

    #[serde(default)]
    pub glossary: GlossaryConfig,

    /// Runtime concurrency limits per pipeline stage.
    #[serde(default)]
    pub runtime: RuntimeConfig,
}

#[derive(Debug, Clone, Deserialize)]
pub struct ProviderConfig {
    #[serde(rename = "type", default = "default_provider_type")]
    pub provider_type: ProviderType,
    pub endpoint: String,
    pub api_key: Option<String>,
}

#[derive(Debug, Clone, Deserialize, PartialEq)]
#[serde(rename_all = "lowercase")]
pub enum ProviderType {
    OpenAI,
    Anthropic,
}

impl Default for ProviderType {
    fn default() -> Self {
        Self::OpenAI
    }
}

#[derive(Debug, Clone, Deserialize)]
pub struct RoleConfig {
    /// References a key in `providers` map
    #[serde(default = "default_provider_name")]
    pub provider: String,
    #[serde(default = "default_model")]
    pub model: String,
    pub reasoning_effort: Option<String>,
}

impl Default for RoleConfig {
    fn default() -> Self {
        Self {
            provider: default_provider_name(),
            model: default_model(),
            reasoning_effort: None,
        }
    }
}

#[derive(Debug, Clone, Default, Deserialize)]
pub struct ContextConfig {
    /// Root directory for the default project store (created if missing).
    pub project_dir: Option<String>,
}

#[derive(Debug, Clone, Default, Deserialize)]
pub struct GlossaryConfig {
    /// Path to TOML file to import terms from on startup.
    pub import_toml: Option<String>,
}

#[derive(Debug, Clone, Deserialize)]
pub struct RuntimeConfig {
    /// Render worker count when LaMa is disabled (CPU-heavy text drawing only).
    #[serde(default = "default_render_workers")]
    pub render_workers: usize,
    /// Render worker count when LaMa is enabled (CoreML/ANE contention-sensitive).
    #[serde(default = "default_render_workers_with_lama")]
    pub render_workers_with_lama: usize,
    /// Maximum number of chapter render jobs buffered in memory.
    ///
    /// A small queue lets translation continue while previous chapters are rendering,
    /// without unbounded memory growth.
    #[serde(default = "default_max_pending_render_jobs")]
    pub max_pending_render_jobs: usize,
}

impl Default for RuntimeConfig {
    fn default() -> Self {
        Self {
            render_workers: default_render_workers(),
            render_workers_with_lama: default_render_workers_with_lama(),
            max_pending_render_jobs: default_max_pending_render_jobs(),
        }
    }
}

impl RuntimeConfig {
    pub fn effective_render_workers(&self, has_lama: bool) -> usize {
        let workers = if has_lama {
            self.render_workers_with_lama
        } else {
            self.render_workers
        };
        workers.max(1)
    }
}

/// Fully resolved provider + model settings for a role.
#[derive(Debug, Clone)]
pub struct ResolvedProvider {
    pub provider_type: ProviderType,
    pub endpoint: String,
    pub api_key: Option<String>,
    pub model: String,
    pub reasoning_effort: Option<String>,
}

fn default_port() -> u16 {
    4319
}
fn default_models_dir() -> String {
    "models".into()
}
fn default_target_lang() -> String {
    "vi".into()
}
fn default_model() -> String {
    "gpt-5.4".into()
}
fn default_provider_name() -> String {
    "default".into()
}
fn default_provider_type() -> ProviderType {
    ProviderType::OpenAI
}
fn default_render_workers() -> usize {
    let cores = std::thread::available_parallelism()
        .map(|n| n.get())
        .unwrap_or(4);
    cores.clamp(1, 8)
}
fn default_render_workers_with_lama() -> usize {
    1
}
fn default_max_pending_render_jobs() -> usize {
    2
}

impl AppConfig {
    pub fn load() -> Result<Self> {
        let config_path = Path::new("config.toml");
        if config_path.exists() {
            let text = std::fs::read_to_string(config_path)?;
            let config: AppConfig = toml::from_str(&text)?;
            tracing::info!("Loaded config from {}", config_path.display());
            Ok(config)
        } else {
            tracing::info!("No config.toml found, using defaults");
            Ok(Self {
                port: default_port(),
                models_dir: default_models_dir(),
                default_target_lang: default_target_lang(),
                providers: HashMap::new(),
                translation: RoleConfig::default(),
                context_agent: None,
                context: ContextConfig::default(),
                glossary: GlossaryConfig::default(),
                runtime: RuntimeConfig::default(),
            })
        }
    }

    /// Resolve a RoleConfig into its full provider details by looking up
    /// `role.provider` in the `[providers]` map.
    pub fn resolve_provider(&self, role: &RoleConfig) -> Result<ResolvedProvider> {
        let provider = self.providers.get(&role.provider).ok_or_else(|| {
            anyhow::anyhow!("Provider {:?} not found in [providers] map", role.provider)
        })?;

        Ok(ResolvedProvider {
            provider_type: provider.provider_type.clone(),
            endpoint: provider.endpoint.clone(),
            api_key: provider.api_key.clone(),
            model: role.model.clone(),
            reasoning_effort: role.reasoning_effort.clone(),
        })
    }
}
