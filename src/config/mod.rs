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
    #[serde(default)]
    pub translation: TranslationConfig,
    #[serde(default)]
    pub canvas_agent: CanvasAgentConfig,
}

#[derive(Debug, Clone, Deserialize)]
pub struct TranslationConfig {
    #[serde(default = "default_endpoint")]
    pub endpoint: String,
    pub api_key: Option<String>,
    #[serde(default = "default_model")]
    pub model: String,
    pub reasoning_effort: Option<String>,
}

#[derive(Debug, Clone, Default, Deserialize)]
pub struct CanvasAgentConfig {
    #[serde(default)]
    pub enabled: bool,
    pub endpoint: Option<String>,
    pub api_key: Option<String>,
    pub model: Option<String>,
    pub reasoning_effort: Option<String>,
}

impl CanvasAgentConfig {
    /// Build a TranslationConfig for the canvas agent, falling back to the
    /// translation config for any fields not explicitly set.
    pub fn resolved_translation(&self, fallback: &TranslationConfig) -> TranslationConfig {
        TranslationConfig {
            endpoint: self.endpoint.clone().unwrap_or_else(|| fallback.endpoint.clone()),
            api_key: self.api_key.clone().or_else(|| fallback.api_key.clone()),
            model: self.model.clone().unwrap_or_else(|| fallback.model.clone()),
            reasoning_effort: self.reasoning_effort.clone().or_else(|| fallback.reasoning_effort.clone()),
        }
    }
}

impl Default for TranslationConfig {
    fn default() -> Self {
        Self {
            endpoint: default_endpoint(),
            api_key: None,
            model: default_model(),
            reasoning_effort: None,
        }
    }
}

fn default_port() -> u16 { 4319 }
fn default_models_dir() -> String { "models".into() }
fn default_target_lang() -> String { "vi".into() }
fn default_endpoint() -> String { "http://localhost:7860/api/provider/openai/v1".into() }
fn default_model() -> String { "gpt-5.4".into() }

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
                translation: TranslationConfig::default(),
                canvas_agent: CanvasAgentConfig::default(),
            })
        }
    }
}
