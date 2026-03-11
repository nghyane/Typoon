use std::path::Path;

use anyhow::Result;
use serde::Deserialize;

#[derive(Debug, Clone, Deserialize)]
pub struct AppConfig {
    #[serde(default = "default_port")]
    pub port: u16,
    #[serde(default = "default_models_dir")]
    pub models_dir: String,
    #[serde(default = "default_cache_dir")]
    pub cache_dir: String,
    #[serde(default = "default_target_lang")]
    pub default_target_lang: String,
    #[serde(default)]
    pub translation: TranslationConfig,
}

#[derive(Debug, Clone, Deserialize)]
pub struct TranslationConfig {
    #[serde(default = "default_endpoint")]
    pub endpoint: String,
    pub api_key: Option<String>,
    #[serde(default = "default_model")]
    pub model: String,
}

impl Default for TranslationConfig {
    fn default() -> Self {
        Self {
            endpoint: default_endpoint(),
            api_key: None,
            model: default_model(),
        }
    }
}

fn default_port() -> u16 { 4319 }
fn default_models_dir() -> String { "models".into() }
fn default_cache_dir() -> String { "cache".into() }
fn default_target_lang() -> String { "vi".into() }
fn default_endpoint() -> String { "http://5.223.45.83:7860/api/provider/openai/v1".into() }
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
                cache_dir: default_cache_dir(),
                default_target_lang: default_target_lang(),
                translation: TranslationConfig::default(),
            })
        }
    }
}
