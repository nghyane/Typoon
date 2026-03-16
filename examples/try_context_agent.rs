//! Manual test for the context agent sub-agent.
//!
//! Usage:
//!   cargo run --example try_context_agent -- "Who is Max?"
//!   cargo run --example try_context_agent -- "How was 선배 translated before?"
//!
//! Requires: config.toml with [context_agent] configured,
//! and data/context.db populated from prior chapter translations.

use anyhow::Result;

#[tokio::main]
async fn main() -> Result<()> {
    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| "info,comic_scan=debug".parse().unwrap()),
        )
        .init();

    let question = std::env::args()
        .nth(1)
        .unwrap_or_else(|| "Who are the main characters and their relationships?".into());

    let project_id = std::env::args().nth(2).unwrap_or_else(|| "default".into());

    tracing::info!("Question: {question:?}");
    tracing::info!("Project: {project_id}");

    // Load config
    let config = comic_scan::config::AppConfig::load()?;

    // Open context store
    let db_path = config
        .context
        .db_path
        .as_deref()
        .unwrap_or("data/context.db");
    let store = comic_scan::storage::context::ContextStore::open(std::path::Path::new(db_path))?;
    tracing::info!("Context store opened: {db_path}");

    // Build context agent provider
    let agent_config = config
        .context_agent
        .as_ref()
        .ok_or_else(|| anyhow::anyhow!("No [context_agent] in config.toml"))?;
    let resolved = config.resolve_provider(agent_config)?;
    let api_key = resolved.api_key.as_deref().unwrap_or("not-needed");

    let provider: Box<dyn comic_scan::llm::Provider> = match resolved.provider_type {
        comic_scan::config::ProviderType::Anthropic => {
            Box::new(comic_scan::llm::anthropic::AnthropicProvider::new(
                &resolved.endpoint,
                api_key,
                &resolved.model,
            )?)
        }
        comic_scan::config::ProviderType::OpenAI => {
            Box::new(comic_scan::llm::openai::OpenAIProvider::new(
                &resolved.endpoint,
                Some(api_key),
                &resolved.model,
            )?)
        }
    };
    tracing::info!("Provider: {} ({})", resolved.model, resolved.endpoint);

    // Run the context agent
    let start = std::time::Instant::now();
    let answer = comic_scan::storage::context::agent::answer_context_question(
        provider.as_ref(),
        &store,
        &project_id,
        &question,
    )
    .await?;
    let elapsed = start.elapsed();

    println!("\n===== Context Agent Answer =====\n");
    if answer.is_empty() {
        println!("(empty — no relevant context found)");
    } else {
        println!("{answer}");
    }
    println!("\n===== {:.1}s =====", elapsed.as_secs_f64());

    Ok(())
}
