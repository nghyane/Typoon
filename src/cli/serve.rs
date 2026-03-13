use anyhow::Result;
use clap::Args;

use crate::{api, config};

#[derive(Args)]
pub struct ServeArgs {
    /// Port to listen on (overrides config.toml)
    #[arg(short, long)]
    pub port: Option<u16>,
}

pub async fn run(args: ServeArgs) -> Result<()> {
    let mut config = config::AppConfig::load()?;
    if let Some(port) = args.port {
        config.port = port;
    }

    let state = api::AppState::new(&config).await?;
    let app = api::router(state);

    let addr = format!("127.0.0.1:{}", config.port);
    tracing::info!("ComicScan listening on {addr}");

    let listener = tokio::net::TcpListener::bind(&addr).await?;
    axum::serve(listener, app).await?;

    Ok(())
}
