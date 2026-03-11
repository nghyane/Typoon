use anyhow::Result;

use comic_scan::{api, config};

#[tokio::main]
async fn main() -> Result<()> {
    tracing_subscriber::fmt::init();

    let config = config::AppConfig::load()?;
    let state = api::AppState::new(&config).await?;
    let app = api::router(state);

    let addr = format!("127.0.0.1:{}", config.port);
    tracing::info!("ComicScan listening on {}", addr);

    let listener = tokio::net::TcpListener::bind(&addr).await?;
    axum::serve(listener, app).await?;

    Ok(())
}
