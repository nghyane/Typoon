use anyhow::Result;
use clap::{Parser, Subcommand};

use comic_scan::cli;

#[derive(Parser)]
#[command(name = "comicscan", about = "Manga/manhwa translation tool")]
struct Cli {
    #[command(subcommand)]
    command: Command,
}

#[derive(Subcommand)]
enum Command {
    /// Translate a manga series or single chapter
    Translate(cli::translate::TranslateArgs),
    /// Start the HTTP API server
    Serve(cli::serve::ServeArgs),
    /// Inspect detection and text masks on a single image
    Inspect(cli::inspect::InspectArgs),
}

#[tokio::main]
async fn main() -> Result<()> {
    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| "warn,comic_scan=info".parse().unwrap()),
        )
        .init();

    let cli = Cli::parse();
    match cli.command {
        Command::Translate(args) => cli::translate::run(args).await,
        Command::Serve(args) => cli::serve::run(args).await,
        Command::Inspect(args) => cli::inspect::run(args).await,
    }
}
