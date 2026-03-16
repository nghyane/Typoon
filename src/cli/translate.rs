use std::path::PathBuf;
use std::sync::Arc;
use std::time::Instant;

use anyhow::Result;
use clap::Args;

use crate::config;
use crate::pipeline;
use crate::pipeline::series::{ChapterResult, ChapterSpec};
use crate::runner::TranslationRunner;

use super::util;

#[derive(Args)]
pub struct TranslateArgs {
    /// Path to series directory (contains chapter subdirs) or single chapter directory
    #[arg(short, long)]
    pub input: PathBuf,

    /// Project name (used for context store)
    #[arg(short, long, default_value = "default")]
    pub project: String,

    /// Translate only this chapter number (default: all chapters)
    #[arg(short, long)]
    pub chapter: Option<usize>,

    /// Source language
    #[arg(short, long, default_value = "ko")]
    pub source: String,

    /// Target language
    #[arg(short, long, default_value = "vi")]
    pub target: String,

    /// Output directory
    #[arg(short, long, default_value = "./output")]
    pub output: PathBuf,
}

pub async fn run(args: TranslateArgs) -> Result<()> {
    let config = config::AppConfig::load()?;
    let runner = Arc::new(TranslationRunner::new(&config).await?);
    let source_lang = pipeline::detect_source_lang(Some(&args.source), &args.target);

    let chapters = discover_chapters(&args.input)?;

    if chapters.is_empty() {
        run_single(&runner, &args, source_lang).await
    } else {
        run_series(&runner, &args, &chapters, source_lang).await
    }
}

// ── Single chapter ──

async fn run_single(
    runner: &Arc<TranslationRunner>,
    args: &TranslateArgs,
    source_lang: &str,
) -> Result<()> {
    if !util::has_images(&args.input) {
        anyhow::bail!("No images found in {}", args.input.display());
    }

    let ch_num = args.chapter.unwrap_or(0);
    println!("═══ ComicScan Translate (single chapter) ═══");
    println!("Input:   {}", args.input.display());
    println!("Project: {}", args.project);
    println!("Chapter: {ch_num}");
    println!("Lang:    {source_lang} → {}", args.target);
    println!();

    let result = pipeline::series::translate_single(
        runner,
        &args.input,
        &args.target,
        source_lang,
        &args.project,
        ch_num,
    )
    .await?;

    std::fs::create_dir_all(&args.output)?;
    for page in &result.pages {
        print_page(page);
        if let Some(rendered) = &page.rendered_image {
            let path = args.output.join(format!("page_{:03}.png", page.page_index));
            rendered.save(&path)?;
        }
    }

    println!(
        "\nTotal: {} bubbles, {} pages in {:.1}s",
        result.num_bubbles, result.num_pages, result.elapsed_s,
    );
    Ok(())
}

// ── Series ──

async fn run_series(
    runner: &Arc<TranslationRunner>,
    args: &TranslateArgs,
    chapters: &[PathBuf],
    source_lang: &str,
) -> Result<()> {
    println!("═══ ComicScan Translate (series) ═══");
    println!(
        "Input:   {} ({} chapters)",
        args.input.display(),
        chapters.len()
    );
    println!("Project: {}", args.project);
    println!("Lang:    {source_lang} → {}", args.target);
    println!("Output:  {}", args.output.display());
    println!();

    let specs = build_work_list(chapters, &args.output, args.chapter)?;

    if specs.is_empty() {
        println!("Nothing to translate.");
        return Ok(());
    }

    let series_start = Instant::now();
    let totals = pipeline::series::translate_batch(
        runner,
        &specs,
        source_lang,
        &args.target,
        &args.project,
        |ch: &ChapterResult| {
            println!(
                "[{}/{}] {} — {} pages, {} bubbles in {:.1}s",
                ch.position, ch.total, ch.label, ch.num_pages, ch.num_bubbles, ch.elapsed_s,
            );
        },
    )
    .await?;

    println!(
        "\n═══ Done: {} bubbles, {} pages in {:.1}s ═══",
        totals.bubbles,
        totals.pages,
        series_start.elapsed().as_secs_f64(),
    );
    Ok(())
}

// ── Helpers ──

fn print_page(page: &pipeline::chapter::ChapterPageOutput) {
    if page.bubbles.is_empty() {
        return;
    }
    println!(
        "── Page {} ({} bubbles) ──",
        page.page_index,
        page.bubbles.len()
    );
    for b in &page.bubbles {
        println!(
            "  [{}] {}px \"{}\" → \"{}\"",
            b.bubble_id,
            b.font_size_px,
            truncate(&b.source_text, 40),
            truncate(&b.translated_text.replace('\n', " "), 50),
        );
    }
}

fn truncate(s: &str, max: usize) -> String {
    s.chars().take(max).collect()
}

fn discover_chapters(dir: &std::path::Path) -> Result<Vec<PathBuf>> {
    if !dir.is_dir() {
        anyhow::bail!("{} is not a directory", dir.display());
    }
    let mut dirs = Vec::new();
    for entry in std::fs::read_dir(dir)? {
        let path = entry?.path();
        if path.is_dir() && util::has_images(&path) {
            dirs.push(path);
        }
    }
    dirs.sort();
    Ok(dirs)
}

fn build_work_list(
    chapters: &[PathBuf],
    output: &std::path::Path,
    only_chapter: Option<usize>,
) -> Result<Vec<ChapterSpec>> {
    let total = chapters.len();
    let mut specs = Vec::new();

    for (idx, ch_dir) in chapters.iter().enumerate() {
        let ch_name = ch_dir.file_name().unwrap().to_string_lossy();
        let ch_num = util::parse_chapter_number(&ch_name).unwrap_or(idx + 1);

        if let Some(only) = only_chapter {
            if ch_num != only {
                continue;
            }
        }

        let ch_output = output.join(&*ch_name);
        if ch_output.exists() && util::has_images(&ch_output) {
            println!(
                "[{}/{}] {ch_name} — skip (already translated)",
                idx + 1,
                total,
            );
            continue;
        }

        specs.push(ChapterSpec {
            dir: ch_dir.clone(),
            output: ch_output,
            chapter_num: ch_num,
            label: ch_name.to_string(),
            position: idx + 1,
            total,
        });
    }

    Ok(specs)
}
