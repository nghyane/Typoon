use std::path::PathBuf;
use std::sync::Arc;
use std::time::Instant;

use anyhow::Result;
use clap::Args;

use crate::config;
use crate::image_io;
use crate::pipeline;
use crate::pipeline::series::ChapterJob;
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

    let chapter_dirs = discover_chapter_dirs(&args.input)?;

    if chapter_dirs.is_empty() {
        run_single(&runner, &args, source_lang).await
    } else {
        run_series(&runner, &args, &chapter_dirs, source_lang).await
    }
}

// ── Single chapter ──

async fn run_single(
    runner: &Arc<TranslationRunner>,
    args: &TranslateArgs,
    source_lang: &str,
) -> Result<()> {
    if !image_io::has_images(&args.input) {
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
        let path = args.output.join(format!("page_{:03}.png", page.page_index));
        page.image.save(&path)?;
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
    chapter_dirs: &[PathBuf],
    source_lang: &str,
) -> Result<()> {
    let total = chapter_dirs.len();
    println!("═══ ComicScan Translate (series) ═══");
    println!("Input:   {} ({total} chapters)", args.input.display());
    println!("Project: {}", args.project);
    println!("Lang:    {source_lang} → {}", args.target);
    println!("Output:  {}", args.output.display());
    println!();

    let (jobs, labels) = build_work_list(chapter_dirs, &args.output, args.chapter, total);

    if jobs.is_empty() {
        println!("Nothing to translate.");
        return Ok(());
    }

    let series_start = Instant::now();
    let totals = pipeline::series::translate_batch(
        runner,
        &jobs,
        source_lang,
        &args.target,
        &args.project,
        |job_index, result| {
            let (label, pos) = &labels[job_index];
            println!(
                "[{pos}/{total}] {label} — {} pages, {} bubbles in {:.1}s",
                result.num_pages, result.num_bubbles, result.elapsed_s,
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

fn print_page(page: &pipeline::types::RenderedPage) {
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
            "  [b{}] {}px \"{}\" → \"{}\"",
            b.idx,
            b.font_size_px,
            truncate(&b.source_text, 40),
            truncate(&b.translated_text.replace('\n', " "), 50),
        );
    }
}

fn truncate(s: &str, max: usize) -> String {
    s.chars().take(max).collect()
}

/// Find subdirectories that contain images (chapter directories).
fn discover_chapter_dirs(dir: &std::path::Path) -> Result<Vec<PathBuf>> {
    if !dir.is_dir() {
        anyhow::bail!("{} is not a directory", dir.display());
    }
    let mut dirs = Vec::new();
    for entry in std::fs::read_dir(dir)? {
        let path = entry?.path();
        if path.is_dir() && image_io::has_images(&path) {
            dirs.push(path);
        }
    }
    dirs.sort();
    Ok(dirs)
}

/// Build pipeline jobs + display labels, skipping already-translated chapters.
/// Returns (jobs, labels) where labels[i] = (chapter_name, display_position).
fn build_work_list(
    chapter_dirs: &[PathBuf],
    output: &std::path::Path,
    only_chapter: Option<usize>,
    total: usize,
) -> (Vec<ChapterJob>, Vec<(String, usize)>) {
    let mut jobs = Vec::new();
    let mut labels = Vec::new();

    for (idx, ch_dir) in chapter_dirs.iter().enumerate() {
        let ch_name = ch_dir.file_name().unwrap().to_string_lossy();
        let ch_num = util::parse_chapter_number(&ch_name).unwrap_or(idx + 1);
        let pos = idx + 1;

        if let Some(only) = only_chapter {
            if ch_num != only {
                continue;
            }
        }

        let ch_output = output.join(&*ch_name);
        if ch_output.exists() && image_io::has_images(&ch_output) {
            println!("[{pos}/{total}] {ch_name} — skip (already translated)");
            continue;
        }

        jobs.push(ChapterJob {
            input_dir: ch_dir.clone(),
            output_dir: ch_output,
            chapter_num: ch_num,
        });
        labels.push((ch_name.to_string(), pos));
    }

    (jobs, labels)
}
