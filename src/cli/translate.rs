use std::path::PathBuf;
use std::sync::{Arc, Mutex};
use std::time::Instant;

use anyhow::Result;
use clap::Args;

use crate::config;
use crate::detection::TextDetector;
use crate::ocr::OcrEngine;
use crate::pipeline;
use crate::pipeline::chapter::{self, PageDetection};
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
    let runner = TranslationRunner::new(&config).await?;
    let source_lang = pipeline::detect_source_lang(Some(&args.source), &args.target);

    // Determine if input is a single chapter or a series directory
    let chapters = discover_chapters(&args.input)?;

    if chapters.is_empty() {
        // Input is a single chapter directory (contains images directly)
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

        translate_single_chapter(
            &runner, &args.input, &args.output,
            &args.project, ch_num, source_lang, &args.target,
        ).await?;
    } else {
        // Series mode: multiple chapter directories
        println!("═══ ComicScan Translate (series) ═══");
        println!("Input:   {} ({} chapters)", args.input.display(), chapters.len());
        println!("Project: {}", args.project);
        println!("Lang:    {source_lang} → {}", args.target);
        println!("Output:  {}", args.output.display());
        println!();

        translate_series(
            &runner, &chapters, &args.output,
            &args.project, args.chapter, source_lang, &args.target,
        ).await?;
    }

    Ok(())
}

// ── Single chapter ──

async fn translate_single_chapter(
    runner: &TranslationRunner,
    input: &PathBuf,
    output: &PathBuf,
    project: &str,
    chapter_num: usize,
    source_lang: &str,
    target_lang: &str,
) -> Result<()> {
    let image_paths = util::discover_images(input)?;
    let images = util::load_images(&image_paths)?;

    let t = Instant::now();
    let det = runner.detector.clone();
    let ocr = runner.ocr.clone();
    let lang = source_lang.to_string();
    let imgs = images.clone();
    let detections = tokio::task::spawn_blocking(move || {
        chapter::detect_chapter(&det, &ocr, &imgs, &lang)
    }).await??;

    let result = chapter::translate_and_render(
        runner, detections, &images,
        target_lang, source_lang,
        Some(project), Some(chapter_num),
    ).await?;

    std::fs::create_dir_all(output)?;
    let mut total = 0;
    for page in &result.pages {
        total += page.bubbles.len();
        print_page(page);
        if let Some(rendered) = &page.rendered_image {
            let path = output.join(format!("page_{:03}.png", page.page_index));
            rendered.save(&path)?;
        }
    }

    println!("\nTotal: {total} bubbles, {} pages in {:.1}s", images.len(), t.elapsed().as_secs_f64());
    Ok(())
}

// ── Series ──

async fn translate_series(
    runner: &TranslationRunner,
    chapters: &[PathBuf],
    output: &PathBuf,
    project: &str,
    only_chapter: Option<usize>,
    source_lang: &str,
    target_lang: &str,
) -> Result<()> {
    let series_start = Instant::now();
    let mut total_bubbles = 0;
    let mut total_pages = 0;

    // Build work list, skip already-translated
    let work: Vec<(usize, &PathBuf, usize, PathBuf)> = chapters
        .iter()
        .enumerate()
        .filter_map(|(idx, ch_dir)| {
            let ch_name = ch_dir.file_name().unwrap().to_string_lossy();
            let ch_num = util::parse_chapter_number(&ch_name).unwrap_or(idx + 1);

            // If --chapter is set, only translate that chapter
            if let Some(only) = only_chapter {
                if ch_num != only { return None; }
            }

            let ch_output = output.join(&*ch_name);
            if ch_output.exists() && util::has_images(&ch_output) {
                println!("[{}/{}] {ch_name} — skip (already translated)", idx + 1, chapters.len());
                None
            } else {
                Some((idx, ch_dir, ch_num, ch_output))
            }
        })
        .collect();

    if work.is_empty() {
        println!("Nothing to translate.");
        return Ok(());
    }

    // Pipeline parallelism: detect chapter N+1 while translating chapter N
    let mut pending: Option<PendingDetection> = None;

    if let Some(&(idx, ch_dir, _, _)) = work.first() {
        pending = start_detection(idx, ch_dir, &runner.detector, &runner.ocr, source_lang, chapters.len());
    }

    for (wi, &(idx, ch_dir, ch_num, ref ch_output)) in work.iter().enumerate() {
        let ch_name = ch_dir.file_name().unwrap().to_string_lossy();

        let (images, detections) = match pending.take() {
            Some(pd) => pd.handle.await??,
            None => detect_inline(ch_dir, &runner.detector, &runner.ocr, source_lang)?,
        };

        // Start detection for next chapter
        if let Some(&(next_idx, next_dir, _, _)) = work.get(wi + 1) {
            pending = start_detection(next_idx, next_dir, &runner.detector, &runner.ocr, source_lang, chapters.len());
        }

        let t = Instant::now();
        let result = chapter::translate_and_render(
            runner, detections, &images,
            target_lang, source_lang,
            Some(project), Some(ch_num),
        ).await?;

        std::fs::create_dir_all(ch_output)?;
        let mut ch_bubbles = 0;
        for page in &result.pages {
            ch_bubbles += page.bubbles.len();
            if let Some(rendered) = &page.rendered_image {
                let path = ch_output.join(format!("page_{:03}.png", page.page_index));
                rendered.save(&path)?;
            }
        }

        total_bubbles += ch_bubbles;
        total_pages += images.len();
        println!(
            "[{}/{}] {ch_name} — {} pages, {ch_bubbles} bubbles in {:.1}s",
            idx + 1, chapters.len(), images.len(), t.elapsed().as_secs_f64(),
        );
    }

    println!(
        "\n═══ Done: {total_bubbles} bubbles, {total_pages} pages in {:.1}s ═══",
        series_start.elapsed().as_secs_f64(),
    );
    Ok(())
}

// ── Helpers ──

fn print_page(page: &chapter::ChapterPageOutput) {
    if page.bubbles.is_empty() { return; }
    println!("── Page {} ({} bubbles) ──", page.page_index, page.bubbles.len());
    for b in &page.bubbles {
        println!(
            "  [{}] {}px \"{}\" → \"{}\"",
            b.bubble_id, b.font_size_px,
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

struct PendingDetection {
    handle: tokio::task::JoinHandle<Result<(Vec<image::DynamicImage>, Vec<PageDetection>)>>,
}

fn start_detection(
    idx: usize,
    ch_dir: &std::path::Path,
    detector: &Arc<Mutex<TextDetector>>,
    ocr: &Arc<OcrEngine>,
    source_lang: &str,
    total: usize,
) -> Option<PendingDetection> {
    let ch_name = ch_dir.file_name().unwrap().to_string_lossy().to_string();
    let image_paths = match util::discover_images(ch_dir) {
        Ok(p) if !p.is_empty() => p,
        Ok(_) => {
            println!("[{}/{}] {ch_name} — skip (no images)", idx + 1, total);
            return None;
        }
        Err(e) => {
            println!("[{}/{}] {ch_name} — skip ({e})", idx + 1, total);
            return None;
        }
    };

    let images = match util::load_images(&image_paths) {
        Ok(imgs) => imgs,
        Err(e) => {
            println!("[{}/{}] {ch_name} — skip ({e})", idx + 1, total);
            return None;
        }
    };

    let det = detector.clone();
    let ocr = ocr.clone();
    let lang = source_lang.to_string();
    let handle = tokio::task::spawn_blocking(move || {
        let detections = chapter::detect_chapter(&det, &ocr, &images, &lang)?;
        Ok((images, detections))
    });

    Some(PendingDetection { handle })
}

fn detect_inline(
    ch_dir: &std::path::Path,
    detector: &Arc<Mutex<TextDetector>>,
    ocr: &Arc<OcrEngine>,
    source_lang: &str,
) -> Result<(Vec<image::DynamicImage>, Vec<PageDetection>)> {
    let image_paths = util::discover_images(ch_dir)?;
    let images = util::load_images(&image_paths)?;
    let det = detector.clone();
    let ocr = ocr.clone();
    let lang = source_lang.to_string();
    let detections = tokio::task::block_in_place(|| {
        chapter::detect_chapter(&det, &ocr, &images, &lang)
    })?;
    Ok((images, detections))
}
