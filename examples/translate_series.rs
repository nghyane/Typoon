//! Translate an entire manga/manhwa series (all chapters in a folder).
//!
//! Usage:
//!   cargo run --release --example translate_series -- \
//!       --input tests/fixtures/ctrlaltresign \
//!       --project ctrlaltresign --source en --target vi \
//!       --output ./output/ctrlaltresign

use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};
use std::time::Instant;

use comic_scan::detection::TextDetector;
use comic_scan::ocr::OcrEngine;
use comic_scan::pipeline::chapter::{self, PageDetection};

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| "warn,comic_scan=info".parse().unwrap()),
        )
        .init();

    let args = parse_args();

    // Discover chapter directories (ch001, ch002, ...)
    let mut chapters = discover_chapters(&args.input)?;
    if chapters.is_empty() {
        anyhow::bail!("No chapter directories found in {}", args.input.display());
    }
    chapters.sort();

    println!("═══ ComicScan Series Translate ═══");
    println!("Input:   {} ({} chapters)", args.input.display(), chapters.len());
    println!("Project: {}", args.project);
    println!("Lang:    {} → {}", args.source_lang, args.target_lang);
    println!("Output:  {}", args.output.display());
    println!();

    // Create runner once, reuse for all chapters
    let config = comic_scan::config::AppConfig::load()?;
    let runner = comic_scan::runner::TranslationRunner::new(&config).await?;

    let source = &args.source_lang;
    let source_lang = comic_scan::pipeline::detect_source_lang(Some(source), &args.target_lang);

    let series_start = Instant::now();
    let mut total_bubbles = 0;
    let mut total_pages = 0;

    // Build work list: (index, ch_dir, ch_num, output_dir) — skip already-translated
    let work: Vec<(usize, &PathBuf, usize, PathBuf)> = chapters
        .iter()
        .enumerate()
        .filter_map(|(idx, ch_dir)| {
            let ch_name = ch_dir.file_name().unwrap().to_string_lossy();
            let ch_num = parse_chapter_number(&ch_name).unwrap_or(idx + 1);
            let ch_output_dir = args.output.join(&*ch_name);
            if ch_output_dir.exists() && has_images(&ch_output_dir) {
                println!("[{}/{}] {} — skip (already translated)", idx + 1, chapters.len(), ch_name);
                None
            } else {
                Some((idx, ch_dir, ch_num, ch_output_dir))
            }
        })
        .collect();

    // Pipeline parallelism: detect chapter N+1 (CPU) while translating chapter N (LLM network).
    // detector/ocr are Arc-wrapped, so we clone handles for the detection task.
    let mut pending_detection: Option<PendingDetection> = None;

    // Kick off detection for the first chapter
    if let Some(&(idx, ch_dir, _, _)) = work.first() {
        pending_detection = start_detection(
            idx, ch_dir, &runner.detector, &runner.ocr, source_lang, &chapters,
        );
    }

    for (wi, &(idx, ch_dir, ch_num, ref ch_output_dir)) in work.iter().enumerate() {
        let ch_name = ch_dir.file_name().unwrap().to_string_lossy();

        // Await detection for this chapter (already running or just completed)
        let (images, detections) = match pending_detection.take() {
            Some(pd) => pd.handle.await??,
            None => {
                // Fallback: detect inline (shouldn't happen in normal flow)
                let mut image_paths = discover_images(ch_dir)?;
                if image_paths.is_empty() {
                    println!("[{}/{}] {} — skip (no images)", idx + 1, chapters.len(), ch_name);
                    continue;
                }
                image_paths.sort();
                let images = load_images(&image_paths)?;
                let det = runner.detector.clone();
                let ocr = runner.ocr.clone();
                let detections = tokio::task::block_in_place(|| {
                    chapter::detect_chapter(&det, &ocr, &images, source_lang)
                })?;
                (images, detections)
            }
        };

        // Start detection for the NEXT chapter while we translate this one
        if let Some(&(next_idx, next_dir, _, _)) = work.get(wi + 1) {
            pending_detection = start_detection(
                next_idx, next_dir, &runner.detector, &runner.ocr, source_lang, &chapters,
            );
        }

        let t = Instant::now();
        let output = chapter::translate_and_render(
            &runner,
            detections,
            &images,
            &args.target_lang,
            source_lang,
            Some(&args.project),
            Some(ch_num),
        )
        .await?;

        // Save rendered pages
        std::fs::create_dir_all(ch_output_dir)?;
        let mut ch_bubbles = 0;
        for page in &output.pages {
            ch_bubbles += page.bubbles.len();
            if let Some(rendered) = &page.rendered_image {
                let path = ch_output_dir.join(format!("page_{:03}.png", page.page_index));
                rendered.save(&path)?;
            }
        }

        total_bubbles += ch_bubbles;
        total_pages += images.len();
        println!(
            "[{}/{}] {} — {} pages, {} bubbles in {:.1}s",
            idx + 1, chapters.len(), ch_name,
            images.len(), ch_bubbles, t.elapsed().as_secs_f64(),
        );
    }

    println!();
    println!(
        "═══ Done: {} bubbles across {} pages in {:.1}s ═══",
        total_bubbles, total_pages, series_start.elapsed().as_secs_f64(),
    );

    Ok(())
}

struct Args {
    input: PathBuf,
    output: PathBuf,
    project: String,
    source_lang: String,
    target_lang: String,
}

fn parse_args() -> Args {
    let args: Vec<String> = std::env::args().collect();
    let mut input = PathBuf::from(".");
    let mut output = PathBuf::from("./output");
    let mut project = "default".to_string();
    let mut source_lang = "en".to_string();
    let mut target_lang = "vi".to_string();

    let mut i = 1;
    while i < args.len() {
        match args[i].as_str() {
            "--input" if i + 1 < args.len() => { input = PathBuf::from(&args[i + 1]); i += 2; }
            "--output" if i + 1 < args.len() => { output = PathBuf::from(&args[i + 1]); i += 2; }
            "--project" if i + 1 < args.len() => { project = args[i + 1].clone(); i += 2; }
            "--source" if i + 1 < args.len() => { source_lang = args[i + 1].clone(); i += 2; }
            "--target" if i + 1 < args.len() => { target_lang = args[i + 1].clone(); i += 2; }
            "--help" | "-h" => {
                eprintln!("Usage: translate_series --input <series_dir> [--output <dir>] [--project <name>] [--source <lang>] [--target <lang>]");
                std::process::exit(0);
            }
            _ => { i += 1; }
        }
    }

    Args { input, output, project, source_lang, target_lang }
}

fn discover_chapters(dir: &Path) -> anyhow::Result<Vec<PathBuf>> {
    if !dir.is_dir() {
        anyhow::bail!("{} is not a directory", dir.display());
    }
    let mut dirs = Vec::new();
    for entry in std::fs::read_dir(dir)? {
        let path = entry?.path();
        if path.is_dir() && has_images(&path) {
            dirs.push(path);
        }
    }
    Ok(dirs)
}

fn discover_images(dir: &Path) -> anyhow::Result<Vec<PathBuf>> {
    let extensions = ["png", "jpg", "jpeg", "webp", "bmp", "tiff"];
    let mut paths = Vec::new();
    for entry in std::fs::read_dir(dir)? {
        let path = entry?.path();
        if path.is_file() {
            if let Some(ext) = path.extension().and_then(|e| e.to_str()) {
                if extensions.contains(&ext.to_lowercase().as_str()) {
                    paths.push(path);
                }
            }
        }
    }
    Ok(paths)
}

fn load_images(paths: &[PathBuf]) -> anyhow::Result<Vec<image::DynamicImage>> {
    paths
        .iter()
        .map(|p| image::open(p).map_err(|e| anyhow::anyhow!("Failed to load {}: {e}", p.display())))
        .collect()
}

fn has_images(dir: &Path) -> bool {
    std::fs::read_dir(dir)
        .map(|entries| {
            entries.filter_map(|e| e.ok()).any(|e| {
                e.path()
                    .extension()
                    .and_then(|ext| ext.to_str())
                    .map(|ext| ["png", "jpg", "jpeg", "webp"].contains(&ext))
                    .unwrap_or(false)
            })
        })
        .unwrap_or(false)
}

fn parse_chapter_number(name: &str) -> Option<usize> {
    name.chars()
        .skip_while(|c| !c.is_ascii_digit())
        .take_while(|c| c.is_ascii_digit())
        .collect::<String>()
        .parse()
        .ok()
}

// ── Pipeline parallelism helpers ──

/// A detection task running in the background via `spawn_blocking`.
/// Returns both the detection results AND the images (moved in, moved back out).
struct PendingDetection {
    handle: tokio::task::JoinHandle<anyhow::Result<(Vec<image::DynamicImage>, Vec<PageDetection>)>>,
}

fn start_detection(
    idx: usize,
    ch_dir: &Path,
    detector: &Arc<Mutex<TextDetector>>,
    ocr: &Arc<OcrEngine>,
    source_lang: &str,
    chapters: &[PathBuf],
) -> Option<PendingDetection> {
    let ch_name = ch_dir.file_name().unwrap().to_string_lossy();
    let mut image_paths = match discover_images(ch_dir) {
        Ok(p) => p,
        Err(e) => {
            println!("[{}/{}] {} — skip ({})", idx + 1, chapters.len(), ch_name, e);
            return None;
        }
    };
    if image_paths.is_empty() {
        println!("[{}/{}] {} — skip (no images)", idx + 1, chapters.len(), ch_name);
        return None;
    }
    image_paths.sort();

    let images = match load_images(&image_paths) {
        Ok(imgs) => imgs,
        Err(e) => {
            println!("[{}/{}] {} — skip ({})", idx + 1, chapters.len(), ch_name, e);
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
