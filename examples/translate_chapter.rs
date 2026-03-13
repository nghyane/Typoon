//! Chapter-level translation: load all images from a folder, translate as one chapter.
//!
//! Usage:
//!   cargo run --example translate_chapter -- --input /path/to/chapter/ \
//!       --project "series-name" --chapter 1 --source ko --target vi --output ./output/

use std::path::{Path, PathBuf};
use std::time::Instant;

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| "warn,comic_scan=info".parse().unwrap()),
        )
        .init();

    let args = parse_args();

    // Discover and sort image files
    let mut image_paths = discover_images(&args.input)?;
    if image_paths.is_empty() {
        anyhow::bail!("No images found in {}", args.input.display());
    }
    image_paths.sort();

    println!("═══ ComicScan Chapter Translate ═══");
    println!("Input:   {} ({} pages)", args.input.display(), image_paths.len());
    println!("Project: {}", args.project);
    println!("Chapter: {}", args.chapter);
    println!("Lang:    {} → {}", args.source_lang, args.target_lang);
    println!("Output:  {}", args.output.display());
    println!();

    // Load all images
    let t = Instant::now();
    let images: Vec<image::DynamicImage> = image_paths
        .iter()
        .map(|p| image::open(p).map_err(|e| anyhow::anyhow!("Failed to load {}: {e}", p.display())))
        .collect::<Result<Vec<_>, _>>()?;
    println!("Loaded {} images in {:.1}s", images.len(), t.elapsed().as_secs_f64());

    // Create runner
    let config = comic_scan::config::AppConfig::load()?;
    let runner = comic_scan::runner::TranslationRunner::new(&config).await?;

    // Detect + OCR
    let t = Instant::now();
    let source_lang = comic_scan::pipeline::detect_source_lang(Some(&args.source_lang), &args.target_lang);
    let detections = tokio::task::block_in_place(|| {
        comic_scan::pipeline::chapter::detect_chapter(&runner.detector, &runner.ocr, &images, source_lang)
    })?;

    // Translate + fit + render
    let output = comic_scan::pipeline::chapter::translate_and_render(
        &runner,
        detections,
        &images,
        &args.target_lang,
        source_lang,
        Some(&args.project),
        Some(args.chapter),
    )
    .await?;
    let translate_secs = t.elapsed().as_secs_f64();

    // Print results and save
    std::fs::create_dir_all(&args.output)?;

    let mut total_bubbles = 0;
    for page in &output.pages {
        if page.bubbles.is_empty() && page.rendered_image.is_none() {
            continue;
        }
        println!("── Page {} ({} bubbles) ──", page.page_index, page.bubbles.len());
        for b in &page.bubbles {
            println!(
                "  [{}] {}px \"{}\" → \"{}\"",
                b.bubble_id,
                b.font_size_px,
                b.source_text.chars().take(40).collect::<String>(),
                b.translated_text.replace('\n', " ").chars().take(50).collect::<String>(),
            );
        }
        total_bubbles += page.bubbles.len();

        if let Some(rendered) = &page.rendered_image {
            let filename = format!("page_{:03}.png", page.page_index);
            let path = args.output.join(&filename);
            rendered.save(&path)?;
            println!("  → {}", path.display());
        }
    }

    println!();
    println!("Total: {} bubbles across {} pages in {:.1}s", total_bubbles, images.len(), translate_secs);

    Ok(())
}

struct Args {
    input: PathBuf,
    output: PathBuf,
    project: String,
    chapter: usize,
    source_lang: String,
    target_lang: String,
}

fn parse_args() -> Args {
    let args: Vec<String> = std::env::args().collect();
    let mut input = PathBuf::from(".");
    let mut output = PathBuf::from("./output");
    let mut project = "default".to_string();
    let mut chapter = 0usize;
    let mut source_lang = "ko".to_string();
    let mut target_lang = "vi".to_string();

    let mut i = 1;
    while i < args.len() {
        match args[i].as_str() {
            "--input" if i + 1 < args.len() => { input = PathBuf::from(&args[i + 1]); i += 2; }
            "--output" if i + 1 < args.len() => { output = PathBuf::from(&args[i + 1]); i += 2; }
            "--project" if i + 1 < args.len() => { project = args[i + 1].clone(); i += 2; }
            "--chapter" if i + 1 < args.len() => { chapter = args[i + 1].parse().unwrap_or(0); i += 2; }
            "--source" if i + 1 < args.len() => { source_lang = args[i + 1].clone(); i += 2; }
            "--target" if i + 1 < args.len() => { target_lang = args[i + 1].clone(); i += 2; }
            "--help" | "-h" => {
                eprintln!("Usage: translate_chapter --input <dir> [--output <dir>] [--project <name>] [--chapter <n>] [--source <lang>] [--target <lang>]");
                std::process::exit(0);
            }
            _ => { i += 1; }
        }
    }

    Args { input, output, project, chapter, source_lang, target_lang }
}

fn discover_images(dir: &Path) -> anyhow::Result<Vec<PathBuf>> {
    if !dir.is_dir() {
        anyhow::bail!("{} is not a directory", dir.display());
    }

    let extensions = ["png", "jpg", "jpeg", "webp", "bmp", "tiff"];
    let mut paths = Vec::new();

    for entry in std::fs::read_dir(dir)? {
        let entry = entry?;
        let path = entry.path();
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
