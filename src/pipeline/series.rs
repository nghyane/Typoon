use std::collections::VecDeque;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::Instant;

use anyhow::Result;
use image::DynamicImage;

use crate::cli::util;
use crate::detection::TextDetector;
use crate::ocr::OcrEngine;
use crate::runner::TranslationRunner;

use super::chapter::{self, ChapterPageOutput, PageDetection};

// ── Public types ──

/// A chapter to translate, with its input/output paths and metadata.
pub struct ChapterSpec {
    pub dir: PathBuf,
    pub output: PathBuf,
    pub chapter_num: usize,
    pub label: String,
    pub position: usize, // 1-based
    pub total: usize,
}

/// Result from translating one chapter.
pub struct ChapterResult {
    pub label: String,
    pub position: usize,
    pub total: usize,
    pub pages: Vec<ChapterPageOutput>,
    pub num_pages: usize,
    pub num_bubbles: usize,
    pub elapsed_s: f64,
}

/// Accumulated totals across all chapters.
#[derive(Default)]
pub struct SeriesTotals {
    pub bubbles: usize,
    pub pages: usize,
}

// ── Main entry points ──

/// Translate a single chapter (detect → translate → render).
/// Returns detailed per-page results.
pub async fn translate_single(
    runner: &Arc<TranslationRunner>,
    input: &Path,
    target_lang: &str,
    source_lang: &str,
    project: &str,
    chapter_num: usize,
) -> Result<ChapterResult> {
    let image_paths = util::discover_images(input)?;
    let images = util::load_images(&image_paths)?;

    let t = Instant::now();
    let det = runner.detector.clone();
    let ocr = runner.ocr.clone();
    let lang = source_lang.to_string();
    let detections =
        tokio::task::block_in_place(|| chapter::detect_chapter(&det, &ocr, &images, &lang))?;

    let result = chapter::translate_and_render(
        runner,
        detections,
        &images,
        target_lang,
        source_lang,
        Some(project),
        Some(chapter_num),
    )
    .await?;

    let num_bubbles = result.pages.iter().map(|p| p.bubbles.len()).sum();
    let num_pages = images.len();

    Ok(ChapterResult {
        label: String::new(),
        position: 0,
        total: 0,
        pages: result.pages,
        num_pages,
        num_bubbles,
        elapsed_s: t.elapsed().as_secs_f64(),
    })
}

/// Translate a batch of chapters with pipeline parallelism:
/// - Detection of chapter N+1 overlaps with translation of chapter N
/// - Rendering runs in a bounded background queue
///
/// Calls `on_done` for each completed chapter (for progress reporting).
pub async fn translate_batch(
    runner: &Arc<TranslationRunner>,
    specs: &[ChapterSpec],
    source_lang: &str,
    target_lang: &str,
    project: &str,
    on_done: impl Fn(&ChapterResult),
) -> Result<SeriesTotals> {
    if specs.is_empty() {
        return Ok(SeriesTotals::default());
    }

    let mut totals = SeriesTotals::default();
    let mut render_backlog = RenderBacklog::new(runner.max_pending_render_jobs(), &on_done);

    // Kick off detection for the first chapter
    let mut pending_detect =
        start_detection(&specs[0], &runner.detector, &runner.ocr, source_lang);

    for (wi, spec) in specs.iter().enumerate() {
        let chapter_start = Instant::now();

        // Await current chapter's detection
        let (images, detections) = match pending_detect.take() {
            Some(pd) => pd.handle.await??,
            None => detect_inline(&spec.dir, &runner.detector, &runner.ocr, source_lang)?,
        };

        // Start detection for next chapter (pipeline overlap)
        if let Some(next) = specs.get(wi + 1) {
            pending_detect =
                start_detection(next, &runner.detector, &runner.ocr, source_lang);
        }

        // Translate + fit (sequential — depends on context from prior chapters)
        let prepared_pages = chapter::translate_and_prepare(
            runner,
            detections,
            &images,
            target_lang,
            source_lang,
            Some(project),
            Some(spec.chapter_num),
        )
        .await?;

        // Enqueue render + save in background
        let render_handle = start_render_save(
            Arc::clone(runner),
            images,
            prepared_pages,
            spec.output.clone(),
        );

        render_backlog
            .enqueue(render_handle, spec, chapter_start, &mut totals)
            .await?;
    }

    render_backlog.finish_all(&mut totals).await?;
    Ok(totals)
}

// ── Internals ──

struct PendingDetection {
    handle: tokio::task::JoinHandle<Result<(Vec<DynamicImage>, Vec<PageDetection>)>>,
}

struct PendingRender {
    handle: tokio::task::JoinHandle<Result<Vec<ChapterPageOutput>>>,
    label: String,
    position: usize,
    total: usize,
    chapter_start: Instant,
}

struct RenderBacklog<'a, F> {
    max_jobs: usize,
    pending: VecDeque<PendingRender>,
    on_done: &'a F,
}

impl<'a, F: Fn(&ChapterResult)> RenderBacklog<'a, F> {
    fn new(max_jobs: usize, on_done: &'a F) -> Self {
        Self {
            max_jobs: max_jobs.max(1),
            pending: VecDeque::new(),
            on_done,
        }
    }

    async fn enqueue(
        &mut self,
        handle: tokio::task::JoinHandle<Result<Vec<ChapterPageOutput>>>,
        spec: &ChapterSpec,
        chapter_start: Instant,
        totals: &mut SeriesTotals,
    ) -> Result<()> {
        self.pending.push_back(PendingRender {
            handle,
            label: spec.label.clone(),
            position: spec.position,
            total: spec.total,
            chapter_start,
        });
        while self.pending.len() > self.max_jobs {
            self.finish_one(totals).await?;
        }
        Ok(())
    }

    async fn finish_all(&mut self, totals: &mut SeriesTotals) -> Result<()> {
        while !self.pending.is_empty() {
            self.finish_one(totals).await?;
        }
        Ok(())
    }

    async fn finish_one(&mut self, totals: &mut SeriesTotals) -> Result<()> {
        let Some(job) = self.pending.pop_front() else {
            return Ok(());
        };
        let pages = job.handle.await??;
        let num_bubbles: usize = pages.iter().map(|p| p.bubbles.len()).sum();
        let num_pages = pages.len();

        totals.bubbles += num_bubbles;
        totals.pages += num_pages;

        (self.on_done)(&ChapterResult {
            label: job.label,
            position: job.position,
            total: job.total,
            pages,
            num_pages,
            num_bubbles,
            elapsed_s: job.chapter_start.elapsed().as_secs_f64(),
        });

        Ok(())
    }
}

fn start_detection(
    spec: &ChapterSpec,
    detector: &Arc<TextDetector>,
    ocr: &Arc<OcrEngine>,
    source_lang: &str,
) -> Option<PendingDetection> {
    let image_paths = match util::discover_images(&spec.dir) {
        Ok(p) if !p.is_empty() => p,
        Ok(_) => {
            tracing::warn!(
                "[{}/{}] {} — skip (no images)",
                spec.position,
                spec.total,
                spec.label
            );
            return None;
        }
        Err(e) => {
            tracing::warn!(
                "[{}/{}] {} — skip ({e})",
                spec.position,
                spec.total,
                spec.label
            );
            return None;
        }
    };

    let images = match util::load_images(&image_paths) {
        Ok(imgs) => imgs,
        Err(e) => {
            tracing::warn!(
                "[{}/{}] {} — skip ({e})",
                spec.position,
                spec.total,
                spec.label
            );
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
    dir: &Path,
    detector: &Arc<TextDetector>,
    ocr: &Arc<OcrEngine>,
    source_lang: &str,
) -> Result<(Vec<DynamicImage>, Vec<PageDetection>)> {
    let image_paths = util::discover_images(dir)?;
    let images = util::load_images(&image_paths)?;
    let det = detector.clone();
    let ocr = ocr.clone();
    let lang = source_lang.to_string();
    let detections =
        tokio::task::block_in_place(|| chapter::detect_chapter(&det, &ocr, &images, &lang))?;
    Ok((images, detections))
}

fn start_render_save(
    runner: Arc<TranslationRunner>,
    images: Vec<DynamicImage>,
    chapter_pages: Vec<ChapterPageOutput>,
    output: PathBuf,
) -> tokio::task::JoinHandle<Result<Vec<ChapterPageOutput>>> {
    tokio::task::spawn_blocking(move || {
        let rendered_pages = chapter::render_prepared_pages(chapter_pages, &images, &runner);
        drop(images);

        std::fs::create_dir_all(&output)?;
        for page in &rendered_pages {
            if let Some(rendered) = &page.rendered_image {
                let path = output.join(format!("page_{:03}.png", page.page_index));
                rendered.save(&path)?;
            }
        }

        Ok(rendered_pages)
    })
}
