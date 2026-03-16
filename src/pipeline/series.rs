use std::collections::VecDeque;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::Instant;

use anyhow::Result;
use image::DynamicImage;

use crate::vision::detection::TextDetector;
use crate::image_io;
use crate::vision::ocr::OcrEngine;
use crate::runner::TranslationRunner;

use super::chapter;
use super::types::*;

// ── Public types ──

/// Minimal input for translating one chapter.
pub struct ChapterJob {
    pub input_dir: PathBuf,
    pub output_dir: PathBuf,
    pub chapter_num: usize,
}

/// Per-chapter translation result.
pub struct ChapterResult {
    pub pages: Vec<RenderedPage>,
    pub num_pages: usize,
    pub num_bubbles: usize,
    pub elapsed_s: f64,
}

/// Accumulated totals across a batch.
#[derive(Default)]
pub struct BatchTotals {
    pub bubbles: usize,
    pub pages: usize,
}

// ── Public entry points ──

/// Translate a single chapter (detect → translate → render).
pub async fn translate_single(
    runner: &Arc<TranslationRunner>,
    input: &Path,
    target_lang: &str,
    source_lang: &str,
    chapter_num: usize,
) -> Result<ChapterResult> {
    let image_paths = image_io::discover_images(input)?;
    let images = image_io::load_images(&image_paths)?;

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
        Some(chapter_num),
    )
    .await?;

    let num_bubbles = result.pages.iter().map(|p| p.bubbles.len()).sum();
    let num_pages = images.len();

    Ok(ChapterResult {
        pages: result.pages,
        num_pages,
        num_bubbles,
        elapsed_s: t.elapsed().as_secs_f64(),
    })
}

/// Translate a batch of chapters with pipeline parallelism.
pub async fn translate_batch(
    runner: &Arc<TranslationRunner>,
    jobs: &[ChapterJob],
    source_lang: &str,
    target_lang: &str,
    on_done: impl Fn(usize, &ChapterResult),
) -> Result<BatchTotals> {
    if jobs.is_empty() {
        return Ok(BatchTotals::default());
    }

    let mut totals = BatchTotals::default();
    let mut render_backlog = RenderBacklog::new(runner.max_pending_render_jobs(), &on_done);

    let mut pending_detect =
        start_detection(&jobs[0].input_dir, &runner.detector, &runner.ocr, source_lang);

    for (i, job) in jobs.iter().enumerate() {
        let chapter_start = Instant::now();

        let (images, detections) = match pending_detect.take() {
            Some(pd) => pd.handle.await??,
            None => detect_inline(&job.input_dir, &runner.detector, &runner.ocr, source_lang)?,
        };

        if let Some(next) = jobs.get(i + 1) {
            pending_detect =
                start_detection(&next.input_dir, &runner.detector, &runner.ocr, source_lang);
        }

        let translated_pages = chapter::translate_chapter(
            runner,
            &detections,
            &images,
            target_lang,
            source_lang,
            Some(job.chapter_num),
        )
        .await?;

        let render_handle = spawn_render_save(
            Arc::clone(runner),
            images,
            translated_pages,
            job.output_dir.clone(),
        );

        render_backlog
            .enqueue(i, render_handle, chapter_start, &mut totals)
            .await?;
    }

    render_backlog.finish_all(&mut totals).await?;
    Ok(totals)
}

// ── Internals ──

struct PendingDetection {
    handle: tokio::task::JoinHandle<Result<(Vec<DynamicImage>, Vec<PageDetections>)>>,
}

struct PendingRender {
    handle: tokio::task::JoinHandle<Result<Vec<RenderedPage>>>,
    job_index: usize,
    chapter_start: Instant,
}

struct RenderBacklog<'a, F> {
    max_jobs: usize,
    pending: VecDeque<PendingRender>,
    on_done: &'a F,
}

impl<'a, F: Fn(usize, &ChapterResult)> RenderBacklog<'a, F> {
    fn new(max_jobs: usize, on_done: &'a F) -> Self {
        Self {
            max_jobs: max_jobs.max(1),
            pending: VecDeque::new(),
            on_done,
        }
    }

    async fn enqueue(
        &mut self,
        job_index: usize,
        handle: tokio::task::JoinHandle<Result<Vec<RenderedPage>>>,
        chapter_start: Instant,
        totals: &mut BatchTotals,
    ) -> Result<()> {
        self.pending.push_back(PendingRender {
            handle,
            job_index,
            chapter_start,
        });
        while self.pending.len() > self.max_jobs {
            self.finish_one(totals).await?;
        }
        Ok(())
    }

    async fn finish_all(&mut self, totals: &mut BatchTotals) -> Result<()> {
        while !self.pending.is_empty() {
            self.finish_one(totals).await?;
        }
        Ok(())
    }

    async fn finish_one(&mut self, totals: &mut BatchTotals) -> Result<()> {
        let Some(job) = self.pending.pop_front() else {
            return Ok(());
        };
        let pages = job.handle.await??;
        let num_bubbles: usize = pages.iter().map(|p| p.bubbles.len()).sum();
        let num_pages = pages.len();

        totals.bubbles += num_bubbles;
        totals.pages += num_pages;

        let result = ChapterResult {
            pages,
            num_pages,
            num_bubbles,
            elapsed_s: job.chapter_start.elapsed().as_secs_f64(),
        };
        (self.on_done)(job.job_index, &result);

        Ok(())
    }
}

fn start_detection(
    dir: &Path,
    detector: &Arc<TextDetector>,
    ocr: &Arc<OcrEngine>,
    source_lang: &str,
) -> Option<PendingDetection> {
    let image_paths = match image_io::discover_images(dir) {
        Ok(p) if !p.is_empty() => p,
        _ => return None,
    };

    let images = match image_io::load_images(&image_paths) {
        Ok(imgs) => imgs,
        Err(_) => return None,
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
) -> Result<(Vec<DynamicImage>, Vec<PageDetections>)> {
    let image_paths = image_io::discover_images(dir)?;
    let images = image_io::load_images(&image_paths)?;
    let det = detector.clone();
    let ocr = ocr.clone();
    let lang = source_lang.to_string();
    let detections =
        tokio::task::block_in_place(|| chapter::detect_chapter(&det, &ocr, &images, &lang))?;
    Ok((images, detections))
}

fn spawn_render_save(
    runner: Arc<TranslationRunner>,
    images: Vec<DynamicImage>,
    translated_pages: Vec<PageTranslations>,
    output: PathBuf,
) -> tokio::task::JoinHandle<Result<Vec<RenderedPage>>> {
    tokio::task::spawn_blocking(move || {
        let rendered_pages = chapter::render_pages(translated_pages, &images, &runner);
        drop(images);

        std::fs::create_dir_all(&output)?;
        for page in &rendered_pages {
            let path = output.join(format!("page_{:03}.png", page.page_index));
            page.image.save(&path)?;
        }

        Ok(rendered_pages)
    })
}
