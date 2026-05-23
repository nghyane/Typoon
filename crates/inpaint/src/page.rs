// SPDX-License-Identifier: GPL-3.0-or-later
//! Per-page inpaint orchestration.
//!
//!   prepared JPEG + raw mask (.bin) + scan msgpack (R2)
//!     → decode → close mask per-block → flood-fill bubbles
//!     → build tiles (REFLECT_101 padded, bucketed) → inpaint each
//!     → compose tiles back into the page → encode PNG → write R2
//!
//! This module replaces the worker-side TS pipeline (mask-close.ts +
//! bubbles.ts + tile.ts + shard.ts + inpaint.ts). Keeping it in-process
//! avoids the 128 MB Worker isolate ceiling — full-page RGBA + Candle
//! tensors easily exceed that.
//!
//! Mask wire format: "MSK1" (4B) + u16LE width (2B) + u16LE height (2B)
//! + W·H uint8.

use std::io::Cursor;
use std::sync::Arc;
use std::time::Instant;

use anyhow::{Context, Result, anyhow};
use image::{ImageBuffer, ImageEncoder, Rgb};
use image::codecs::png::PngEncoder;
use image::imageops::FilterType;
use serde::Deserialize;

use crate::Inpainter;
use crate::s3::S3;

// ── Constants (must match workers/inpaint/src/constants.ts) ────────────────

/// Pad the AOT-GAN inference tile up to a multiple of this. The Rust
/// `Inpainter::inpaint` (Candle) requires `pad_mod = 8` internally — tiles
/// are reflect-padded to that boundary inside `inpaint()`. There is no
/// fixed bucket list: the model handles any 8-multiple size, so we just
/// `snap_up` the (bbox + PAD_AROUND_BUBBLE) crop and inpaint it natively.
/// Previously we truncated to [128,192,256,384] and silently dropped the
/// rest of large bubbles — black-output bug (see Python TiledInpainter,
/// `packages/typoon-vision/typoon/vision/erasers/inpaint.py`).
const SNAP_MOD: usize         = 8;
const MIN_CONTEXT_PX: i32     = 32;
const CONTEXT_FRAC: f32       = 0.50;
/// AOT-GAN needs real unmasked context inside the same inference canvas.
/// Once mask density climbs past ~50–60%, it tends to synthesize muddy or
/// near-black output. Grow crop context until normal tiles land below this.
const TARGET_MASK_DENSITY: f32 = 0.42;
const DEFAULT_AOT_CANVAS: usize = 512;
const REGION_MERGE_DISTANCE: i32 = 96;
const MAX_REGIONS_PER_PAGE: usize = 3;
const FLAT_STD_THRESHOLD: f32 = 10.0;
const FLAT_MIN_SAMPLES: usize = 32;
const DEFAULT_AOT_REGION_CONCURRENCY: usize = 2;
const CLOSE_RADIUS_MIN: i32   = 2;

fn close_radius_frac(class_name: Option<&str>) -> f64 {
    // Defaults to dialogue when the scan msgpack omits "class" (legacy).
    match class_name {
        Some("sfx")       => 0.15,
        Some("narration") => 0.12,
        _                 => 0.10,
    }
}

// ── Scan msgpack subset ───────────────────────────────────────────────────
//
// We only need bbox + class. Everything else in BubbleGroup (polygon, text,
// typesetting hints) is unused for mask close + tile build.

#[derive(Deserialize)]
struct ScanGroupSlim {
    #[serde(default)]
    idx:   i32,
    bbox:  [i32; 4],
    #[serde(default)]
    class: Option<String>,
}

#[derive(Deserialize)]
struct ScanPageSlim {
    groups: Vec<ScanGroupSlim>,
}

// ── Public entry ──────────────────────────────────────────────────────────

#[derive(serde::Serialize)]
pub struct PageResult {
    pub output_key:  String,
    pub bubbles:     usize,
    pub tiles_shape: Vec<String>,
}

/// Per-stage wall-clock timings, all in milliseconds. Returned by the
/// probe endpoint so we can decide whether to keep the per-page fan-out
/// architecture or batch pages inside the container.
#[derive(serde::Serialize, Default)]
pub struct PageTimings {
    pub r2_get_ms:           u64,
    pub decode_ms:           u64,
    pub mask_close_ms:       u64,
    pub flood_fill_ms:       u64,
    pub tiles_build_ms:      u64,
    pub tiles_count:         usize,
    pub candle_inference_ms: u64,   // sum across all tiles
    pub compose_ms:          u64,
    pub png_encode_ms:       u64,
    pub r2_put_ms:           u64,
    pub total_ms:            u64,
    pub image_w:             u32,
    pub image_h:             u32,
    pub bubbles_count:       usize,
}

pub async fn inpaint_page(
    s3:        &S3,
    inpainter: Arc<Inpainter>,
    job_id:    u64,
    page_index: u32,
) -> Result<PageResult> {
    let pad4 = format!("{:04}", page_index);
    let prepared_key = format!("prepared/{job_id}/{pad4}.jpg");
    let mask_key     = format!("mask/{job_id}/{pad4}.bin");
    let scan_key     = format!("scan/{job_id}/{pad4}.msgpack");
    let output_key   = format!("inpaint/{job_id}/{pad4}.png");

    // 1. Fetch inputs in parallel (async, on the caller's runtime).
    let (img_bytes, mask_bytes, scan_bytes) = tokio::try_join!(
        s3.get(&prepared_key),
        s3.get(&mask_key),
        s3.get(&scan_key),
    ).context("R2 fetch failed")?;

    // 2. CPU-bound section: decode JPEG, close mask, flood-fill, build
    //    tiles, run Candle inpaint per tile, compose, encode PNG.
    //    Wrapped in spawn_blocking so the tokio worker thread stays free
    //    to drive concurrent requests on this container.
    //
    //    Important: do NOT call `s3.get/put` here. reqwest's Client is
    //    bound to the outer runtime; calling it from inside a nested
    //    runtime would trigger "dispatch task is gone".
    let (png_bytes, bubbles_count, tiles_shape) = tokio::task::spawn_blocking(move || {
        run_page_sync(
            inpainter.as_ref(),
            img_bytes.to_vec(), mask_bytes.to_vec(), scan_bytes.to_vec(),
        )
    }).await
        .context("inpaint join error")?
        .context("inpaint sync stage failed")?;

    // 3. Upload result (async).
    s3.put(&output_key, png_bytes, "image/png").await?;

    Ok(PageResult { output_key, bubbles: bubbles_count, tiles_shape })
}

// ── Chapter batch ─────────────────────────────────────────────────────────

#[derive(serde::Serialize)]
pub struct ChapterPageOutcome {
    pub page_index: u32,
    #[serde(flatten, skip_serializing_if = "Option::is_none")]
    pub ok:    Option<PageResult>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub error: Option<String>,
}

#[derive(serde::Serialize)]
pub struct ChapterResult {
    pub results:           Vec<ChapterPageOutcome>,
    pub wall_total_ms:     u64,
    pub concurrency_used:  usize,
}

/// Inpaint a list of pages inside one container call. Pages run concurrently
/// up to `concurrency`. Each page's failure is recorded in its
/// `ChapterPageOutcome.error`; siblings keep running. The outer pipeline
/// treats the whole step as failed iff any page errored, and retries the
/// whole step — idempotent because R2 keys are deterministic.
///
/// Why batch: the worker → container hop has a fixed ~5 ms overhead and each
/// container Durable Object handles one HTTP request at a time. Pushing N
/// pages per request keeps a single container instance saturated across its
/// full vCPU budget (4 × standard-4) without paying per-page hop or
/// suffering container-spawn storms.
pub async fn inpaint_chapter(
    s3:          &Arc<S3>,
    inpainter:   Arc<Inpainter>,
    job_id:      u64,
    page_indices: Vec<u32>,
    concurrency: usize,
) -> Result<ChapterResult> {
    let started = std::time::Instant::now();
    let concurrency = concurrency.clamp(1, 16);
    let sem = Arc::new(tokio::sync::Semaphore::new(concurrency));

    let mut handles = Vec::with_capacity(page_indices.len());
    for page_index in page_indices {
        let permit = sem.clone();
        let s3     = s3.clone();
        let inp    = inpainter.clone();
        handles.push(tokio::spawn(async move {
            let _p = permit.acquire_owned().await.unwrap();
            let result = inpaint_page(&s3, inp, job_id, page_index).await;
            match result {
                Ok(r)  => ChapterPageOutcome { page_index, ok: Some(r), error: None },
                Err(e) => ChapterPageOutcome { page_index, ok: None, error: Some(format!("{e:#}")) },
            }
        }));
    }

    let mut results = Vec::with_capacity(handles.len());
    for h in handles {
        match h.await {
            Ok(o)  => results.push(o),
            Err(e) => results.push(ChapterPageOutcome {
                page_index: 0, ok: None, error: Some(format!("join error: {e}")),
            }),
        }
    }
    results.sort_by_key(|o| o.page_index);

    Ok(ChapterResult {
        results,
        wall_total_ms:    started.elapsed().as_millis() as u64,
        concurrency_used: concurrency,
    })
}

/// Sync section — fully self-contained on a worker thread. No async, no
/// network. Owns all inputs by value and returns the encoded PNG.
fn run_page_sync(
    inpainter: &Inpainter,
    img_bytes: Vec<u8>, mask_bytes: Vec<u8>, scan_bytes: Vec<u8>,
) -> Result<(Vec<u8>, usize, Vec<String>)> {
    let dbg = DebugDump::from_env();
    let img = image::load_from_memory(&img_bytes)
        .context("JPEG decode failed")?
        .to_rgb8();
    let (w, h) = (img.width() as usize, img.height() as usize);

    let (orig_mask, mw, mh) = decode_mask_bin(&mask_bytes)?;
    if mw != w || mh != h {
        return Err(anyhow!("shape mismatch: img={w}x{h} mask={mw}x{mh}"));
    }
    dbg.dump_mask("00-orig-mask", &orig_mask, w, h);

    let scan: ScanPageSlim = rmp_serde::from_slice(&scan_bytes)
        .context("scan msgpack decode failed")?;

    let img_raw = img.as_raw();
    let closed  = close_mask_per_block(&orig_mask, w, h, &scan.groups, img_raw);
    dbg.dump_mask("01-closed-mask", &closed, w, h);
    let regions = build_inpaint_regions(&closed, w, h, &scan.groups);

    let mut composite: Vec<u8> = img.into_raw();
    dbg.dump_rgb("02-input-rgb", &composite, w, h);
    let mut tiles_shape = Vec::with_capacity(regions.len());

    if regions.is_empty() {
        let png = encode_png(&composite, w, h)?;
        return Ok((png, 0, tiles_shape));
    }

    let mut jobs = Vec::new();
    for (bi, bb) in regions.iter().enumerate() {
        if let Some(fill) = flat_fill_color(&composite, &closed, w, h, bb) {
            fill_region(&mut composite, &closed, w, bb, fill);
            tiles_shape.push("flat-fill".to_string());
            tracing::info!(region = bi, "flat inpaint fill");
            continue;
        }

        let (tile_rgb, tile_mask, tile_w, tile_h, pad_box) =
            build_tile(&composite, &closed, w, h, bb);
        tiles_shape.push(format!("{tile_w}x{tile_h}"));
        let mask_on = tile_mask.iter().filter(|&&v| v >= 127).count();
        let mask_density = mask_on as f32 / (tile_w * tile_h) as f32;
        tracing::info!(
            bubble = bi,
            tile = %format!("{tile_w}x{tile_h}"),
            crop = %format!("{}x{}", pad_box.x1 - pad_box.x0 + 1, pad_box.y1 - pad_box.y0 + 1),
            mask_density,
            "inpaint tile start",
        );

        dbg.dump_rgb(&format!("tile-{bi:02}-rgb"),   &tile_rgb, tile_w, tile_h);
        dbg.dump_mask(&format!("tile-{bi:02}-mask"), &tile_mask, tile_w, tile_h);
        jobs.push(AotJob { idx: bi, tile_rgb, tile_mask, tile_w, tile_h, pad_box });
    }

    let conc = aot_region_concurrency();
    for chunk in jobs.chunks(conc) {
        let outputs = std::thread::scope(|scope| {
            let mut handles = Vec::with_capacity(chunk.len());
            for job in chunk {
                handles.push(scope.spawn(move || -> Result<AotOutput> {
                    let started = Instant::now();
                    let out_rgb = inpainter
                        .inpaint(&job.tile_rgb, &job.tile_mask, job.tile_w as u32, job.tile_h as u32)
                        .with_context(|| format!("inpaint tile {}x{} failed", job.tile_w, job.tile_h))?;
                    Ok(AotOutput {
                        idx: job.idx,
                        tile_w: job.tile_w,
                        tile_h: job.tile_h,
                        pad_box: job.pad_box,
                        out_rgb,
                        inference_ms: started.elapsed().as_millis() as u64,
                    })
                }));
            }
            let mut outputs = Vec::with_capacity(handles.len());
            for handle in handles {
                outputs.push(handle.join().expect("AOT thread panicked")?);
            }
            Ok::<Vec<AotOutput>, anyhow::Error>(outputs)
        })?;

        for out in outputs {
            tracing::info!(bubble = out.idx, inference_ms = out.inference_ms, "inpaint tile done");
            dbg.dump_rgb(&format!("tile-{:02}-out", out.idx), &out.out_rgb, out.tile_w, out.tile_h);
            compose_tile(&mut composite, w, &closed, &out.pad_box, out.tile_w, out.tile_h, &out.out_rgb);
        }
    }

    dbg.dump_rgb("99-final-rgb", &composite, w, h);

    let png = encode_png(&composite, w, h)?;
    Ok((png, regions.len(), tiles_shape))
}

// ── Local debug dumps ─────────────────────────────────────────────────────
//
// When INPAINT_DEBUG_DIR is set, run_page_sync writes intermediate buffers
// as PNG into that directory. Each file is prefixed so they sort in pipeline
// order: 00-orig-mask, 01-closed-mask, 02-input-rgb, tile-00-rgb/mask/out,
// …, 99-final-rgb. Production has the env var unset → zero overhead.

struct DebugDump {
    dir: Option<std::path::PathBuf>,
}

struct AotJob {
    idx:       usize,
    tile_rgb:  Vec<u8>,
    tile_mask: Vec<u8>,
    tile_w:    usize,
    tile_h:    usize,
    pad_box:   BBox,
}

struct AotOutput {
    idx:          usize,
    tile_w:       usize,
    tile_h:       usize,
    pad_box:      BBox,
    out_rgb:      Vec<u8>,
    inference_ms: u64,
}

fn aot_region_concurrency() -> usize {
    std::env::var("AOT_REGION_CONCURRENCY")
        .ok()
        .and_then(|v| v.parse::<usize>().ok())
        .filter(|&v| v > 0)
        .unwrap_or(DEFAULT_AOT_REGION_CONCURRENCY)
        .min(MAX_REGIONS_PER_PAGE)
}

impl DebugDump {
    fn from_env() -> Self {
        Self { dir: std::env::var_os("INPAINT_DEBUG_DIR").map(Into::into) }
    }

    fn dump_mask(&self, name: &str, mask: &[u8], w: usize, h: usize) {
        let Some(dir) = self.dir.as_ref() else { return };
        let _ = std::fs::create_dir_all(dir);
        let path = dir.join(format!("{name}.png"));
        // Mask is single-channel u8; encode as Luma8.
        let buf: image::ImageBuffer<image::Luma<u8>, &[u8]> =
            image::ImageBuffer::from_raw(w as u32, h as u32, mask)
                .expect("mask shape");
        let _ = buf.save(&path);
    }

    fn dump_rgb(&self, name: &str, rgb: &[u8], w: usize, h: usize) {
        let Some(dir) = self.dir.as_ref() else { return };
        let _ = std::fs::create_dir_all(dir);
        let path = dir.join(format!("{name}.png"));
        let buf: image::ImageBuffer<image::Rgb<u8>, &[u8]> =
            image::ImageBuffer::from_raw(w as u32, h as u32, rgb)
                .expect("rgb shape");
        let _ = buf.save(&path);
    }
}

// ── Traced variants (used by the /probe endpoint) ─────────────────────────

/// Sync section with per-stage timings. Mirrors `run_page_sync` step-for-step
/// so probe output reflects the production hot path verbatim.
fn run_page_sync_traced(
    inpainter: &Inpainter,
    img_bytes: Vec<u8>, mask_bytes: Vec<u8>, scan_bytes: Vec<u8>,
) -> Result<(Vec<u8>, PageTimings)> {
    let mut t = PageTimings::default();

    let t0 = Instant::now();
    let img = image::load_from_memory(&img_bytes)
        .context("JPEG decode failed")?
        .to_rgb8();
    let (w, h) = (img.width() as usize, img.height() as usize);
    t.decode_ms = t0.elapsed().as_millis() as u64;
    t.image_w = img.width();
    t.image_h = img.height();

    let (orig_mask, mw, mh) = decode_mask_bin(&mask_bytes)?;
    if mw != w || mh != h {
        return Err(anyhow!("shape mismatch: img={w}x{h} mask={mw}x{mh}"));
    }

    let scan: ScanPageSlim = rmp_serde::from_slice(&scan_bytes)
        .context("scan msgpack decode failed")?;

    let t1 = Instant::now();
    let img_raw = img.as_raw();
    let closed = close_mask_per_block(&orig_mask, w, h, &scan.groups, img_raw);
    t.mask_close_ms = t1.elapsed().as_millis() as u64;

    let t2 = Instant::now();
    let regions = build_inpaint_regions(&closed, w, h, &scan.groups);
    t.flood_fill_ms  = t2.elapsed().as_millis() as u64;
    t.bubbles_count  = regions.len();

    let mut composite: Vec<u8> = img.into_raw();

    if !regions.is_empty() {
        let t3 = Instant::now();
        let mut tiles = Vec::new();
        for bb in &regions {
            if let Some(fill) = flat_fill_color(&composite, &closed, w, h, bb) {
                fill_region(&mut composite, &closed, w, bb, fill);
            } else {
                tiles.push(build_tile(&composite, &closed, w, h, bb));
            }
        }
        t.tiles_build_ms = t3.elapsed().as_millis() as u64;
        t.tiles_count    = tiles.len();

        let t4 = Instant::now();
        let mut composed_outputs: Vec<(usize, usize, BBox, Vec<u8>)> = Vec::with_capacity(tiles.len());
        for (rgb, mask, tile_w, tile_h, pad_box) in tiles.iter() {
            let out_rgb = inpainter
                .inpaint(rgb, mask, *tile_w as u32, *tile_h as u32)
                .with_context(|| format!("inpaint tile {tile_w}x{tile_h} failed"))?;
            composed_outputs.push((*tile_w, *tile_h, *pad_box, out_rgb));
        }
        t.candle_inference_ms = t4.elapsed().as_millis() as u64;

        let t5 = Instant::now();
        for (tile_w, tile_h, pad_box, out_rgb) in &composed_outputs {
            compose_tile(&mut composite, w, &closed, pad_box, *tile_w, *tile_h, out_rgb);
        }
        t.compose_ms = t5.elapsed().as_millis() as u64;
    }

    let t6 = Instant::now();
    let png = encode_png(&composite, w, h)?;
    t.png_encode_ms = t6.elapsed().as_millis() as u64;

    Ok((png, t))
}

/// Async wrapper for the probe endpoint. Same shape as `inpaint_page` but
/// returns granular per-stage timings instead of `PageResult`. Writes to a
/// per-run probe key so production artifacts aren't clobbered.
pub async fn inpaint_page_traced(
    s3:        &S3,
    inpainter: Arc<Inpainter>,
    job_id:    u64,
    page_index: u32,
    run_index: u32,
) -> Result<PageTimings> {
    let pad4 = format!("{:04}", page_index);
    let prepared_key = format!("prepared/{job_id}/{pad4}.jpg");
    let mask_key     = format!("mask/{job_id}/{pad4}.bin");
    let scan_key     = format!("scan/{job_id}/{pad4}.msgpack");
    let output_key   = format!("inpaint-probe/{job_id}/{pad4}.r{run_index:03}.png");

    let t_total = Instant::now();

    let t_get = Instant::now();
    let (img_bytes, mask_bytes, scan_bytes) = tokio::try_join!(
        s3.get(&prepared_key),
        s3.get(&mask_key),
        s3.get(&scan_key),
    ).context("R2 fetch failed")?;
    let r2_get_ms = t_get.elapsed().as_millis() as u64;

    let (png_bytes, mut timings) = tokio::task::spawn_blocking(move || {
        run_page_sync_traced(
            inpainter.as_ref(),
            img_bytes.to_vec(), mask_bytes.to_vec(), scan_bytes.to_vec(),
        )
    }).await
        .context("inpaint join error")?
        .context("inpaint sync stage failed")?;

    let t_put = Instant::now();
    s3.put(&output_key, png_bytes, "image/png").await?;
    let r2_put_ms = t_put.elapsed().as_millis() as u64;

    timings.r2_get_ms = r2_get_ms;
    timings.r2_put_ms = r2_put_ms;
    timings.total_ms  = t_total.elapsed().as_millis() as u64;
    Ok(timings)
}

// ── Mask binary header ────────────────────────────────────────────────────

fn decode_mask_bin(buf: &[u8]) -> Result<(Vec<u8>, usize, usize)> {
    if buf.len() < 8 || &buf[0..4] != b"MSK1" {
        return Err(anyhow!("bad mask magic"));
    }
    let w = u16::from_le_bytes([buf[4], buf[5]]) as usize;
    let h = u16::from_le_bytes([buf[6], buf[7]]) as usize;
    let body = &buf[8..];
    if body.len() != w * h {
        return Err(anyhow!("mask size {} != {}x{}", body.len(), w, h));
    }
    Ok((body.to_vec(), w, h))
}

// ── Mask close (port of workers/inpaint/src/mask-close.ts) ────────────────

fn dilate_rect(src: &[u8], w: usize, h: usize, r: i32) -> Vec<u8> {
    if r <= 0 { return src.to_vec(); }
    let r = r as usize;
    let mut h1 = vec![0u8; w * h];
    for y in 0..h {
        let row = y * w;
        for x in 0..w {
            let lo = x.saturating_sub(r);
            let hi = (x + r).min(w - 1);
            let mut on = 0u8;
            for xx in lo..=hi {
                if src[row + xx] != 0 { on = 1; break; }
            }
            h1[row + x] = on;
        }
    }
    let mut out = vec![0u8; w * h];
    for x in 0..w {
        for y in 0..h {
            let lo = y.saturating_sub(r);
            let hi = (y + r).min(h - 1);
            let mut on = 0u8;
            for yy in lo..=hi {
                if h1[yy * w + x] != 0 { on = 1; break; }
            }
            out[y * w + x] = on;
        }
    }
    out
}

fn erode_rect(src: &[u8], w: usize, h: usize, r: i32) -> Vec<u8> {
    if r <= 0 { return src.to_vec(); }
    let r = r as usize;
    let mut h1 = vec![0u8; w * h];
    for y in 0..h {
        let row = y * w;
        for x in 0..w {
            let lo = x.saturating_sub(r);
            let hi = (x + r).min(w - 1);
            let mut on = 1u8;
            for xx in lo..=hi {
                if src[row + xx] == 0 { on = 0; break; }
            }
            h1[row + x] = on;
        }
    }
    let mut out = vec![0u8; w * h];
    for x in 0..w {
        for y in 0..h {
            let lo = y.saturating_sub(r);
            let hi = (y + r).min(h - 1);
            let mut on = 1u8;
            for yy in lo..=hi {
                if h1[yy * w + x] == 0 { on = 0; break; }
            }
            out[y * w + x] = on;
        }
    }
    out
}

fn close_radius_for(group: &ScanGroupSlim) -> i32 {
    let [x1, y1, x2, y2] = group.bbox;
    let short = (x2 - x1).min(y2 - y1).max(0) as f64;
    let frac  = close_radius_frac(group.class.as_deref());
    ((short * frac).round() as i32).max(CLOSE_RADIUS_MIN)
}

fn close_mask_per_block(
    mask: &[u8], w: usize, h: usize,
    groups: &[ScanGroupSlim],
    img: &[u8],
) -> Vec<u8> {
    // bin[i] ∈ {0,1}
    let mut bin = vec![0u8; w * h];
    for i in 0..w * h {
        bin[i] = if mask[i] >= 127 { 1 } else { 0 };
    }
    let mut result = bin.clone();

    for b in groups {
        let r = close_radius_for(b);
        let [bx1, by1, bx2, by2] = b.bbox;
        let px0 = (bx1 - r).max(0) as usize;
        let py0 = (by1 - r).max(0) as usize;
        let px1 = ((bx2 + r) as usize).min(w);
        let py1 = ((by2 + r) as usize).min(h);
        if px1 <= px0 || py1 <= py0 { continue; }
        let pw = px1 - px0;
        let ph = py1 - py0;

        // Check if the original mask in this bbox is suspiciously dense
        // (polygon-fill fallback from scan stage). If so, regenerate a
        // tighter mask from the image content via stroke detection.
        let bbox_area = (bx2 - bx1).max(0) as usize * (by2 - by1).max(0) as usize;
        let mut bbox_mask_on = 0usize;
        for y in by1.max(0) as usize..by2.min(h as i32) as usize {
            for x in bx1.max(0) as usize..bx2.min(w as i32) as usize {
                if bin[y * w + x] != 0 { bbox_mask_on += 1; }
            }
        }
        let bbox_density = if bbox_area > 0 {
            bbox_mask_on as f32 / bbox_area as f32
        } else { 0.0 };

        // Block-local patch
        let mut patch = vec![0u8; pw * ph];
        for y in 0..ph {
            for x in 0..pw {
                patch[y * pw + x] = bin[(py0 + y) * w + (px0 + x)];
            }
        }

        // When the mask is nearly a full rectangle (polygon fill),
        // detect actual strokes from the image instead.
        if bbox_density > 0.85 {
            let strokes = detect_strokes_in_bbox(img, w, h, &[bx1, by1, bx2, by2], r);
            for y in 0..ph {
                for x in 0..pw {
                    patch[y * pw + x] = strokes[(py0 + y) * w + (px0 + x)];
                }
            }
        }

        let dilated = dilate_rect(&patch, pw, ph, r);
        let mut closed = erode_rect(&dilated, pw, ph, r);
        fill_enclosed_holes(&mut closed, pw, ph);

        // Outsider guard: any other group's centre inside this closed bbox?
        let bridges = groups.iter().any(|o| {
            if o.idx == b.idx { return false; }
            let cx = (o.bbox[0] + o.bbox[2]) as f64 / 2.0;
            let cy = (o.bbox[1] + o.bbox[3]) as f64 / 2.0;
            cx >= px0 as f64 && cx < px1 as f64 && cy >= py0 as f64 && cy < py1 as f64
        });

        for y in 0..ph {
            for x in 0..pw {
                if closed[y * pw + x] == 0 { continue; }
                let gi = (py0 + y) * w + (px0 + x);
                if bin[gi] == 0 && bridges { continue; }
                result[gi] = 1;
            }
        }
    }

    // Materialise as 0/255 (caller threshold is >= 127).
    let mut out = vec![0u8; w * h];
    for i in 0..w * h { out[i] = if result[i] != 0 { 255 } else { 0 }; }
    out
}

/// Stroke detection from image content. Used when the original mask is
/// suspiciously dense (>85% in the group bbox), indicating a polygon
/// full-fill fallback from the scan stage. Detects ink/dark/saturated
/// pixels within the bbox region.
fn detect_strokes_in_bbox(
    img: &[u8], img_w: usize, img_h: usize,
    bbox: &[i32; 4],
    pad: i32,
) -> Vec<u8> {
    let [bx1, by1, bx2, by2] = *bbox;
    let mut out = vec![0u8; img_w * img_h];
    let x0 = (bx1 - pad).max(0) as usize;
    let y0 = (by1 - pad).max(0) as usize;
    let x1 = (bx2 + pad).min(img_w as i32 - 1) as usize;
    let y1 = (by2 + pad).min(img_h as i32 - 1) as usize;
    for y in y0..=y1 {
        for x in x0..=x1 {
            let i = (y * img_w + x) * 3;
            let r = img[i] as i16;
            let g = img[i + 1] as i16;
            let b = img[i + 2] as i16;
            let mx = r.max(g).max(b);
            let mn = r.min(g).min(b);
            let sat = mx - mn;
            let lum = (0.299 * r as f32 + 0.587 * g as f32 + 0.114 * b as f32) as i16;
            let is_ink = ((r - b) > 20 && sat > 30 && lum < 160)
                       || (lum < 80);
            if is_ink {
                out[y * img_w + x] = 1;
            }
        }
    }
    out
}

fn fill_enclosed_holes(mask: &mut [u8], w: usize, h: usize) {
    if w == 0 || h == 0 { return; }

    let mut outside = vec![0u8; w * h];
    let mut stack = Vec::new();
    for x in 0..w {
        push_outside(mask, &mut outside, &mut stack, x);
        push_outside(mask, &mut outside, &mut stack, (h - 1) * w + x);
    }
    for y in 0..h {
        push_outside(mask, &mut outside, &mut stack, y * w);
        push_outside(mask, &mut outside, &mut stack, y * w + (w - 1));
    }

    while let Some(i) = stack.pop() {
        let x = i % w;
        let y = i / w;
        if x > 0     { push_outside(mask, &mut outside, &mut stack, i - 1); }
        if x + 1 < w { push_outside(mask, &mut outside, &mut stack, i + 1); }
        if y > 0     { push_outside(mask, &mut outside, &mut stack, i - w); }
        if y + 1 < h { push_outside(mask, &mut outside, &mut stack, i + w); }
    }

    for i in 0..w * h {
        if mask[i] == 0 && outside[i] == 0 {
            mask[i] = 1;
        }
    }
}

fn push_outside(mask: &[u8], outside: &mut [u8], stack: &mut Vec<usize>, i: usize) {
    if mask[i] == 0 && outside[i] == 0 {
        outside[i] = 1;
        stack.push(i);
    }
}

// ── Flood-fill (port of workers/inpaint/src/bubbles.ts) ───────────────────

#[derive(Debug, Clone, Copy)]
struct BBox { x0: i32, y0: i32, x1: i32, y1: i32 }

fn find_bubbles(mask: &[u8], w: usize, h: usize) -> Vec<BBox> {
    let mut visited = vec![0u8; w * h];
    let mut out     = Vec::new();
    let mut stack: Vec<usize> = Vec::with_capacity(1024);

    for py in 0..h {
        for px in 0..w {
            let idx = py * w + px;
            if visited[idx] != 0 || mask[idx] < 127 { continue; }
            let mut x0 = px as i32; let mut y0 = py as i32;
            let mut x1 = px as i32; let mut y1 = py as i32;
            stack.clear();
            stack.push(idx);
            visited[idx] = 1;
            while let Some(cur) = stack.pop() {
                let cx = (cur % w) as i32;
                let cy = (cur / w) as i32;
                if cx < x0 { x0 = cx; }
                if cx > x1 { x1 = cx; }
                if cy < y0 { y0 = cy; }
                if cy > y1 { y1 = cy; }
                if cx > 0 {
                    let n = cur - 1;
                    if visited[n] == 0 && mask[n] >= 127 { visited[n] = 1; stack.push(n); }
                }
                if (cx as usize) < w - 1 {
                    let n = cur + 1;
                    if visited[n] == 0 && mask[n] >= 127 { visited[n] = 1; stack.push(n); }
                }
                if cy > 0 {
                    let n = cur - w;
                    if visited[n] == 0 && mask[n] >= 127 { visited[n] = 1; stack.push(n); }
                }
                if (cy as usize) < h - 1 {
                    let n = cur + w;
                    if visited[n] == 0 && mask[n] >= 127 { visited[n] = 1; stack.push(n); }
                }
            }
            out.push(BBox { x0, y0, x1, y1 });
        }
    }
    out
}

fn build_inpaint_regions(mask: &[u8], w: usize, h: usize, groups: &[ScanGroupSlim]) -> Vec<BBox> {
    let mut regions = Vec::new();
    for g in groups {
        if let Some(bb) = mask_bbox_in_group(mask, w, h, g) {
            regions.push(bb);
        }
    }
    if regions.is_empty() {
        regions = find_bubbles(mask, w, h);
    }
    merge_regions(regions)
}

fn mask_bbox_in_group(mask: &[u8], w: usize, h: usize, group: &ScanGroupSlim) -> Option<BBox> {
    let r = close_radius_for(group).max(REGION_MERGE_DISTANCE / 4);
    let [x1, y1, x2, y2] = group.bbox;
    let px0 = (x1 - r).max(0) as usize;
    let py0 = (y1 - r).max(0) as usize;
    let px1 = ((x2 + r) as usize).min(w.saturating_sub(1));
    let py1 = ((y2 + r) as usize).min(h.saturating_sub(1));
    let mut out: Option<BBox> = None;
    for y in py0..=py1 {
        for x in px0..=px1 {
            if mask[y * w + x] < 127 { continue; }
            out = Some(match out {
                Some(bb) => BBox {
                    x0: bb.x0.min(x as i32),
                    y0: bb.y0.min(y as i32),
                    x1: bb.x1.max(x as i32),
                    y1: bb.y1.max(y as i32),
                },
                None => BBox { x0: x as i32, y0: y as i32, x1: x as i32, y1: y as i32 },
            });
        }
    }
    out
}

fn merge_regions(mut regions: Vec<BBox>) -> Vec<BBox> {
    regions.sort_by_key(|b| (b.y0, b.x0));
    let mut changed = true;
    while changed {
        changed = false;
        'outer: for i in 0..regions.len() {
            for j in i + 1..regions.len() {
                if bbox_distance(&regions[i], &regions[j]) <= REGION_MERGE_DISTANCE {
                    let merged = union_bbox(&regions[i], &regions[j]);
                    regions[i] = merged;
                    regions.remove(j);
                    changed = true;
                    break 'outer;
                }
            }
        }
    }
    while regions.len() > MAX_REGIONS_PER_PAGE {
        let mut best = (0usize, 1usize, i32::MAX);
        for i in 0..regions.len() {
            for j in i + 1..regions.len() {
                let d = bbox_distance(&regions[i], &regions[j]);
                if d < best.2 { best = (i, j, d); }
            }
        }
        let merged = union_bbox(&regions[best.0], &regions[best.1]);
        regions[best.0] = merged;
        regions.remove(best.1);
    }
    regions
}

fn union_bbox(a: &BBox, b: &BBox) -> BBox {
    BBox { x0: a.x0.min(b.x0), y0: a.y0.min(b.y0), x1: a.x1.max(b.x1), y1: a.y1.max(b.y1) }
}

fn bbox_distance(a: &BBox, b: &BBox) -> i32 {
    let dx = if a.x1 < b.x0 { b.x0 - a.x1 } else if b.x1 < a.x0 { a.x0 - b.x1 } else { 0 };
    let dy = if a.y1 < b.y0 { b.y0 - a.y1 } else if b.y1 < a.y0 { a.y0 - b.y1 } else { 0 };
    dx.max(dy)
}

fn snap_up(v: i32, m: usize) -> usize {
    let m = m as i32;
    (((v + m - 1) / m) * m).max(0) as usize
}

/// Build a tile crop from the page + mask.
///
/// AOT-GAN is local-context sensitive: a crop that is mostly mask gives the
/// generator little real texture to condition on and commonly collapses to
/// muddy/black output. Grow the crop until the mask occupies a safe fraction
/// of the inference canvas, expanding further on the available sides when a
/// component sits near a page edge. The mask is never expanded here; only the
/// surrounding context is.
fn build_tile(
    composite: &[u8],
    mask:      &[u8],
    src_w:     usize,
    src_h:     usize,
    bb:        &BBox,
) -> (Vec<u8>, Vec<u8>, usize, usize, BBox) {
    let bw = bb.x1 - bb.x0 + 1;
    let bh = bb.y1 - bb.y0 + 1;
    let short = bw.min(bh) as f32;
    let mut context = ((short * CONTEXT_FRAC).round() as i32).max(MIN_CONTEXT_PX);
    let mut crop = crop_with_context(bb, context, src_w, src_h);

    for _ in 0..16 {
        let w0 = crop_width(&crop) as usize;
        let h0 = crop_height(&crop) as usize;
        let tile_w = snap_up(w0 as i32, SNAP_MOD).max(SNAP_MOD);
        let tile_h = snap_up(h0 as i32, SNAP_MOD).max(SNAP_MOD);
        let mask_area = count_mask_in_box(mask, src_w, &crop);
        let density = mask_area as f32 / (tile_w * tile_h) as f32;
        let full_page = crop.x0 == 0
            && crop.y0 == 0
            && crop.x1 == src_w as i32 - 1
            && crop.y1 == src_h as i32 - 1;
        if density <= TARGET_MASK_DENSITY || full_page {
            break;
        }
        context = context + (context / 2).max(16);
        crop = crop_with_context(bb, context, src_w, src_h);
    }

    let x0 = crop.x0;
    let y0 = crop.y0;
    let x1 = crop.x1;
    let y1 = crop.y1;
    let w0 = crop_width(&crop) as usize;
    let h0 = crop_height(&crop) as usize;
    let canvas = aot_canvas_size();
    let tile_w = canvas;
    let tile_h = canvas;

    let mut crop_rgb = vec![0u8; w0 * h0 * 3];
    let mut crop_mask = vec![0u8; w0 * h0];

    for ty in 0..h0 {
        let src_y = y0 as usize + ty;
        for tx in 0..w0 {
            let src_x = x0 as usize + tx;
            let src_rgb_i = (src_y * src_w + src_x) * 3;
            let src_msk_i =  src_y * src_w + src_x;
            let di = ty * w0 + tx;
            crop_rgb[di * 3    ] = composite[src_rgb_i    ];
            crop_rgb[di * 3 + 1] = composite[src_rgb_i + 1];
            crop_rgb[di * 3 + 2] = composite[src_rgb_i + 2];
            crop_mask[di]        = mask[src_msk_i];
        }
    }
    let rgb = resize_rgb(&crop_rgb, w0, h0, tile_w, tile_h, FilterType::Triangle);
    let msk = resize_mask_nearest(&crop_mask, w0, h0, tile_w, tile_h);
    (rgb, msk, tile_w, tile_h, BBox { x0, y0, x1, y1 })
}

fn aot_canvas_size() -> usize {
    std::env::var("AOT_CANVAS")
        .ok()
        .and_then(|v| v.parse::<usize>().ok())
        .filter(|&v| v >= SNAP_MOD)
        .map(|v| snap_up(v as i32, SNAP_MOD))
        .unwrap_or(DEFAULT_AOT_CANVAS)
}

fn resize_rgb(src: &[u8], w: usize, h: usize, out_w: usize, out_h: usize, filter: FilterType) -> Vec<u8> {
    let img: ImageBuffer<Rgb<u8>, Vec<u8>> = ImageBuffer::from_raw(w as u32, h as u32, src.to_vec())
        .expect("RGB crop shape");
    image::imageops::resize(&img, out_w as u32, out_h as u32, filter).into_raw()
}

fn resize_mask_nearest(src: &[u8], w: usize, h: usize, out_w: usize, out_h: usize) -> Vec<u8> {
    let img: ImageBuffer<image::Luma<u8>, Vec<u8>> = ImageBuffer::from_raw(w as u32, h as u32, src.to_vec())
        .expect("mask crop shape");
    image::imageops::resize(&img, out_w as u32, out_h as u32, FilterType::Nearest)
        .into_raw()
        .into_iter()
        .map(|v| if v >= 127 { 255 } else { 0 })
        .collect()
}

fn crop_with_context(bb: &BBox, context: i32, src_w: usize, src_h: usize) -> BBox {
    BBox {
        x0: (bb.x0 - context).max(0),
        y0: (bb.y0 - context).max(0),
        x1: (bb.x1 + context).min(src_w as i32 - 1),
        y1: (bb.y1 + context).min(src_h as i32 - 1),
    }
}

fn crop_width(bb: &BBox) -> i32 { bb.x1 - bb.x0 + 1 }
fn crop_height(bb: &BBox) -> i32 { bb.y1 - bb.y0 + 1 }

fn count_mask_in_box(mask: &[u8], src_w: usize, bb: &BBox) -> usize {
    let mut n = 0usize;
    for y in bb.y0..=bb.y1 {
        let row = y as usize * src_w;
        for x in bb.x0..=bb.x1 {
            if mask[row + x as usize] >= 127 {
                n += 1;
            }
        }
    }
    n
}

fn flat_fill_color(rgb: &[u8], mask: &[u8], w: usize, h: usize, bb: &BBox) -> Option<[u8; 3]> {
    let mut samples = Vec::new();
    let x0 = bb.x0.max(0) as usize;
    let y0 = bb.y0.max(0) as usize;
    let x1 = bb.x1.min(w as i32 - 1) as usize;
    let y1 = bb.y1.min(h as i32 - 1) as usize;
    for y in y0..=y1 {
        for x in x0..=x1 {
            let i = y * w + x;
            if mask[i] >= 127 { continue; }
            let adjacent =
                (x > 0     && mask[i - 1] >= 127) ||
                (x + 1 < w && mask[i + 1] >= 127) ||
                (y > 0     && mask[i - w] >= 127) ||
                (y + 1 < h && mask[i + w] >= 127);
            if !adjacent { continue; }
            let p = i * 3;
            samples.push([rgb[p], rgb[p + 1], rgb[p + 2]]);
        }
    }
    if samples.len() < FLAT_MIN_SAMPLES { return None; }
    let n = samples.len() as f32;
    let mut mean = [0f32; 3];
    for s in &samples {
        mean[0] += s[0] as f32; mean[1] += s[1] as f32; mean[2] += s[2] as f32;
    }
    mean[0] /= n; mean[1] /= n; mean[2] /= n;
    let mut var = 0f32;
    for s in &samples {
        for c in 0..3 {
            let d = s[c] as f32 - mean[c];
            var += d * d;
        }
    }
    let std = (var / (n * 3.0)).sqrt();
    if std > FLAT_STD_THRESHOLD { return None; }
    Some([mean[0].round() as u8, mean[1].round() as u8, mean[2].round() as u8])
}

fn fill_region(composite: &mut [u8], mask: &[u8], w: usize, bb: &BBox, color: [u8; 3]) {
    let h = mask.len() / w;
    let x0 = bb.x0.max(0) as usize;
    let y0 = bb.y0.max(0) as usize;
    let x1 = bb.x1.min(w as i32 - 1) as usize;
    let y1 = bb.y1.min(h as i32 - 1) as usize;
    for y in y0..=y1 {
        for x in x0..=x1 {
            let mi = y * w + x;
            if mask[mi] < 127 { continue; }
            let alpha = feather_alpha(mask, w, x, y, 3);
            let p = mi * 3;
            for c in 0..3 {
                let old = composite[p + c] as f32;
                let new = color[c] as f32;
                composite[p + c] = (old * (1.0 - alpha) + new * alpha)
                    .round()
                    .clamp(0.0, 255.0) as u8;
            }
        }
    }
}

/// Paste tile_rgb back into composite, only at pixels where the inpaint mask
/// is set. Composite is RGB stride 3.
fn compose_tile(
    composite: &mut [u8],
    src_w:     usize,
    orig_mask: &[u8],
    pad_box:   &BBox,
    tile_w:    usize,
    tile_h:    usize,
    tile_rgb:  &[u8],
) {
    let w0 = (pad_box.x1 - pad_box.x0 + 1) as usize;
    let h0 = (pad_box.y1 - pad_box.y0 + 1) as usize;
    let crop_out = resize_rgb(tile_rgb, tile_w, tile_h, w0, h0, FilterType::Triangle);
    for ty in 0..h0 {
        let dst_y = pad_box.y0 as usize + ty;
        for tx in 0..w0 {
            let dst_x = pad_box.x0 as usize + tx;
            let mi    = dst_y * src_w + dst_x;
            if orig_mask[mi] < 127 { continue; }
            let alpha = feather_alpha(orig_mask, src_w, dst_x, dst_y, 3);
            let di = (dst_y * src_w + dst_x) * 3;
            let ti = (ty * w0 + tx) * 3;
            for c in 0..3 {
                let old = composite[di + c] as f32;
                let new = crop_out[ti + c] as f32;
                composite[di + c] = (old * (1.0 - alpha) + new * alpha)
                    .round()
                    .clamp(0.0, 255.0) as u8;
            }
        }
    }
}

fn feather_alpha(mask: &[u8], w: usize, x: usize, y: usize, radius: i32) -> f32 {
    let h = mask.len() / w;
    let mut min_dist = radius + 1;
    for dy in -radius..=radius {
        for dx in -radius..=radius {
            let d = dx.abs() + dy.abs();
            if d == 0 || d > radius { continue; }
            let nx = x as i32 + dx;
            let ny = y as i32 + dy;
            let outside = nx < 0 || ny < 0 || nx >= w as i32 || ny >= h as i32;
            if outside || mask[ny as usize * w + nx as usize] < 127 {
                min_dist = min_dist.min(d);
            }
        }
    }
    if min_dist > radius { 1.0 } else { min_dist as f32 / (radius + 1) as f32 }
}

// ── PNG encode ────────────────────────────────────────────────────────────

fn encode_png(rgb: &[u8], w: usize, h: usize) -> Result<Vec<u8>> {
    let buf: ImageBuffer<Rgb<u8>, &[u8]> = ImageBuffer::from_raw(w as u32, h as u32, rgb)
        .ok_or_else(|| anyhow!("RGB buffer shape mismatch"))?;
    let mut out = Vec::with_capacity(w * h);
    PngEncoder::new(Cursor::new(&mut out))
        .write_image(buf.as_raw(), w as u32, h as u32, image::ExtendedColorType::Rgb8)
        .context("PNG encode failed")?;
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::fill_enclosed_holes;

    #[test]
    fn fills_inside_closed_stroke() {
        let mut mask = vec![0u8; 25];
        for x in 1..4 {
            mask[5 + x] = 1;
            mask[15 + x] = 1;
        }
        for y in 1..4 {
            mask[y * 5 + 1] = 1;
            mask[y * 5 + 3] = 1;
        }

        fill_enclosed_holes(&mut mask, 5, 5);

        assert_eq!(mask[12], 1);
        assert_eq!(mask[0], 0);
    }

    #[test]
    fn leaves_open_stroke_interior_unfilled() {
        let mut mask = vec![0u8; 25];
        for x in 1..4 {
            mask[5 + x] = 1;
            mask[15 + x] = 1;
        }
        for y in 1..4 {
            mask[y * 5 + 1] = 1;
        }

        fill_enclosed_holes(&mut mask, 5, 5);

        assert_eq!(mask[12], 0);
    }
}
