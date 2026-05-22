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
const PAD_AROUND_BUBBLE: i32  = 16;
/// Pad as a fraction of the bubble's shorter edge. AOT-GAN regresses to
/// near-black when the mask covers >55% of the inference tile; adding
/// pad proportional to bubble size keeps mask density inside that
/// envelope even for full-bbox-text logos. Empirically 0.30 holds a
/// fully-masked square bubble at ~40% tile density (Python reference:
/// packages/typoon-vision/typoon/vision/erasers/inpaint.py uses
/// context_px=16 + a bucket-relative resize; we get the same effect by
/// growing the crop instead, since the Rust backend has no fixed bucket).
const PAD_FRAC: f32           = 0.30;
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

    let closed  = close_mask_per_block(&orig_mask, w, h, &scan.groups);
    dbg.dump_mask("01-closed-mask", &closed, w, h);
    let bubbles = find_bubbles(&closed, w, h);

    let mut composite: Vec<u8> = img.into_raw();
    dbg.dump_rgb("02-input-rgb", &composite, w, h);
    let mut tiles_shape = Vec::with_capacity(bubbles.len());

    if bubbles.is_empty() {
        let png = encode_png(&composite, w, h)?;
        return Ok((png, 0, tiles_shape));
    }

    for (bi, bb) in bubbles.iter().enumerate() {
        let (tile_rgb, tile_mask, tile_w, tile_h, pad_box) =
            build_tile(&composite, &orig_mask, w, h, bb);
        tiles_shape.push(format!("{tile_w}x{tile_h}"));

        dbg.dump_rgb(&format!("tile-{bi:02}-rgb"),   &tile_rgb, tile_w, tile_h);
        dbg.dump_mask(&format!("tile-{bi:02}-mask"), &tile_mask, tile_w, tile_h);

        // Mask density check. With adaptive PAD_FRAC, normal bubbles land
        // well below this threshold. The boundary-median fallback is kept
        // for pathological cases (mask near 100% even after pad expand —
        // can happen when the bubble bbox already covers the entire page
        // edge, leaving no room to grow). For flat-colour logos this
        // fallback is visually correct; for everything else it at least
        // avoids the model's near-black collapse.
        let on = tile_mask.iter().filter(|&&v| v >= 127).count();
        let on_frac = on as f32 / (tile_w * tile_h) as f32;
        const DENSE_THRESHOLD: f32 = 0.70;

        let out_rgb = if on_frac >= DENSE_THRESHOLD {
            boundary_median_fill(&tile_rgb, &tile_mask, tile_w, tile_h)
        } else {
            inpainter
                .inpaint(&tile_rgb, &tile_mask, tile_w as u32, tile_h as u32)
                .with_context(|| format!("inpaint tile {tile_w}x{tile_h} failed"))?
        };
        dbg.dump_rgb(&format!("tile-{bi:02}-out"), &out_rgb, tile_w, tile_h);

        compose_tile(&mut composite, w, &orig_mask, &pad_box, tile_w, tile_h, &out_rgb);
    }

    dbg.dump_rgb("99-final-rgb", &composite, w, h);

    let png = encode_png(&composite, w, h)?;
    Ok((png, bubbles.len(), tiles_shape))
}

// ── Boundary-median fill (fallback for dense masks) ───────────────────────
//
// When the mask covers >55% of a tile, AOT-GAN can't reconstruct a
// plausible fill — it lacks unmasked context to anchor on. For text-as-
// logo bubbles where the surrounding background is flat-coloured, taking
// the median of the boundary band is both faster and visually correct.
//
// Algorithm:
//   1. Find pixels just outside the masked region (mask < 127, but with
//      ≥1 masked 4-neighbour). Cap to a band of `BAND_PX` pixels wide.
//   2. Take the median R, G, B independently across that band.
//   3. Fill every masked pixel with that median; leave unmasked pixels.

fn boundary_median_fill(rgb: &[u8], mask: &[u8], w: usize, h: usize) -> Vec<u8> {
    const BAND_PX: usize = 8;

    // Collect boundary samples — pixels NOT in the mask but adjacent to it.
    let mut rs: Vec<u8> = Vec::with_capacity(w + h);
    let mut gs: Vec<u8> = Vec::with_capacity(w + h);
    let mut bs: Vec<u8> = Vec::with_capacity(w + h);
    for y in 0..h {
        for x in 0..w {
            let i = y * w + x;
            if mask[i] >= 127 { continue; }
            // 4-neighbour check
            let adjacent_to_mask =
                (x > 0     && mask[i - 1]     >= 127) ||
                (x + 1 < w && mask[i + 1]     >= 127) ||
                (y > 0     && mask[i - w]     >= 127) ||
                (y + 1 < h && mask[i + w]     >= 127);
            if !adjacent_to_mask { continue; }
            // Walk outward up to BAND_PX in each direction, capture pixels
            // we know are not masked.
            for dy in 0..=BAND_PX as i32 {
                for &sign in &[-1i32, 1] {
                    let ny = y as i32 + dy * sign;
                    if ny < 0 || ny >= h as i32 { continue; }
                    let j = (ny as usize) * w + x;
                    if mask[j] < 127 {
                        let p = j * 3;
                        rs.push(rgb[p]); gs.push(rgb[p + 1]); bs.push(rgb[p + 2]);
                    }
                }
            }
        }
    }

    // Median; on empty boundary (whole-tile mask), fall back to global mean
    // of any unmasked pixels, or pure white.
    fn median(mut v: Vec<u8>) -> u8 {
        if v.is_empty() { return 255; }
        v.sort_unstable();
        v[v.len() / 2]
    }
    let (mr, mg, mb) = if !rs.is_empty() {
        (median(rs), median(gs), median(bs))
    } else {
        // No 4-adjacent boundary: scan every unmasked pixel.
        let (mut r, mut g, mut b) = (Vec::new(), Vec::new(), Vec::new());
        for i in 0..w * h {
            if mask[i] < 127 {
                let p = i * 3;
                r.push(rgb[p]); g.push(rgb[p + 1]); b.push(rgb[p + 2]);
            }
        }
        if r.is_empty() { (255, 255, 255) } else { (median(r), median(g), median(b)) }
    };

    let mut out = rgb.to_vec();
    for i in 0..w * h {
        if mask[i] >= 127 {
            let p = i * 3;
            out[p]     = mr;
            out[p + 1] = mg;
            out[p + 2] = mb;
        }
    }
    out
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
    let closed = close_mask_per_block(&orig_mask, w, h, &scan.groups);
    t.mask_close_ms = t1.elapsed().as_millis() as u64;

    let t2 = Instant::now();
    let bubbles = find_bubbles(&closed, w, h);
    t.flood_fill_ms  = t2.elapsed().as_millis() as u64;
    t.bubbles_count  = bubbles.len();

    let mut composite: Vec<u8> = img.into_raw();

    if !bubbles.is_empty() {
        let t3 = Instant::now();
        let tiles: Vec<_> = bubbles.iter()
            .map(|bb| build_tile(&composite, &orig_mask, w, h, bb))
            .collect();
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
            compose_tile(&mut composite, w, &orig_mask, pad_box, *tile_w, *tile_h, out_rgb);
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

fn close_mask_per_block(mask: &[u8], w: usize, h: usize, groups: &[ScanGroupSlim]) -> Vec<u8> {
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

        // Block-local patch
        let mut patch = vec![0u8; pw * ph];
        for y in 0..ph {
            for x in 0..pw {
                patch[y * pw + x] = bin[(py0 + y) * w + (px0 + x)];
            }
        }
        let dilated = dilate_rect(&patch, pw, ph, r);
        let closed  = erode_rect(&dilated, pw, ph, r);

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
                if cx < x0 { x0 = cx; } if cx > x1 { x1 = cx; }
                if cy < y0 { y0 = cy; } if cy > y1 { y1 = cy; }
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

fn snap_up(v: i32, m: usize) -> usize {
    let m = m as i32;
    (((v + m - 1) / m) * m).max(0) as usize
}

/// Build a tile crop from the page + mask.
///
/// Pad is **adaptive**: a fraction of the bbox short edge, floor at
/// `PAD_AROUND_BUBBLE`. The mask itself is just cropped (no expansion),
/// so a large bubble whose mask covers ~100% of its bbox gets a tile
/// large enough that unmasked context outside the bbox brings density
/// down to where AOT-GAN can reconstruct.
///
/// Empirical: AOT-GAN collapses to ~black around mask density ≥ 55%.
/// PAD_FRAC=0.3 keeps a fully-masked square bubble at ~40% tile density.
///
/// Tile size is snapped up to SNAP_MOD so the Candle backend's
/// `pad_mod = 8` internal pad is a no-op.
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
    let pad   = (short * PAD_FRAC).round() as i32;
    let pad   = pad.max(PAD_AROUND_BUBBLE);

    let x0 = (bb.x0 - pad).max(0);
    let y0 = (bb.y0 - pad).max(0);
    let x1 = (bb.x1 + pad).min(src_w as i32 - 1);
    let y1 = (bb.y1 + pad).min(src_h as i32 - 1);
    let w0 = (x1 - x0 + 1) as usize;
    let h0 = (y1 - y0 + 1) as usize;
    let tile_w = snap_up(w0 as i32, SNAP_MOD).max(SNAP_MOD);
    let tile_h = snap_up(h0 as i32, SNAP_MOD).max(SNAP_MOD);

    let mut rgb  = vec![0u8; tile_w * tile_h * 3];
    let mut msk  = vec![0u8; tile_w * tile_h];

    for ty in 0..tile_h {
        let sy_i = if ty < h0 { ty as i32 } else { 2 * (h0 as i32 - 1) - ty as i32 };
        let sy = sy_i.clamp(0, h0 as i32 - 1) as usize;
        let src_y = y0 as usize + sy;
        for tx in 0..tile_w {
            let sx_i = if tx < w0 { tx as i32 } else { 2 * (w0 as i32 - 1) - tx as i32 };
            let sx = sx_i.clamp(0, w0 as i32 - 1) as usize;
            let src_x = x0 as usize + sx;
            let src_rgb_i = (src_y * src_w + src_x) * 3;
            let src_msk_i =  src_y * src_w + src_x;
            let di = ty * tile_w + tx;
            rgb[di * 3    ] = composite[src_rgb_i    ];
            rgb[di * 3 + 1] = composite[src_rgb_i + 1];
            rgb[di * 3 + 2] = composite[src_rgb_i + 2];
            msk[di]         = mask[src_msk_i];
        }
    }
    (rgb, msk, tile_w, tile_h, BBox { x0, y0, x1, y1 })
}

/// Paste tile_rgb back into composite, only at pixels where the ORIGINAL
/// mask is set. Composite is RGB stride 3.
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
    for ty in 0..h0.min(tile_h) {
        let dst_y = pad_box.y0 as usize + ty;
        for tx in 0..w0.min(tile_w) {
            let dst_x = pad_box.x0 as usize + tx;
            let mi    = dst_y * src_w + dst_x;
            if orig_mask[mi] < 127 { continue; }
            let di = (dst_y * src_w + dst_x) * 3;
            let ti = (ty * tile_w + tx) * 3;
            composite[di    ] = tile_rgb[ti    ];
            composite[di + 1] = tile_rgb[ti + 1];
            composite[di + 2] = tile_rgb[ti + 2];
        }
    }
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
