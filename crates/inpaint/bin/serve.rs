// SPDX-License-Identifier: GPL-3.0-or-later
//! HTTP server for the inpaint container.
//!
//! Endpoints:
//!   GET  /health                                       → liveness
//!   POST /inpaint-page?job_id=N&page_index=I           → run a single page
//!   POST /probe?job_id=N&page_index=I&n=K&concurrency=C → benchmark — same
//!        per-page pipeline, but returns granular per-stage timings so we
//!        can decide whether to consolidate fan-out at the workflow vs.
//!        container level. K iterations × C concurrent. Probe writes to
//!        `inpaint-probe/{job}/...` so production artifacts aren't touched.
//!
//! Per-page work (decode JPEG, close mask, flood-fill, tile, inpaint each,
//! compose, encode PNG, upload) lives in `crate::page`. R2 access via the
//! lightweight `crate::s3` client — no FUSE.

use std::{path::PathBuf, sync::Arc, time::{Instant, SystemTime}};

use anyhow::Result;
use axum::{
    Router,
    extract::{Query, State},
    http::StatusCode,
    response::{IntoResponse, Response, Json},
    routing::{get, post},
};
use serde::{Deserialize, Serialize};
use tokio::{net::TcpListener, sync::Semaphore};
use tracing::info;

use typoon_inpaint::{
    Inpainter,
    page::{inpaint_page, inpaint_chapter, inpaint_page_traced, PageTimings},
    s3::S3,
};

#[derive(Clone)]
struct AppState {
    inpainter: Arc<Inpainter>,
    s3:        Arc<S3>,
    /// Wall-clock time since the model finished loading. Useful for the
    /// probe payload: a fresh container reports a small uptime, a warm one
    /// reports minutes.
    started_at: Arc<SystemTime>,
}

#[derive(Deserialize)]
struct PageQuery {
    job_id:     u64,
    page_index: u32,
}

async fn health(State(state): State<AppState>) -> impl IntoResponse {
    let uptime_s = SystemTime::now()
        .duration_since(*state.started_at)
        .map(|d| d.as_secs())
        .unwrap_or(0);
    Json(serde_json::json!({
        "ok": true,
        "service": "inpaint",
        "uptime_s": uptime_s,
    }))
}

async fn inpaint_page_handler(
    Query(q):    Query<PageQuery>,
    State(state): State<AppState>,
) -> Response {
    // inpaint_page itself uses spawn_blocking internally for the CPU-bound
    // section. Stay on the caller's runtime here so reqwest's hyper client
    // (bound to that runtime) can drive R2 GET/PUT calls.
    match inpaint_page(&state.s3, state.inpainter.clone(), q.job_id, q.page_index).await {
        Ok(r)  => Json(r).into_response(),
        Err(e) => (StatusCode::INTERNAL_SERVER_ERROR, format!("{e:#}")).into_response(),
    }
}

// ── /inpaint-chapter ──────────────────────────────────────────────────────

#[derive(Deserialize)]
struct ChapterBody {
    job_id:       u64,
    page_indices: Vec<u32>,
    /// In-instance concurrency cap. Defaults to 4 — saturates a standard-4
    /// (4 vCPU) instance which is where probe data placed the sweet spot.
    /// Clamped to 1..16 by `inpaint_chapter`.
    concurrency:  Option<usize>,
}

async fn inpaint_chapter_handler(
    State(state): State<AppState>,
    Json(body):   Json<ChapterBody>,
) -> Response {
    let conc = body.concurrency.unwrap_or(4);
    match inpaint_chapter(
        &state.s3,
        state.inpainter.clone(),
        body.job_id,
        body.page_indices,
        conc,
    ).await {
        Ok(r)  => Json(r).into_response(),
        Err(e) => (StatusCode::INTERNAL_SERVER_ERROR, format!("{e:#}")).into_response(),
    }
}

// ── /candle-self-test ────────────────────────────────────────────────────
//
// Feeds the Candle inpainter a synthetic RGB tile + mask and dumps the
// result so we can verify the forward pass independently of the production
// pipeline (S3, mask close, tile build, compose). Three modes via `pattern`:
//
//   gray        — uniform mid-gray (128) + fully-masked. Healthy output is
//                 still gray-ish; black output ⇒ model regressed to zero.
//   gradient    — vertical gradient + fully-masked. Healthy output is a
//                 smoothly varying field; black ⇒ confirms the dead-output
//                 hypothesis.
//   half-mask   — gray with mask covering only the lower half. Healthy
//                 output keeps the top half unchanged (compose handles it
//                 in production, but inpainter respects mask).
//
// Writes synthetic-{pattern}-in.png and synthetic-{pattern}-out.png into
// INPAINT_DEBUG_DIR. Returns the in-memory tile statistics so the caller
// can spot black output without downloading the PNG.

#[derive(Deserialize, Default)]
struct CandleProbeQuery {
    /// One of: gray | gradient | half-mask. Default: gradient.
    pattern: Option<String>,
    /// Tile size; must be one of the AOT-GAN buckets. Default 384.
    size:    Option<u32>,
}

#[derive(Serialize)]
struct CandleProbeResponse {
    pattern:        String,
    size:           u32,
    /// Mean RGB of the inpainter output. Black ⇒ ~(0,0,0).
    out_mean_rgb:   [u8; 3],
    /// Min/max per channel — tighter bounds reveal saturated outputs.
    out_min_rgb:    [u8; 3],
    out_max_rgb:    [u8; 3],
    /// Wall ms for `Inpainter::inpaint` itself, no I/O.
    candle_ms:      u64,
}

async fn candle_self_test_handler(
    Query(q):    Query<CandleProbeQuery>,
    State(state): State<AppState>,
) -> Response {
    let pattern = q.pattern.unwrap_or_else(|| "gradient".to_string());
    let size    = q.size.unwrap_or(384).clamp(64, 512) as usize;
    let n       = size * size;

    let mut rgb  = vec![0u8; n * 3];
    let mut mask = vec![0u8; n];

    match pattern.as_str() {
        "gray" => {
            rgb.fill(128);
            mask.fill(255);
        }
        "half-mask" => {
            rgb.fill(128);
            // Cover the lower half only.
            for y in (size / 2)..size {
                let row = y * size;
                for x in 0..size { mask[row + x] = 255; }
            }
        }
        _ /* gradient */ => {
            for y in 0..size {
                let v = ((y * 255) / (size - 1)) as u8;
                let row = y * size * 3;
                for x in 0..size {
                    rgb[row + x * 3    ] = v;
                    rgb[row + x * 3 + 1] = v;
                    rgb[row + x * 3 + 2] = v;
                }
            }
            mask.fill(255);
        }
    }

    // Dump inputs first so caller can verify even on inpaint failure.
    if let Some(dir) = std::env::var_os("INPAINT_DEBUG_DIR") {
        let _ = std::fs::create_dir_all(&dir);
        let dir = std::path::PathBuf::from(dir);
        let rgb_buf: image::ImageBuffer<image::Rgb<u8>, &[u8]> =
            image::ImageBuffer::from_raw(size as u32, size as u32, rgb.as_slice()).unwrap();
        let _ = rgb_buf.save(dir.join(format!("synthetic-{pattern}-in.png")));
        let mask_buf: image::ImageBuffer<image::Luma<u8>, &[u8]> =
            image::ImageBuffer::from_raw(size as u32, size as u32, mask.as_slice()).unwrap();
        let _ = mask_buf.save(dir.join(format!("synthetic-{pattern}-mask.png")));
    }

    let inp = state.inpainter.clone();
    let rgb_in  = rgb.clone();
    let mask_in = mask.clone();
    let sz = size as u32;
    let result = tokio::task::spawn_blocking(move || {
        let t = std::time::Instant::now();
        let out = inp.inpaint(&rgb_in, &mask_in, sz, sz);
        (out, t.elapsed().as_millis() as u64)
    }).await;

    let (out_rgb, candle_ms) = match result {
        Ok((Ok(out), ms)) => (out, ms),
        Ok((Err(e), _))   => return (StatusCode::INTERNAL_SERVER_ERROR, format!("inpaint err: {e:#}")).into_response(),
        Err(e)            => return (StatusCode::INTERNAL_SERVER_ERROR, format!("join: {e}")).into_response(),
    };

    // Stats.
    let mut sum   = [0u64; 3];
    let mut min_v = [255u8; 3];
    let mut max_v = [0u8; 3];
    for i in 0..n {
        for c in 0..3 {
            let v = out_rgb[i * 3 + c];
            sum[c] += v as u64;
            if v < min_v[c] { min_v[c] = v; }
            if v > max_v[c] { max_v[c] = v; }
        }
    }
    let mean = [
        (sum[0] / n as u64) as u8,
        (sum[1] / n as u64) as u8,
        (sum[2] / n as u64) as u8,
    ];

    if let Some(dir) = std::env::var_os("INPAINT_DEBUG_DIR") {
        let dir = std::path::PathBuf::from(dir);
        let out_buf: image::ImageBuffer<image::Rgb<u8>, &[u8]> =
            image::ImageBuffer::from_raw(size as u32, size as u32, out_rgb.as_slice()).unwrap();
        let _ = out_buf.save(dir.join(format!("synthetic-{pattern}-out.png")));
    }

    Json(CandleProbeResponse {
        pattern, size: sz,
        out_mean_rgb: mean,
        out_min_rgb:  min_v,
        out_max_rgb:  max_v,
        candle_ms,
    }).into_response()
}

// ── /probe ────────────────────────────────────────────────────────────────

#[derive(Deserialize)]
struct ProbeQuery {
    job_id:      u64,
    page_index:  u32,
    /// Total iterations to run. K=1 → cold-ish reading. K=10 → warm steady state.
    n:           Option<u32>,
    /// Max in-flight iterations. C=1 → serial baseline. C>1 → contention.
    concurrency: Option<u32>,
}

#[derive(Serialize)]
struct ProbeRun {
    iteration:        u32,
    wall_ms:          u64,
    error:            Option<String>,
    #[serde(flatten, skip_serializing_if = "Option::is_none")]
    timings:          Option<PageTimings>,
}

#[derive(Serialize)]
struct ProbeResponse {
    job_id:      u64,
    page_index:  u32,
    n:           u32,
    concurrency: u32,
    uptime_s:    u64,
    wall_total_ms: u64,
    runs:        Vec<ProbeRun>,
}

async fn probe_handler(
    Query(q):    Query<ProbeQuery>,
    State(state): State<AppState>,
) -> Response {
    let n           = q.n.unwrap_or(5).clamp(1, 64);
    let concurrency = q.concurrency.unwrap_or(1).clamp(1, 32);
    let sem         = Arc::new(Semaphore::new(concurrency as usize));
    let total_start = Instant::now();

    let mut handles = Vec::with_capacity(n as usize);
    for i in 0..n {
        let permit_sem = sem.clone();
        let s3        = state.s3.clone();
        let inp       = state.inpainter.clone();
        let job_id    = q.job_id;
        let page_idx  = q.page_index;
        handles.push(tokio::spawn(async move {
            let _permit = permit_sem.acquire_owned().await.unwrap();
            let wall = Instant::now();
            let result = inpaint_page_traced(&s3, inp, job_id, page_idx, i).await;
            let wall_ms = wall.elapsed().as_millis() as u64;
            match result {
                Ok(t)  => ProbeRun { iteration: i, wall_ms, error: None,                       timings: Some(t) },
                Err(e) => ProbeRun { iteration: i, wall_ms, error: Some(format!("{e:#}")),     timings: None    },
            }
        }));
    }

    let mut runs = Vec::with_capacity(n as usize);
    for h in handles {
        match h.await {
            Ok(r)  => runs.push(r),
            Err(e) => runs.push(ProbeRun {
                iteration: 0, wall_ms: 0, error: Some(format!("join: {e}")), timings: None,
            }),
        }
    }
    runs.sort_by_key(|r| r.iteration);

    let uptime_s = SystemTime::now()
        .duration_since(*state.started_at)
        .map(|d| d.as_secs())
        .unwrap_or(0);

    Json(ProbeResponse {
        job_id:      q.job_id,
        page_index:  q.page_index,
        n, concurrency,
        uptime_s,
        wall_total_ms: total_start.elapsed().as_millis() as u64,
        runs,
    }).into_response()
}

#[tokio::main]
async fn main() -> Result<()> {
    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::from_default_env()
                .add_directive("typoon_inpaint=info".parse()?)
                .add_directive("serve=info".parse()?),
        )
        .init();

    let weights_path = PathBuf::from(
        std::env::var("MODEL_PATH").unwrap_or_else(|_| "/app/model.safetensors".to_string()),
    );
    let fp16 = std::env::var("FP16").map(|v| v == "1" || v.eq_ignore_ascii_case("true")).unwrap_or(false);
    let port: u16 = std::env::var("PORT").ok().and_then(|p| p.parse().ok()).unwrap_or(8000);

    info!(path = %weights_path.display(), fp16, port, "loading model...");
    let inpainter = Arc::new(Inpainter::load(&weights_path, fp16)?);
    info!("model ready");

    let s3 = Arc::new(S3::from_env()?);
    info!("S3 client ready");

    let state = AppState {
        inpainter,
        s3,
        started_at: Arc::new(SystemTime::now()),
    };
    let app = Router::new()
        .route("/health",            get(health))
        .route("/inpaint-page",      post(inpaint_page_handler))
        .route("/inpaint-chapter",   post(inpaint_chapter_handler))
        .route("/bench",             post(probe_handler))
        .route("/candle-self-test",  get(candle_self_test_handler))
        .with_state(state);

    let listener = TcpListener::bind(format!("0.0.0.0:{port}")).await?;
    info!(addr = %listener.local_addr()?, "listening");
    axum::serve(listener, app).await?;
    Ok(())
}
