# RFC-003: Rust Rewrite — API Server Architecture

## Status

Approved

## Context

Current Python + Rust hybrid has fundamental limitations for API server:
- GIL blocks parallel vision pipeline
- Python heap: 300-600MB per chapter, limits concurrency
- LazyPageProvider reconstructs pages from source — fragile, caused multiple bugs
- Duplicate normalize logic between Rust stitch and Python lazy provider
- Runtime dependencies: Python, venv, numpy, cv2, ultralytics, paddlepaddle

## Decision

Full Rust rewrite. Single binary, zero Python dependency.

## Architecture

```
crates/
├── core/              Vision pipeline (CPU/GPU bound)
│   ├── stitch.rs      Stitch source pages → mmap'd strip
│   ├── detect.rs      Detector trait + backend selection
│   ├── detect/
│   │   ├── onnx.rs    ONNX Runtime (Windows/Linux)
│   │   └── coreml.rs  CoreML via Swift bridge (macOS)
│   ├── cut.rs         Safe cut planner (bubble-aware)
│   ├── erase.rs       Inpaint text regions
│   ├── merge.rs       Group text lines → bubbles
│   ├── ocr.rs         OCR on bubble crops
│   └── strip.rs       StitchedStrip + zero-copy page access
│
├── render/            Text overlay (existing, ~1700 LOC)
│   ├── layout.rs      Word wrap, font measurement
│   ├── overlay.rs     Draw translated text
│   ├── border.rs      Bubble border detection
│   └── fit.rs         Font size fitting
│
├── translate/         LLM translation (async, network bound)
│   ├── provider.rs    Provider trait
│   ├── openai.rs      OpenAI API
│   ├── gemini.rs      Gemini API
│   ├── prompt.rs      Prompt templates
│   └── session.rs     Chapter translation session
│
├── storage/           Persistence
│   └── sqlite.rs      rusqlite — projects, translations, glossary
│
├── api/               HTTP server
│   ├── main.rs        axum server, routes, middleware
│   └── handlers.rs    Request handlers
│
├── cli/               CLI interface
│   └── main.rs        clap — detect, translate, serve commands
│
├── bridge/            Platform-specific (macOS only)
│   ├── Package.swift
│   └── Sources/
│       └── CoreMLBridge/
│           ├── Bridge.swift    CoreML inference
│           └── include/
│               └── bridge.h    C API header
│
└── Cargo.toml         Workspace
```

## Stack

| Concern | Crate | Why |
|---|---|---|
| Error handling | `anyhow` + `thiserror` | `?` propagation, no boilerplate |
| Serialization | `serde` + `serde_json` + `toml` | Derive macros, zero parse code |
| Logging | `tracing` + `tracing-subscriber` | Structured, async-aware, span timing |
| HTTP server | `axum` + `tower-http` | Extractor pattern, middleware stack |
| HTTP client | `reqwest` | Async, streaming, retry |
| Async runtime | `tokio` | Implicit via axum/reqwest |
| Parallel CPU | `rayon` | par_iter for vision pipeline |
| Image | `image` + `imageproc` + `fast_image_resize` | Decode/encode, draw, SIMD resize |
| ML inference | `ort` (ONNX Runtime) | Cross-platform, CUDA/DirectML/CPU |
| Database | `rusqlite` | SQLite, bundled, no external dep |
| CLI | `clap` | Derive macros |

## Platform Backends

| Platform | Inference | GPU |
|---|---|---|
| macOS | CoreML (Swift bridge) | Neural Engine + Metal |
| Windows | ONNX Runtime | DirectML / CUDA |
| Linux | ONNX Runtime | CUDA / CPU |

Backend selected at compile time via `#[cfg(target_os)]`.

## CoreML Bridge

Swift package compiled to static lib. Exposes C API only:
- `cs_model_load(path) → handle`
- `cs_model_free(handle)`
- `cs_detect(handle, image, w, h) → detections`
- `cs_detections_free(ptr)`

Rust calls via `extern "C"`. Ownership: Swift retains model, Rust
calls free via Drop. No shared mutable state.

## Memory Model

### Scan time
- Source pages loaded sequentially, fed to stitch
- Strip: single mmap'd buffer, RSS = touched pages only
- Scan reads strip sequentially → OS prefetch effective
- Strip freed after cut plan computed

### Translation time
- Zero image data in RAM
- Only text + metadata (~50KB per chapter)
- LookAt agent: load single page on-demand from source

### Render time
- One page at a time from strip (zero-copy pointer)
- Working buffer: ~2MB per page, reused
- Parallel via rayon if multiple pages

### Per-request RSS: ~10-20MB (vs 300-600MB Python)

## Pipeline Flow

```
Request
  │
  ├─ Vision (rayon thread pool, sync)
  │   load pages → stitch → mmap strip
  │   → detect (CoreML/ONNX) → merge → OCR
  │   → cut plan → bubble metadata
  │   → free strip
  │
  ├─ Translation (tokio async)
  │   → build prompts from bubble text
  │   → call LLM API (OpenAI/Gemini)
  │   → parse responses
  │
  ├─ Render (rayon, sync)
  │   → load strip again (mmap, cheap)
  │   → per-page: erase + overlay translated text
  │   → encode output (PNG/WebP)
  │   → stream response
  │
  └─ Response
```

## Migration Path

### Phase 1: Core vision (port from Python)
- `stitch.rs` — exists, extend with mmap
- `detect.rs` — new, ONNX Runtime via `ort`
- `merge.rs` — port text_grouping.py (~500 LOC)
- `cut.rs` — port engine.py cut planner (~100 LOC)
- `strip.rs` — mmap'd strip, zero-copy page access

### Phase 2: CoreML bridge (macOS)
- Swift bridge package
- `detect_coreml.rs` FFI wrapper

### Phase 3: Translation (port from Python)
- `openai.rs`, `gemini.rs` — reqwest + serde
- `prompt.rs` — string templates
- `session.rs` — chapter translation orchestration

### Phase 4: Storage + API
- `sqlite.rs` — port from Python sqlite store
- `api/` — axum server, handlers

### Phase 5: CLI
- `cli/` — clap, replaces typer CLI

Each phase is independently testable. Phase 1 can be validated
against Python pipeline output for correctness.

## Risks

- Text grouping heuristics (merge.rs) are the most complex port
  (~500 LOC Python with many edge cases). Needs thorough test
  fixtures from current pipeline output.
- Font rendering differences between platforms. Current Rust render
  uses ab_glyph — keep as-is.
- CoreML model format may differ from ONNX. Need export both
  formats from same source model.

## Verification

Each phase:
- Unit tests with fixture images from current pipeline
- Compare output against Python pipeline (snapshot tests via `insta`)
- Benchmark: throughput, RSS, latency per chapter

## Deliverable

Single binary: `comicscan`
- `comicscan detect <path>` — vision only
- `comicscan translate <path>` — full pipeline
- `comicscan serve` — local web UI + API server
- ~20MB binary + model files (~50MB)

## Distribution Model

Self-hosted desktop app. User downloads binary, runs `comicscan serve`,
opens browser at localhost:8080.

### Hybrid monetization

**Free tier (BYOK):** User provides own LLM API key (OpenAI/Gemini).
App calls LLM directly. Zero cost for us. Unlimited usage.

**Paid tier (Cloud):** User subscribes ($5/mo). App routes LLM calls
through `api.comicscan.com` proxy. No API key needed. We markup LLM
cost (~$0.02/chapter → $0.05).

### Proxy server

Lightweight — only forwards LLM requests, no vision compute:

```
App (local) → api.comicscan.com → OpenAI/Gemini
               ├─ Verify subscription (license key)
               ├─ Forward request
               └─ Track usage per user
```

Single VPS ($5-20/mo) handles thousands of users.

### API changes for hybrid

```
PUT /api/config
  { "mode": "cloud", "license_key": "cs_..." }
  or
  { "mode": "byok", "provider": "openai", "api_key": "sk-..." }
```

App switches LLM endpoint based on mode. Vision always runs local
(Free/Pro tiers).

## Pricing Tiers

| Tier | Vision | Translation | User needs | Price |
|---|---|---|---|---|
| Free | Local | BYOK (own API key) | Download binary + LLM key | $0 |
| Pro | Local | Cloud proxy | Download binary | $5/mo |
| Cloud | Server (GPU) | Server | Browser only | $15/mo |

### Cloud tier

Full SaaS — user opens `app.comicscan.com`, no install.

Same codebase, different deployment:

```bash
comicscan serve                # Free/Pro: local, localhost:8080
comicscan serve --cloud        # Cloud: production, auth required
```

Cloud-specific modules behind feature flag:

```rust
#[cfg(feature = "cloud")]
mod auth;          // JWT/session verification
#[cfg(feature = "cloud")]
mod s3_storage;    // source images + rendered output on S3
```

Vision pipeline, translation, render — identical across all tiers.

### Cloud infrastructure

```
Browser → app.comicscan.com
            ├─ API server (axum)
            ├─ Vision workers (GPU instances, T4/A10)
            ├─ LLM proxy
            └─ Storage (S3)
```

Minimal viable:
- 1 GPU instance (T4): $150-300/mo — ~50 concurrent users
- 1 VPS (API + proxy): $20/mo
- S3: ~$5/mo
- Total: ~$200-350/mo
- Break even: 25 Cloud users × $15 = $375/mo

### Cost per chapter (Cloud tier)

- GPU scan+render: ~$0.002 (5-10s GPU time)
- LLM translation: ~$0.02
- Total: ~$0.025/chapter
- $15/mo budget: ~600 chapters/user/mo
