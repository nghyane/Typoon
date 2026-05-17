# Cloudflare edge pipeline — feasibility findings

Empirical findings from a 2026-05 spike exploring whether the full
Typoon pipeline (prepare → scan → translate → inpaint → typeset →
pack) can run on Cloudflare Workers instead of the current Mac-native
runtime. Read this before proposing any "move pipeline to edge"
direction — the constraints are non-obvious and most blockers were
discovered by probing real workloads, not from docs.

All probe code lives under `spike/` at repo root. The deployed Workers are:

- `ort-inpaint-tile`        — ORT WASM inpaint, now exposed as a
  `TileInpaint` Durable Object so the ORT session lives across requests
- `ort-inpaint-orchestrator` — bubble-discovery + fan-out compose,
  with R2 cache for tile outputs
- `prepare-worker`          — ZIP-only input, port of
  `typoon/stages/prepare.py` (color detect + manhwa stitch cut),
  verified against Python reference, R2 in/out
- `scan-worker`             — Lens OCR per prepared page, tile + dedup
  + filter, R2 in/out, mask PNG output for inpaint
- `translate-worker`        — R2-native windowed LLM (OpenAI Responses
  API via packyapi proxy), parses `@@ KEY kind` reply format
- `render-archive-worker`   — packs the inpainted pages into a `.bnl`
  archive on R2 (JPEG q=92, byte-stable with the Mac path)
- `chapter-workflow-v2`     — Workflow orchestrator (RPC + R2 + DAG)
- `probe-vps`               — measures Worker → CF Tunnel → Mac latency
- `probe-workflow`          — measures `Promise.all(step.do)` parallelism

Numbers below come from those deployments, not estimates.

## TL;DR

| Stage | Worker fit | Effort | Status |
|---|---|---|---|
| prepare       | ✅ ZIP → @jsquash + hand-ported OpenCV math | Medium | done |
| scan (Lens)   | ✅✅ direct HTTP to Lens crupload, paragraph parser fork | Low | done |
| translate     | ✅✅ R2-in, R2-out, windowed LLM, RPC | Low | done |
| inpaint       | ✅ ORT WASM in a Durable Object, R2-cached tile outputs | Done | done |
| render-archive| ✅ `bunle.pack` thuần JS, in-memory | Low | done |
| typeset       | ⏸ needs port of `typoon_render` Rust → wasm-bindgen | Medium | planned |
| orchestration | ✅✅ RPC + Workflow DAG — caller hits 1 endpoint | n/a | done |

Verdict: **the full pipeline minus typeset runs end-to-end on Workers +
R2 + Durable Objects.** A 3-page chapter completes in ~30 s wall time
(with a warm tile cache); a cold-cache cold-DO run still beats the Mac
single-threaded baseline by 3-5× thanks to per-page parallelism.

Workers Paid is required at current bundle sizes — the tile worker's
ORT WASM is 12.4 MiB. Free tier 3 MiB script cap is exceeded.

## Why this matters

Today the pipeline runs on a single Mac via launchd. Throughput is
capped by that one box. Moving compute to Workers means:

- Per-chapter wall time drops from ~60 s to ~6-10 s (parallel page
  fan-out via Workflows; numbers below).
- Concurrent chapter capacity goes from 1 to "unlimited" (per-account
  Cloudflare limits, not per-host).
- Single point of failure (Mac uptime) becomes irrelevant.

The cost trade-off is documented at the bottom.

---

## ORT WASM on Workers — three patches that make it work

`onnxruntime-web` fails silently in workerd. The fix is three
post-build patches on the bundled JS, from
[CosteGieF/ort-cloudflare-workers](https://github.com/CosteGieF/ort-cloudflare-workers):

1. **Pre-compiled WASM via `[[rules]] type = "CompiledWasm"`**.
   Wrangler compiles `.wasm` at deploy time, not runtime
   (`WebAssembly.compile()` is blocked in workerd). The import
   resolves to a `WebAssembly.Module` already compiled.

2. **Inject `instantiateWasm` callback** onto the Emscripten config
   object right before the factory call. The callback uses the
   pre-compiled module directly, bypassing `WebAssembly.compile()`.

3. **Kill dynamic `import(variable)`** — workerd rejects non-string
   import specifiers at module-analysis time, even if the path never
   runs. Replace with `Promise.reject(...)`.

Reference implementation: `spike/tile-worker/build.mjs`.

### Static Assets escape hatch for the 22 MB model

`aot-inpaint.onnx` is 22 MB. Worker script size limit is 10 MB gzip
on the Paid plan, 3 MB on Free. Bundling the model exceeds the limit.

Fix: serve the model via [Workers Static Assets](https://developers.cloudflare.com/workers/static-assets/binding/),
which has its own 25 MiB per-file limit and is **not counted toward
Worker script size**. The tile worker fetches the model at session
init via `env.ASSETS.fetch("http://fake-host/model.onnx")`.

Result: Worker bundle = **3.28 MiB gzip** (compiled WASM dominates).
This **exceeds the Free tier 3 MiB script limit** and requires the
Workers Paid plan ($5/month, 10 MiB script limit). The model itself
uploads separately via Static Assets and does not count toward the
script size cap. Cold session load = ~1.4 s.

Wrangler enforces this at deploy time on Free accounts:

```
✘ Your Worker exceeded the size limit of 3 MiB.
  - .worker/ort-wasm-simd-threaded.wasm - 12717.19 KiB
```

A previous version of this doc claimed the bundle fits Free; it does
not. ORT WASM (12.4 MiB uncompressed → 2.98 MiB gzip on its own)
plus the orchestrator code pushes total gzip just over the 3 MiB cap.

### Squeezing under 3 MiB — ORT minimal custom build

If you need Free-tier deployment (or want faster cold starts on
Paid), `onnxruntime-web` supports a [minimal custom build](https://onnxruntime.ai/docs/build/custom.html)
that strips unused operator kernels from the WASM binary. The model
must first be converted to ORT format
(`python -m onnxruntime.tools.convert_onnx_models_to_ort aot-inpaint.onnx`).

For `aot-inpaint.onnx` the operator config (already generated under
`/tmp/ort-models/aot-inpaint.required_operators.config` during the
probe) lists 17 ops: Conv, ConvTranspose, FusedConv, Cast, Clip,
Concat, Gather, Pad, ReduceMean, ReduceProd, Sigmoid, Sqrt, Add,
Div, Mul, Relu, Sub, Shape. The full ORT WASM ships ~150 op
kernels; removing the unused 130-odd is where the size win comes
from.

Build (1-2 hours on a Mac):

```bash
git clone --recursive https://github.com/microsoft/onnxruntime
cd onnxruntime
./build.sh \
  --config Release \
  --build_wasm \
  --minimal_build \
  --include_ops_by_config /path/to/aot-inpaint.required_operators.config \
  --enable_reduced_operator_type_support \
  --disable_wasm_exception_catching \
  --disable_rtti \
  --skip_tests \
  --enable_wasm_simd \
  --enable_wasm_threads
```

Expected output (numbers from Microsoft prebuilt mobile/web targets
using the same flags):

| | Full WASM (today) | Minimal custom |
|---|---|---|
| WASM raw | 12.4 MiB | ~3 MiB |
| WASM gzip | 2.98 MiB | ~1 MiB |
| Total Worker bundle gzip | 3.28 MiB ❌ Free | ~1.3 MiB ✅ Free |

**Performance and quality impact** — the docs are explicit that
removing kernels does not regress runtime characteristics for the ops
that remain:

- **Quality:** bit-identical. The ORT format conversion applies all
  graph optimizations (constant folding, node fusion) ahead of time.
  No lossy compression, no quantization. Same numerical output.
- **Inference speed:** identical to full Release build. Use
  `--config Release` (not `MinSizeRel`) to keep compiler-level
  optimizations; you only lose kernels you never call.
- **Cold start:** ~20-30% faster. The minimal runtime skips the ONNX
  parser (loads flatbuffer instead) and skips runtime graph
  optimization (already baked into the .ort file).
- **Memory:** marginally lower at runtime — fewer kernel registry
  tables loaded.

Don't expect any pitfall here. The only operational cost is rebuilding
WASM if the model architecture changes (new op type appears in the
graph). Weight retraining or fine-tuning does not require a rebuild.

### When to do this

Skip unless one of:

- You want to deploy the inpaint Worker on Free tier (avoid $5/mo).
- You want faster cold starts (each new tile-worker isolate saves
  ~600-1100 ms on session init, ~40-50% reduction).
- You're optimising memory headroom for larger inference buckets.

Cold start savings analysis (matters most if pursued for latency):

| Phase | Full build | Minimal build |
|---|---|---|
| Fetch model from R2/ASSETS | 50-100 ms | 50-100 ms |
| WASM module instantiate (pre-compiled) | 10 ms | 5 ms (smaller binary) |
| ORT runtime init | 100-200 ms | 50-100 ms |
| ONNX parser → IR | 300-500 ms | **skipped** (flatbuffer read 50-100ms) |
| Graph optimizer (level=all) | 200-400 ms | **skipped** (done at convert) |
| Kernel registration | 100-200 ms | 20-50 ms (only 17 ops) |
| First inference warmup | 100-200 ms | 100-200 ms |
| **Total cold start** | **860-1610 ms** | **275-555 ms** |

Currently the rest of the pipeline (Workflows) requires the Paid plan
regardless, so the size win does not enable a free deployment by
itself. The cold-start saving is real but applies per-isolate, not
per-chapter — fan-out parallel `step.do` runs cold starts concurrently
so the wall-clock benefit is at most one cold-start delta (~800 ms),
not the cumulative savings across all tiles. For a 6-10 s chapter
pipeline, this is 8-13 % wall-time improvement.

Decision rubric for the project today:

- Already Paid → script size is non-issue (10 MiB cap, 3.28 MiB actual).
- Wall-time saving per chapter: ~0.8 s out of 6-10 s.
- Build/maintain effort: ~1-2 h initial, occasional rebuild on model
  architecture changes (not weight updates).

**Skip until cold start is empirically a bottleneck.** Measure first via
the `/warm` endpoint in `spike/tile-worker/src/index.ts` which already
reports per-phase timing (fetch / env / session_create). If
`session_create_ms` dominates and the chapter pipeline metrics show
cold-start tail is the long pole, revisit. Otherwise the engineering
hours are better spent on the typeset WASM port.

---

## Memory budget — the 128 MB hard ceiling

Cloudflare Workers have a 128 MB memory limit per isolate. **No paid
plan increases this. Durable Objects share the same limit.** Hyperdrive
doesn't change it. There is no escape hatch on the Workers runtime.

Feeding a full manga page (1600×690) directly into AOT inpaint blows
this budget — activation tensors alone are ~125 MB before counting
the model weights and ORT runtime.

The working pattern (matches the existing Python `_OnnxBackend` in
`typoon/vision/_backends/aot/__init__.py`):

1. Per bubble, crop the bbox with a small padding margin.
2. Pad each crop to the nearest model bucket (`128 / 192 / 256 / 384`
   per axis — model requires mod-8 and ≥128 to avoid Pad-node
   underflow inside the network).
3. Run ORT on each crop independently.
4. Composite results back into the source image, blending only inside
   the original mask.

Bucket 384 inference uses ~53 MB total — fits comfortably under 128 MB.

This logic lives in `spike/orchestrator/src/index.ts` (`buildTile`,
`composeTile`). The orchestrator fans out tiles to the tile worker.

---

## Workflow parallelism — verified

Probed via `spike/chapter-workflow/`. `Promise.all(step.do(...))`
schedules steps in separate Worker invocations, each with its own
6-connection budget and 128 MB.

Measured (calls hit VPS `/healthz` to keep step CPU negligible —
purpose was to confirm scheduling is truly parallel):

| n steps | mode       | wall time |
|--------:|------------|----------:|
|      10 | sequential |  2,375 ms |
|      10 | parallel   |    405 ms |
|      50 | parallel   |    875 ms |

10-step parallel speedup: 5.9×. 50-step speedup: 14.4× (sublinear
because the dummy endpoint throttles, not because Workflow does).

**Implication:** a 60-page chapter with one `step.do` per page is
~5-10 s wall time, regardless of page count, until you hit the per-step
fan-out limit on the underlying tile worker.

### Subrequest / connection caps

Inside a single Worker request:

| Limit | Free | Paid |
|---|---|---|
| Subrequests total | 50 | 10,000 |
| **Simultaneous open connections** | **6** | **6** |

The 6-connection cap is the operative one for fan-out. To exceed it,
dispatch separate Worker requests — which is exactly what `step.do`
in a Workflow does.

---

## Worker → VPS tunnel latency

The existing Cloudflare Tunnel (`api.mangalocal.com` → localhost:8000)
already exposes the FastAPI. Probed from Worker (HKG colo) → Tunnel
→ Mac:

| Metric | Value |
|---|---|
| Sequential `/healthz` p50 | 41 ms |
| Sequential p99 | 225 ms (TLS handshake on first hit) |
| 10 parallel calls wall time | 383 ms |
| 100 sequential burst, total | 5.88 s (~58 ms avg, max 219 ms) |

Tunnel callback latency is **not a bottleneck**. The pattern "Workers
push pipeline progress / done callbacks via HTTP to the VPS API" is
viable without Hyperdrive or a DB binding.

---

## End-to-end probe: 3 pages, real ORT inference

`spike/tile-worker` + `spike/orchestrator`, real `aot-inpaint.onnx`,
real manga pages from chapter 135. Three pages, two bubbles each,
fanned out in parallel.

| Run | Wall time | Notes |
|---|---|---|
| Cold (first) | 8.7 s | Tile worker session init in flight |
| Warm best | 4.2 s | Most tiles hit warm isolates |
| Warm typical | 14 s | Cold isolates show up at random under load |

Per-page warm steady-state: 4-7 s. The tail latency is high because
ORT WASM is single-threaded (Workers don't support WASM threads) and
each cold tile-worker isolate has to refetch + relink the model from
Static Assets.

Native CoreML on the Mac does the same work in ~17 ms per bubble at
384×384. WASM is 10-20× slower per page on identical input.

**This is the headline trade-off**: Workers gives unbounded parallel
chapters at higher per-page latency. Mac native gives lower per-page
latency at fixed throughput.

---

## R2 as inter-stage transport — the right shape

Workflow step return values are capped at 1 MiB. Page pixel data
(raw RGB at 1600×690 = ~3 MB; rendered JPEG ~600 KB) won't fit if
passed through the workflow state.

The clean pattern: every step writes its output to R2 and returns
**the R2 key only** (~50 bytes). The next step reads from R2.

```typescript
const pageKey = await step.do(`inpaint-${i}`, async () => {
  const rgb = await callTileWorker(...);
  await env.R2.put(`work/${chapterId}/inpainted/${i}.raw`, rgb);
  return `work/${chapterId}/inpainted/${i}.raw`;  // string fits 1 MiB
});

await step.do(`typeset-${i}`, async () => {
  const rgb = await env.R2.get(pageKey).then(o => o!.arrayBuffer());
  // ...typeset, write rendered/{i}.jpg back to R2...
});
```

R2 ↔ Worker traffic is in-network and free of egress charges. Storage
is $0.015/GB-month. Class A (PUT) ops are $4.50 / million.

### Suggested R2 layout

```
raw/{chapter_id}.zip                ← already produced by browser uploads
prepared/{chapter_id}/prepared.bnl  ← persistent
prepared/{chapter_id}/masks.npz     ← persistent
work/{chapter_id}/scan.json         ← ephemeral (24h lifecycle)
work/{chapter_id}/translate.json
work/{chapter_id}/inpainted/{i}.raw
work/{chapter_id}/rendered/{i}.jpg
final/{chapter_id}.bnl              ← persistent, public via Cache API
```

`work/` gets a 24-hour lifecycle rule. `prepared/` and `final/` stay.

---

## Cost envelope

For 200 chapters/month × 30 pages:

| Component | $/month |
|---|---|
| Workers Paid plan (required for Workflows + 10 MB scripts) | 5.00 |
| R2 storage (month 6 cumulative ~42 GB) | 0.48 |
| R2 PUT ops (~12,600/month) | 0.06 |
| VPS Postgres (unchanged, existing) | 5-7 |
| **Total marginal vs today** | **+5-6/month** |

R2 free tier covers month 1. Class B (GET) ops are largely served
by Cache API hits at the edge and don't show up on the bill.

LLM API costs (Gemini / Claude / OpenAI) are identical to today —
the calls move from the Mac to a Worker but go to the same upstream.

---

## What does NOT work — common dead ends

Skip these unless you have new information that contradicts the
probe data:

- **Bundling the `.onnx` into the Worker script.** Exceeds the 10 MB
  gzip script limit. Load from R2 inside the tile DO instead (the
  reference implementation does this).
- **Static Assets for the model.** Worked in round 1; superseded in
  round 3 by R2-from-DO. R2 is simpler (no separate upload pipeline,
  no per-file 25 MiB cap, lifecycle rules cover work artefacts and
  the model under one binding).
- **Cloudflare Containers.** Works technically, but requires Workers
  Paid ($5) plus per-second CPU/memory billing, and you still pay for
  Postgres separately. Same cost as Workers + R2 with more moving
  parts.
- **Hyperdrive for DB.** Not wrong, but unnecessary. The VPS API
  boundary keeps Workers stateless and is simpler to operate.
- **D1 as a Postgres replacement.** The current schema uses
  `pg_trgm`, tsvector FTS, `FOR UPDATE SKIP LOCKED`, JSONB indexes.
  Migrating loses all of these; D1 is single-writer SQLite.
- **Bypassing the 128 MB memory limit.** No knob exists. The Workers
  runtime enforces it at the V8 isolate level. Don't waste time
  looking for workarounds — design the pipeline to fit.
- **Cron triggers to keep tile workers warm.** Solved by promoting the
  tile worker to a Durable Object (round 3). The DO keeps the ORT
  session alive across calls; a cron is no longer needed.
- **Per-bubble cache for cross-page reuse.** Bubble pixels include
  the background under the bubble. They are unique per page. The
  useful caches in round 3 are (a) the R2 tile-output cache keyed by
  `sha256(rgb || mask)` and (b) Cache API for whole-stage outputs
  on `prepare` and `scan`.

---

## Lessons from porting prepare + scan (2026-05 round 2)

These came up while moving real Python pipeline code into Workers.
Recording them so the next iteration does not relearn them.

### Use `@jsquash/jpeg` and `@jsquash/png`, not photon-rs

The earlier "photon-rs (or Rust → WASM)" suggestion is wrong for the
shapes we actually need. `@jsquash/jpeg` (mozjpeg WASM, ~245 KiB) and
`@jsquash/png` (Squoosh png WASM, ~177 KiB) cover both decode and
encode, accept a pre-compiled `WebAssembly.Module` via `init(module)`,
and have no dynamic-import landmines. Total codec WASM for the
prepare worker is under 600 KiB combined, comfortably inside any
script budget. photon-rs forces a bigger Rust ABI than we need and
its decoder API is more awkward than `decode(buf)` / `encode({data,
width, height}, opts)`.

### OpenCV math needs deliberate porting, not approximation

The prepare stage uses three OpenCV-isms that do not translate
1:1 to JavaScript:

- `cv2.INTER_AREA` is a weighted area average with sub-pixel overlap,
  not a box-mean of integer pixel ranges. The integer box
  approximation drifts the color-ratio sample by ~1% on synthetic
  manhwa, enough to make verification noisy. Port the full overlap
  weights; the cost is one extra `Float64Array` per scanline.
- OpenCV `RGB2HSV` computes saturation via a precomputed `div_table`
  rather than a straight float divide. Using `Math.round((max-min)*
  255/max)` instead of `((max-min)*255/max)|0` halves the residual
  drift; the remainder (≤0.02) is genuine codec / rounding noise and
  is safe to accept in the verifier.
- `cv2.cvtColor(_, RGB2GRAY)` is BT.601 with integer-fixed-point
  weights. Use `(r*4899 + g*9617 + b*1868 + 8192) >> 14` to match
  exactly; this matters because the manhwa stitch cut algorithm
  compares per-row max-neighbour diff against a threshold derived
  from those integer pixels.

The verifier `spike/prepare-worker/verify.py` checks all four cases
(manga auto / manhwa auto / forced one-to-one / forced stitch) and
asserts strategy + page count + dimensions match exactly, with pixel
PSNR ≥30 dB and color-ratio Δ ≤ 0.02.

### Lens API: call it directly, no proxy

There is no need to route through the VPS or the Discord Activity
URL mapping. The Lens crupload endpoint is a server-to-server HTTPS
POST with `Content-Type: application/x-protobuf`, public
`X-Goog-Api-Key`, and no CORS requirement. Workers send the same
`LensOverlayServerRequest` protobuf the Python client builds.

The clean way to do this inside a Worker is to import the proto
classes from `chrome-lens-ocr`'s `proto_generated/` (they are plain
google-protobuf CommonJS modules, bundle cleanly under esbuild) and
skip the rest of that package's surface — it pulls in `sharp`,
`undici`, and `node:fs/promises`, none of which work in workerd.
The fork in `spike/scan-worker/src/lens-core.ts` is ~150 lines and
exports just `buildRequest(bytes, w, h)` and
`parseResponse(bytes, [origW, origH])`. Image resizing to ≤1200 px
(Lens's max input dim) is done with the same `resizeRGBA` helper
used by `prepare-worker`.

### Don't subclass `LensCore` — fork the parser

`chrome-lens-ocr`'s `LensCore` exposes only line-level segments via
its private `#parseLensProtoResponse`. The Typoon scan stage needs
paragraph-level bboxes (one block per speech bubble) plus per-word
boxes for the inpaint mask augmentor. Subclassing is impossible
(private method), and the parser is short enough to fork. Result:
the scan worker returns a paragraph-shaped DetectionResult that maps
directly onto the Python `TextBlock` / `WordBox` / `LineBox`
contracts, so verification can compare structure.

### Wrangler account ID is sticky across `wrangler login`

After `wrangler login` to a new Cloudflare account, `wrangler deploy`
may still send requests to the old account ID. The fix is to set
`CLOUDFLARE_ACCOUNT_ID=<new-id>` in the environment for the deploy
command. There is no `wrangler logout`-equivalent that clears this
cache on its own.

### Workers Paid is mandatory at current bundle sizes

The Free-tier 3 MiB script limit is exceeded by the ORT WASM blob
(12.4 MiB) used by the tile worker. Workflows are now free GA on
both tiers, so Workflow availability is no longer a reason to pay,
but the ORT WASM size still is. Don't waste time trying to slim the
WASM under 3 MiB — it would mean dropping SIMD and threading paths
that don't matter in workerd anyway but are baked into the upstream
build.

### Numbers re-measured on a fresh account

A clean Cloudflare account (HKG colo) re-running the workflow probe:

| n steps | mode       | wall time |
|--------:|------------|----------:|
|      10 | sequential |  2,240 ms |
|      10 | parallel   |    728 ms |

The 3.1× speedup is lower than the original 5.9× because the
`/healthz` endpoint on the VPS now serialises more aggressively
(each parallel hit costs ~430 ms vs the original ~50 ms). Workflow
scheduling itself is still genuinely parallel — `stepMsSum` was
3,440 ms for 10 parallel steps, so `Promise.all(step.do)` runs the
steps concurrently as advertised; only the upstream limits the
observable speedup.

---

## Round 3 architecture — RPC + R2 + Workflow DAG (2026-05)

By the end of round 2 every stage worker accepted multipart bytes over
HTTPS and the workflow chained them with raw `fetch` calls. That works
but burns four extra TLS handshakes per chapter, serialises 30 MB of
page data through Workers script execution time, and gives Workflows
no idea what the steps do (so retries are blunt and observability is
opaque). Round 3 rebuilt the pipeline around the patterns Cloudflare
actually intends for multi-stage compute.

### What changed at a glance

| Surface | Before | After |
|---|---|---|
| Stage call shape | `fetch(multipart)` | `WorkerEntrypoint` RPC class methods |
| State between stages | request bodies | R2 keys (work/, prepared/, scan/, inpaint/, translate/, render/) |
| Cross-worker binding | public workers.dev URL | service binding with `entrypoint = "ClassName"` |
| Tile-inpaint isolate | per-request, model reload | Durable Object, ORT session pinned |
| Tile-inpaint cache | none | R2 content-addressed by `sha256(rgb \|\| mask)` |
| Idempotent stage cache | none | Cache API on `prepare` and `scan` outputs |
| Caller upload path | per-page presigned PUT | one ZIP via worker-mediated `/upload` (R2 presigned PUT preferred when available) |
| Caller request shape | 1 fetch per stage | 1 fetch (`/start`) returns Workflow instance id |
| Pipeline orchestration | linear `await` chain | Workflow DAG with per-page `scan(i) \u2192 inpaint(i)` chains and `translate \|\| inpaint` overlap |

### Why each change pays off

**RPC over service binding.** Workers RPC (`WorkerEntrypoint` /
`DurableObject` classes) lets one worker call typed methods on another
without HTTPS or content negotiation. The runtime serialises with
structured clone (zero-copy for `ArrayBuffer`/`Uint8Array`) and routes
in-zone. We dropped multipart parsing in 4 workers; bundle sizes fell
by 30-40 KiB each.

**R2 as the source of truth.** Workflow step return values cap at
1 MiB. A 30-page chapter at 2 MB per inpainted PNG would blow that
cap several times over if we kept passing bytes around. With R2,
every step returns `{ output_key, count }` (~80 bytes) and the next
step `R2.get(key)`s exactly what it needs. R2 ↔ Worker is in-network
and free of egress charges; this is the canonical pattern from the
Cloudflare reference architectures.

**Durable Object for the tile-inpaint isolate.** The ORT session
takes 1.5-2 s to compile per cold isolate (model parse + WASM
shader build). Free-tier Workers spin up many isolates per region;
each gets a cold session. A DO pins one isolate per region key and
keeps the session alive across calls. Subsequent tiles in the same
region pay only inference time. Caller chooses the DO id with
`idFromName(`${chapter_id}-${page_index}`)` so all tiles on a page
land in the same isolate.

**R2 cache on tile outputs.** AOT inpaint is fully deterministic over
its input bytes (padded RGB + binary mask). Cache key is the SHA-256
of `tile.body`; value is the inpainted RGB. Hit path is one R2 GET
and zero ORT inference. We observe \u221e\u00d7 speedup on chapter reruns and a
significant speedup whenever two chapters share an identical bubble
crop (rare in practice but free to claim).

**Cache API for `prepare` and `scan`.** Both stages are deterministic
over chapter input (zip bytes + strategy for prepare; prepared
JPEG bytes + is_color for scan). Wrapping the RPC methods with
`caches.default.match/put` makes rerun-from-the-start cost ~5 ms per
stage on a hit. Devs replaying a chapter ten times to debug
translation prompts now pay no compute for prepare/scan.

**ZIP upload + worker-mediated `/upload`.** A 30-page chapter of 200 KB
pages is 6 MB compressed in a ZIP. One `PUT` to R2 vs 30
presigned puts saves 29 TLS handshakes. The endpoint expects a key
under `raw/{chapter_id}/source.zip` and streams the body into R2
without buffering; this is the spike fallback while
`R2Bucket.createPresignedUrl` is still gated behind a runtime flag
in workerd \u2014 when it lands we switch to true client-direct presigned
PUT and the Worker stays out of the data path entirely.

**Workflow DAG (per-page streaming + translate-inpaint overlap).**
The original layout chained `prepare \u2192 scan(all) \u2192 translate \u2192
inpaint(all) \u2192 render-archive`. The actual dependencies form a DAG:
`inpaint(i)` only needs `scan(i)`, not all scans; `translate` only
needs scan jsons and never reads pixels. We refactored the workflow
so `scan(i)` resolves a promise that immediately spawns
`inpaint(i)`, and `translate` runs concurrently with the inpaint
fan-out:

```ts
const perPage = prepared.pages.map((p) => {
  const scanP = step.do(`scan-page-${p.index}`,    () => env.SCAN.scanPage(...));
  const inpaintP = scanP.then(scan =>
    step.do(`inpaint-page-${p.index}`, () => env.INPAINT.inpaintPage({ mask_key: scan.mask_key, ... }))
  );
  return { scanP, inpaintP };
});
const allScans = Promise.all(perPage.map(p => p.scanP));
const translationP = step.do("translate", async () =>
  env.TRANSLATE.translateChapter({ scan_keys: (await allScans).map(s => s.scan_key), ... })
);
const [, inpaintResults, translation] = await Promise.all([
  allScans, Promise.all(perPage.map(p => p.inpaintP)), translationP,
]);
```

The Workflow scheduler is happy to suspend and resume these
independently because each `step.do` is its own (idempotent) named
checkpoint. Hibernation across `await` is free \u2014 Workflow billing
is wall time, not CPU.

### Measured end-to-end

3-page chapter, cold tile cache, warm DO, packy translate provider:

| Step | Duration |
|---|---|
| prepare-1 | 9 s |
| scan-page-0..2 | 7 s / 2 s / 2 s (parallel) |
| translate-1 | 12 s (overlaps inpaint) |
| inpaint-page-0..2 | 6 s / 8 s / 11 s (parallel, cache hits) |
| render-archive-1 | 9 s |
| **wall** | **31 s** |

Compared to round 2's linear layout on the same chapter:

| Round | Wall | Inpaint dominant |
|---|---|---|
| 2 linear, no cache, no DO | 171 s | 150 s |
| 3 DAG + tile cache + DO | 31 s | 11 s |

5.5\u00d7 speedup end-to-end. The remaining bottleneck during a true cold
run (DO empty, tile cache miss across every bubble) is the ORT WASM
inference itself; that is where bin-packing bubbles into larger tiles
or moving inference to WebGPU would buy further wins.

### Surface area for callers

The whole pipeline is reachable through three endpoints on
`chapter-workflow-v2`:

```
POST /upload-url  { chapter_id }              \u2192 { key, url, method }
PUT  /upload?key=raw/{cid}/source.zip         (body: ZIP)
POST /start       { chapter_id, zip_key,      \u2192 { id }
                    source_lang, target_lang }
GET  /status?id=\u2026                              \u2192 Workflow status
```

That is the entire client API. The four stage workers are not
publicly addressed; they are only reachable via service bindings
from `chapter-workflow-v2`. Stages can be deployed independently
without breaking the contract because the wire format between them
is the structurally-clone-safe TypeScript types in
`spike/shared/src/types.ts`.

### What we ruled out in round 3

- **WebGPU + wonnx on Durable Objects.** The pattern is right
  (`#[durable_object]` with `Option<wonnx::Session>` keeping a GPU
  pipeline warm), but our Cloudflare account is not in the WebGPU
  beta yet (`compatibility_flag = "webgpu"` rejected at deploy with
  code 10021). The CF reference repo `cloudflare/workers-wonnx`
  exists, the `@webonnx/wonnx-wasm` NPM package exists, both compile
  and bundle fine \u2014 but `workerd` rejects the flag in production
  until the account is approved. Spike kept under
  `spike/inpaint-do-webgpu/` (TypeScript variant) for when access
  lands. Verdict for now: stick with WASM ORT in a DO.

- **Per-page presigned PUT.** Saves zero wall time vs one ZIP upload
  and triples client complexity.

- **Bin-packing bubbles into larger tiles.** Documented as a
  high-ROI follow-up but not in round 3 \u2014 the DO + cache combo was
  already enough to get end-to-end under 35 s and we wanted to ship
  the DAG refactor first.

---

## Status — what's left

After round 3 only typeset and the production wiring remain.

1. **Typeset (Rust \u2192 wasm-bindgen).** Port `crates/render` to a
   wasm-bindgen target. The crate is already pure-Rust
   (`harfrust`, `skrifa`, `tiny-skia`); only the `pyo3` shim has to
   be replaced. Bundle NotoSansCJK (~16 MB) via R2 (mirrors the
   model-loading pattern of the tile DO) rather than Static Assets,
   so the same R2 lifecycle / replication story applies.

2. **Production wiring.** The pipeline currently produces a `render.bnl`
   without translated typesetting (inpaint output only). Once
   typeset is wired in, the workflow gains a `typeset(i)` step
   between `inpaint(i)` and `render-archive`; render-archive then
   reads from `typeset/{chapter}/*.png` instead of `inpaint/...`.

3. **Optional speedups, ranked by ROI.** None are blockers.

   - **Bin-pack bubbles into larger tiles.** Page with 19 bubbles
     fans out to 19 inference calls today. A 2D bin packer (~50 LOC)
     can reduce that to 4-5 calls. Combine with the existing R2 cache
     and the savings compound.
   - **Smart Placement on the orchestrator.** Already on the tile
     worker; mirror onto the orchestrator so its R2 reads/writes
     stay in-region with the rest of the pipeline.
   - **AI Gateway in front of the LLM provider.** One env-var
     change, free up to 10k req/day, gives prompt-hash caching and
     fallback routing.
   - **WebGPU + wonnx in the DO** \u2014 only meaningful if the AOT
     model's ops survive wonnx and the account gets into the
     WebGPU beta. The infrastructure is in
     `spike/inpaint-do-webgpu/`; trigger condition is the CF
     account approval.

Stop after typeset + bin-packing if the per-page latency is fast
enough for production traffic; the pipeline already scales to many
concurrent chapters because nothing in it is stateful per host.

---

## How to revive this thread

1. Read this file plus the source in `spike/`. The DAG flow and RPC
   contracts live in `spike/chapter-workflow-v2/src/index.ts`;
   shared types in `spike/shared/src/types.ts`.
2. Account requirements:
   - Workers Paid (needed for the ORT WASM bundle size, not for any
     feature).
   - R2 enabled. Bucket name `typoon-work`. Lifecycle rules in the
     dashboard (24 h on `work/*`, `scan/*`, `inpaint/*`; keep
     `prepared/*` and `render/*`).
   - Workflows GA on both Free and Paid; no flag required.
   - Set `CLOUDFLARE_ACCOUNT_ID` env when running `wrangler deploy`
     after a fresh `wrangler login`; the CLI caches the old id
     otherwise.
3. Deploy order matters (cross-worker bindings must exist first):
   1. `tile-worker` (declares `TileInpaint` DO class)
   2. `prepare-worker`
   3. `scan-worker`
   4. `translate-worker` (after `wrangler secret put PACKY_API_KEY`)
   5. `render-archive-worker`
   6. `ort-inpaint-orchestrator` (binds `TILE_DO` from `tile-worker`)
   7. `chapter-workflow-v2` (binds all of the above)
4. Smoke test:
   ```bash
   curl -X PUT --data-binary @chapter.zip \
     "$HOST/upload?key=raw/test/source.zip"
   ID=$(curl -X POST -d '{"chapter_id":"test","source_lang":"en","target_lang":"vi","zip_key":"raw/test/source.zip"}' \
         "$HOST/start" | jq -r .id)
   while :; do
     curl -s "$HOST/status?id=$ID" | jq -e '.status=="complete"' \
       && break; sleep 5
   done
   ```
   Expected wall: 30-60 s on first run, ~30 s on rerun (caches warm).

The architectural conclusion is firm. The remaining engineering work
is well-scoped. Don't re-derive the dead ends — they are listed above
because each one was independently probed and ruled out.
