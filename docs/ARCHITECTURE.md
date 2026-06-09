# Typoon v3 — Architecture

## Goals

- 1 chapter (zip → bnl) end-to-end ≤ 2 min @ 100 pages.
- AOT-GAN ground truth except pure flat-colour bubbles.
- Single vendor: Cloudflare.

## Stack

| Layer    | Choice                                            |
|----------|---------------------------------------------------|
| Edge     | Cloudflare Workers (paid tier)                    |
| Compute  | Cloudflare Container DO, `standard-4` (4 vCPU × 12 GB) |
| Storage  | Cloudflare R2                                     |
| LLM      | Gemini Flash 2.5 (brief), Pro 2.5 (translate)     |

Compute reached via Worker service binding. No public URL, no bearer.

## Flow

```
zip → R2 raw/
        │
        ▼  Workflow ChapterPipeline
prepare-scan  (Container, ~61s)  → prepared/, scan/, storyboard/
        │
        ├─ brief      (Worker, ~38s) ─┐
        └─ translate  (Worker, ~38s) ─┤
                                       ▼
render-pack   (Container, ~14s)  → render/{job}.bnl
                                       │
                                  finalize (api)
```

Wall ≈ 115s. Same container instance per `job_id` keeps AOT model warm
across both calls.

## Components

```
crates/typoon_native/         1 Rust wheel (PyO3)
  src/inpaint/{flat_fill, aot, dispatch}.rs
  src/render/typesetting.rs
  src/pack/bnl.rs

python/typoon/                1 Python package
  domain.py                   ChapterImageManifest, InpaintPlan, InpaintTier
  artifacts.py                ArtifactSink, NullSink, FileSink
  scan/{detect, group, route, plan, storyboard}.py
  render/{erase, text, pack}.py
  storage.py                  only place R2 / boto3 appears
  server.py                   FastAPI: /prepare-scan, /render-pack, /health
  cli/e2e.py                  probe (FileSink injection)

workers/
  api/                        D1, upload, auth, callbacks
  pipeline/                   ChapterPipeline workflow
  brief/                      LLM vision    (stable, untouched)
  translate/                  LLM translate (stable, untouched)
  shared/                     types + r2 + llm helpers
  compute/                    Container DO running typoon-compute image
    Dockerfile                python:3.12-slim + native wheel
                              + aot-gan.safetensors + comic-detr.onnx
    wrangler.toml             Container class binding
    src/index.ts              TypoonCompute extends Container
```

Only `workers/compute/` decodes images.

## Contracts

### ChapterImageManifest (~5 KB / 100 pages, inline in workflow params)

```ts
interface ChapterImageManifest {
  job_id:           number;
  strategy:         "one_to_one" | "stitch";
  is_color:         boolean;
  source_lang_hint: string;
  pages: { index: number; width: number; height: number; source: number[] }[];
}
function preparedKey(job_id: number, i: number): string {
  return `prepared/${job_id}/${String(i).padStart(4, "0")}.jpg`;
}
```

### Coordinate space

Every persisted artifact uses **prepared_space** (pixels of
`prepared/{job}/{i:04d}.jpg`). `lens_space` exists only inside
`LensBlocksDetector.detect()`. `tile_space` only inside AOT runtime.

### InpaintTier

```python
class InpaintTier(IntEnum):
    FLAT_FILL = 0
    AOT_GAN   = 1

def pick_tier(group, page_img) -> int:
    if group.shape_kind == "burst": return 1
    interior = sample_polygon_interior(page_img, group.polygon, group.text_mask)
    if interior.size < 100: return 1
    return 0 if interior.std(axis=0).max() < 8 else 1
```

Stored in `InpaintPlan.groups[i].inpaint_tier`; dispatched by
`crates/typoon_native/src/inpaint/dispatch.rs`.

### Container endpoints

```
POST /prepare-scan?job_id=N
  { job_id, zip_key, source_lang_hint }
  → { manifest, scan_keys[], storyboard_keys[] }

POST /render-pack?job_id=N
  { job_id, manifest, scan_keys[], translate_key }
  → { archive_key, size_bytes, pages }

GET /health → { ok: true }
```

Worker side:

```ts
const c = getContainer(env.COMPUTE, String(job_id));
await c.fetch(`https://c/prepare-scan?job_id=${job_id}`, { method: "POST", ... });
```

### R2 layout

```
raw/{job}/source.zip                    7d
prepared/{job}/{i:04d}.jpg              7d
scan/{job}/{i:04d}.msgpack              30d   (plan embedded)
storyboard/{job}/{n:02d}.jpg            7d
brief/{job}/*.json                      keep
ctx/{job}/{input,output}.json.gz        keep
translate/{job}.json                    90d
render/{job}.bnl                        keep
jobs/{job}/error.json                   30d
```

No `inpaint/*.png` or `typeset/*.png`. Render-pack does erase + text +
JPEG + BNL in memory, one call.

## Artifacts (probe only)

Stages emit through one hook. No `if debug:` branches.

```python
class ArtifactSink(Protocol):
    def image(self, key: str, img: np.ndarray) -> None: ...
    def json (self, key: str, obj: dict)       -> None: ...
    def bytes(self, key: str, data: bytes)     -> None: ...
```

- Prod: `server.py` injects `NullSink()`. Zero cost.
- Probe: `cli/e2e.py` injects `FileSink("debug-runs/<run-id>/")`.

```
debug-runs/<run-id>/
  01_prepare/{i:04d}.jpg            normalized page
  02_detect/{i:04d}.jpg             page + bbox overlay
  03_group/{i:04d}.jpg              page + polygon + tier label + text_mask
  03_group/{i:04d}.json             InpaintPlan decoded
  04_storyboard/composite.jpg       composite sent to brief LLM
  05_translate/translate.json       LLM output
  06_render/{i:04d}.jpg             page after erase + text
  final/output.bnl                  packed archive
```

## Deploy

```bash
maturin build --release --target x86_64-unknown-linux-gnu  # wheel for container

cd workers/api       && wrangler deploy
cd workers/pipeline  && wrangler deploy
cd workers/brief     && wrangler deploy
cd workers/translate && wrangler deploy
cd workers/compute   && wrangler deploy   # builds + pushes Docker image
```

## Failure modes

| Failure              | Recovery                              |
|----------------------|---------------------------------------|
| Container OOM        | step.do retry, fresh instance         |
| Container cold ~3s   | within step.do timeout (5 min)        |
| R2 transient         | boto3 retry → step.do retry           |
| LLM 429              | step.do retry with backoff            |
| Compute > 10 min     | mark failed, notifyError              |

Workflow checkpoints between every stage.
