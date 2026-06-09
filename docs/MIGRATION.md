# Typoon v3 — Migration

From current Cloudflare-Container-only stack (4 separate containers)
to one consolidated Container DO + clean Python/Rust split. LLM
workers (`brief`, `translate`) are stable — do not touch.

## Current → target

```
crates/inpaint/                      → crates/typoon_native/
python/typoon_inpaint/                → python/typoon/
workers/media/        (delete)        ┐
workers/scan/         (delete)        ├─ all subsumed by workers/compute/
workers/inpaint/      (delete)        │
workers/typeset-pack/ (delete)        ┘
workers/pipeline/     rewrite         (call env.COMPUTE binding)
workers/api/          keep
workers/brief/        keep            (LLM stable)
workers/translate/    keep            (LLM stable)
workers/shared/       extend          (+ manifest.ts)
workers/compute/      new             (Container DO + Dockerfile)
```

## Phases (one at a time, verify each)

### 1. Rust wheel — `crates/typoon_native`

Rename `crates/inpaint` → `crates/typoon_native`. Consolidate:

- `src/inpaint/{flat_fill, aot, dispatch}.rs`
- `src/render/typesetting.rs`
- `src/pack/bnl.rs`
- `src/lib.rs` exports: `inpaint_page`, `render_page`, `pack_bnl`.

Update root `pyproject.toml`: `manifest-path = "crates/typoon_native/Cargo.toml"`,
`module-name = "typoon.typoon_native"`.

Verify: `maturin build --release` green. Single-image CLI still works.

### 2. Python package — `python/typoon`

Rename `python/typoon_inpaint` → `python/typoon`. Split:

- `domain.py` — pure data: `ChapterImageManifest`, `InpaintPlan`, `InpaintTier`, …
- `artifacts.py` — `ArtifactSink`, `NullSink`, `FileSink`.
- `scan/{detect, group, route, plan, storyboard}.py`.
- `render/{erase, text, pack}.py` — wrap `typoon_native`.
- `storage.py` — R2 boto3 (only place R2 appears).
- `server.py` — FastAPI: `POST /prepare-scan`, `POST /render-pack`, `GET /health`.
  Injects `NullSink()`. No bearer auth (Container binding handles it).
- `cli/e2e.py` — probe; injects `FileSink("debug-runs/<run-id>/")`.

Stages take `sink: ArtifactSink = NullSink()` last. No `if debug:` branches.

Verify: `python -m typoon.cli.e2e --zip path/to/chapter.zip --lang ja`
produces all 6 directories under `debug-runs/<id>/`.

### 3. Compute container — `workers/compute`

```
workers/compute/
  Dockerfile        # python:3.12-slim + wheel + aot-gan.safetensors + comic-detr.onnx
  wrangler.toml     # Container class binding, standard-4
  src/index.ts      # TypoonCompute extends Container; route by ?job_id=
```

```ts
import { Container, getContainer } from "@cloudflare/containers";

export class TypoonCompute extends Container {
  defaultPort = 8080;
  sleepAfter  = "5m";
}

export default {
  async fetch(req: Request, env: Env) {
    const job = new URL(req.url).searchParams.get("job_id") ?? "default";
    return getContainer(env.COMPUTE, job).fetch(req);
  }
};
```

Verify: `wrangler deploy`, hit `/health` via Worker binding.

### 4. Workflow + shared

- `workers/shared/src/manifest.ts` — `ChapterImageManifest` + `preparedKey`.
- Rewrite `workers/pipeline/src/index.ts`:
  - Remove DO bindings for old containers.
  - Add `COMPUTE` Container binding.
  - Two `step.do` calls: `prepare-scan` → (brief ∥ translate) → `render-pack`.
- `workers/api/` — strip any references to removed containers.

### 5. E2E

Trigger one real chapter via `api`. Verify:

- `prepared/`, `scan/`, `storyboard/`, `brief/`, `translate/`, `render/{job}.bnl` written.
- Wall ≤ 130s.
- BNL opens in reader.

### 6. Cleanup

After E2E green:

```
git rm -r workers/{media,scan,inpaint,typeset-pack}
git rm -r python/typoon_inpaint     # renamed
git rm -r crates/inpaint            # renamed
```
