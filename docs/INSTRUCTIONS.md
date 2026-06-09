# Typoon v3 — Coding rules

## Naming

| Concept              | Name                          |
|----------------------|-------------------------------|
| Python package       | `typoon`                      |
| PyO3 native module   | `typoon.typoon_native`        |
| Rust crate           | `typoon_native`               |
| Container DO class   | `TypoonCompute`               |
| Worker binding       | `env.COMPUTE`                 |
| TS shared package    | `@typoon/shared`              |

No `_v3`, `_new`, `_py`, `_legacy` suffixes.

## Module boundaries (Python)

- `domain` — pure data classes. No IO, no native imports.
- `artifacts` — `ArtifactSink` protocol + `NullSink` + `FileSink`. No stage logic.
- `scan.*`, `render.*` — orchestration. May import `typoon_native`, `storage`, `artifacts`.
- `storage` — only place R2 / boto3 / filesystem appears.
- `server` — FastAPI routing only. Injects `NullSink()`. Handlers call
  `scan.run_prepare_scan` / `render.run_render_pack`.
- `cli.*` — dev probes. Injects `FileSink`. May reach anywhere.

Stage signature: `def stage(...args, sink: ArtifactSink = NullSink()) -> ...`.
**No `if debug:` branches.**

## Module boundaries (Workers)

- `shared/` — types + helpers. No state, no DO.
- `api/` — HTTP, D1, auth, callbacks. No image bytes in memory.
- `pipeline/` — Workflow only. No direct R2 IO; delegates to `env.COMPUTE`.
- `brief/`, `translate/` — single LLM stage. Stateless. **Stable, do not modify.**
- `compute/` — Container DO. Routes by `?job_id=` for warm affinity.

## Forbidden

- Resurrecting `typoon_inpaint_py`, `typoon-vision`, `typoon-stages`,
  or any removed compat shim.
- Coordinate transforms outside `LensBlocksDetector.detect()`.
  Every persisted coord is `prepared_space`.
- Intermediate `inpaint/*.png` or `typeset/*.png` on R2.
- Image decode/encode in a Worker isolate.
- Two RPC endpoints with overlapping args. `ChapterImageManifest` is
  the only carry-along.
- `manager` / `service` / `helper` / `utils` / `processor` modules
  outside the boundaries above.
- `if debug:` branches in stage code. Use `ArtifactSink`.

## Probe + tests

- Probe: `python -m typoon.cli.e2e --zip <path>` runs the same code
  paths `server.py` uses. Only the sink differs.
- Visual artifacts: `debug-runs/<run-id>/` with 6 dirs (see ARCHITECTURE.md
  §Artifacts).
- No unit tests for architecture; review enforces.
- Tests allowed for:
  - Deterministic contracts (msgpack roundtrip, `pick_tier` thresholds,
    `preparedKey` derivation).
  - Bug regressions.
  - Native correctness vs reference (e.g. `flat_fill` matches numpy).

## Commits

Conventional commits with scope. Reference decisions:

```
feat(scan): pick_tier heuristic
refactor(render): split erase / text / pack
chore(crates): rename inpaint -> typoon_native
docs(architecture): record CF Containers decision  (See D1.)
```

## Deploy

```sh
maturin build --release --target x86_64-unknown-linux-gnu -o wheels/

cd workers/api       && wrangler deploy
cd workers/pipeline  && wrangler deploy
cd workers/brief     && wrangler deploy
cd workers/translate && wrangler deploy
cd workers/compute   && wrangler deploy
```

## Secrets

```
wrangler secret put ANTHROPIC_API_KEY  # brief + translate
wrangler secret put GEMINI_API_KEY     # brief + translate
```

R2 access for the container is via Workers binding, not API keys.
