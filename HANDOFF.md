# Handoff — Typoon v2 Refocus

Do not read docs by default. Docs are context only and may be outdated; read
them only when the user explicitly asks. The active source of truth is the user,
current code, and `AGENTS.md`.

## Current goal

Improve v2. Do not continue the Rust rewrite.

Main decision:

```text
RawSource -> prepare -> PreparedChapter -> page-local pipeline -> RenderedChapter
```

The main pipeline only accepts `PreparedChapter`. Raw source images must pass
through `prepare` first.

## Rules

- `prepare` is the only stage allowed to stitch/cut multiple files.
- Main pipeline is page-local: detect -> group -> OCR -> translate -> render.
- Every stage must emit visual debug artifacts under `debug-runs/<run-id>/`.
- Visual E2E is the primary verification.
- Do not keep replaced legacy paths alive after the new path passes E2E.

Forbidden in active design:

```text
strip as source of truth
LazyPageProvider in main render path
duplicate normalization
cross-page merge in scan/render
compatibility layer for wrong concepts
hacky surface fixes without visual proof
```

## Why this reset

v3 failed because refactors kept legacy concepts alive. Multiple plausible
architectures coexisted, so agents added glue and surface patches instead of
removing wrong concepts. v2 must be made observable first, then simplified stage
by stage.

## Next steps

Current demolition status:

- Removed old raw-source strip/lazy-provider pipeline.
- Removed `typoon/app/`, `typoon/engine.py`, `typoon/ports.py`, old CLI pipeline
  resolution, `LazyPageProvider`, `StitchedStrip`, old merge path, and tests that
  protected those concepts.
- Added architecture boundaries: `typoon/runs/`, `typoon/stages/`,
  `typoon/adapters/vision_runtime.py`, and `typoon/adapters/connectors.py`.
- `translate` and `detect` commands intentionally fail until the new
  PreparedChapter pipeline exists.

Next build steps:

1. Add `typoon/runs/artifacts.py` with `ArtifactSink`, `FileArtifactSink`, and
   run manifest/layout creation.
2. Add `typoon/domain/prepared.py` with `PreparedPage` and `PreparedChapter`.
3. Add visual E2E run layout:

   ```text
   debug-runs/<run-id>/
     manifest.json
     01_prepare/
     02_detect/
     03_group/
     04_ocr/
     05_translate/
     06_render/
     final/
   ```

4. Add `typoon/stages/prepare.py` and a `prepare` command that outputs
   `PreparedChapter`:

   ```text
   PreparedChapter/
     manifest.json
     pages/0000.png
     pages/0001.png
   ```

5. Only after prepare is visually verifiable, add page-local scan path.

## Working style

For each task:

1. Read relevant files first.
2. Change one stage/layer only.
3. Produce visual artifacts.
4. Run E2E/test for that stage.
5. Report briefly: changed files, debug output path, verification, next step.

