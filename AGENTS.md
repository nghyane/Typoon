# Agent Instructions — Typoon v2

Do not read docs by default. Documentation can be outdated; only read docs when
the user explicitly asks for them or the task requires a specific referenced
document. Treat current code, the user's request, and this file as the active
source of truth.

## Current direction

Improve Typoon v2. Do not continue the Rust rewrite.

Agents must follow the new architecture. Do not invent a parallel structure,
compatibility layer, or temporary glue path to keep old concepts alive.

The accepted pipeline is:

```text
RawSource -> prepare -> PreparedChapter -> page-local pipeline -> RenderedChapter
```

The main pipeline accepts only `PreparedChapter`. Raw source images must pass
through `prepare` first.

## Required package architecture

New code must fit this structure:

```text
typoon/
  domain/    stable data contracts and pure domain types
  stages/    use-case orchestration: prepare, scan, translate, render
  adapters/  external systems: sources, storage, LLMs, model/runtime bindings
  runs/      run manifests and materialized artifacts for visual verification
  cli/       command interface only
```

Existing packages can be migrated incrementally, but new architecture code must
not be placed in broad modules such as `engine.py`, `service.py`, `manager.py`,
`processor.py`, `helper.py`, or `utils.py`.

The legacy project pipeline has been intentionally removed. Do not recreate
these modules or concepts:

```text
typoon/app/
typoon/engine.py
typoon/ports.py
typoon/cli/pipeline.py
typoon/cli/resolve.py
typoon/cli/utils.py
LazyPageProvider
StitchedStrip
Engine
AppService
ResumePolicy
```

Use `typoon/adapters/vision_runtime.py` for vision model runtime access and
`typoon/runs/events.py` for event/hook contracts that still support LLM code.

Preferred names:

- Data/contracts: `PreparedChapter`, `PreparedPage`, `DetectedPage`,
  `GroupedPage`, `OcrPage`, `TranslatedPage`, `RenderedPage`.
- Stage functions: `prepare_chapter`, `scan_page`, `scan_chapter`,
  `translate_chapter`, `render_page`, `render_chapter`.
- Artifact writing: `ArtifactSink`, `FileArtifactSink`, `RunManifest`.

Use plain functions for stateless stages. Use classes only for stateful things
such as model runtimes, stores, and filesystem artifact sinks.

Before adding a module, classify it as `domain`, `stages`, `adapters`, `runs`,
or `cli`. If it does not fit one of those categories, stop and ask.

## Dependency rules

Allowed direction:

```text
cli -> stages
cli -> runs
stages -> domain
stages -> adapters
stages -> runs
adapters -> domain
runs -> domain optional
domain -> standard library only unless an existing domain type already requires more
```

Forbidden direction:

```text
domain -> stages/adapters/runs/cli
adapters -> stages/cli
runs -> stages/cli
stages -> cli
```

Keep package public APIs small. Export intentional entry points from
`__init__.py`; do not make agents import deep private helpers when a stage or
contract API exists.

## Hard rules

- `prepare` is the only stage allowed to stitch or cut multiple source files.
- The main pipeline is page-local: detect -> group -> OCR -> translate -> render.
- Every stage must emit visual artifacts under `debug-runs/<run-id>/`.
- Visual E2E is the primary verification.
- Change one stage or layer at a time.
- Do not keep replaced legacy paths alive after the new path passes E2E.
- Delete dead legacy code before adding replacement architecture that could be
  confused with it.
- If legacy is still on the active runtime path, replace it at one boundary and
  delete the replaced part in the same task after verification.
- After `prepare`, raw source is dead. Later stages must use prepared page
  pixels only.
- Coordinates, masks, boxes, polygons, OCR, visual context, and render input for
  a page must all refer to the same prepared page image.
- Do not add tests just to enforce architecture. Keep architecture enforcement
  in this file and in code review. Add tests only for runnable behavior,
  deterministic contracts, or bugs that need regression coverage.

## Forbidden in active design

```text
strip as source of truth
LazyPageProvider in main render path
duplicate normalization
cross-page merge in scan/render
compatibility layer for wrong concepts
hacky surface fixes without visual proof
new manager/service/processor/helper/utils modules
new broad engine responsibilities
parallel old/new runtime paths
root ports.py bucket
app/workflows project pipeline
```

## Required run artifact layout

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

## Working style

For each task:

1. Read relevant files first.
2. Change one stage or layer only.
3. Produce visual artifacts when the changed stage is runnable.
4. Run E2E/test for that stage.
5. Report briefly: changed files, debug output path, verification, next step.
