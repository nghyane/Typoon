# Hard rules

## Pipeline boundary

- `prepare` is the only stage allowed to stitch or cut multiple source files.
- After `prepare`, raw source is dead. All later stages use prepared page pixels only.
- The main pipeline is strictly page-local: one prepared page → one output page.
- No cross-page merge in scan, OCR, or render.

## Coordinate integrity

- Coordinates, masks, boxes, polygons, OCR crops, and render input for a page
  must all refer to the **same prepared page image**.
- Never mix tile-local and full-image coordinates.

## Code structure

- New code must live in `domain/`, `stages/`, `adapters/`, `runs/`, or `cli/`.
- Do not create `engine.py`, `service.py`, `manager.py`, `processor.py`,
  `helper.py`, or `utils.py`.
- Use plain functions for stateless stages.
- Use classes only for stateful things: model runtimes, stores, artifact sinks.

## Verification

- Visual E2E is the primary verification method.
- Every stage must emit visual artifacts under `debug-runs/<run-id>/`.
- A stage is not done unless it runs from a repeatable CLI command and writes
  inspectable images.
- Unit tests only for small deterministic logic or regression coverage of bugs.

## Dead code

- Do not keep replaced legacy paths alive after the new path passes E2E.
- Delete dead code in the same task, not later.
- Do not create compatibility layers or glue for wrong concepts.

## Forbidden — do not recreate

```text
strip as source of truth
LazyPageProvider in main render path
duplicate normalization
cross-page merge in scan/render
typoon/app/
typoon/engine.py
typoon/ports.py
typoon/cli/pipeline.py
typoon/cli/resolve.py
typoon/cli/utils.py
Engine, AppService, ResumePolicy
parallel old/new runtime paths
```

## Why these rules exist

v3 failed because refactors kept old concepts alive. Multiple architectures
coexisted and agents added glue instead of removing wrong concepts. These rules
exist to prevent that pattern from recurring.
