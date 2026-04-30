# Architecture

## Pipeline

```text
RawSource
  └─ prepare_chapter()          stages/prepare.py
       └─ PreparedChapter       domain/prepared.py
            └─ scan_page()      adapters/vision_runtime.py → vision/*
            └─ translate        translation/*
            └─ render           vision/erase.py + render (not yet wired)
```

`prepare` is the only stage that reads raw source files.
All later stages consume `PreparedChapter` pages only.

## Package layout

```text
typoon/
  domain/       pure data contracts — no external deps beyond stdlib
  stages/       use-case orchestration: prepare, scan, translate, render
  adapters/     concrete external systems: sources, LLMs, model runtimes
  runs/         run manifests and artifact sinks
  cli/          CLI entry points only — no business logic
  vision/       CV models and grouping pipeline (called from adapters/stages)
  translation/  LLM translation pipeline
  llm/          LLM client adapters
```

## Dependency directions

Allowed:

```text
cli       → stages, runs
stages    → domain, adapters, runs
adapters  → domain
runs      → domain (optional)
domain    → stdlib only
```

Forbidden:

```text
domain    → stages / adapters / runs / cli
adapters  → stages / cli
runs      → stages / cli
stages    → cli
vision    → stages / cli  (vision is a library, not an orchestrator)
```

## Key types

| Type | Location | Role |
|---|---|---|
| `PreparedPage` | `domain/prepared.py` | one canonical page: index, file path, width, height |
| `PreparedChapter` | `domain/prepared.py` | ordered tuple of PreparedPage + root path |
| `VisualTextGroup` | `vision/types.py` | one detected+OCR'd text group with all bbox/mask variants |
| `ArtifactSink` | `runs/artifacts.py` | protocol for writing debug images and JSON |
| `FileArtifactSink` | `runs/artifacts.py` | writes to `debug-runs/<run-id>/` on disk |
| `VisionRuntime` | `adapters/vision_runtime.py` | owns scanner, eraser, YOLO model instances |

## Implemented stages

### prepare (`stages/prepare.py`)

- reads raw pages via `RawChapterSource` protocol
- writes one PNG per page to `PreparedChapter/pages/`
- writes `manifest.json`
- writes debug artifacts to `01_prepare/`
- currently 1:1 (no stitch/cut) — stitch/cut is planned but not yet implemented

CLI: `typoon prepare <input_folder> [--out <dir>] [--run-id <id>]`

### inspect-vision (`cli/commands.py → inspect_vision`)

Runs the full vision grouping pipeline on a folder of images and writes
four-panel debug images (units / groups+scopes / masks / erased) plus
`detections.json`.

CLI: `typoon inspect-vision <folder> [--out <dir>] [--limit N]`

### translate, render

Not yet connected to `PreparedChapter`. The `translate` CLI command is
explicitly disabled and prints an error.

## Debug run layout

```text
debug-runs/<run-id>/
  manifest.json
  01_prepare/
    prepared_manifest.json
    groups.json
    prepared_0000.png
    row_cost.png        (placeholder until stitch/cut is implemented)
    cuts_overlay.png    (placeholder)
  02_detect/
  03_group/
  04_ocr/
  05_translate/
  06_render/
  final/
```

## Models

Resolved from `~/.typoon/models/` (or config `models_dir`):

| File | Used for |
|---|---|
| `ppocr-det.mlpackage` / `ppocr-det.safetensors` | PP-OCR text detection (CoreML > MLX > ONNX) |
| `ppocr-det-config.json` | det config |
| `bubble-scope-yolov8m.mlpackage` / `.pt` | YOLO bubble scope detection |
| `AOT-*.onnx` (aot/) | text inpainting / erasing |
