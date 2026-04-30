# RFC-002: Visual E2E Workflow

## Status

Accepted

## Problem

v2/v3 bugs were often patched by inference. Without mandatory visual artifacts,
agents changed heuristics without seeing detection, grouping, OCR crops, masks,
or rendered output.

## Decision

Visual E2E is the primary verification. Unit tests are secondary and only cover
small deterministic logic.

Every stage must emit artifacts under:

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

## Required artifacts

### Prepare

```text
groups.json
row_cost.png
cuts_overlay.png
prepared_*.png
```

### Detect

```text
page_0000_source.png
page_0000_boxes.png
page_0000_polygons.png
detections.json
```

### Group/OCR

```text
page_0000_groups.png
page_0000_group_labels.png
ocr_crops/*.png
ocr.json
```

### Translate

```text
request.xml or request.json
response.xml or response.json
translations.json
```

### Render

```text
erase_mask.png
inpainted.png
text_layout.png
rendered.png
```

## Definition of done

A stage is not done unless:

- it runs from a repeatable command
- it writes visual artifacts
- it writes intermediate JSON/XML
- the user can inspect the output image
- replaced old code is removed after the new path passes E2E

