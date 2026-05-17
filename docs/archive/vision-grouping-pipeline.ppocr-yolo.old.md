# Vision grouping pipeline

Read this before touching `typoon/vision/text_grouping.py`,
`typoon/vision/tiling.py`, or `typoon/vision/bubble_scope.py`.

## Overview

YOLO detects bubble boundaries as **scope hints** only.
Final fit regions and erase masks are built entirely from PP-OCR text units.

```text
image
  └─ detect_raw_text_units      tiled PP-OCR detection → deduplicate_regions
  └─ ocr_units_for_filtering    quick OCR crop pass for noise filter
  └─ filter_units               unit_quality: size, confidence, alnum
  └─ detect_scopes              YOLO predict → merge overlapping boxes
  └─ split_units_crossing_scopes  split wide units crossing distinct bubbles
  └─ assign_units_to_scopes     score-based assignment, two-pass ambiguity
  └─ build_groups               union → fit_bbox → balance in scope
  └─ ocr_groups                 full OCR on group ocr_bbox crop
  └─ final_filter_groups        accept/reject with reason code
       └─ to_visual_text_groups export VisualTextGroup list
```

Entry point: `build_page_scan_state()` → `PageScanState`
Public output: `to_visual_text_groups(state)` → `list[VisualTextGroup]`

---

## Tiling (`tiling.py`)

Large pages are cut into overlapping 1280×1280 tiles (400px overlap).
Regions from each tile are shifted into full-image coordinates by
`offset_regions_2d`.

### `deduplicate_regions` — two passes

**Pass 1 — fragment absorption (runs first)**

Tile seams produce small fragments (~20px wide) that are x-adjacent to their
parent region but have IoU = 0. Standard IoU dedup cannot remove them.

Absorption fires when all hold:
- fragment width <= 30px
- fragment width <= 55% of the adjacent (larger) region width
- x gap between them <= 4px
- y overlap >= 50% of the smaller height

Fragment polygon is merged into the parent; fragment is dropped.

This pass runs **before** IoU dedup so the parent cannot be dropped before
the fragment is absorbed.

**Pass 2 — IoU dedup**

If IoU > 0.5 between two regions, keep the higher-confidence one.

---

## YOLO scope detection (`bubble_scope.py`)

```python
results = model.predict(image, imgsz=imgsz, conf=conf, iou=0.5)
```

After prediction, overlapping YOLO boxes are merged via Union-Find:
- IoU >= 0.25 **or** one box >= 65% inside the other → merge
- Merged scope = union bbox, max confidence

---

## Unit splitting (`split_units_crossing_scopes`)

Wide units (width >= 1.5× median) that span two separate bubbles are split
at the midpoint between scope boundaries.

### Nested scope guard

A unit may overlap both a large scope and a smaller scope fully contained
within it. Without a guard this would split the unit incorrectly.

Fix: before splitting, remove any qualifying scope whose x-range is fully
contained within another qualifying scope. Only truly separate scopes trigger
a split.

---

## Scope assignment (`assign_units_to_scopes`)

### Pass 1 — scoring

Each unit is scored against each scope:

```
score = 0.70 × center + 0.25 × inside + 0.05 × iou
```

| Component | Weight | Meaning |
|---|---|---|
| `center` | 0.70 | 1.0 if unit centroid is inside scope bbox |
| `inside` | 0.25 | fraction of unit area covered by scope |
| `iou` | 0.05 | standard IoU |

Skip if `inside < 0.18` and no center hit.
Assign `None` (free) if best score < 0.45.
Mark ambiguous if `score_best - score_second < 0.08` and `score_second > 0.35`.

### Pass 2 — neighbor support

Ambiguous units are resolved by counting already-assigned neighbors that
line up in the same column or row. The scope with more neighbors wins,
provided `best_support > 0` and either beats the second or has score >= 0.55.

---

## Group construction and fit bbox (`build_groups`)

```text
raw_bbox  = union of all unit boxes in the group (no padding)

pad       = fit_padding(group_boxes, page_w, page_h)
            ├─ tall pages (h/w > 2.5):  clamp(0.18 × median_unit_h, 4, 18)
            └─ normal pages:            clamp(0.12 × median_unit_h, 2, 10)

fit_bbox  = expand(raw_bbox, pad, page_w, page_h)   ← expand all 4 sides
          → clamp to scope_bbox if scoped            ← never exceed YOLO scope
          → _balance_fit_in_scope(fit, scope, pad)   ← correct asymmetric gap

ocr_bbox  = raw_bbox + 10% padding, expanded asymmetrically toward scope edges
            where text may be under-detected, clipped by neighboring unit boxes

median_angle = median of _polygon_angle() across all units in group
               stored as TextGroup.median_angle, used in final filter
```

### `_balance_fit_in_scope`

When PP-OCR misses units on one side, fit is skewed: gap is small on the
detect-poor side and large on the other. The function expands only the **tight
side** to match the **roomy side**. It never shrinks any side.

Activation (all three must hold per axis):

1. `|left_gap - right_gap| >= 8px` — imbalance is meaningful
2. `min(gap) <= pad` — the tight side is genuinely close to detected text
3. `max(gap) > pad` — the roomy side actually has space

---

## Final group filter (`final_filter_groups`)

Evaluation order for **free (unscoped)** groups:

| Check | Condition | Result |
|---|---|---|
| `_looks_like_system_card` | >= 3 units, >= 4 words, uppercase-heavy, large bbox | accept |
| `free_skewed` | `\|median_angle\| > 20°` | reject |
| `_looks_like_narration` | conf >= 0.70, >= 3 words, >= 3 distinct words, alnum >= 10 & >= 55% density | accept |
| `free_low_conf` | conf < 0.35 | reject |
| `free_large_sfx_like` | area > 2.5% or width > 24% or height > 16% of page | reject |
| `free_short_low_conf` | <= 2 chars and conf < 0.80 | reject |
| `free_tiny` | bbox < 20px on either side | reject |

Scoped groups only check `ocr_empty` and `ocr_no_alnum`.

### `free_skewed`

Background content (exam papers, posters, tilted signage) has consistent
tilt ~-30° to -47°. Checked via `TextGroup.median_angle` (median of unit
polygon angles). Runs before `_looks_like_narration` so horizontal narration
text (≈ 0°) is unaffected.

### `_looks_like_narration`

Caption/narration boxes outside bubbles are wide (would hit
`free_large_sfx_like`) but are clearly readable text. Accepted when:
- OCR confidence >= 0.70
- >= 3 words with >= 3 distinct words (rejects repeated SFX like "HA HA HA")
- alnum char count >= 10
- alnum density >= 55%

---

## Output types (`vision/types.py`)

`VisualTextGroup` — final exported group:

| Field | Content |
|---|---|
| `text` | OCR text |
| `confidence` | OCR confidence |
| `text_bbox` | raw union bbox of unit boxes |
| `fit_bbox` | padded, balanced, scope-clamped render bbox |
| `ocr_bbox` | asymmetrically expanded crop used for OCR |
| `mask_bbox` | union of text pixel masks |
| `erase_bbox` | union of dilated erase masks |
| `scope_bbox` | YOLO scope bbox if scoped, else None |
| `text_masks` | list of `TextMask` (pixel-level) per unit |
| `erase_masks` | dilated masks for inpainting |
| `source` | `"scoped"` or `"free"` |

---

## Known edge cases

| Symptom | Root cause | Fix |
|---|---|---|
| Fragment unit ~20px beside main unit | Tile seam, IoU = 0 between adjacent regions | `deduplicate_regions` fragment absorption (pass 1) |
| Fragment reappears after dedup | Nested YOLO scope causes `split_units_crossing_scopes` to re-split the merged unit | Nested scope guard in split function |
| fit bbox skewed left or right in bubble | PP-OCR misses units on one side | `_balance_fit_in_scope` |
| Narration/caption box rejected as large SFX | `free_large_sfx_like` threshold too aggressive for wide caption boxes | `_looks_like_narration` exit before size filter |
| Background exam/poster text accepted | Tilted text (~-35°) passes size and confidence checks | `free_skewed` filter on `median_angle` |
