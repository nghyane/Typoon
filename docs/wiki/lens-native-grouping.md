# Lens-native grouping

Read this before touching `typoon/vision/detectors/lens_blocks.py` or
`typoon/vision/groupers/lens_native.py`.

Companion to [Vision grouping pipeline](vision-grouping-pipeline.md) (which
documents the PP-OCR + YOLO bubble-scope pipeline used by the `offline`,
`bing` and `manga_ja` presets). This page covers the **`lens` preset** —
detector and grouper specific to Google Lens output. Lens ships text
together with detection, so there is no separate OCR step.

```text
RGB page
  └─ LensBlocksDetector
       └─ tile 900×900 with 200px overlap (Lens resizes >1000px down)
       └─ _ocr_tile         per-tile call, output_format="detailed",
                            ocr_language = lens_lang_hint(source_lang)
       └─ _dedup_raw        pairwise IoU dedup over tile overlap
       └─ TextBlock[]       page-coord, with words[] and lines[]
       └─ _filter_blocks    tiny / decoration / huge / cross_column
       └─ _recover_row_gaps  re-OCR rows where Lens dropped glyphs
  └─ LensNativeGrouper
       └─ _block_to_group   1 block → 1 BubbleGroup (+ class profile, masks)
       └─ _merge_tategaki_columns   chain-cluster vertical columns
```

Entry points:
- `LensBlocksDetector.detect(image, lang) → DetectionResult`
- `LensNativeGrouper.group(image, detection, lang) → tuple[BubbleGroup, ...]`

The output `tuple[BubbleGroup, ...]` is what `stages.scan` consumes.

---

## Why Lens needs its own grouper

Lens returns each **paragraph** as a single block with:
- per-block bbox + rotation
- per-line geometry (`LineBox`)
- per-word geometry (`WordBox`)
- recognised text in source language

For horizontal scripts this is 1 paragraph = 1 bubble. For **tategaki**
(Japanese vertical) and **vertical Chinese**, Lens emits **one block per
column** — a 3-column bubble becomes 3 separate blocks. We re-merge those
columns ourselves because:

- The translator must see the full sentence (one column has no grammar).
- The renderer needs a single bbox covering the whole bubble for fit.
- Reading order across columns is right-to-left (RTL); the grouper sorts
  text RTL after merge.

---

## Detector: post-processing

Lens caps its input around 1000px. We tile pages at 900px tall with
200px overlap (constants `_TILE_H`, `_OVERLAP` in `lens_blocks.py`).
Each tile call returns paragraphs in tile-local **normalised** coords;
the detector applies post-processing in this order:

1. `_norm_geom_to_pixels` — rotation-aware page-AABB.
2. `_dedup_raw` — pairwise IoU dedup over tile overlap.
3. `_filter_blocks` — tiny / decoration / huge / `cross_column`.
4. `_recover_row_gaps` — re-OCR rows where Lens dropped glyphs.

### `_norm_geom_to_pixels` — rotation-aware AABB

Lens reports a paragraph's geometry as `(center_x, center_y, width,
height, rotation_z)` where **`width` and `height` are in the text
box's own axes**, not the page. For axis-aligned text the rotation is
≈ 0 and width/height map straight to page extents. For rotated text —
SFX (10–45°), side-rotated watermarks (~90°) — naively using
width/height as a page AABB swaps the visual extents.

Example: the `manhuaren.com` watermark on a regression fixture is
reported as 93 × 14 (along its reading axis) with `angle_deg ≈ -89.74`.
Without rotation handling, the AABB became 93 wide × 14 tall on page;
filtering called it `tiny_bbox` by accident. After rotation the AABB
is 14 × 93 — the correct on-page extent.

```text
if |angle_deg| < 0.5:
    cheap path: (cx ± w/2, cy ± h/2)
else:
    corners = [(±w/2, ±h/2)] rotated by angle_deg around centre
    page_aabb = (min(xs), min(ys), max(xs), max(ys))
```

This affects every geometry consumer (block bbox, line bbox, word
bbox). Downstream signals like `_classify_block` (aspect-based SFX
detection) and the erase mask now see the true on-page footprint.

### `_dedup_raw` — pairwise IoU

Inside `lens_blocks.py`:

```text
sort by len(text) desc
for each block, drop if either:
  - _iou_self(b, kept) > 0.5
  - _iou_self(b, kept) > 0.05 AND b.text in kept.text
```

`_iou_self(a, b) = intersection / area(a)` measures how much of `a` is
inside `b`. This catches duplicates of the same paragraph emitted on two
tiles.

### `_drop_cross_column_artifacts` — line-subset dedup

`_dedup_raw` misses one specific Lens failure mode: at the boundary
between two tiles, Lens occasionally fabricates a phantom "horizontal
paragraph" stitched from the **tail line of multiple tategaki columns**.
Each constituent line bbox sits inside a different real column
paragraph, but pairwise bbox-IoU never reaches 0.5 because the phantom
spans across the columns.

Detection signal: paragraph whose lines are absorbed by ≥2 other
paragraphs.

```text
for each block b with len(b.lines) >= 2:
  for each line in b.lines:
    if line bbox is inside any other paragraph
       with area(line ∩ other) / area(line) >= 0.70:
         count that paragraph as "absorbing"
  if absorbing parents >= 2:
    drop b with reason "cross_column"
```

Constants: `_CROSS_COLUMN_MIN_LINES_ABSORBED = 2`,
`_CROSS_COLUMN_LINE_INSIDE_RATIO = 0.70`.

**Why text substring is not part of the signal**: Lens often mis-OCRs
partial glyphs at tile boundaries (e.g. emits `军答` for what should be
just `答`), so a substring match would miss. Geometry alone is
reliable.

### Language hint (`_lens_lang_hint`)

Lens's `ocr_language` parameter is set from the upstream `source_lang`:

| `source_lang` | Lens `ocr_language` |
|---|---|
| `None` / `""` / `en*` | `""` (auto) |
| `ja`, `ja-JP` | `ja` |
| `zh`, `zh-CN`, `zh-Hans` | `zh-Hans` |
| `zh-Hant`, `zh-TW`, `zh-HK` | `zh-Hant` |
| `ko`, `ko-KR` | `ko` |
| `vi`, `vi-VN` | `vi` |
| other | `""` (auto) |

English (and unset) stay on auto-detect because manga pages routinely
mix scripts — a JP page may have EN SFX in the same panel, and a hard
`en` hint suppresses the non-Latin recognition. JP / CN / KO / VI pass
through as an explicit hint to bias Lens on row crops where the
auto-detector has less context (row recovery path).

Empirically the hint produces identical full-page output to auto on
clean 900×900 tiles. Its benefit is on the row crops in the recovery
path, where there's less context for auto to anchor on.

### Row recognition gap recovery (`_recover_row_gaps`)

Lens occasionally drops glyphs around dense decoration runs (e.g. CJK
ellipsis `······`) and emits only the unaffected suffix on that row.
The row's bbox and y-range stay correct so we can localise the missed
glyphs.

Detection: `_suspicious_line_indices` flags non-edge rows whose width
is below `_ROW_GAP_SHORT_RATIO=0.5` of the median of the block's other
rows. Edges are excluded — first and last rows are often legitimately
short (ragged justification, closing punctuation).

Recovery, per suspicious row:

```text
crop_bbox = (block.x1, line.y1 - pad_y, block.x2, line.y2 + pad_y)
            # full block width, line y-range + 45% line height padding
crop = image[crop_bbox]
if min(crop.shape[:2]) < 200:
    crop = upscale_lanczos(crop, scale = ceil(200 / min_dim))
new_text = lens.process_image(crop, ocr_language=lens_lang_hint(source_lang))
if new_text and new_text != line.text:
    replace block.lines[i].text with new_text
    block.text = " ".join(lines.text)
```

The crop is upscaled to ≥200px on the short axis with Lanczos
resampling. Without this, Lens returns the same incomplete text as the
full-page pass — the recognizer needs surface area to re-segment the
glyphs around the decoration run. **Empirically a full-block re-OCR
does not recover the missed glyphs**, only a tight single-row crop
does (verified on the `难道······他 才是` regression fixture).

Word bboxes are intentionally NOT rewritten — they correspond to the
glyphs Lens did detect. The mask builder
(`lens_native._build_glyph_mask`) handles the recognition gap on its
own via row-gap-aware mask stretching (next section).

Constants: `_ROW_GAP_SHORT_RATIO = 0.5`, `_ROW_GAP_MIN_LINES = 3`,
`_ROW_REOCR_MIN_DIM = 200`, `_ROW_REOCR_PAD_Y_FRAC = 0.45`,
`_ROW_REOCR_PAD_X_PX = 6`.

---

## Grouper: class detection per block

Before any merge, each Lens block becomes one `BubbleGroup` with a class
attached. Classes drive `shape_kind` (which the renderer reads for halo
intensity and fit tolerance) and erase-mask dilation.

`_classify_block` in `lens_native.py`, ordered by specificity:

1. `|rotation_deg| > 5°` → **sfx**. Only SFX get angled typesetting in
   manga (survey: 4/4 angled blocks were SFX).
2. `char_count ≤ 10 AND aspect (w/h) ≥ 1.4` → **sfx**. Short text in
   a wide bbox is the canonical horizontal SFX pattern.
3. `char_count > 30` → **narration**. Long captions, no bubble.
4. Default → **dialogue**.

Class profiles (`_PROFILES`):

| Class | shape_kind | dilate fraction | max dilate px |
|---|---|---|---|
| sfx | burst (glow halo) | 0.08 | 20 |
| dialogue | dialogue (plain) | 0.04 | 14 |
| narration | dialogue | 0.06 | 18 |

### Glyph mask: word union + row-gap stretch

`_build_glyph_mask` in `lens_native.py` builds the per-block erase mask:

1. **Word union base**: paint each `WordBox` bbox into a block-local
   uint8 buffer; same for an expanded-support buffer (`in_bounds`)
   that pads each word by `~12%/18%` of font size on x/y. This
   mirrors `koharu.refine_segmentation_mask`.
2. **Row-gap stretch**: for each row flagged by
   `_suspicious_line_indices` (same signal as detector recovery),
   paint a full-block-width band into both `base` and `in_bounds` at
   the line's y-range. This wipes glyphs Lens dropped around dense
   decoration. Necessary even after text recovery because re-OCR
   leaves word bboxes untouched (they still only cover the
   originally-recognised glyphs).
3. **Dilate + clip**: dilate `base` by `~10%` font size, intersect
   with `in_bounds` — mask grows around glyphs but cannot bleed past
   per-word/row headroom.

Falls back to a full block rect when `block.words` is empty.

---

## Direction inference (`_infer_text_direction`)

Two signals, in order:

1. **Strict aspect**: `h > w * 2.0`. Unambiguous tategaki column.
   Empirical survey on chapter 112 (10 JP blocks): aspect h/w ranges
   from 3.4 to 10.7 — all clearly above 2.0.
2. **CJK script + `h > w`**. Catches short 1–2 character columns
   (e.g. 「滚」, 「前辈」) where the strict aspect rule misses. CJK
   script = Hiragana, Katakana, CJK Unified Ideographs, Extension A,
   Compatibility Ideographs. Japanese and Chinese manga are the
   dominant tategaki sources; Korean / Latin / Vietnamese in manga are
   typeset horizontally, so CJK is a reliable secondary signal.

For non-CJK horizontal prose the bbox is usually wider than tall, so
neither rule fires.

---

## Tategaki column merging (`_merge_tategaki_columns`)

A chain-clustering algorithm with two structural guards. Replaces the
older union-find approach which used a fixed 100px x-gap threshold.

### Step 1 — chain clustering

```text
vertical_columns = [g for g in groups if g.text_direction == "vertical"]
sort vertical_columns by x_centre DESCENDING       # tategaki reads RTL

clusters = []
for column in sorted order:
    # Evaluate every open cluster; join the one with the strongest
    # y-overlap that also passes the x-gap budget. Evaluating ALL
    # open clusters (not just the last) handles interleaving — a
    # column from a different bubble sitting between two clusters by
    # x doesn't fork the right one.
    best = max(c for c in clusters if compatible(column, c))
            by max y_overlap_ratio(column, member) for member in c
    if best exists:
        best.append(column)
    else:
        clusters.append([column])
```

`compatible(column, members)` requires:

1. `y_overlap_ratio(column, m) >= 0.50` for at least one `m` in the
   cluster (overlap against the **closest** member, not the union, so a
   tall outlier doesn't dominate).
2. Edge-to-edge `x_gap(column, leftmost(members)) <= gap_cap`, where
   `gap_cap = max(80, 2.0 × min(width_a, width_b))`. The cap scales with
   column width so 4K+ pages with proportionally wider columns and gaps
   are still caught, while the 80px floor protects low-res pages.
3. Gap may be slightly negative (`-2`) to tolerate 1px overlap from
   rounding.

The cluster extends **leftward** (RTL reading order), and the x-gap is
measured against the leftmost current member.

### Step 2 — cluster guards

Two post-checks. A failing cluster falls back to singletons (the
original 1-column groups), never to a partial merge.

**Font guard** (`_passes_font_guard`):

```text
for each member, derive glyph size:
  - if vertical: bbox width   (column width ≈ glyph width)
  - else:        typesetting.font_size_px

if max(sizes) / min(sizes) > 1.8: reject
```

Reasoning: same-bubble columns share a glyph size; cross-bubble columns
typically don't. We deliberately do **not** use
`typesetting.font_size_px` for vertical members because Lens reports one
"line" per tategaki column, so its line height equals the column height
(glyph size × character count), not the glyph size itself.

**Outsider guard** (`_passes_outsider_guard`):

```text
cluster_bbox = union of member bboxes
for each vertical column NOT in this cluster:
  centre = (x_centre, y_centre) of column
  if centre is inside cluster_bbox: reject
```

Reasoning: the cluster bbox spans only its members. If another vertical
column's centre lies inside it, the cluster crossed a bubble boundary —
some bridge member has y-range that pulls the bbox over a neighbouring
bubble.

**Safety cap**: clusters with > `_MAX_COLUMNS` (=6) members are also
dropped to singletons. Real bubbles rarely exceed 6 columns; larger
clusters are almost always artefacts.

### Step 3 — merge surviving clusters

`_merge_group_list`:

```text
bbox          = union of member bboxes
text          = "\n".join(member.text for m sorted by -x)   # RTL
text_masks    = concatenated tuple of all member text masks
erase_masks   = concatenated tuple of all member erase masks
shape_kind    = members[0].shape_kind                      # all vertical → dialogue
rotation_deg  = members[0].rotation_deg
typesetting:
  line_count        = sum of member line counts
  font_size_px      = from the member with most lines
  avg_chars_per_line = total_chars / total_lines
text_direction = "vertical"
```

---

## Mask layers: text vs erase

Each `BubbleGroup` carries two mask tuples with distinct consumers:

| Field | Consumer | Purpose |
|---|---|---|
| `text_masks` | render | "where do glyphs sit" — drives font fit and per-glyph layout. Tight; multiple components per block are fine. |
| `erase_masks` | AOT-GAN inpaint | "what to wipe from canvas" — must cover stroke anti-alias and any glyphs OCR may have missed in a row. Should be a **single connected component per bubble** so AOT inpaints a contiguous patch. |

Both are derived from the same `_build_glyph_mask` base (word union +
row-gap stretch + dilate + clip). The erase variant is then dilated
isotropically by `class.profile.erase_dilate_fraction × font_size`
(capped). The class profile sets the multiplier:

- **sfx** — 0.08 / 20px cap (glow halo, big surrounding area)
- **dialogue** — 0.04 / 14px cap (clean bubble background)
- **narration** — 0.06 / 18px cap (caption on art, mid)

The row-gap stretch step (block-local, see "Glyph mask" above) is what
keeps the erase mask **single-component** even when Lens drops glyphs
mid-row. Without it, the dilate alone couldn't bridge a 100+ px gap
between word bboxes and AOT would leave the dropped glyphs visible.

On both probe fixtures every block's erase mask is exactly 1 connected
component, verified by `connectedComponentsWithStats`.

---

## Considered and rejected

Two ideas evaluated during this iteration that did **not** ship,
recorded so they're not re-litigated:

### Re-OCR the entire block (instead of single row)

When recovering rows Lens dropped glyphs on, one option is to
re-process the full block crop. Probed in
`scripts/probe_lens_reocr_block_vs_row.py` on the `难道······他 才是`
fixture:

| Variant | Crop | Scale | Output for missed row |
|---|---|---|---|
| Block native res | 206 × 124 | ×1 | `才是` (still misses) |
| Block upscaled | 824 × 496 | ×4 | `才是` (still misses) |
| **Row upscaled** | 1010 × 235 | ×5 | `难道………………他才是` ✓ |

Resolution is not the problem — **context is**. Given multi-line
input, Lens's recognizer keeps dropping glyphs around the dense
decoration run. Only when the crop is reduced to a single row does
Lens segment glyph-by-glyph and recover. Row recovery is therefore
mandatory; full-block recovery isn't a viable alternative.

### `cv2.morphologyEx(MORPH_CLOSE)` on erase mask

Proposed as defence against gaps between word bboxes that survived
the isotropic dilate (font-large + sparse-word case). Probed in
`scripts/probe_lens_close_erase.py` across all 19 blocks of the two
probe fixtures:

- Component count: **1 → 1** on every block (already single-component
  after the existing dilate).
- Pixels added: 0–8 per block (sub-pixel noise; invisible at output
  resolution).

Conclusion: a no-op on real data. Not added — YAGNI. The trigger to
revisit would be a render fixture showing visible black artefacts
between glyphs after AOT (font ≥ 60px + word gap > 2 × dilate radius,
which the current radius cap of 20px would only fail on extreme SFX).

---

## Tunable constants

All in `lens_native.py` unless noted.

| Constant | Default | Meaning |
|---|---|---|
| `_SFX_MAX_CHARS` | 10 | classify ≤ this as SFX if aspect wide |
| `_SFX_MIN_ASPECT` | 1.4 | aspect threshold for short→SFX rule |
| `_SFX_ROTATION_OVERRIDE` | 5° | any block tilted past this is SFX |
| `_SHORT_MAX_CHARS` | 30 | dialogue/narration boundary |
| `_Y_OVERLAP_MIN` | 0.50 | min y-overlap fraction inside cluster |
| `_X_GAP_FLOOR_PX` | 80 | absolute floor for x-gap budget |
| `_X_GAP_WIDTH_MULT` | 2.0 | x-gap budget multiplier on min column width |
| `_FONT_RATIO_MAX` | 1.8 | max glyph-size ratio inside cluster |
| `_MAX_COLUMNS` | 6 | safety cap on cluster size |
| `_ROW_GAP_SHORT_RATIO` | 0.5 | suspicious row width ratio vs median (mask + detector) |
| `_ROW_GAP_MIN_LINES` | 3 | minimum lines per block to apply row-gap heuristics |
| `_CROSS_COLUMN_MIN_LINES_ABSORBED` | 2 | in `lens_blocks.py` |
| `_CROSS_COLUMN_LINE_INSIDE_RATIO` | 0.70 | in `lens_blocks.py` |
| `_ROW_REOCR_MIN_DIM` | 200 | upscale target for row recovery crops (`lens_blocks.py`) |
| `_ROW_REOCR_PAD_Y_FRAC` | 0.45 | vertical padding around row crop (`lens_blocks.py`) |
| `_ROW_REOCR_PAD_X_PX` | 6 | horizontal padding around row crop (`lens_blocks.py`) |
| `_LENS_LANG_HINTS` | ja, zh-Hans, zh-Hant, ko, vi | source-lang → Lens `ocr_language` map |

Empirical bases:

- Aspect h/w for tategaki: 3.4–10.7 (ch112 JP survey, 10 blocks). 2.0
  is well below the floor.
- Same-bubble column glyph-size ratio: within ~10% (consistent typeset).
  Cross-bubble ratio: typically ≥ 1.8× when bubbles have distinct sizes.
- X-gap absolute floor 80px holds for ≤ 2K pages. Width-scaled term
  takes over above ~40px-wide columns.

---

## Debug probes

Several probes live under `scripts/probe_lens_*.py`. Each writes to
`debug-runs/` and is the primary verification tool when changing the
matching code path.

| Script | Purpose | Artefacts |
|---|---|---|
| `probe_lens_bubble.py <image>` | End-to-end detect + group + mask overlay; the default smoke test. | `detect.{png,json}`, `group.{png,json}`, `masks.png`, `merge_edges.json`, `summary.json` |
| `probe_lens_rowgap.py` | Visualises mask variants A (current), B (row-aware), C (full block) on the probe2 problem block. | `probe_A_rowgap.png` |
| `probe_lens_reocr_row.py` | Re-OCRs a suspicious row crop with Lens; prints before/after text. | `probe_reocr_row.png` |
| `probe_lens_reocr_block_vs_row.py` | Compares full-block re-OCR (native and ×4 upscale) vs single-row upscale. Documents why row beats block. | console output |
| `probe_lens_reocr_lang.py` | Tries multiple `ocr_language` hints on the same row crop. Documents `_lens_lang_hint` choice. | console output |
| `probe_lens_fullpage_lang.py` | Compares full-page detect with auto vs explicit hint across two fixtures. | console output |
| `probe_lens_close_erase.py` | Quantifies the effect of `MORPH_CLOSE` on erase masks (component count, delta pixels). | `probe_closing_erase.png` |

Run any with `python scripts/<probe>.py`. The bubble probe takes an
image path; the others operate on the fixtures under
`debug-runs/lens_bubble_probe{,2}/source.png`.

Always visually diff the output of `probe_lens_bubble.py` against a
known-good run before merging changes to detector filters or grouper
merge rules.

---

## Test fixtures

Synthetic tests live in:

- `tests/test_lens_filters.py` — detector filters including
  `cross_column` rejection and rotation-aware AABB.
- `tests/test_lens_row_recovery.py` — language hint map +
  `_suspicious_line_indices` + row re-OCR splice (stubbed Lens API).
- `tests/test_lens_grouper.py` — class detection per block.
- `tests/test_lens_grouper_tategaki.py` — direction inference, chain
  compat, font + outsider guards, mixed direction.
- `tests/test_lens_grouper_words.py` — per-word mask building.
- `tests/test_lens_grouper_typesetting.py` — `TypesettingHint` plumbing.
- `tests/test_lens_grouper_refined_mask.py` — glyph mask refinement
  including row-gap stretch.
- `tests/test_lens_block_classification.py` — class boundary cases.

All tests are network-free; the detector tests use synthetic
`TextBlock`s, not real Lens calls.
