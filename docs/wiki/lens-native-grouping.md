# Lens-native grouping

Read this before touching `typoon/vision/detectors/lens/` or
`typoon/vision/groupers/_spatial_join.py`.

The `lens` preset is the only active vision stack. PP-OCR + YOLO,
Bing reverse-image, row-gap recovery, and the `manga_ocr` / `offline`
presets were removed during the lens refactor. The only sibling preset
still wired is `ctd_manga` (offline JP via CTD).

```text
RGB page
  └─ LensBlocksDetector  (typoon/vision/detectors/lens/)
       ├─ Phase A: tile_pass.run   coarse Lens OCR on 900-tall tiles
       │     dedup → filters.apply → TextBlock[]
       ├─ Comic-DETR side-detector (parallel)
       └─ Phase B: bubble_pass.run  re-OCR DETR anchors that Phase A
             left EMPTY or PARTIAL (gap > 0.7 × line height)
  └─ LensNativeGrouper  (groupers/lens_native.py)
       └─ spatial_join  cluster DETR anchors, container polygon,
                        per-member oriented mask
```

Entry points:
- `LensBlocksDetector.detect(image, lang) → DetectionResult`
- `LensNativeGrouper.group(image, detection, lang) → tuple[BubbleGroup, ...]`

## Detector — two phases

### Phase A: `tile_pass.run`

Tiles the page 900 px tall × 200 overlap, OCRs every tile, dedups
across overlaps by length-weighted bbox containment, then runs
`filters.apply`:

- `tiny_bbox` — < 25 × 18, < 700 px²
- `decoration_only` — no letter / digit
- `huge_bbox` — area / char > 6000 (art region)
- `cross_column` — paragraph whose lines sit inside ≥ 2 other
  paragraphs (tile-boundary phantom that fuses tategaki column tails)

`text_direction` from Lens proto is mapped to page-aligned axes
(`writing_direction` projected through `rotation_deg`). **The value
is unreliable for tilted tategaki** — Phase B and the grouper both
re-detect direction from word geometry rather than trusting this
field.

### Phase B: `bubble_pass.run`

Walks Comic-DETR anchors (`text_bubble` > `bubble` > `text_free`,
clusters dedup'd by IoU > 0.5) and classifies each:

- `COMPLETE` — Lens word_union covers the anchor (no edge gap)
- `EMPTY`    — no Lens block centre inside
- `PARTIAL`  — block(s) inside but max edge gap > 0.7 × median line h

Only EMPTY + PARTIAL anchors trigger a re-OCR HTTP call. The crop is
upscaled to ≥ 200 px short side; recovered blocks replace the Phase-A
members of that anchor.

This pass is what fixes:
- bubbles Lens missed entirely
- bubbles where Lens dropped glyphs at one edge (top/bottom/left/right
  partial coverage)
- bubbles where Lens scrambled tategaki reading order (full re-OCR on
  the bubble crop gives Lens enough context to get direction right)

## Grouper — spatial join

### Anchor reduction

Comic-DETR routinely emits 2–3 overlapping boxes per balloon. We
cluster all regions by IoU > 0.7 and pick one anchor per cluster with
precedence:

```
text_free  >  bubble  >  text_bubble
```

`text_free` wins whenever it appears — 3-class overlap (bubble +
text_bubble + text_free) is the canonical caption signature, not a
balloon. `text_bubble`-only clusters are dropped (Lens word geometry
is tighter than the DETR inner rect).

### Block assignment

Anchors walk innermost-first (smallest area). Each anchor claims
unassigned Lens blocks whose centre falls inside its bbox. Remaining
blocks become singleton groups.

### Shape kind — DETR `bubble` overrides per-block SFX

`_derive_shape_kind` picks `dialogue` whenever the cluster anchor is
`bubble`, regardless of what `classify_block` says about individual
members. The per-block rule (`rotation > 5°` → SFX) is tuned for sound
effects; tategaki bubbles tilted for art style trip it and collapse to
a tight burst OBB even when DETR clearly outlined a speech balloon.
DETR `bubble` is the strongest available "this is a balloon" signal —
trust it so the text_bubble hint + ellipse path can fire.

Without an anchor we keep the previous behaviour (any SFX member
promotes the group to `burst`); singletons keep `block_to_group` /
`classify_block` decisions because they have no DETR confirmation.

### Container polygon vs. mask — DIFFERENT shapes

Two outputs per group, both derived from Lens geometry (or a DETR
text_bubble hint when available), **never** from the outer DETR bubble
bbox or pixel border-scan:

| | container polygon | text_masks / erase_masks |
|---|---|---|
| consumer | Rust render fit | AOT-GAN inpaint |
| base shape | inscribed ellipse in `text_bubble` (dialogue + sane hint) <br> or OBB / word_union per group | OBB / word_union per member |
| padding | `0.20 × median_glyph_short` dialogue, `0.08 ×` burst (min 4 px) | `0.08 × median_glyph_short` (min 1 px) |
| orientation | OBB when tilted non-column | OBB when tilted non-column |

When a `bubble` cluster has a co-located `text_bubble` (matched by
containment ≥ 0.80, not IoU — the outer bubble box includes the
balloon tail so IoU peaks ~ 0.5 even for a perfect inner rect), the
text_bubble rect becomes the container hint. The hint is **union-expanded
with `word_union`** before use so the result always covers every glyph
Lens read; DETR jitter that leaves a few pixels of glyph outside the
raw text_bubble can no longer clip text. Dialogue + sane merged hint +
aspect in `[0.4, 2.5]` → 24-vertex inscribed ellipse so the translated
string wraps to the balloon's rounded interior. SFX and column
tategaki keep the tight OBB path.

Hint guard (`_hint_is_sane`): dialogue only; merged-hint area
≤ 4 × word_union area (rejects DETR over-shoots onto neighbouring
captions). Coverage / `hint ≥ word_union` guards are implicit from the
union expansion.

Both share the same OBB construction (`_group_obb` / `_word_axis_obb`
→ centres + median glyph) when the hint is unavailable — single source
of truth for orientation and base extent; only padding differs.
Container > mask just enough to give translated text breathing room
without crowding neighbour balloons.

text_masks and erase_masks are the **same TextMask tuple** — there is
no separate glyph-aware mask. Render uses container.polygon for fit;
the per-member mask shape doesn't matter to it.

### Rotation axis — measured, not Lens-reported

`block.rotation_deg` from the Lens proto is unreliable: tategaki
columns regularly report `rot ≈ 0` while the page itself is tilted, and
horizontal SFX scrambled by tile-edge artefacts can falsely report
non-zero angles.

`_is_column_layout(members)` measures the **actual** word axis: sort
word centres by x, take vector from first to last centre. `|dy| >= |dx|`
→ column layout (tategaki) → mask is axis-aligned AABB (Lens already
projected rotation through `geometry.norm_bbox`, no further rotation
needed).

`_word_axis_obb(member, pad)` builds the rotated rect from the same
vector for non-column tilted text:

- long axis = first→last centre vector, extended by `half_glyph + pad`
  at each end
- short axis = `median(min(word_w, word_h))` (the glyph's cross-axis
  dimension), centred on the axis line, ± `half_glyph + pad`

Projecting raw word corners onto the OBB frame inflates the rect —
Lens word bboxes are axis-aligned AABBs of rotated glyphs, so their
corners sit well outside the visible glyph stripe. **Use centres +
median glyph, never corner projections.**

### Why `block.bbox` for mask fallback is wrong

`block.bbox` is the **paragraph** AABB Lens computes from its rotated
corners. For tategaki it's often visibly larger than the word union
because Lens pads the paragraph for the rounded balloon shape. Always
fall back to `_word_union([m])`, never `m.bbox`, when no OBB is
available — see the regression where panel 4 of the happymh2 probe
showed pink magenta tategaki masks bleeding past the bubble outline.

## What was removed (don't reintroduce)

- `_recover_row_gaps` / `_suspicious_line_indices` — Phase B re-OCR
  covers this case with full context.
- `_empty_bubble_recovery` (standalone) — folded into `bubble_pass`
  as the `EMPTY` action.
- `_glyph_mask.py` (koharu-style word-union + dilate refine) —
  filled-rect masks ship the same erase coverage with less code.
- `border::detect_edge_insets` in Rust — container polygon is now the
  drawable area directly. No `bubble_mask` argument, no
  `BubbleIndex::expand_area`.
- `BubbleGroup.fit_box` / `erase_box` / `text_box` — `polygon` IS the
  fit/erase/text area.
- `text_direction` from Lens proto as a routing signal — use
  `_is_column_layout` instead.
- PP-OCR + YOLO + Bing detectors and `manga_ja`/`offline`/`bing`
  presets.

## Probe

```
python -m scripts.probes.lens_group <image.png> [--out <dir>] [--lang ja]
```

Output: `source.png`, `raw.json` (Lens + DETR), `overview.png` (2×2 grid):

1. Lens raw — word ⊂ line ⊂ paragraph + word-axis polygon for tilted
2. Comic-DETR regions + empty-bubble flag
3. Render polygon per group (ellipse when bubble+text_bubble fit fires,
   OBB when tilted, AABB otherwise)
4. text/erase mask (= the same TextMask tuple) over source

After the probe, audit cross-group container overlap:

```
python -m scripts.probes.lens_group.audit_overlap debug-runs/<run-id> ...
```

Writes `overlap_audit.png` (red intersection over translucent green
containers) and `overlap_report.json` (`verdict: ok | warn | fail`,
per-pair ratios). Verdict thresholds: warn ≥ 1%, fail ≥ 5% of the
smaller container's area. Container overlap is a hard regression
signal — re-run before merging any change that touches
`_container_polygon`, padding factors, or hint guards.

The probe modules under `scripts/probes/lens_group/` mirror the
production geometry helpers — when you change `_word_axis_obb` or
`_is_column_layout`, update the probe too so panel 1 stays honest.
