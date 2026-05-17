# Render container fit — what is correct and what is wrong

Read this before touching:
- `typoon/vision/groupers/_spatial_join.py` — anchor reduction,
  container polygon, calibrated-pad post-pass.
- `crates/render/src/{fit,layout,overlay}.rs` — fit binary search,
  drawable area, glyph painting.
- `typoon/stages/render.py` — page-level render driver.

This wiki is the production playbook. Every rule below was learned
from a regression on real fixtures (happymh, happymh2, happymh3,
wowpic1, mangabuzz). Do not re-derive — the cost of doing so is one
of those regressions reappearing.

## Foundation: Lens is the source of truth

Every fit / pad / hint / mask decision derives from **Lens output**:

| What | Source |
|---|---|
| Container polygon | Lens word_union (+ optional DETR text_bubble hint) |
| Font size hint | Lens word-bbox shorter side median (`_typesetting_hint`) |
| Reading order | Lens block sequence (tategaki RTL, yokogaki LTR) |
| Mask | Lens word geometry + small dilation |

DETR `bubble` / `text_bubble` / `text_free` are **anchors for grouping**,
not geometry. When DETR is reliable (text_bubble matches with bubble)
it acts as a hint to expand the container around the balloon
interior. When DETR misses or over-shoots, Lens word geometry alone
still produces a correct render.

**Never invent geometry the detector did not see.** Resizing a polygon
based on a ratio computed-from-Lens "to match ideal" is fine. Resizing
based on a hardcoded aesthetic ("balloon should be 3:1 aspect") is
not.

## Three things that must scale atomically

When you change a container, you must change all three together OR
prove the invariant `polygon ⊇ word_union ⊇ mask` still holds:

1. **`BubbleGroup.polygon`** — drawable area passed to the Rust fitter.
2. **`BubbleGroup.erase_masks`** — tight per-member rectangles, glyph
   stripe + small dilation. AOT inpaint operates on these.
3. **`BubbleGroup.typesetting`** (Lens hint) — fit ceiling for the
   binary-search font picker.

The calibrated-pad post-pass only **expands** polygons, never shrinks.
Expand-only preserves `polygon ⊇ word_union` automatically and keeps
the mask inside the polygon. If you ever introduce a shrink path you
must rebuild masks atomically — there is no shortcut.

Failure modes when you violate this:

| Operation | Polygon | Mask | Hint | Visible symptom |
|---|---|---|---|---|
| Polygon shrink, mask unchanged | smaller | unchanged | unchanged | Mask leaks **outside** polygon → cross-group overlap audit fails. Magenta neighbour bleed. |
| Polygon expand, mask unchanged | bigger | tight | unchanged | OK. Mask stays inside; render writes into the new area. AOT erases what it needs. |
| Polygon swap (ellipse → AABB) | shape kind changed | unchanged | unchanged | Renderer's ellipse-fit scale (0.85) stops applying → text overflows the rounded corners. |
| Hint changed without polygon | unchanged | unchanged | new | Fit picks wrong ceiling for the actual polygon size. Adjacent panels go inconsistent. |

## Anchor precedence

```
text_free  >  bubble  >  text_bubble
```

The cluster's representative anchor comes from this order — class
beats confidence within the cluster.

- `text_free` wins whenever it appears — the 3-class overlap (bubble
  + text_bubble + text_free) is the canonical **caption** signature.
  Captions never get the inscribed-ellipse path.
- `bubble` wins when it appears without text_free. If a `text_bubble`
  is co-located (containment ≥ 0.80), its bbox becomes the inner-rect
  hint for the merged-ellipse container.
- `text_bubble` standalone wins when DETR missed the outer balloon
  outline. Its bbox doubles as the inner-rect hint so the same
  merged-ellipse path fires — without this fallback, two tategaki
  strands sharing one balloon become adjacent singletons that overlap
  each other (mangabuzz 374/1/9 `離せっ` + `一人で歩けるっ`).

When the precedence collapses (no anchor at all), each Lens block
becomes a singleton group.

## The hint sanity guard (real signal, not heuristic)

`_hint_is_sane` decides whether a DETR text_bubble hint expands the
container around the balloon or is rejected back to the word_union +
pad fallback.

**Use centre-containment, not area ratio.**

Production rule:

> The merged hint must not contain the **centre of any block outside
> this group**. If it does, DETR over-shot onto a neighbouring balloon.

Sparse-text balloons routinely have hint-area / word_union-area
ratios of 4–10× because the glyph is one or two characters in a
normal-sized balloon (single-character grunts: `くそっ`, `?`, `…`).
The earlier `4 × word_union` cap killed these legitimate hints and
forced the fitter into a tiny word_union pad — yellow polygon
miniature, render fit way smaller than the balloon.

The `100.0×` cap that remains is emergency-only: catches DETR
returning the full page when no outsider block happens to lie inside.
Centre-containment is the actual guard.

Worked example (happymh group `くそっ`):
- DETR text_bubble: `64 × 83`
- word_union: `19 × 61`
- ratio: 5,312 / 1,159 ≈ 4.6×
- Old cap (4×): **reject** → fallback to word_union + pad (27 × 26)
- New centre check: no outsider block inside → **accept** → polygon ≈
  text_bubble (65 × 84). Yellow now matches magenta in the probe.

## Calibrated pad — per-page glyph/container ratio

Manga letterers maintain a consistent **ratio** between glyph height
and balloon interior across the page. Big balloons get big fonts,
small balloons get small fonts — same proportion. We recover that
ratio from groups that DO have DETR hints (SOURCE), then apply it to
groups that DON'T (FALLBACK).

### Pool

Only groups where `_container_polygon` accepted the hint contribute
samples:

```
ratio_sample = glyph_short / polygon_short
```

Filter `0.05 ≤ ratio ≤ 0.45` (drops decoration / partial OCR below,
SFX bleed above). Trimmed mean 10–90% over filtered samples. Require
≥ 3 samples; below that the ratio isn't trustworthy, fall back to
heuristic pad.

SFX (burst) **is excluded** from the body pool. SFX glyphs fill a
much higher fraction of their balloon — pooling them would skew the
body cap. (A separate SFX pool exists in the contract but isn't
applied yet; SFX containers stay heuristic.)

### Apply

For each FALLBACK dialogue group:

```
target_short = glyph_short / body_ratio
if target_short < 20 px:           skip — balloon too small
if target_short ≤ current_short:   skip — already big enough
new_polygon = expand_short_axis(word_union, target_short)
if new_polygon overlaps any neighbour bbox: reject
```

Constraints:
- **Short axis only.** Long axis stays at word_union extent. Aspect
  preserved by the longer axis — never force a square.
- **Expand only.** Shrinking would risk clipping the mask (mask is
  inside word_union which is inside the polygon).
- **Rolling-state neighbour check.** Each iteration sees the latest
  bbox list; two neighbours can't both grow into each other's slack.
- **Walk top-down** (deterministic order on bbox.y) so the result is
  reproducible across runs.
- **Atomic rebuild**: only polygon + bbox change via
  `dataclasses.replace`. Mask and typesetting hint stay as-is — the
  invariant holds because we only expanded.

Worked example (mangabuzz `! = / = !`):
- glyph_short: 11
- body_ratio: 0.20 (page-level trimmed mean)
- target_short: 55 px
- word_union: 26 × 44
- current polygon: 27 × 44 (word_union + 0 pad after clip)
- new polygon: 55 × 63
- no neighbour overlap → accept

## Rust fitter — what the polygon means at fit time

`crates/render/src/fit.rs` consumes the polygon plus a hint:

1. `DrawableArea::from_polygon` — extracts AABB, oriented dimensions,
   and an `is_ellipse` flag (polygon > 4 vertices → 24-vert inscribed
   ellipse). `size()` applies a `0.85` scale when `is_ellipse=true`
   so centred text doesn't poke through the ellipse curve at top/
   bottom (the AABB of an ellipse includes the four corners that lie
   outside the curve).

2. `fit_page_areas` computes one `page_body_cap` from the median of
   in-range per-bubble hints (10–60 px window). Every bubble's
   `fit_area` then uses this single number as `hi_bound`.

3. `fit_area` binary-searches font size within `[MIN_FONT_SIZE,
   hi_bound]`. Source font size never inflates — only shrinks when the
   translated text won't wrap into the polygon at the cap.

4. `HINT_MAX_GROWTH = 1.15` — production grew slightly above source
   size for VI legibility (stacked tone marks need vertical room).

**Why per-bubble hint as-is, not ratio × box_short:**

I tried using a per-bubble ratio-derived hint (each bubble gets its
own ratio × its own polygon_short). It was unstable. The page-median
cap is the correct mechanism because manga letterers pick ONE body
size per page; bubbles only shrink below that cap, never inflate
above. The calibrated-pad pass already handles per-bubble scaling
by adjusting the **polygon**; doing it again at the hint level
double-counts.

## Hyphen / dash break opportunities

`layout.rs::wrap_text` splits at `- – — _ \u{00AD}` inside a word so a
hyphenated compound (`28-Shōnan`) can wrap as `28-` + `Shōnan`
instead of forcing a tiny font. Hyphen stays attached to the LEFT
chunk.

`fit.rs::needs_char_break` uses `longest_atom_width` (atom-level
measurement, not whole-word) so the binary search isn't blocked by a
long hyphenated token.

## Romaji macron strip

`typoon/stages/translate.py::_normalize_for_render` performs `NFD →
strip U+0304 (combining macron) → NFC` before render. The embedded
font has no Latin Extended-A precomposed macrons; without the strip,
`Shōwa` shapes to glyph 0 (no outline, full advance) and renders as
"Sh wa". Vietnamese never uses U+0304 — strip is lossless.

## Mask pad per shape

`_MASK_PAD_FACTOR` is shape-kind-aware:

- `dialogue`: 0.20 × glyph_short
- `burst`: 0.08 × glyph_short

Matches `_CONTAINER_PAD_FACTOR` exactly so the mask is never wider
than the container polygon. Burst SFX use the tighter factor because
tilted SFX with the 0.20 factor produced visible cross-group overlap
on dense pages (audit happymh2 #3∩#4).

`_MASK_PAD_MIN_PX = 2` covers Lens word-bbox under-coverage on
diagonal strokes / ascenders / diacritics, plus anti-aliased glyph
edges. Below 2 px the mask leaves a faint ghost after AOT inpaint.

## Probes — verify before touching

Two probes, one purpose: verify the geometry before any production
change.

### `scripts/probes/lens_group/`

Single-page 2×2 overview + overlap audit.

```
python -m scripts.probes.lens_group <image> --out debug-runs/<name>
python -m scripts.probes.lens_group.audit_overlap debug-runs/<name>
```

Panels:
1. Lens raw (word ⊂ line ⊂ paragraph)
2. Comic-DETR regions (3 classes)
3. Container polygon per group (yellow polylines — what render sees)
4. Mask overlay (filled erase area)

Audit thresholds: warn ≥ 1%, fail ≥ 5% of smaller polygon area.
Container overlap is the hard regression signal.

### `scripts/probes/calibrated_pad/`

Reports per-group calibration role + proposed pad, plus the page
ratio.

```
python -m scripts.probes.calibrated_pad <image> --out debug-runs/<name>
bash scripts/probes/calibrated_pad.sh                                  # full corpus
```

Overlay colours:
- **Magenta** — DETR text_bubble raw bbox (ground truth, SOURCE
  groups only)
- **Yellow** — production current polygon (what render uses)
- **Cyan** — proposed polygon for FALLBACK groups (preview only)
- **Grey** — Lens word_union AABB

Yellow ≈ magenta on SOURCE groups means the hint sanity guard passed.
Yellow noticeably smaller than magenta means a hint was rejected —
investigate the centre-containment guard.

### `scripts/probes/full_page/`

End-to-end single-image render: detect → group → translate → erase →
render. Writes the full artifact tree under `debug-runs/<name>/`.

```
python -m scripts.probes.full_page <image> --out fp_<name> --target-lang vi
python -m scripts.probes.full_page <image> --out fp_<name> --no-translate  # stub
```

The `--no-translate` flag echoes source text so erase + render can be
verified without an LLM round trip.

## Forbidden patterns (will regress)

```
shrink polygon without rebuilding mask
swap ellipse polygon for AABB without re-checking ellipse fit
mask_pad > container_pad
hint area-ratio cap (use centre containment)
per-bubble ratio × box_short as hint (use page-median cap)
forcing square aspect on expanded fallback polygons
chapter-level state in scan/group (page-local pipeline rule)
ignoring overlap audit before committing geometry changes
"tall-narrow swap" that lies to the fitter about width/height
HINT_MAX_GROWTH > 1.20 (VI inflates visibly beyond source)
mask pad < 2 px floor (ghost after AOT inpaint)
```

## When to expand the system

New container case (e.g. a fifth DETR class)? Walk this checklist:

1. Does it produce a hint bbox? → goes through `_hint_is_sane`.
2. Does it qualify for the body ratio pool? → set `used_hint=True`
   only when the bbox is ground-truth balloon geometry (not e.g.
   text_free caption).
3. Does it have a special shape (caption ≠ balloon)? → branch
   `_derive_shape_kind` and `_apply_calibrated_pad` skip.
4. Update `lens-native-grouping.md` precedence table.
5. Add a fixture + probe before committing.

New target language (e.g. KO)? Walk this checklist:

1. Glyph metrics differ → review `HINT_MAX_GROWTH`. Hangul stacks
   higher than Latin → may need 1.10 instead of 1.15.
2. Add the script to `_normalize_for_render` macron-equivalents (KO
   uses combining tone marks differently — check NFD output).
3. Font coverage → audit cmap for the script's range.
4. Test with the corpus's hardest pages (dense + sparse text mix).

## What I tried that did NOT work (do not retry)

1. **Per-bubble ratio × polygon_short as Rust FitHint.font_size_px.**
   Unstable because Lens hints vary more across a single page than
   real source font sizes vary. Use page-median cap instead.

2. **Resize SOURCE polygons to match the page ratio.** Their polygon
   IS the ground truth; resizing destroys the DETR signal we
   calibrated against. Only FALLBACK groups get resized.

3. **Shrink FALLBACK polygons when `target_short < current_short`.**
   word_union is the floor; anything smaller clips the glyph stripe.
   Expand-only.

4. **Aspect-preserving expansion** (scale both axes by the same
   factor). Tategaki word_union 30×200 with target 100 → 100×666 is
   nonsense. Expand short axis only; long axis follows word_union.

5. **Static pre-normalize neighbour list.** Two adjacent FALLBACK
   bubbles both pass the static check, both grow, end up overlapping.
   Use rolling-state neighbours.

6. **One median-of-medians (page hint cap) feeding ratio-of-medians
   (per-bubble target).** Double-aggregation. The page cap is the
   final mechanism; if you also resize polygons by ratio, you
   double-count the calibration.

7. **Container normalize phase that touches polygon only and trusts
   the mask to follow.** Mask must scale atomically OR you must prove
   the invariant `polygon ⊇ word_union ⊇ mask` holds (expand-only is
   the only proof-by-construction).

8. **`_HINT_AREA_RATIO_MAX = 4.0` as the hint guard.** Sparse-text
   balloons in normal-sized bubbles fail. Use centre-containment.

9. **Per-chapter ratio aggregation.** Violates the page-local
   pipeline rule. Per-page flat-pool words with trimmed mean is
   stable enough on real fixtures.

## Tunable constants — current production values

In `typoon/vision/groupers/_spatial_join.py`:

| Constant | Value | Rationale |
|---|---|---|
| `_CONTAINER_PAD_FACTOR["dialogue"]` | 0.20 | Translated text usually longer than source; needs breathing room |
| `_CONTAINER_PAD_FACTOR["burst"]` | 0.08 | SFX rarely expand; tight pad avoids cross-overlap |
| `_CONTAINER_PAD_MIN_PX` | 4 | Floor for glyph_short=0 cases |
| `_MASK_PAD_FACTOR` | match container | Mask ⊆ polygon invariant |
| `_MASK_PAD_MIN_PX` | 2 | Anti-alias + Lens under-coverage |
| `_HINT_AREA_RATIO_MAX` | 100.0 | Emergency-only (centre containment is the real guard) |
| `_HINT_CONTAINMENT_MIN` | 0.80 | text_bubble matched as inner rect |
| `_RATIO_PHYSICAL_MIN/MAX` | 0.05 / 0.45 | Body ratio physical band |
| `_RATIO_TRIM_LOW/HIGH` | 0.10 / 0.90 | Trimmed mean window |
| `_RATIO_MIN_SOURCES` | 3 | Below this, body ratio not trusted |
| `_CALIBRATED_MIN_SHORT_PX` | 20 | Minimum sensible balloon size |
| `_ANCHOR_DEDUP_IOU` | 0.7 | DETR cluster collapse |
| `_ROTATION_AABB_DEG` | 1.0 | Below this, OBB collapses to AABB |
| `_ELLIPSE_VERTICES` | 24 | < 1 px deviation from true ellipse at print res |
| `_ELLIPSE_ASPECT_MIN/MAX` | 0.4 / 2.5 | Outside this band, ellipse degenerates |

In `crates/render/src/fit.rs`:

| Constant | Value | Rationale |
|---|---|---|
| `MIN_FONT_SIZE` | 8 | Below this, VI diacritics merge |
| `ABS_MAX_FONT_SIZE` | 96 | Print resolution ceiling |
| `MAX_FONT_PAGE_FRACTION` | 0.05 | 5% of page width — typical body cap |
| `HEIGHT_OVERFLOW_TOLERANCE` | 0.0 | Hard contract: text must not exit polygon |
| `HINT_MAX_GROWTH` | 1.15 | VI tone marks need vertical room |
| `HINT_OUTLIER_MIN/MAX` | 10 / 60 | Filter Lens column-split / SFX bleed |

In `crates/render/src/layout.rs`:

| Constant | Value | Rationale |
|---|---|---|
| `DEFAULT_INSET` | 2.0 | Polygon → drawable margin |
| `WIDTH_OVERFLOW_TOLERANCE` | 0.0 | Inscribed ellipse hard contract |
| `LINE_LEADING_FRAC` | 0.05 | Stacked diacritics breathing room |
| `LINE_HEIGHT_MULTIPLIER` | 1.6 | Budget math (not exact baseline) |
| `ELLIPSE_FIT_SCALE` | 0.85 | Inscribed-rect-of-ellipse fit |

## Reading list (in order)

1. This file.
2. `docs/wiki/lens-native-grouping.md` — detector + grouper internals.
3. `typoon/vision/groupers/_spatial_join.py` — the algorithm.
4. `crates/render/src/fit.rs` — the fit binary search.
5. `crates/render/src/overlay/mod.rs` — glyph painting + rotation.
6. Run all three probes on a fresh fixture before changing anything.
