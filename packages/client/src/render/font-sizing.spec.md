# Font sizing — layered spec

Status: living spec for the render text-sizing path. Source of truth for *why*
each number lives where. Code is the source of truth for *current values*.

## Problem

Localising KR/JA/ZH comics → VI. The only reliable signal is the **source text
geometry** (OCR): position, per-line height, line count, footprint. The speech
**bubble** is NOT reliable (white-on-white bubbles make background/DETR bleed to
the whole page), so it must never drive sizing — only loosely clamp it.

Different source scripts render at different visual sizes for the same pixel
height (KR/ZH square glyphs look larger than VI Latin; EN ≈ VI). Translations are
usually longer than the source and must keep their size by using free space
rather than shrinking.

## Three independent levers (each lives in exactly one layer)

| Lever | Job | Lives in | Nature |
|---|---|---|---|
| `glyphScale` | cross-script glyph-size ratio (KR→VI < 1, EN→VI ≈ 1) | T1 typography | **constant** per language pair |
| `readableFloorPx` | absolute legibility floor for already-small text | T1 defines, T3 enforces | absolute px |
| `expand` | longer translation keeps size via free space | T4 fit, only here | length-driven |
| `leadingRatio` | line-height headroom for VI tone marks | T1 → composer | ratio (Phase 3) |

`glyphScale` is proportional, so it is a flat multiplier at every source size.
"Already-small source" is NOT the scale's problem — it is the floor's problem.
(The earlier per-size *taper* conflated these two and is removed.)

## Layers

```
T1 TYPOGRAPHY INTENT   render/languageProfile.ts   (pure language; no pixels/bubble)
   out: glyphScale, readableFloorPx, leadingRatio, expansionAllowance, max fractions

T2 ANCHOR GEOMETRY     render/fitGeometry.ts        (pure geometry; no language)
   in:  source footprint + centre + line count (reliable); containerBBox (loose)
   out: anchorRegion = footprint × expansionAllowance, centred on source
        softCeiling  = containerBBox (clamp only)
        sourceFontPx = source line height

T3 FONT INTENT         render/fitLayout.ts:fontIntentFor   (T1 + T2 → target)
   targetFontPx = clamp(sourceFontPx × glyphScale, readableFloorPx, placementMax)

T4 FIT / EXPAND        render/fitLayout.ts (orchestrator)  (+ bubbleShape, lineComposer)
   fit target in anchorRegion (shape-aware); if long, expand region inside
   softCeiling to keep target; only then shrink, never below readableFloorPx.

T5 RENDER              render/textLayer.ts          (draw only)
```

Dependency direction (no cycles):
`languageProfile → (none)`, `fitGeometry → domain`, `fontIntent → T1,T2`,
`fitLayout → T3,T2,bubbleShape,lineComposer`, `textLayer → T4`.

Forbidden: T1 knows pixels/bubble; T2 knows language; T3 fits; T5 sizes.

## Implementation phases

- **Phase 1 (this change): font-intent correctness.**
  Remove taper. `glyphScale` flat per language pair. Enforce `readableFloorPx`
  as the target floor (was `MIN_FONT_SIZE`). Verify on p2-r1/p2-r3.
- **Phase 2 (done): geometry unification.** Removed the dead `geometryGrow*`
  profile fields (a defunct parallel mechanism, never read). The anchor-region
  growth is now language-aware via `expansionAllowanceX/Y` (T1), injected into
  `textFitRect` (T2) instead of hard-coded constants. `containerBBox` is the
  soft ceiling, applied only as a clamp in `fitLayout.boundsForExpansion`.
- **Phase 3 (done): VI leading.** `leadingRatio` (T1, latin 1.06) multiplies the
  font line-height once in `fitLayout` into an effective `FontProfile` used by all
  measure/compose/output, so Vietnamese stacked diacritics get headroom without
  touching font metrics or the composer's contract.

## Current glyphScale values (T1, tunable)

```
SFX                1.00   (dramatic by design)
KR (hangul)→VI     0.70   (square dense glyphs read large)
JA/ZH dense→VI     0.88
Latin→VI           0.80   (tall light line-box vs bold high-x-height target)
other→Latin        0.94
→ hangul target    0.96
→ CJK target       1.02
```
