"""Calibrated-pad probe.

Diagnoses the "fit container too tight/loose" question by computing a
per-page calibration ratio from Lens samples and showing what
container size each fallback group WOULD get under a calibrated pad —
without touching production grouper code.

Algorithm:
  1. Run production detect + group on one image.
  2. For each group, classify into calibration source or fallback:
       * SOURCE: anchor has a text_bubble (bubble+inner-rect OR
         text_bubble-standalone). The DETR-detected balloon outline IS
         the ground-truth container; glyph_short / text_bubble_short
         contributes to the page ratio pool.
       * FALLBACK: anchor=bubble without text_bubble, anchor=text_free,
         or singleton (no anchor). Current container = word_union +
         heuristic pad. We compute the *proposed* container using
         `target_short = glyph_short / page_ratio`.
  3. Trimmed mean (10-90%) over SOURCE pool → `body_ratio`,
     `sfx_ratio`.
  4. For each FALLBACK group, emit:
       current_polygon vs proposed_polygon, current_short vs
       target_short, delta_pad.
  5. Write `calibration.json` + `overlay.png` (yellow = current,
     cyan = proposed, magenta = SOURCE balloon outline).

Probe ONLY — no production code change.

Run:
    python -m scripts.probes.calibrated_pad <image>
        [--out debug-runs/<name>]
"""
