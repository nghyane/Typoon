"""Top-left panel: Lens raw geometry (word ⊂ line ⊂ paragraph).

For each block we draw:
  * word bboxes (gold, 1px)
  * line bboxes (cyan, 1px)
  * paragraph AABB (magenta, 2px) — axis-aligned as-is from Lens
  * for tilted non-column blocks: an oriented green polygon computed
    from the actual word-first→word-last vector (the true tilt axis).
    This is what the container polygon will also use.
"""

from __future__ import annotations

import math

import cv2
import numpy as np

from typoon.vision.contracts import TextBlock
from typoon.vision.groupers._spatial_join import _is_column_layout

from .draw import PALETTE, draw_box, draw_legend, label_panel


_ROT_THRESHOLD = 1.0


def render(
    canvas: np.ndarray,
    blocks: list[TextBlock],
    rejected: list[tuple[TextBlock, str]] | None = None,
) -> np.ndarray:
    out = canvas.copy()
    for b, reason in (rejected or []):
        x1, y1, x2, y2 = (int(v) for v in b.bbox)
        cv2.rectangle(out, (x1, y1), (x2, y2), (255, 0, 0), 1)
        cv2.putText(
            out, f"REJ {reason}", (x1 + 2, max(12, y1 - 2)),
            cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255, 0, 0), 1, cv2.LINE_AA,
        )
    for b in blocks:
        for w in b.words:
            draw_box(out, w.bbox, PALETTE["word"], thickness=1)
        for ln in b.lines:
            draw_box(out, ln.bbox, PALETTE["line"], thickness=1)
        draw_box(
            out, b.bbox, PALETTE["paragraph"],
            label=f"rot={b.rotation_deg:.0f}",
            thickness=2,
        )
        # Oriented polygon only when the word layout is genuinely tilted
        # (not a column) and rotation is non-trivial.
        if abs(b.rotation_deg) > _ROT_THRESHOLD and not _is_column_layout([b]):
            _draw_oriented(out, b)
    draw_legend(out, [
        ("paragraph (AABB)", PALETTE["paragraph"]),
        ("line",             PALETTE["line"]),
        ("word",             PALETTE["word"]),
        ("oriented polygon", (0, 255, 0)),
        ("rejected",         (255, 0, 0)),
    ])
    label_panel(out, "1. Lens raw — word / line / paragraph")
    return out


def _draw_oriented(canvas: np.ndarray, block: TextBlock) -> None:
    """Draw the oriented polygon derived from word-first→word-last vector."""
    poly = _word_axis_polygon(block)
    if poly is None:
        return
    pts = np.array([[int(p[0]), int(p[1])] for p in poly], dtype=np.int32)
    cv2.polylines(canvas, [pts], True, (0, 255, 0), 2)


def _word_axis_polygon(block: TextBlock):
    """OBB tight to the actual glyph stripe — centres + median glyph size.

    Mirrors ``_word_axis_obb`` in the production grouper so the probe
    panel matches what the renderer actually receives.
    """
    if not block.words:
        return None
    centres = sorted(
        [((w.bbox[0] + w.bbox[2]) / 2.0, (w.bbox[1] + w.bbox[3]) / 2.0)
         for w in block.words],
        key=lambda c: c[0],
    )
    if len(centres) < 2:
        return None
    dx = centres[-1][0] - centres[0][0]
    dy = centres[-1][1] - centres[0][1]
    length = math.hypot(dx, dy)
    if length < 1:
        return None
    ux, uy = dx / length, dy / length
    vx, vy = -uy, ux

    import statistics
    glyph_half = statistics.median(
        min(max(1, w.bbox[2] - w.bbox[0]),
            max(1, w.bbox[3] - w.bbox[1]))
        for w in block.words
    ) / 2.0

    cx0, cy0 = centres[0]
    cxN, cyN = centres[-1]
    u_start = cx0 * ux + cy0 * uy - glyph_half
    u_end   = cxN * ux + cyN * uy + glyph_half
    v_mid   = (cx0 * vx + cy0 * vy + cxN * vx + cyN * vy) / 2.0
    v_min   = v_mid - glyph_half
    v_max   = v_mid + glyph_half

    return [
        (u * ux + v * vx, u * uy + v * vy)
        for u, v in [(u_start, v_min), (u_end, v_min),
                     (u_end, v_max), (u_start, v_max)]
    ]
