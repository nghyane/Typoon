"""Bottom-left panel: final per-group containers fed to render."""

from __future__ import annotations

import cv2
import numpy as np

from typoon.vision.contracts import BubbleGroup

from .draw import PALETTE, draw_legend, label_panel


def render(canvas: np.ndarray, groups: tuple[BubbleGroup, ...]) -> np.ndarray:
    out = canvas.copy()
    for i, g in enumerate(groups):
        pts = np.array(
            [[int(p[0]), int(p[1])] for p in g.polygon], dtype=np.int32,
        )
        cv2.polylines(out, [pts], True, PALETTE["container"], 2)
        x1, y1, x2, y2 = g.bbox
        w, h = x2 - x1, y2 - y1
        ts = g.typesetting
        font = ts.font_size_px if ts else 0
        label = f"#{i} {g.shape_kind[:1]} {g.text_direction[:1].upper()} {w}x{h} fs={font}"
        cv2.putText(
            out, label, (x1 + 2, max(12, y1 - 2)),
            cv2.FONT_HERSHEY_SIMPLEX, 0.4, PALETTE["container"], 1, cv2.LINE_AA,
        )
    draw_legend(out, [
        ("container polygon (render input)", PALETTE["container"]),
    ])
    label_panel(out, "3. Render polygon — ellipse / OBB / AABB (shape-aware)")
    return out
