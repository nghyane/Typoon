"""ObbPerLineStrategy — per-line oriented bbox mask for burst/SFX groups.

Ported directly from _erase_masks_from_words in _spatial_join.py.
SFX glyphs are widely spaced; a dilation large enough to bridge them
would bleed into adjacent art. Per-line OBBs stay tight.
"""

from __future__ import annotations

import cv2
import numpy as np

from typoon.vision.contracts import BubbleGroup, TextBlock, TextMask
from typoon.vision.groupers._spatial_join import (
    _contains_center,
    _is_column_layout,
    _line_anchored_obb,
    _MASK_PAD_FACTOR,
    _MASK_PAD_MIN_PX,
    _median_glyph_size,
)

__all__ = ["ObbPerLineStrategy"]

Bbox    = tuple[int, int, int, int]
Polygon = tuple[tuple[float, float], ...]


class ObbPerLineStrategy:
    name = "obb_per_line"

    def build(
        self,
        group: BubbleGroup,
        members: tuple[TextBlock, ...],
        image: np.ndarray | None,
    ) -> tuple[TextMask, ...]:
        shape_kind = group.shape_kind
        members_l  = list(members)

        glyph  = _median_glyph_size(members_l)
        factor = _MASK_PAD_FACTOR.get(shape_kind, _MASK_PAD_FACTOR["dialogue"])
        pad    = max(_MASK_PAD_MIN_PX, int(glyph * factor))

        masks: list[TextMask] = []

        def _push_aabb(b: Bbox) -> None:
            x1, y1, x2, y2 = b
            x1 -= pad; y1 -= pad; x2 += pad; y2 += pad
            w, h = max(1, x2 - x1), max(1, y2 - y1)
            img = np.full((h, w), 255, dtype=np.uint8)
            masks.append(TextMask(x=x1, y=y1, image=img))

        def _push_obb(obb: Polygon) -> None:
            xs = [p[0] for p in obb]; ys = [p[1] for p in obb]
            x1 = int(min(xs)); y1 = int(min(ys))
            x2 = int(max(xs)) + 1; y2 = int(max(ys)) + 1
            w, h = max(1, x2 - x1), max(1, y2 - y1)
            img = np.zeros((h, w), dtype=np.uint8)
            local = np.array(
                [[int(p[0]) - x1, int(p[1]) - y1] for p in obb], dtype=np.int32,
            )
            cv2.fillPoly(img, [local], 255)
            masks.append(TextMask(x=x1, y=y1, image=img))

        for m in members:
            if _is_column_layout([m]):
                _push_aabb(m.bbox)
                continue
            if not m.lines:
                _push_aabb(m.bbox)
                continue
            for line in m.lines:
                words_in_line = [
                    w.bbox for w in m.words if _contains_center(line.bbox, w.bbox)
                ]
                obb = _line_anchored_obb(words_in_line, line.bbox, pad) \
                      if len(words_in_line) >= 2 else None
                if obb is not None:
                    _push_obb(obb)
                else:
                    _push_aabb(line.bbox)

        return tuple(masks)
