"""OversizeSfxScorer — reject large decorative SFX that overlap dialogue bubbles.

Decorative SFX (sound effects rendered as large stylised art) differ from
real SFX in two ways that together form a high-precision signal:

  1. SFX bbox area >> typical group area on the page.
     Lens OCR detects the text fragment inside the SFX art, not the art
     itself — so word glyph size is unreliable. Instead compare the SFX
     bbox area against the median area of dialogue groups on the same page.
     Decorative SFX bboxes are 3× or more the dialogue area median.

  2. The SFX bbox overlaps an adjacent dialogue bubble.
     Decorative SFX are placed between or over speech balloons by the
     artist. When their bbox intersects another group's bbox they will
     erase art inside a bubble they don't belong to.

Both conditions must fire together to avoid false positives:
  - Large SFX in empty space → safe to keep (art, no erase collision)
  - Small SFX overlapping a bubble → just proximity, not oversized art
"""

from __future__ import annotations

import statistics

from typoon.domain.filter import GroupSignal, ScoringContext
from typoon.vision.contracts import BubbleGroup


__all__ = ["OversizeSfxScorer"]

# SFX bbox area / page body area median above which the SFX is "oversized"
_AREA_RATIO_THRESHOLD = 2.5

# Minimum overlap area (px²) with another group to trigger rejection
_OVERLAP_MIN_PX = 500


def _bbox_overlap(a: tuple, b: tuple) -> int:
    ix1 = max(a[0], b[0]); iy1 = max(a[1], b[1])
    ix2 = min(a[2], b[2]); iy2 = min(a[3], b[3])
    return max(0, ix2 - ix1) * max(0, iy2 - iy1)


def _area(bbox: tuple) -> int:
    return max(1, (bbox[2] - bbox[0]) * (bbox[3] - bbox[1]))


class OversizeSfxScorer:
    """Reject decorative SFX that are oversized AND overlap dialogue bubbles."""

    name = "oversize_sfx"

    def score(
        self,
        group: BubbleGroup,
        ctx: ScoringContext,
    ) -> GroupSignal | None:
        if group.shape_kind != "burst":
            return None

        sfx_area = _area(group.bbox)

        # Page body area median from dialogue groups
        body_areas = [
            _area(g.bbox)
            for g in ctx.page_groups
            if g is not group and g.shape_kind == "dialogue"
        ]
        if not body_areas:
            return None

        body_median = int(statistics.median(body_areas))
        ratio = sfx_area / body_median

        if ratio < _AREA_RATIO_THRESHOLD:
            return None

        # Check overlap with any other group
        max_overlap = 0
        overlap_text = ""
        for g in ctx.page_groups:
            if g is group:
                continue
            ov = _bbox_overlap(group.bbox, g.bbox)
            if ov > max_overlap:
                max_overlap = ov
                overlap_text = g.text[:20]

        if max_overlap < _OVERLAP_MIN_PX:
            return None

        return GroupSignal(
            scorer=self.name,
            reason=(
                f"oversized SFX: area={sfx_area}px² vs body_median={body_median}px² "
                f"(ratio={ratio:.1f}x), "
                f"overlap={max_overlap}px with {repr(overlap_text)}"
            ),
            score=1.0,
            hard=True,
        )
