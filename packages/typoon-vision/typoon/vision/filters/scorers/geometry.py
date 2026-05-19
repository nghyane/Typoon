"""Geometry scorer — tiny area + single short word.

Catches partial OCR hits on panel borders or stray ink fragments
(e.g. "IN", "MA") that are too small to be real story content.

Threshold kept conservative: area < 1500 AND words <= 1 AND chars <= 2.
Combining three conditions prevents false positives on real SFX which
may be small but have >= 3 chars or occupy more area.
"""

from __future__ import annotations

from typoon.domain.filter import GroupSignal, ScoringContext
from typoon.vision.contracts import BubbleGroup


__all__ = ["GeometryScorer"]

_AREA_MAX  = 1500
_WORD_MAX  = 1
_CHARS_MAX = 2


class GeometryScorer:
    name = "geometry"

    def score(
        self,
        group: BubbleGroup,
        ctx: ScoringContext,
    ) -> GroupSignal | None:
        x1, y1, x2, y2 = group.bbox
        area    = (x2 - x1) * (y2 - y1)
        n_words = len(group.text.split())
        n_chars = sum(1 for c in group.text if not c.isspace())

        if area < _AREA_MAX and n_words <= _WORD_MAX and n_chars <= _CHARS_MAX:
            return GroupSignal(
                scorer=self.name,
                reason=(
                    f"tiny fragment: area={area} < {_AREA_MAX}, "
                    f"words={n_words}, chars={n_chars}"
                ),
                score=1.0,
                hard=True,
            )
        return None
