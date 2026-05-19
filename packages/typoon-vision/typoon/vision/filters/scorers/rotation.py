"""Rotation scorer — extreme rotation + single word.

Catches upside-down / near-inverted OCR fragments from logo art or
scan artifacts.  Real tategaki stays <= 90°; real tilted SFX is <= 45°.
Combining with single-word guard prevents false positives on legitimately
rotated multi-word text.
"""

from __future__ import annotations

from typoon.domain.filter import GroupSignal, ScoringContext
from typoon.vision.contracts import BubbleGroup


__all__ = ["RotationScorer"]

_ROT_THRESHOLD = 100.0   # degrees absolute
_WORD_MAX      = 1


class RotationScorer:
    name = "rotation"

    def score(
        self,
        group: BubbleGroup,
        ctx: ScoringContext,
    ) -> GroupSignal | None:
        rot     = abs(group.rotation_deg)
        n_words = len(group.text.split())

        if rot > _ROT_THRESHOLD and n_words <= _WORD_MAX:
            return GroupSignal(
                scorer=self.name,
                reason=f"extreme rotation {rot:.1f}° with single word",
                score=1.0,
                hard=True,
            )
        return None
