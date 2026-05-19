"""Watermark / scanlation-credit scorer.

Hard-rejects groups whose text contains known platform watermark tokens.
Uses hard=True because watermark text is unambiguous — no aggregation needed.
"""

from __future__ import annotations

from typoon.domain.filter import GroupSignal, ScoringContext
from typoon.vision.contracts import BubbleGroup


__all__ = ["WatermarkScorer"]

_TOKENS = frozenset({
    "DO NOT MIRROR",
    "DO NOT REPOST",
    "MANGA STREAM",
    "MANGASTREAM",
    "SCANLATION",
    "SCANS.NET",
    "READMANGA",
    "MANGADEX",
    "BAOZIMH",
    "MANHUAGUI",
    "COPYMANGA",
})


class WatermarkScorer:
    name = "watermark"

    def score(
        self,
        group: BubbleGroup,
        ctx: ScoringContext,
    ) -> GroupSignal | None:
        txt = group.text.upper()
        for tok in _TOKENS:
            if tok in txt:
                return GroupSignal(
                    scorer=self.name,
                    reason=f"watermark token {tok!r} in text",
                    score=1.0,
                    hard=True,
                )
        return None
