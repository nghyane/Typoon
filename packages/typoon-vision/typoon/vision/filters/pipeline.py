"""Group filter pipeline — evaluate a tuple of scorers against a page's groups."""

from __future__ import annotations

from typoon.domain.filter import (
    FilterResult,
    GroupSignal,
    RejectVerdict,
    Scorer,
    ScoringContext,
)
from typoon.vision.contracts import BubbleGroup


__all__ = ["GroupFilter"]


class GroupFilter:
    """Run N scorers against each group, aggregate → keep or reject.

    Aggregation rules (in order):
      1. Any scorer emits ``hard=True`` → reject immediately.
      2. Sum of all signal scores >= threshold → reject.
      3. Otherwise keep.

    Scorers are stateless; context carries page-level information.
    """

    def __init__(
        self,
        scorers: tuple[Scorer, ...],
        threshold: float = 1.0,
    ) -> None:
        self._scorers   = scorers
        self._threshold = threshold

    @property
    def scorers(self) -> tuple[Scorer, ...]:
        return self._scorers

    @property
    def threshold(self) -> float:
        return self._threshold

    def evaluate(
        self,
        groups: tuple[BubbleGroup, ...],
        ctx: ScoringContext,
    ) -> FilterResult:
        kept:     list[BubbleGroup]                       = []
        rejected: list[tuple[BubbleGroup, RejectVerdict]] = []

        for g in groups:
            signals: list[GroupSignal] = []
            hard_reject = False

            for sc in self._scorers:
                sig = sc.score(g, ctx)
                if sig is None:
                    continue
                signals.append(sig)
                if sig.hard:
                    hard_reject = True
                    break   # no need to evaluate further

            total = sum(s.score for s in signals)

            if hard_reject or total >= self._threshold:
                verdict = RejectVerdict(
                    signals=tuple(signals),
                    total=total,
                    threshold=self._threshold,
                )
                rejected.append((g, verdict))
            else:
                kept.append(g)

        return FilterResult(
            kept=tuple(kept),
            rejected=tuple(rejected),
        )
