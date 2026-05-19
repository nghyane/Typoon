"""Group filter contracts — pure domain, no vision deps.

Defines the Scorer protocol, signal/verdict data types, FilterResult,
and ScoringContext. Implementations live in vision/filters/.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Protocol, runtime_checkable

if TYPE_CHECKING:
    import numpy as np
    from typoon.vision.contracts import BubbleGroup


__all__ = [
    "GroupSignal",
    "RejectVerdict",
    "FilterResult",
    "ScoringContext",
    "Scorer",
]


@dataclass(frozen=True, slots=True)
class GroupSignal:
    """Evidence from one scorer about one group.

    ``score`` is additive: multiple scorers each contribute 0.0–1.0.
    Scores are summed by the pipeline; if the total meets the threshold
    the group is rejected.  ``hard=True`` bypasses aggregation and
    rejects immediately regardless of other scorers.
    """
    scorer: str
    reason: str
    score:  float        # 0.0 = clean, 1.0 = certain noise
    hard:   bool = False # True → reject immediately, skip aggregate


@dataclass(frozen=True, slots=True)
class RejectVerdict:
    """Why a group was rejected — full audit trail."""
    signals:   tuple[GroupSignal, ...]
    total:     float   # sum of signal scores
    threshold: float   # threshold that was applied


@dataclass(frozen=True, slots=True)
class FilterResult:
    """Output of GroupFilter.evaluate()."""
    kept:     tuple[BubbleGroup, ...]
    rejected: tuple[tuple[BubbleGroup, RejectVerdict], ...]

    @property
    def n_kept(self) -> int:
        return len(self.kept)

    @property
    def n_rejected(self) -> int:
        return len(self.rejected)

    def rejection_summary(self) -> list[dict]:
        return [
            {
                "text":      g.text[:60],
                "bbox":      list(g.bbox),
                "total":     v.total,
                "signals":   [{"scorer": s.scorer, "reason": s.reason, "score": s.score}
                               for s in v.signals],
            }
            for g, v in self.rejected
        ]


@dataclass(frozen=True, slots=True)
class ScoringContext:
    """All information a scorer may need — extensible without breaking callers.

    Add new fields with defaults; existing scorers ignore fields they
    don't know about.
    """
    page_size:   tuple[int, int]
    page_groups: tuple[BubbleGroup, ...]
    image:       "np.ndarray | None" = field(default=None, compare=False)
    blocks:      "tuple | None"      = field(default=None, compare=False)
    # blocks: tuple[TextBlock, ...] — kept as Any to avoid vision import in domain
    # future: ocr_result, detect_metadata, render_hints, ...


@runtime_checkable
class Scorer(Protocol):
    """One orthogonal noise signal."""
    name: str

    def score(
        self,
        group: BubbleGroup,
        ctx: ScoringContext,
    ) -> GroupSignal | None:
        """Return a GroupSignal if the group looks noisy, else None."""
        ...
