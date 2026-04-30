"""MaskStore — pixel-level mask data keyed by bubble identity.

Lives in adapter layer. Never imported by domain types.
Scan stage populates it; render stage consumes it.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from typoon.vision.types import TextMask

MaskKey = tuple[int, int]  # (page_index, bubble_idx)


@dataclass(frozen=True)
class BubbleMasks:
    erase_masks: tuple[TextMask, ...]
    text_masks:  tuple[TextMask, ...]


class MaskStore:
    """In-memory mask store for one chapter scan."""

    def __init__(self) -> None:
        self._data: dict[MaskKey, BubbleMasks] = {}

    def put(self, page_index: int, bubble_idx: int, masks: BubbleMasks) -> None:
        self._data[(page_index, bubble_idx)] = masks

    def get(self, page_index: int, bubble_idx: int) -> BubbleMasks | None:
        return self._data.get((page_index, bubble_idx))

    def __len__(self) -> int:
        return len(self._data)
