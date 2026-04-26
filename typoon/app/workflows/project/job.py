"""Pipeline job — flows through feeder → scanner → translator_and_renderer."""

from __future__ import annotations

from dataclasses import dataclass

from ....domain.bubble import Page
from ....ports import ChapterSource
from ....vision.chapter_images import LazyPageProvider


@dataclass
class _Job:
    chapter: float
    source: ChapterSource
    project_id: int
    t0: float
    action: str
    pages: list[Page] | None = None
    images: LazyPageProvider | None = None
    pairs: list[tuple[str, str]] | None = None
