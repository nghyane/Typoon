"""Prepare raw source images into a Chapter of prepared pages."""

from __future__ import annotations

from typing import Protocol

import cv2
import numpy as np

from typoon.domain.prepared import Chapter, Page
from typoon.paths import ChapterPaths
from typoon.runs.artifacts import ArtifactSink


class RawChapterSource(Protocol):
    def page_count(self) -> int: ...
    def load_page(self, index: int) -> np.ndarray: ...


def prepare_chapter(
    source: RawChapterSource,
    cp: ChapterPaths,
    *,
    source_label: str = "",
    artifacts: ArtifactSink | None = None,
) -> Chapter:
    """Write one prepared PNG per raw page. Returns PreparedChapter."""
    cp.pages.mkdir(parents=True, exist_ok=True)

    pages: list[Page] = []
    for index in range(source.page_count()):
        image = source.load_page(index)
        h, w  = image.shape[:2]
        dest  = cp.page(index)
        _write_rgb_png(dest, image)
        pages.append(Page(index=index, file=f"pages/{dest.name}", width=w, height=h))
        if artifacts is not None:
            artifacts.write_image("01_prepare", f"prepared_{index:04d}.png", image)

    chapter = Chapter(root=cp.root, source=source_label, pages=tuple(pages))

    if artifacts is not None:
        artifacts.write_json("01_prepare", "groups.json", {
            "version": 1,
            "strategy": "one_to_one",
            "groups": [[p.index] for p in pages],
        })

    return chapter


def _write_rgb_png(path, image: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR) if image.ndim == 3 else image
    if not cv2.imwrite(str(path), bgr):
        raise RuntimeError(f"Failed to write prepared page: {path}")
