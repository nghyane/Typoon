"""Prepare raw source images into canonical PreparedChapter pages."""

from __future__ import annotations

from pathlib import Path
from typing import Protocol

import cv2
import numpy as np

from typoon.domain.prepared import PreparedChapter, PreparedPage, write_prepared_chapter
from typoon.runs.artifacts import ArtifactSink


class RawChapterSource(Protocol):
    def page_count(self) -> int: ...
    def load_page(self, index: int) -> np.ndarray: ...


def prepare_chapter(
    source: RawChapterSource,
    out_dir: Path,
    *,
    source_label: str = "",
    artifacts: ArtifactSink | None = None,
) -> PreparedChapter:
    """Write one prepared PNG per raw page.

    This first implementation intentionally does not stitch or cut. It creates
    the canonical PreparedChapter boundary so later stages cannot read raw
    source images directly.
    """
    out_dir = Path(out_dir)
    pages_dir = out_dir / "pages"
    pages_dir.mkdir(parents=True, exist_ok=True)

    pages: list[PreparedPage] = []
    for index in range(source.page_count()):
        image = source.load_page(index)
        h, w = image.shape[:2]
        rel = f"pages/{index:04d}.png"
        out = out_dir / rel
        _write_rgb_png(out, image)
        pages.append(PreparedPage(index=index, file=rel, width=w, height=h))
        if artifacts is not None:
            artifacts.write_image("01_prepare", f"prepared_{index:04d}.png", image)

    chapter = PreparedChapter(root=out_dir, source=source_label, pages=tuple(pages))
    write_prepared_chapter(chapter)

    data = chapter.to_manifest()
    if artifacts is not None:
        artifacts.write_json("01_prepare", "prepared_manifest.json", data)
        artifacts.write_json("01_prepare", "groups.json", {
            "version": 1,
            "strategy": "one_to_one",
            "groups": [[page.index] for page in pages],
        })
        artifacts.write_image("01_prepare", "row_cost.png", _blank_debug_image("row cost pending stitch/cut"))
        artifacts.write_image("01_prepare", "cuts_overlay.png", _blank_debug_image("no cuts: one raw page to one prepared page"))
    return chapter


def _write_rgb_png(path: Path, image: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR) if image.ndim == 3 else image
    if not cv2.imwrite(str(path), bgr):
        raise RuntimeError(f"Failed to write prepared page: {path}")


def _blank_debug_image(label: str) -> np.ndarray:
    image = np.full((120, 640, 3), 255, dtype=np.uint8)
    cv2.putText(image, label, (16, 68), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (20, 20, 20), 1, cv2.LINE_AA)
    return image
