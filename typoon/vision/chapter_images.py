"""Chapter page image providers.

StitchedStrip: temporary scan buffer — stitch source pages, scan, free.
LazyPageProvider: lazy runtime provider — reconstructs one logical page
at a time from source images + cut plan. No chapter-sized buffer in RAM.
"""

from __future__ import annotations

import numpy as np



class LazyPageProvider:
    """Loads and crops logical pages from source images on demand."""

    __slots__ = ("_source", "_offsets", "_heights", "_ranges", "width", "_alive")

    def __init__(
        self,
        source,
        source_heights: list[int],
        target_width: int,
        page_ranges: list[tuple[int, int]],
    ) -> None:
        self._source = source
        self._heights = source_heights
        self._ranges = page_ranges
        self.width = target_width
        self._alive = True
        self._offsets: list[int] = []
        y = 0
        for h in source_heights:
            self._offsets.append(y)
            y += h

    def page_count(self) -> int:
        return len(self._ranges)

    def page(self, index: int) -> np.ndarray:
        """Load and return one logical page image."""
        if not self._alive:
            raise RuntimeError(f"Page {index} image already freed")
        y_start, y_end = self._ranges[index]
        parts: list[np.ndarray] = []
        for src_i, src_y in enumerate(self._offsets):
            src_h = self._heights[src_i]
            src_end = src_y + src_h
            if src_end <= y_start or src_y >= y_end:
                continue
            img = _normalize_page(self._source.load_page(src_i), self.width, src_h)
            crop_y1 = max(y_start, src_y) - src_y
            crop_y2 = min(y_end, src_end) - src_y
            parts.append(img[crop_y1:crop_y2])
        if not parts:
            raise IndexError(f"No image data for logical page {index}")
        return parts[0] if len(parts) == 1 else np.concatenate(parts, axis=0)

    def page_height(self, index: int) -> int:
        return self._ranges[index][1] - self._ranges[index][0]

    def free(self) -> None:
        self._alive = False

    @property
    def alive(self) -> bool:
        return self._alive



class StitchedStrip:
    """Temporary stitched buffer for scanning. Free after preprocess."""

    __slots__ = ("_buffer", "_heights", "width")

    def __init__(self, buffer: np.ndarray, heights: list[int]) -> None:
        self._buffer = buffer
        self._heights = heights
        self.width = buffer.shape[1]

    @staticmethod
    def from_pages(images: list[np.ndarray]) -> StitchedStrip:
        """Stitch page images into one vertical strip."""
        if len(images) == 1:
            return StitchedStrip(buffer=images[0], heights=[images[0].shape[0]])
        try:
            return _stitch_rust(images)
        except Exception:
            return _stitch_numpy(images)

    @property
    def image(self) -> np.ndarray:
        return self._buffer

    @property
    def heights(self) -> list[int]:
        return self._heights

    def free(self) -> None:
        self._buffer = None



def _stitch_rust(images: list[np.ndarray]) -> StitchedStrip:
    import typoon_render
    result = typoon_render.stitch_pages(images)
    return StitchedStrip(
        buffer=result.image,
        heights=[int(h) for h in result.heights],
    )


def _stitch_numpy(images: list[np.ndarray]) -> StitchedStrip:
    from statistics import median

    if len(images) == 1:
        return StitchedStrip(buffer=images[0], heights=[images[0].shape[0]])

    widths = [img.shape[1] for img in images]
    target_w = int(median(widths))

    normalized = [_normalize_page(img, target_w) for img in images]
    heights = [img.shape[0] for img in normalized]
    buffer = np.concatenate(normalized, axis=0)
    return StitchedStrip(buffer=buffer, heights=heights)


def _normalize_page(img: np.ndarray, target_w: int, target_h: int | None = None) -> np.ndarray:
    """Single source of truth for width normalization.

    Matches Rust stitch_pages logic: wider pages bilinear-resized down,
    narrower pages white-padded on the right.
    """
    import cv2

    h, w = img.shape[:2]
    if w == target_w:
        return img[:target_h] if target_h is not None else img
    if w > target_w:
        new_h = target_h if target_h is not None else int(round(h * target_w / w))
        return cv2.resize(img, (target_w, new_h))
    if target_h is not None and target_h != h:
        img = cv2.resize(img, (w, target_h))
        h = target_h
    pad = np.full((h, target_w - w, 3), 255, dtype=np.uint8)
    return np.concatenate([img, pad], axis=1)
