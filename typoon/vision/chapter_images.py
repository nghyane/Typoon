"""ChapterImages — owns the stitched buffer, provides per-page views.

Single owner of the pixel data for a chapter. Pages get lightweight
numpy views into this buffer (no copies). Call free() after erase
to release the backing memory.
"""

from __future__ import annotations

import numpy as np


class ChapterImages:
    """Owns the stitched image buffer. Provides zero-copy page views."""

    __slots__ = ("_buffer", "_offsets", "_heights", "width")

    def __init__(self, buffer: np.ndarray, heights: list[int]) -> None:
        self._buffer = buffer
        self._heights = heights
        self.width = buffer.shape[1]
        # Precompute y-offsets
        self._offsets: list[int] = []
        y = 0
        for h in heights:
            self._offsets.append(y)
            y += h

    @staticmethod
    def from_pages(images: list[np.ndarray]) -> ChapterImages:
        """Build from individual page images. Uses Rust for >1 page."""
        if len(images) == 1:
            return ChapterImages(buffer=images[0], heights=[images[0].shape[0]])

        try:
            return _stitch_rust(images)
        except Exception:
            return _stitch_numpy(images)

    @staticmethod
    def from_source(source) -> ChapterImages:
        """Load all pages from a ChapterSource and stitch."""
        images = [source.load_page(i) for i in range(source.page_count())]
        return ChapterImages.from_pages(images)

    def page_count(self) -> int:
        return len(self._heights)

    def page(self, index: int) -> np.ndarray:
        """Zero-copy view of page pixels. Valid until free()."""
        y = self._offsets[index]
        h = self._heights[index]
        return self._buffer[y:y + h]

    def page_height(self, index: int) -> int:
        return self._heights[index]

    def page_offset(self, index: int) -> int:
        return self._offsets[index]

    @property
    def image(self) -> np.ndarray:
        """Full stitched buffer (for scanner)."""
        return self._buffer

    @property
    def heights(self) -> list[int]:
        return self._heights

    def free(self) -> None:
        """Release the backing buffer."""
        self._buffer = None

    @property
    def alive(self) -> bool:
        return self._buffer is not None


def _stitch_rust(images: list[np.ndarray]) -> ChapterImages:
    """Stitch via Rust — single allocation, no Python intermediate."""
    import typoon_render
    result = typoon_render.stitch_pages(images)
    return ChapterImages(
        buffer=result.image,
        heights=[int(h) for h in result.heights],
    )


def _stitch_numpy(images: list[np.ndarray]) -> ChapterImages:
    """Fallback stitch in pure numpy (single page or no Rust)."""
    from statistics import median
    import cv2

    if len(images) == 1:
        return ChapterImages(buffer=images[0], heights=[images[0].shape[0]])

    widths = [img.shape[1] for img in images]
    target_w = int(median(widths))

    normalized: list[np.ndarray] = []
    for img in images:
        w = img.shape[1]
        if w == target_w:
            normalized.append(img)
        elif w > target_w:
            scale = target_w / w
            new_h = int(img.shape[0] * scale)
            normalized.append(cv2.resize(img, (target_w, new_h)))
        else:
            pad = np.full((img.shape[0], target_w - w, 3), 255, dtype=np.uint8)
            normalized.append(np.concatenate([img, pad], axis=1))

    heights = [img.shape[0] for img in normalized]
    buffer = np.concatenate(normalized, axis=0)
    return ChapterImages(buffer=buffer, heights=heights)
