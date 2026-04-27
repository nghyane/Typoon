"""Chapter page image providers.

StitchedStrip: temporary scan buffer — stitch source pages, scan, free.
LazyPageProvider: lazy runtime provider — reconstructs one logical page
at a time from source images + cut plan. No chapter-sized buffer in RAM.
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np


@dataclass(frozen=True)
class _Slice:
    """One source page contribution to a logical page."""
    src_index: int
    crop_y1: int
    crop_y2: int


class LazyPageProvider:
    """Reconstructs logical pages on demand from source + cut plan."""

    __slots__ = ("_source", "_pages", "_target_w", "_alive")

    def __init__(
        self, source, target_w: int, pages: list[list[_Slice]],
    ) -> None:
        self._source = source
        self._target_w = target_w
        self._pages = pages
        self._alive = True

    @staticmethod
    def build(
        source, source_heights: list[int], target_w: int,
        page_ranges: list[tuple[int, int]],
    ) -> LazyPageProvider:
        offsets = _cumulative_offsets(source_heights)
        pages: list[list[_Slice]] = []
        for y_start, y_end in page_ranges:
            slices: list[_Slice] = []
            for i, off in enumerate(offsets):
                src_end = off + source_heights[i]
                if src_end <= y_start or off >= y_end:
                    continue
                slices.append(_Slice(
                    src_index=i,
                    crop_y1=max(y_start, off) - off,
                    crop_y2=min(y_end, src_end) - off,
                ))
            pages.append(slices)
        return LazyPageProvider(source, target_w, pages)

    def page_count(self) -> int:
        return len(self._pages)

    def page(self, index: int) -> np.ndarray:
        if not self._alive:
            raise RuntimeError(f"Page {index} image already freed")
        parts = [self._load_slice(s) for s in self._pages[index]]
        if len(parts) == 1:
            return parts[0]
        w = max(p.shape[1] for p in parts)
        return np.concatenate([_pad_width(p, w) for p in parts], axis=0)

    def page_height(self, index: int) -> int:
        return sum(s.crop_y2 - s.crop_y1 for s in self._pages[index])

    def free(self) -> None:
        self._alive = False

    @property
    def alive(self) -> bool:
        return self._alive

    def _load_slice(self, s: _Slice) -> np.ndarray:
        img = self._source.load_page(s.src_index)
        if img.shape[1] > self._target_w:
            img = _resize_width(img, self._target_w)
        return img[s.crop_y1:s.crop_y2]


class StitchedStrip:
    """Temporary stitched buffer for scanning. Free after preprocess."""

    __slots__ = ("_buffer", "_heights", "width")

    def __init__(self, buffer: np.ndarray, heights: list[int]) -> None:
        self._buffer = buffer
        self._heights = heights
        self.width = buffer.shape[1]

    @staticmethod
    def from_pages(images: list[np.ndarray]) -> StitchedStrip:
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


def _cumulative_offsets(heights: list[int]) -> list[int]:
    offsets = []
    y = 0
    for h in heights:
        offsets.append(y)
        y += h
    return offsets


def _resize_width(img: np.ndarray, target_w: int) -> np.ndarray:
    h, w = img.shape[:2]
    new_h = int(round(h * target_w / w))
    return cv2.resize(img, (target_w, new_h))


def _pad_width(img: np.ndarray, target_w: int) -> np.ndarray:
    w = img.shape[1]
    if w >= target_w:
        return img
    pad = np.full((img.shape[0], target_w - w, 3), 255, dtype=np.uint8)
    return np.concatenate([img, pad], axis=1)


def _normalize_page(img: np.ndarray, target_w: int) -> np.ndarray:
    """Normalize width for stitching: resize wide, pad narrow."""
    w = img.shape[1]
    if w == target_w:
        return img
    if w > target_w:
        return _resize_width(img, target_w)
    return _pad_width(img, target_w)


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

    target_w = int(median(img.shape[1] for img in images))
    normalized = [_normalize_page(img, target_w) for img in images]
    heights = [img.shape[0] for img in normalized]
    return StitchedStrip(buffer=np.concatenate(normalized, axis=0), heights=heights)
