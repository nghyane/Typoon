"""Prepare raw source images into a Chapter of prepared WebP-lossless pages.

Strategy auto-detection:
  - Sample pages at 25%, 50%, 75% of chapter.
  - Measure color ratio: fraction of pixels with HSV saturation > SAT_THRESHOLD.
  - If color ratio >= COLOR_RATIO_THRESHOLD → webtoon/manhwa → stitch strategy.
  - Otherwise → one_to_one (standard manga page-per-file).

Stitch strategy:
  - vstack all raw slices into one long strip.
  - Re-cut using vectorized pixel-comparison (stitchtoon algorithm):
      valid row = max neighbor diff <= threshold AND row spread <= threshold
      confirmed row = window consecutive valid rows (avoids single-row false positives)
      nearest confirmed row to each target cut point wins; hard cut fallback if none.
  - Guarantees no bubble is split at a prepared page boundary.

Output is a directory of `<i:04d>.webp` files (WebP lossless), ready for
`bunle.pack_dir` passthrough. Prepared pixels are the canonical coordinate
source for all downstream stages.
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal, Protocol

import cv2
import numpy as np
from PIL import Image

from typoon.domain.prepared import Chapter, Page
from typoon.runs.artifacts import ArtifactSink

# ── Constants ─────────────────────────────────────────────────────────

_COLOR_RATIO_THRESHOLD = 0.15
_SAT_THRESHOLD         = 30       # HSV saturation, out of 255

_MAX_PAGE_HEIGHT = 4096           # target prepared page height for stitch
_MIN_PAGE_HEIGHT = 2048           # minimum — prevents cluster of cuts in whitespace
_SENSITIVITY     = 97             # 0-100; higher = stricter valid-row check
_WINDOW          = 10             # consecutive valid rows required to confirm
_X_MARGINS       = 10             # pixels to ignore on each side during detection


class RawChapterSource(Protocol):
    def page_count(self) -> int: ...
    def load_page(self, index: int) -> np.ndarray: ...


# ── Public API ────────────────────────────────────────────────────────


def prepare_chapter(
    source: RawChapterSource,
    out_dir: Path,
    *,
    strategy: Literal["auto", "one_to_one", "stitch"] = "auto",
    source_label: str = "",
    artifacts: ArtifactSink | None = None,
) -> Chapter:
    """Write `<i:04d>.webp` (lossless) into `out_dir`. Returns PreparedChapter."""
    out_dir.mkdir(parents=True, exist_ok=True)

    if strategy == "auto":
        strategy = _detect_strategy(source)

    if strategy == "stitch":
        pages, groups = _prepare_stitch(source, out_dir, artifacts)
    else:
        pages, groups = _prepare_one_to_one(source, out_dir, artifacts)

    chapter = Chapter(source=source_label, pages=tuple(pages))

    if artifacts is not None:
        artifacts.write_json("01_prepare", "groups.json", {
            "version": 1,
            "strategy": strategy,
            "groups": groups,
        })

    return chapter


# ── Strategy detection ────────────────────────────────────────────────


def _detect_strategy(source: RawChapterSource) -> Literal["one_to_one", "stitch"]:
    n = source.page_count()
    if n == 0:
        return "one_to_one"
    indices = sorted({n // 4, n // 2, 3 * n // 4})
    ratios  = [_color_ratio(source.load_page(i)) for i in indices]
    return "stitch" if sum(ratios) / len(ratios) >= _COLOR_RATIO_THRESHOLD else "one_to_one"


def _color_ratio(image: np.ndarray) -> float:
    small = cv2.resize(image, (256, 256)) if min(image.shape[:2]) > 256 else image
    hsv   = cv2.cvtColor(small, cv2.COLOR_RGB2HSV)
    return float(np.count_nonzero(hsv[:, :, 1] > _SAT_THRESHOLD)) / (hsv.shape[0] * hsv.shape[1])


# ── one_to_one ────────────────────────────────────────────────────────


def _prepare_one_to_one(
    source: RawChapterSource,
    out_dir: Path,
    artifacts: ArtifactSink | None,
) -> tuple[list[Page], list[list[int]]]:
    pages:  list[Page]      = []
    groups: list[list[int]] = []
    for i in range(source.page_count()):
        img  = source.load_page(i)
        h, w = img.shape[:2]
        _write_lossless(out_dir / f"{i:04d}.webp", img)
        pages.append(Page(index=i, width=w, height=h))
        groups.append([i])
        if artifacts is not None:
            artifacts.write_image("01_prepare", f"prepared_{i:04d}.png", img)
    return pages, groups


# ── stitch ────────────────────────────────────────────────────────────


def _prepare_stitch(
    source: RawChapterSource,
    out_dir: Path,
    artifacts: ArtifactSink | None,
) -> tuple[list[Page], list[list[int]]]:
    raw = [source.load_page(i) for i in range(source.page_count())]
    w   = _modal_width(raw)
    raw = [_resize_to_width(img, w) for img in raw]

    strip     = np.concatenate(raw, axis=0)
    total_h   = strip.shape[0]
    confirmed = _confirmed_rows(strip)
    boundaries = _raw_boundaries(raw)

    pages:  list[Page]      = []
    groups: list[list[int]] = []
    prev    = 0
    out_idx = 0
    target  = _MAX_PAGE_HEIGHT

    while target < total_h:
        lo  = prev + _MIN_PAGE_HEIGHT
        hi  = min(prev + int(_MAX_PAGE_HEIGHT * 1.5), total_h)
        cut = _nearest_confirmed(confirmed, lo, hi, prev + _MAX_PAGE_HEIGHT)
        if cut is None:
            cut = _nearest_confirmed(confirmed, lo, total_h, prev + _MAX_PAGE_HEIGHT)
        if cut is None:
            cut = prev + _MAX_PAGE_HEIGHT  # hard fallback: no valid row anywhere

        _write_segment(strip, prev, cut, out_dir, out_idx, pages, groups, boundaries, artifacts)
        out_idx += 1
        prev    = cut
        target  = prev + _MAX_PAGE_HEIGHT

    if prev < total_h:
        _write_segment(strip, prev, total_h, out_dir, out_idx, pages, groups, boundaries, artifacts)

    return pages, groups


def _write_segment(
    strip: np.ndarray,
    start: int, end: int,
    out_dir: Path,
    out_idx: int,
    pages: list[Page],
    groups: list[list[int]],
    boundaries: list[tuple[int, int]],
    artifacts: ArtifactSink | None,
) -> None:
    seg  = strip[start:end]
    h, w = seg.shape[:2]
    _write_lossless(out_dir / f"{out_idx:04d}.webp", seg)
    pages.append(Page(index=out_idx, width=w, height=h))
    groups.append([i for i, (rs, re) in enumerate(boundaries) if rs < end and re > start])
    if artifacts is not None:
        artifacts.write_image("01_prepare", f"prepared_{out_idx:04d}.png", seg)


# ── Pixel-comparison cut detection (stitchtoon algorithm) ─────────────


def _confirmed_rows(strip: np.ndarray) -> np.ndarray:
    """Bool array (H,): True where a window of consecutive valid rows starts."""
    gray = cv2.cvtColor(strip, cv2.COLOR_RGB2GRAY)
    arr  = gray[:, _X_MARGINS: gray.shape[1] - _X_MARGINS] if _X_MARGINS > 0 else gray

    threshold = int(255 * (1 - _SENSITIVITY / 100))
    diff      = np.abs(np.diff(arr.astype(np.int16), axis=1))
    valid     = (diff.max(axis=1) <= threshold) & (
        (arr.max(axis=1).astype(np.int16) - arr.min(axis=1).astype(np.int16)) <= threshold
    )

    H   = len(valid)
    cum = np.concatenate([[0], np.cumsum(valid.astype(np.int32))])
    confirmed = np.zeros(H, dtype=bool)
    confirmed[: H - _WINDOW + 1] = (cum[_WINDOW:] - cum[: H - _WINDOW + 1]) == _WINDOW
    return confirmed


def _nearest_confirmed(
    confirmed: np.ndarray, lo: int, hi: int, target: int,
) -> int | None:
    lo, hi = max(lo, 0), min(hi, len(confirmed))
    if lo >= hi:
        return None
    indices = np.where(confirmed[lo:hi])[0]
    if len(indices) == 0:
        return None
    abs_idx = indices + lo
    return int(abs_idx[np.argmin(np.abs(abs_idx - target))])


# ── Helpers ───────────────────────────────────────────────────────────


def _modal_width(images: list[np.ndarray]) -> int:
    widths = [img.shape[1] for img in images]
    return max(set(widths), key=widths.count)


def _resize_to_width(image: np.ndarray, w: int) -> np.ndarray:
    h, cw = image.shape[:2]
    if cw == w:
        return image
    return cv2.resize(image, (w, round(h * w / cw)), interpolation=cv2.INTER_AREA)


def _raw_boundaries(images: list[np.ndarray]) -> list[tuple[int, int]]:
    boundaries, row = [], 0
    for img in images:
        h = img.shape[0]
        boundaries.append((row, row + h))
        row += h
    return boundaries


def _write_lossless(path: Path, image: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    rgb = image if image.ndim == 3 else cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    Image.fromarray(rgb, mode="RGB").save(path, format="WEBP", lossless=True, quality=100)
