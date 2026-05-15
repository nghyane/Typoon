"""Lens block detector.

Fetches text blocks from the Google Lens reverse-engineered API. Lens
returns blocks already segmented per bubble with recognised text, so
this detector both detects AND recognises in one call.

Uses `output_format="detailed"` to get:
  - paragraph bbox + angle_deg (bubble geometry + rotation hint)
  - lines (per-line text, ignored downstream; lines roll up into paragraph)
  - words with per-glyph bbox (drives tight erase masks in lens_native grouper)
  - content_language (Lens-detected source language, propagated to scan)

Tile strategy: 720×900 tiles with 200px overlap to stay under Lens's
1000px resize threshold; dedup across overlaps by length-weighted bbox
containment so a bubble split across tiles wins from the longer copy.

Filters (from poc_lens_v3 empirical evidence):
  - tiny_bbox       — < 25×18 (decoration star/dots)
  - decoration_only — only symbols, no letter/digit
  - huge_bbox       — area/char > 6000 (Lens hallucinated on art region)
"""

from __future__ import annotations

import asyncio
import importlib
import logging
import os
import unicodedata
from collections import Counter
from dataclasses import dataclass
from functools import cache

import numpy as np

from ..contracts import DetectionResult, LineBox, TextBlock, WordBox


__all__ = ["LensBlocksDetector", "LensUnavailableError"]

logger = logging.getLogger(__name__)


# ─── Constants ────────────────────────────────────────────────────────────


_DEFAULT_ENDPOINT = "https://lensfrontend-pa.googleapis.com/v1/crupload"

# Lens resizes any input over ~1000px on the longest axis.
_TILE_H = 900
_OVERLAP = 200
_MAX_CONCURRENT = 15
_DEDUP_IOU = 0.5
_SUBSTRING_IOU = 0.05

# Filter thresholds (from poc_lens_v3 on 9 fixture pages).
_MIN_BBOX_W      = 25
_MIN_BBOX_H      = 18
_MIN_BBOX_AREA   = 700
_MAX_AREA_PER_CHAR = 6000

_DECORATION_CHARS = frozenset("★☆●○◎◇◆□■▲△▽▼※・…—–-_=+×÷")


# ─── Errors ───────────────────────────────────────────────────────────────


class LensUnavailableError(RuntimeError):
    """Raised when Lens dependency missing or upstream unreachable."""


# ─── Filter helpers ───────────────────────────────────────────────────────


@cache
def _is_letter_or_digit(ch: str) -> bool:
    if ch in _DECORATION_CHARS or ch.isspace():
        return False
    if ch.isalnum():
        return True
    return unicodedata.category(ch).startswith(("L", "N"))


def _is_decoration_only(text: str) -> bool:
    return not any(_is_letter_or_digit(c) for c in text)


def _bbox_too_small(bbox: tuple[int, int, int, int]) -> bool:
    x1, y1, x2, y2 = bbox
    w, h = x2 - x1, y2 - y1
    return w < _MIN_BBOX_W or h < _MIN_BBOX_H or w * h < _MIN_BBOX_AREA


def _bbox_too_large_for_text(bbox: tuple[int, int, int, int], text: str) -> bool:
    x1, y1, x2, y2 = bbox
    area = max(1, (x2 - x1) * (y2 - y1))
    chars = max(1, sum(1 for c in text if not c.isspace()))
    return area / chars > _MAX_AREA_PER_CHAR


def _filter_blocks(
    blocks: list[TextBlock],
) -> tuple[list[TextBlock], list[tuple[TextBlock, str]]]:
    kept:     list[TextBlock] = []
    rejected: list[tuple[TextBlock, str]] = []
    for b in blocks:
        text = b.text or ""
        if _bbox_too_small(b.bbox):
            rejected.append((b, "tiny_bbox"))
        elif _is_decoration_only(text):
            rejected.append((b, "decoration_only"))
        elif _bbox_too_large_for_text(b.bbox, text):
            rejected.append((b, "huge_bbox"))
        else:
            kept.append(b)
    return kept, rejected


# ─── Tile dedup ───────────────────────────────────────────────────────────


def _iou_self(a: tuple[int, ...], b: tuple[int, ...]) -> float:
    """Intersection / area(a) — measures how much of `a` is inside `b`."""
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    if ix1 >= ix2 or iy1 >= iy2:
        return 0.0
    inter = (ix2 - ix1) * (iy2 - iy1)
    area = max(1, (a[2] - a[0]) * (a[3] - a[1]))
    return inter / area


@dataclass(slots=True)
class _RawBlock:
    """Tile-local intermediate before page-coordinate translation."""
    bbox:         tuple[int, int, int, int]
    text:         str
    confidence:   float
    rotation_deg: float
    words:        tuple[WordBox, ...]
    lines:        tuple[LineBox, ...]


def _dedup_raw(blocks: list[_RawBlock]) -> list[_RawBlock]:
    sorted_b = sorted(blocks, key=lambda b: -len(b.text))
    kept: list[_RawBlock] = []
    for b in sorted_b:
        drop = False
        for k in kept:
            iou = _iou_self(b.bbox, k.bbox)
            if iou > _DEDUP_IOU:
                drop = True
                break
            if iou > _SUBSTRING_IOU and b.text in k.text:
                drop = True
                break
        if not drop:
            kept.append(b)
    return kept


# ─── Tile iteration ───────────────────────────────────────────────────────


def _iter_tiles(
    image: np.ndarray,
    tile_h: int = _TILE_H,
    overlap: int = _OVERLAP,
):
    h = image.shape[0]
    step = tile_h - overlap
    y = 0
    while y < h:
        y_end = min(y + tile_h, h)
        if y_end - y < 100:
            break
        yield y, image[y:y_end].copy()
        if y_end == h:
            break
        y += step


# ─── Detector ─────────────────────────────────────────────────────────────


class LensBlocksDetector:
    """Lens-as-detector with built-in recognition.

    Reuses chrome-lens-py via async API. Lazy API construction so
    construction is cheap; first detect() call may pay endpoint patching cost.
    """

    name = "lens_blocks"

    def __init__(
        self,
        *,
        endpoint: str | None = None,
        max_concurrent: int = _MAX_CONCURRENT,
    ) -> None:
        self._endpoint = (
            endpoint or os.environ.get("LENS_ENDPOINT") or _DEFAULT_ENDPOINT
        )
        self._max_concurrent = max_concurrent
        self._api: object | None = None

    # Public ---------------------------------------------------------------

    async def detect(self, image: np.ndarray, lang: str | None) -> DetectionResult:
        """Run Lens OCR on the image.

        `lang` is currently ignored — Lens auto-detects script per tile,
        which produces better results than a forced hint (verified on
        mixed JP/EN manga pages with embedded Japanese SFX). The detected
        language is surfaced via `DetectionResult.detected_lang`.
        """
        api = await self._get_api()
        h, w = image.shape[:2]

        tiles = list(_iter_tiles(image))
        try:
            async with asyncio.TaskGroup() as tg:
                tasks = [
                    tg.create_task(_ocr_tile(api, tile, origin_y, w))
                    for origin_y, tile in tiles
                ]
        except* Exception as eg:
            raise LensUnavailableError(
                f"lens_blocks detector failed: {eg.exceptions[0]!r}"
            ) from eg

        per_tile: list[tuple[list[_RawBlock], str | None]] = [t.result() for t in tasks]
        raw_blocks: list[_RawBlock] = [b for blocks, _ in per_tile for b in blocks]
        deduped = _dedup_raw(raw_blocks)

        all_blocks = [
            TextBlock(
                bbox=b.bbox,
                polygon=None,
                confidence=b.confidence,
                text=b.text,
                detector=self.name,
                rotation_deg=b.rotation_deg,
                words=b.words,
                lines=b.lines,
            )
            for b in deduped
        ]
        kept, rejected = _filter_blocks(all_blocks)

        tile_langs = [lang for _, lang in per_tile if lang]
        detected_lang = (
            Counter(tile_langs).most_common(1)[0][0] if tile_langs else None
        )

        return DetectionResult(
            blocks=tuple(kept),
            text_already_recognized=True,
            page_size=(w, h),
            rejected=tuple(rejected),
            detected_lang=detected_lang,
        )

    # Internal -------------------------------------------------------------

    async def _get_api(self):
        if self._api is None:
            self._patch_endpoint()
            try:
                from chrome_lens_py import LensAPI
            except ImportError as e:
                raise LensUnavailableError(
                    "chrome-lens-py not installed; install or switch pipeline preset"
                ) from e
            self._api = LensAPI(max_concurrent=self._max_concurrent)
        return self._api

    def _patch_endpoint(self) -> None:
        """Repoint chrome-lens-py at the configured endpoint.

        Idempotent: only reloads request_handler if the constant changed,
        so repeated detector instances don't churn the import system.
        """
        try:
            constants = importlib.import_module("chrome_lens_py.constants")
        except ImportError:
            return
        if constants.LENS_CRUPLOAD_ENDPOINT == self._endpoint:
            return
        constants.LENS_CRUPLOAD_ENDPOINT = self._endpoint
        request_handler = importlib.import_module(
            "chrome_lens_py.core.request_handler"
        )
        importlib.reload(request_handler)


# ─── Tile worker ──────────────────────────────────────────────────────────


async def _ocr_tile(
    api,
    tile: np.ndarray,
    origin_y: int,
    page_width: int,
) -> tuple[list[_RawBlock], str | None]:
    """OCR one tile; let Lens auto-detect the script.

    Returns (blocks, detected_language). Per-tile errors are non-fatal:
    Lens commonly returns nothing on a tile of pure art, which is fine.
    """
    try:
        result = await api.process_image(
            tile,
            ocr_language="",  # "" = Lens auto-detect (DEFAULT_OCR_LANG)
            output_format="detailed",
        )
    except Exception as e:
        logger.warning("lens tile failed at y=%d: %s", origin_y, e)
        return [], None

    tile_h = tile.shape[0]
    out: list[_RawBlock] = []
    for paragraph in result.get("detailed_blocks") or []:
        block = _paragraph_to_raw(paragraph, origin_y, page_width, tile_h)
        if block is not None:
            out.append(block)

    detected_lang = _extract_content_language(result)
    return out, detected_lang


def _paragraph_to_raw(
    paragraph: dict,
    origin_y: int,
    page_width: int,
    tile_h: int,
) -> _RawBlock | None:
    text = (paragraph.get("text") or "").replace("\n", " ").strip()
    if not text:
        return None
    geom = paragraph.get("geometry") or {}
    bbox = _norm_geom_to_pixels(geom, origin_y, page_width, tile_h)
    if bbox is None:
        return None

    words = _collect_words(paragraph, origin_y, page_width, tile_h)
    lines = _collect_lines(paragraph, origin_y, page_width, tile_h)
    rotation = float(geom.get("angle_deg") or 0.0)

    return _RawBlock(
        bbox=bbox,
        text=text,
        confidence=1.0,  # Lens does not surface per-block confidence
        rotation_deg=rotation,
        words=words,
        lines=lines,
    )


def _collect_lines(
    paragraph: dict, origin_y: int, page_width: int, tile_h: int,
) -> tuple[LineBox, ...]:
    """Per-line geometry → LineBox in page pixels."""
    out: list[LineBox] = []
    for line in paragraph.get("lines") or []:
        text = (line.get("text") or "").strip()
        if not text:
            continue
        geom = line.get("geometry")
        if not geom:
            continue
        bbox = _norm_geom_to_pixels(geom, origin_y, page_width, tile_h)
        if bbox is None:
            continue
        out.append(LineBox(
            bbox=bbox,
            text=text,
            rotation_deg=float(geom.get("angle_deg") or 0.0),
        ))
    return tuple(out)


def _collect_words(
    paragraph: dict, origin_y: int, page_width: int, tile_h: int,
) -> tuple[WordBox, ...]:
    """Walk paragraph → lines → words, project bboxes to page pixels."""
    out: list[WordBox] = []
    for line in paragraph.get("lines") or []:
        for word in line.get("words") or []:
            text = (word.get("text") or "").strip()
            if not text:
                continue
            geom = word.get("geometry")
            if not geom:
                continue
            bbox = _norm_geom_to_pixels(geom, origin_y, page_width, tile_h)
            if bbox is None:
                continue
            out.append(WordBox(bbox=bbox, text=text))
    return tuple(out)


def _norm_geom_to_pixels(
    geom: dict, origin_y: int, page_width: int, tile_h: int,
) -> tuple[int, int, int, int] | None:
    """Convert Lens normalised geometry (center+size in [0, 1]) to page pixels."""
    try:
        cx = float(geom["center_x"]) * page_width
        cy = float(geom["center_y"]) * tile_h
        bw = float(geom["width"]) * page_width
        bh = float(geom["height"]) * tile_h
    except (KeyError, TypeError, ValueError):
        return None
    x1 = max(0, int(cx - bw / 2))
    x2 = min(page_width, int(cx + bw / 2))
    y1 = origin_y + int(cy - bh / 2)
    y2 = origin_y + int(cy + bh / 2)
    if x2 <= x1 or y2 <= y1:
        return None
    return (x1, y1, x2, y2)


def _extract_content_language(result: dict) -> str | None:
    """chrome-lens-py exposes content_language on the raw protobuf objects."""
    raw = result.get("raw_response_objects")
    if raw is None:
        return None
    text = getattr(raw, "text", None)
    if text is None:
        return None
    lang = getattr(text, "content_language", None) or None
    if isinstance(lang, str):
        lang = lang.strip()
    return lang or None
