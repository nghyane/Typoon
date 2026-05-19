"""Lens tile pass (Phase A) — gross localisation across a whole page.

Lens resizes any input over ~1000px to 1000px on the longest axis;
straight full-page calls lose tiny SFX glyphs. The page is therefore
sliced into 720 × 900 vertical tiles with 200 px overlap, each tile
OCR'd independently, and the per-tile blocks deduplicated by
length-weighted bbox containment.

What this pass produces is **coarse**: it might miss glyph edges at
tile splits and gets tategaki direction wrong inside complex
bubbles. The bubble pass (Phase B) authoritatively re-OCRs any DETR
region that ends up incomplete.
"""

from __future__ import annotations

import asyncio
import logging
from collections import Counter

import numpy as np

from ...contracts import LineBox, TextBlock, WordBox
from .geometry import Frame, paragraph_to_raw
from .retry import lens_call_with_retry
from .types import RawBlock


__all__ = ["run", "iter_tiles"]

logger = logging.getLogger(__name__)


# Tile geometry (px).
_TILE_H = 900
_OVERLAP = 200
# IoU thresholds for tile-edge dedup.
_DEDUP_IOU = 0.5
_SUBSTRING_IOU = 0.05


async def run(
    api,
    image: np.ndarray,
    lang_hint: str,
) -> tuple[list[TextBlock], str | None]:
    """OCR every tile concurrently, dedup, return TextBlocks + detected lang."""
    h, w = image.shape[:2]
    tiles = list(iter_tiles(image))
    if not tiles:
        return [], None

    per_tile = await asyncio.gather(*[
        _ocr_tile(api, tile, origin_y, w, h, lang_hint)
        for origin_y, tile in tiles
    ])
    raw = [b for blocks, _ in per_tile for b in blocks]
    deduped = _dedup_raw(raw)

    blocks = [
        TextBlock(
            bbox=b.bbox, polygon=None, confidence=b.confidence,
            text=b.text, detector="lens_blocks/tile",
            rotation_deg=b.rotation_deg, words=b.words, lines=b.lines,
            text_direction=b.text_direction,
        )
        for b in deduped
    ]

    tile_langs = [tl for _, tl in per_tile if tl]
    detected_lang = (
        Counter(tile_langs).most_common(1)[0][0] if tile_langs else None
    )
    return blocks, detected_lang


def iter_tiles(
    image: np.ndarray, tile_h: int = _TILE_H, overlap: int = _OVERLAP,
):
    """Yield (origin_y, tile_pixels). Tail tiles shorter than 100 px are dropped."""
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


async def _ocr_tile(
    api, tile: np.ndarray, origin_y: int,
    page_w: int, page_h: int, lang_hint: str,
) -> tuple[list[RawBlock], str | None]:
    try:
        result = await lens_call_with_retry(
            api, tile,
            ocr_language=lang_hint, output_format="detailed",
            label=f"lens tile y={origin_y}",
        )
    except Exception as e:
        logger.warning("lens tile failed at y=%d after retries: %s", origin_y, e)
        return [], None

    tile_h = tile.shape[0]
    frame = Frame(
        origin_x=0, origin_y=origin_y,
        frame_w=page_w, frame_h=tile_h, scale=1,
    )
    raw_paragraphs = _raw_paragraphs(result)

    out: list[RawBlock] = []
    for i, paragraph in enumerate(result.get("detailed_blocks") or []):
        raw_para = raw_paragraphs[i] if i < len(raw_paragraphs) else None
        block = paragraph_to_raw(paragraph, raw_para, frame, (page_w, page_h))
        if block is not None:
            out.append(block)

    return out, _extract_content_language(result)


# ─── Dedup across tile overlaps ───────────────────────────────────────────


def _dedup_raw(blocks: list[RawBlock]) -> list[RawBlock]:
    """Length-weighted bbox containment dedup.

    Sort by text length descending; drop a block if (a) IoU-self with
    any kept block > _DEDUP_IOU, or (b) its text is a substring of a
    kept block with non-trivial overlap.
    """
    sorted_b = sorted(blocks, key=lambda b: -len(b.text))
    kept: list[RawBlock] = []
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


def _iou_self(a: tuple[int, ...], b: tuple[int, ...]) -> float:
    """Intersection / area(a) — fraction of `a` that lies inside `b`."""
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    if ix1 >= ix2 or iy1 >= iy2:
        return 0.0
    inter = (ix2 - ix1) * (iy2 - iy1)
    area = max(1, (a[2] - a[0]) * (a[3] - a[1]))
    return inter / area


# ─── Lens response helpers ────────────────────────────────────────────────


def _raw_paragraphs(result: dict) -> list:
    """Return ``text_layout.paragraphs`` from the raw proto, if surfaced."""
    raw = result.get("raw_response_objects")
    if raw is None:
        return []
    text = getattr(raw, "text", None)
    if text is None:
        return []
    layout = getattr(text, "text_layout", None)
    if layout is None:
        return []
    return list(getattr(layout, "paragraphs", []) or [])


def _extract_content_language(result: dict) -> str | None:
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
