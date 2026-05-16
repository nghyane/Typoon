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
  - cross_column    — paragraph whose lines sit inside ≥2 other paragraphs
                      (tile-boundary artefact gluing tategaki column tails).
"""

from __future__ import annotations

import asyncio
import importlib
import logging
import math
import os
import statistics
import unicodedata
from collections import Counter
from dataclasses import dataclass
from functools import cache

import numpy as np
from PIL import Image as _PILImage

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

# Cross-column artefact: at least this many of a paragraph's lines must
# land inside (line area / parent area) other paragraphs to call it a
# tile-boundary union hallucination.
_CROSS_COLUMN_MIN_LINES_ABSORBED = 2
_CROSS_COLUMN_LINE_INSIDE_RATIO  = 0.70  # fraction of line area that must overlap parent

# Row recognition gap: a row whose width is below this fraction of the
# block's other rows' median width is flagged for re-OCR. Empirically
# Lens drops glyphs around dense ellipsis runs (e.g. "难道······他 才是")
# and emits only the unaffected suffix; the row geometry stays correct
# so the y-band still localises the missed glyphs.
_ROW_GAP_SHORT_RATIO = 0.5
_ROW_GAP_MIN_LINES   = 3      # need >=3 lines to compute a meaningful median
_ROW_REOCR_MIN_DIM   = 200    # min(h, w) target for the re-OCR crop (upscale below)
_ROW_REOCR_PAD_Y_FRAC = 0.45  # vertical breathing room around the row crop
_ROW_REOCR_PAD_X_PX   = 6

# Mapping from our `source_lang` to Lens `ocr_language`. English / unset
# stays "" (auto) so mixed-script pages (e.g. JP manga with EN SFX) keep
# both scripts. Other source languages pass through as an explicit hint.
_LENS_LANG_HINTS: dict[str, str] = {
    "ja":      "ja",
    "ja-JP":   "ja",
    "zh":      "zh-Hans",
    "zh-CN":   "zh-Hans",
    "zh-Hans": "zh-Hans",
    "zh-Hant": "zh-Hant",
    "zh-TW":   "zh-Hant",
    "zh-HK":   "zh-Hant",
    "ko":      "ko",
    "ko-KR":   "ko",
    "vi":      "vi",
    "vi-VN":   "vi",
}


def _lens_lang_hint(source_lang: str | None) -> str:
    """Map our `source_lang` to a Lens `ocr_language` argument.

    Returns ``""`` (auto-detect) for English or unset — manga pages
    routinely mix scripts (JP dialogue with EN sound effects) and a
    hard English hint suppresses non-Latin recognition. JP/CN/KO/VI
    pass through as explicit hints; unknown locales fall through to
    auto as well.
    """
    if not source_lang:
        return ""
    if source_lang.lower().startswith("en"):
        return ""
    return _LENS_LANG_HINTS.get(source_lang, "")


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

    kept, cross_rejected = _drop_cross_column_artifacts(kept)
    rejected.extend(cross_rejected)
    return kept, rejected


def _bbox_inside_ratio(child: tuple[int, int, int, int],
                       parent: tuple[int, int, int, int]) -> float:
    """Fraction of `child` area that lies inside `parent`."""
    cx1, cy1, cx2, cy2 = child
    px1, py1, px2, py2 = parent
    ix1, iy1 = max(cx1, px1), max(cy1, py1)
    ix2, iy2 = min(cx2, px2), min(cy2, py2)
    if ix1 >= ix2 or iy1 >= iy2:
        return 0.0
    inter = (ix2 - ix1) * (iy2 - iy1)
    area = max(1, (cx2 - cx1) * (cy2 - cy1))
    return inter / area


def _drop_cross_column_artifacts(
    blocks: list[TextBlock],
) -> tuple[list[TextBlock], list[tuple[TextBlock, str]]]:
    """Drop paragraphs whose lines are absorbed by ≥2 other paragraphs.

    Lens occasionally emits a phantom horizontal "paragraph" near a tile
    overlap by stitching together the tail line of multiple adjacent
    tategaki columns. Each constituent line bbox geometrically sits inside
    a different real column paragraph, but pairwise bbox-IoU never reaches
    the dedup threshold because the phantom spans across the columns.

    Signal: line bboxes contained in ≥2 distinct other paragraphs.
    Substring text match is intentionally not required — Lens often
    mis-OCRs the partial glyphs in the union (e.g. "军答" instead of
    "答"), so we rely on geometry only.
    """
    if len(blocks) < 3:
        return list(blocks), []

    kept: list[TextBlock] = []
    rejected: list[tuple[TextBlock, str]] = []
    for i, b in enumerate(blocks):
        if len(b.lines) < _CROSS_COLUMN_MIN_LINES_ABSORBED:
            kept.append(b)
            continue
        absorbing_parents: set[int] = set()
        for ln in b.lines:
            for j, other in enumerate(blocks):
                if j == i:
                    continue
                if _bbox_inside_ratio(ln.bbox, other.bbox) >= _CROSS_COLUMN_LINE_INSIDE_RATIO:
                    absorbing_parents.add(j)
                    break
        if len(absorbing_parents) >= _CROSS_COLUMN_MIN_LINES_ABSORBED:
            rejected.append((b, "cross_column"))
        else:
            kept.append(b)
    return kept, rejected


# ─── Row-level recognition gap recovery ───────────────────────────────────


def _suspicious_line_indices(block: TextBlock) -> list[int]:
    """Non-edge rows whose width is anomalously short vs the rest.

    Lens occasionally drops glyphs around a dense run of decoration
    characters (e.g. CJK ellipsis 「······」), keeping only the
    unaffected suffix on that row. The row's bbox geometry stays
    correct so we can localise the dropped glyphs by re-OCRing the
    full-width band at that y-range.

    First / last lines are excluded — those are often legitimately
    short (closing punctuation, ragged justification at paragraph
    edges) and shouldn't trigger recovery.
    """
    lines = block.lines
    if len(lines) < _ROW_GAP_MIN_LINES:
        return []
    widths = [max(1, l.bbox[2] - l.bbox[0]) for l in lines]
    out: list[int] = []
    for i in range(1, len(lines) - 1):
        others = widths[:i] + widths[i + 1:]
        median_w = statistics.median(others)
        if widths[i] / median_w < _ROW_GAP_SHORT_RATIO:
            out.append(i)
    return out


def _row_recover_crop_bbox(
    block: TextBlock,
    line_idx: int,
    page_w: int,
    page_h: int,
) -> tuple[int, int, int, int]:
    """Crop spanning the full block width at the row's y-range plus padding."""
    line = block.lines[line_idx]
    lh = max(1, line.bbox[3] - line.bbox[1])
    pad_y = int(lh * _ROW_REOCR_PAD_Y_FRAC)
    x1 = max(0, block.bbox[0] - _ROW_REOCR_PAD_X_PX)
    x2 = min(page_w, block.bbox[2] + _ROW_REOCR_PAD_X_PX)
    y1 = max(0, line.bbox[1] - pad_y)
    y2 = min(page_h, line.bbox[3] + pad_y)
    return (x1, y1, x2, y2)


async def _reocr_row(
    api,
    image: np.ndarray,
    crop_bbox: tuple[int, int, int, int],
    lang_hint: str,
) -> str:
    """Re-OCR one row crop; return concatenated paragraph text.

    Crops shorter than `_ROW_REOCR_MIN_DIM` on the short axis are
    upscaled with Lanczos resampling. Lens recognition quality drops
    sharply below ~200px on the short side; the upscale is the only
    thing that consistently recovers missed glyphs in the
    full-page→full-row path.
    """
    x1, y1, x2, y2 = crop_bbox
    crop = image[y1:y2, x1:x2]
    if crop.size == 0:
        return ""
    h, w = crop.shape[:2]
    if min(h, w) < _ROW_REOCR_MIN_DIM:
        scale = max(1, int(np.ceil(_ROW_REOCR_MIN_DIM / min(h, w))))
        pil = _PILImage.fromarray(crop).resize(
            (w * scale, h * scale), _PILImage.LANCZOS
        )
        crop = np.asarray(pil)
    try:
        result = await api.process_image(
            crop,
            ocr_language=lang_hint,
            output_format="detailed",
        )
    except Exception as e:
        logger.warning("lens row re-OCR failed: %s", e)
        return ""
    paragraphs = result.get("detailed_blocks") or []
    chunks: list[str] = []
    for p in paragraphs:
        t = (p.get("text") or "").replace("\n", " ").strip()
        if t:
            chunks.append(t)
    return "  ".join(chunks)


async def _recover_row_gaps(
    api,
    image: np.ndarray,
    blocks: list[TextBlock],
    lang_hint: str,
) -> list[TextBlock]:
    """Re-OCR suspicious rows; splice recovered text back into the block.

    Only mutates `block.text` and `block.lines[i].text` — geometry and
    word-level bboxes are preserved. Words are intentionally NOT
    rewritten: their bboxes correspond to glyphs Lens did detect, and
    grouper mask building handles the recognition gap on its own
    (row-aware glyph mask in `lens_native._build_glyph_mask`).
    """
    page_h, page_w = image.shape[:2]
    out: list[TextBlock] = []
    for block in blocks:
        sus = _suspicious_line_indices(block)
        if not sus:
            out.append(block)
            continue
        new_lines = list(block.lines)
        recovered_any = False
        for idx in sus:
            crop_bbox = _row_recover_crop_bbox(block, idx, page_w, page_h)
            new_text = await _reocr_row(api, image, crop_bbox, lang_hint)
            if not new_text or new_text == new_lines[idx].text:
                continue
            old = new_lines[idx]
            new_lines[idx] = LineBox(
                bbox=old.bbox,
                text=new_text,
                rotation_deg=old.rotation_deg,
            )
            recovered_any = True
            logger.info(
                "lens row recovery: %r -> %r", old.text, new_text,
            )
        if not recovered_any:
            out.append(block)
            continue
        merged_text = " ".join(l.text for l in new_lines if l.text.strip())
        out.append(
            TextBlock(
                bbox=block.bbox,
                polygon=block.polygon,
                confidence=block.confidence,
                text=merged_text,
                detector=block.detector,
                text_mask=block.text_mask,
                rotation_deg=block.rotation_deg,
                words=block.words,
                lines=tuple(new_lines),
                text_direction=block.text_direction,
            )
        )
    return out


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

        `lang` is the upstream source-language hint (BCP-47-ish, e.g.
        ``"zh-Hans"``, ``"ja"``). It maps via `_lens_lang_hint` to a
        Lens `ocr_language` argument; English / unset stay on Lens
        auto-detect so mixed-script pages (JP dialogue + EN SFX) keep
        both scripts. The Lens-detected language is still surfaced via
        `DetectionResult.detected_lang`.

        After the main tile pass, blocks with a suspiciously short
        non-edge line are re-OCRed at row granularity to recover glyphs
        Lens dropped around dense decoration runs (e.g. ellipsis).
        """
        api = await self._get_api()
        h, w = image.shape[:2]
        lang_hint = _lens_lang_hint(lang)

        tiles = list(_iter_tiles(image))
        try:
            async with asyncio.TaskGroup() as tg:
                tasks = [
                    tg.create_task(
                        _ocr_tile(api, tile, origin_y, w, lang_hint)
                    )
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

        tile_langs = [tl for _, tl in per_tile if tl]
        detected_lang = (
            Counter(tile_langs).most_common(1)[0][0] if tile_langs else None
        )

        # Row-level recovery for blocks where Lens dropped glyphs in the
        # middle of a line. The hint passed here prefers the upstream
        # `lang` over Lens's per-page detection, but falls back to it.
        recovery_hint = lang_hint or _lens_lang_hint(detected_lang)
        kept = await _recover_row_gaps(api, image, kept, recovery_hint)

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
    lang_hint: str = "",
) -> tuple[list[_RawBlock], str | None]:
    """OCR one tile.

    Returns (blocks, detected_language). Per-tile errors are non-fatal:
    Lens commonly returns nothing on a tile of pure art, which is fine.
    `lang_hint` is the Lens `ocr_language` argument; ``""`` = auto.
    """
    try:
        result = await api.process_image(
            tile,
            ocr_language=lang_hint,
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
    """Convert Lens normalised geometry (center+size in [0, 1]) to page pixels.

    Lens reports ``width`` and ``height`` in the text box's **own axes**,
    paired with ``rotation_z`` (radians, converted to ``angle_deg``
    upstream). For axis-aligned text the rotation is ≈0 and the bbox is
    width×height around (cx, cy). For rotated text — manga SFX (10–45°),
    side-rotated watermarks (~90°) — naively using width/height as page
    AABB swaps the visual extents (a watermark "manhuaren.com" rotated
    90° gets reported as 93px wide × 14px tall when its on-page extent
    is the opposite).

    We rotate the four corners around the centre and take the axis-aligned
    page bounding box. Result: bbox always reflects on-page pixel extent.
    """
    try:
        cx = float(geom["center_x"]) * page_width
        cy = float(geom["center_y"]) * tile_h
        bw = float(geom["width"]) * page_width
        bh = float(geom["height"]) * tile_h
    except (KeyError, TypeError, ValueError):
        return None
    angle_deg = float(geom.get("angle_deg") or 0.0)
    if abs(angle_deg) < 0.5:
        # Cheap path — axis-aligned (most paragraphs).
        x1 = cx - bw / 2
        x2 = cx + bw / 2
        y1 = cy - bh / 2
        y2 = cy + bh / 2
    else:
        rad = math.radians(angle_deg)
        cos_t = math.cos(rad)
        sin_t = math.sin(rad)
        hx, hy = bw / 2, bh / 2
        # Four corners of the local rect, rotated around (0, 0)
        corners = [(-hx, -hy), (hx, -hy), (hx, hy), (-hx, hy)]
        xs = [cx + x * cos_t - y * sin_t for x, y in corners]
        ys = [cy + x * sin_t + y * cos_t for x, y in corners]
        x1, x2 = min(xs), max(xs)
        y1, y2 = min(ys), max(ys)
    x1_i = max(0, int(x1))
    x2_i = min(page_width, int(x2))
    y1_i = origin_y + int(y1)
    y2_i = origin_y + int(y2)
    if x2_i <= x1_i or y2_i <= y1_i:
        return None
    return (x1_i, y1_i, x2_i, y2_i)


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
