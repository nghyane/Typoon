"""Scan stage — run the configured vision pipeline across a prepared chapter.

Pure async, structured concurrency via asyncio.TaskGroup. Pages process
independently (page_gate semaphore bounds RAM). Per page: detect → group
→ recognize sequential (each step depends on the previous).
"""

from __future__ import annotations

import asyncio
import logging
from collections import Counter
from dataclasses import dataclass

import cv2
import numpy as np

from typoon.adapters.mask_store import BubbleMasks, MaskStore
from typoon.adapters.prepared_reader import PreparedReader
from typoon.domain import scan
from typoon.domain.prepared import Chapter as PreparedChapter
from typoon.domain.scan import BubbleGeometry, PageGeometry
from typoon.runs.artifacts import ArtifactSink
from typoon.runs.events import Hook, PageDone
from typoon.vision.contracts import BubbleGroup, DetectionResult
from typoon.vision.runtime import VisionRuntime


__all__ = ["ScanOutput", "scan_chapter"]

logger = logging.getLogger(__name__)


# PaddleOCR DBNet (CoreML) input has a RangeDim of 128..2048 on both axes —
# a tile shorter than 128px on either side will fault deep in the model with
# "Size (32) of dimension (2) is not in allowed range". `prepare` slices
# webtoon strips into ~4k-tall chunks plus a tail; when the strip length
# divides evenly the tail can collapse to a few px (we've seen 8 and 15px).
# Those slivers never carry text, so we skip them.
_MIN_PAGE_DIM = 128


# ─── Output type ──────────────────────────────────────────────────────────


@dataclass(frozen=True, slots=True)
class ScanOutput:
    """Output of scan_chapter — pure data, no persistence logic."""
    chapter:        scan.Chapter
    masks:          MaskStore
    geometry:       list[PageGeometry]
    detected_lang:  str | None = None  # majority vote across pages, if surfaced

    def bubble_records(self) -> list[dict]:
        return [
            {
                "page_index":             b.page_index,
                "bubble_idx":             b.idx,
                "source_text":            b.source_text,
                "confidence":             b.confidence,
                "shape_kind":             b.shape_kind,
                "rotation_deg":           b.rotation_deg,
                "src_font_size_px":       b.src_font_size_px,
                "src_line_count":         b.src_line_count,
                "src_avg_chars_per_line": b.src_avg_chars_per_line,
                "text_direction":          b.text_direction,
            }
            for b in self.chapter.all_bubbles
        ]

    def geometry_records(self) -> list[dict]:
        return [
            {
                "page_index": pg.page_index,
                "width":  pg.width,
                "height": pg.height,
                "bubbles": [
                    {
                        "bubble_idx":             bg.bubble_idx,
                        "polygon":                bg.polygon,
                        "rotation_deg":           bg.rotation_deg,
                        "src_font_size_px":       bg.src_font_size_px,
                        "src_line_count":         bg.src_line_count,
                        "src_avg_chars_per_line": bg.src_avg_chars_per_line,
                        "text_direction":          bg.text_direction,
                    }
                    for bg in pg.bubbles
                ],
            }
            for pg in self.geometry
        ]


# ─── Per-page artefacts, kept off the hot path ────────────────────────────


@dataclass(frozen=True, slots=True)
class _PageOutput:
    page:         scan.Page
    geometry:     PageGeometry
    bubble_masks: tuple[BubbleMasks, ...]
    detected_lang: str | None = None


# ─── Entry point ──────────────────────────────────────────────────────────


async def scan_chapter(
    prepared: PreparedChapter,
    reader: PreparedReader,
    runtime: VisionRuntime,
    *,
    source_lang: str | None = None,
    chapter_id: int = 0,
    hook: Hook | None = None,
    artifacts: ArtifactSink | None = None,
) -> ScanOutput:
    """Run the configured vision pipeline on every prepared page.

    Page-parallel via asyncio.TaskGroup, bounded by runtime.page_gate.
    Per-page state is local; `MaskStore.put` is the only shared mutation
    and is gated by an asyncio.Lock so the call site stays simple.
    """
    total = prepared.page_count
    page_results: list[_PageOutput | None] = [None] * total
    masks_store = MaskStore()
    masks_lock  = asyncio.Lock()

    async def scan_one(index: int) -> None:
        async with runtime.page_gate:
            image = await asyncio.to_thread(reader.read_rgb, index)
            h, w = image.shape[:2]
            detection = None

            if min(h, w) < _MIN_PAGE_DIM:
                page_results[index] = _PageOutput(
                    page=scan.Page(index=index, width=w, height=h, bubbles=()),
                    geometry=PageGeometry(page_index=index, width=w, height=h, bubbles=()),
                    bubble_masks=(),
                )
            else:
                detection = await _detect(runtime, image, source_lang)
                groups = await _group(runtime, image, detection, source_lang)
                if runtime.recognizer is not None:
                    groups = await runtime.recognizer.recognize(image, groups, source_lang)
                page_results[index] = _assemble_page(
                    index, w, h, groups, detection.detected_lang,
                )

                if artifacts is not None:
                    await asyncio.to_thread(
                        _write_artifacts, artifacts, index, image, detection, groups,
                    )

            assert page_results[index] is not None
            result = page_results[index]
            async with masks_lock:
                for bubble, bm in zip(result.page.bubbles, result.bubble_masks):
                    masks_store.put(index, bubble.idx, bm)
                # Store CTD UNet bubble mask if provided (ctd_blocks detector)
                if detection is not None and detection.bubble_mask is not None:
                    masks_store.put_bubble_mask(index, detection.bubble_mask)

            if hook is not None:
                hook.on(PageDone(
                    chapter_id=chapter_id, stage="scan",
                    page_index=index, page_total=total,
                ))

    async with asyncio.TaskGroup() as tg:
        for i in range(total):
            tg.create_task(scan_one(i))

    pages    = tuple(r.page for r in page_results)  # type: ignore[union-attr]
    geometry = [r.geometry for r in page_results]    # type: ignore[union-attr]
    detected_lang = _vote_lang(
        [r.detected_lang for r in page_results if r is not None]
    )

    if (
        source_lang
        and detected_lang
        and _scripts_differ(source_lang, detected_lang)
    ):
        logger.warning(
            "scan: source_lang=%r but detector saw %r across the chapter "
            "(different scripts — likely wrong chapter language).",
            source_lang, detected_lang,
        )

    if artifacts is not None:
        await asyncio.to_thread(
            artifacts.write_json, "04_ocr", "ocr_all_pages.json",
            [
                {
                    "page": p.index,
                    "bubbles": [
                        {"idx": b.idx, "text": b.source_text, "confidence": b.confidence}
                        for b in p.bubbles
                    ],
                }
                for p in pages
            ],
        )

    return ScanOutput(
        chapter=scan.Chapter(prepared=prepared, pages=pages),
        masks=masks_store,
        geometry=geometry,
        detected_lang=detected_lang,
    )


# ─── Pipeline step wrappers ───────────────────────────────────────────────


async def _detect(
    runtime: VisionRuntime, image: np.ndarray, lang: str | None,
) -> DetectionResult:
    async with runtime.detect_gate:
        return await runtime.detector.detect(image, lang)


async def _group(
    runtime: VisionRuntime,
    image: np.ndarray,
    detection: DetectionResult,
    lang: str | None,
) -> tuple[BubbleGroup, ...]:
    return await runtime.grouper.group(image, detection, lang)


# ─── BubbleGroup → domain.scan.Bubble ─────────────────────────────────────


def _assemble_page(
    index: int,
    width: int,
    height: int,
    groups: tuple[BubbleGroup, ...],
    detected_lang: str | None,
) -> _PageOutput:
    bubbles:      list[scan.Bubble]      = []
    geom_list:    list[BubbleGeometry]   = []
    bubble_masks: list[BubbleMasks]      = []

    for i, g in enumerate(groups):
        polygon_list = [list(p) for p in g.polygon]

        ts = g.typesetting
        src_font  = ts.font_size_px if ts else 0
        src_lines = ts.line_count if ts else 0
        src_chars = ts.avg_chars_per_line if ts else 0.0

        # Strip inline noise tokens (watermark domains, logo emoji,
        # scanlation tags) that Lens OCR appended to dialogue text.
        # This keeps the story content while making is_auto_skip accurate.
        from typoon.stages.noise import strip_noise_tokens
        clean_text = strip_noise_tokens(g.text)

        bubble = scan.Bubble(
            idx=i,
            page_index=index,
            source_text=clean_text,
            confidence=g.confidence,
            polygon=polygon_list,
            shape_kind=g.shape_kind,
            rotation_deg=g.rotation_deg,
            src_font_size_px=src_font,
            src_line_count=src_lines,
            src_avg_chars_per_line=src_chars,
            text_direction=g.text_direction,
        )
        bubbles.append(bubble)
        geom_list.append(BubbleGeometry(
            bubble_idx=i,
            polygon=polygon_list,
            rotation_deg=g.rotation_deg,
            src_font_size_px=src_font,
            src_line_count=src_lines,
            src_avg_chars_per_line=src_chars,
            text_direction=g.text_direction,
        ))
        bubble_masks.append(BubbleMasks(
            erase_masks=tuple(g.erase_masks),
            text_masks=tuple(g.text_masks),
        ))

    return _PageOutput(
        page=scan.Page(
            index=index, width=width, height=height,
            bubbles=tuple(bubbles),
        ),
        geometry=PageGeometry(
            page_index=index, width=width, height=height,
            bubbles=tuple(geom_list),
        ),
        bubble_masks=tuple(bubble_masks),
        detected_lang=detected_lang,
    )


def _vote_lang(samples: list[str | None]) -> str | None:
    """Majority vote across per-page detected languages.

    Lens can mis-detect on art-heavy pages; the per-chapter majority is
    a stronger signal than any single page. Returns None if no pages
    surfaced a language.
    """
    valid = [s for s in samples if s]
    if not valid:
        return None
    return Counter(valid).most_common(1)[0][0]


def _normalize_lang(code: str) -> str:
    """Strip region tag for comparison: 'zh-CN' → 'zh', 'en-US' → 'en'."""
    return code.lower().split("-", 1)[0]


# Coarse script grouping for cross-language warning. Two languages in the
# same script (es/en/vi/pt/fr → Latin) are commonly confused by Lens
# tile-by-tile detection and a warning produces more noise than signal.
# We only warn when the detector sees a *different script* than declared
# — that's a strong indicator the wrong chapter language was chosen.
_LANG_SCRIPT: dict[str, str] = {
    "ja": "japanese",
    "ko": "korean",
    "zh": "han",
    "th": "thai",
    "ar": "arabic",
    "he": "hebrew",
    "ru": "cyrillic",
    "uk": "cyrillic",
    "bg": "cyrillic",
    # Default for everything not listed: latin (en/es/fr/vi/pt/it/de/...).
}


def _script_of(code: str) -> str:
    return _LANG_SCRIPT.get(_normalize_lang(code), "latin")


def _scripts_differ(a: str, b: str) -> bool:
    return _script_of(a) != _script_of(b)


# ─── Visual artefacts ─────────────────────────────────────────────────────


_PALETTE = [
    (255, 80, 80), (80, 200, 255), (80, 255, 120), (255, 200, 0),
    (200, 80, 255), (255, 140, 0), (0, 220, 200), (255, 80, 180),
]
_RED       = (255, 70, 70)
_MAGENTA   = (255, 0, 200)
_GREY      = (140, 140, 140)


def _write_artifacts(
    artifacts: ArtifactSink,
    index: int,
    image: np.ndarray,
    detection: DetectionResult,
    groups: tuple[BubbleGroup, ...],
) -> None:
    """Write debug overlays + JSON. Runs inside asyncio.to_thread (cv2 is sync)."""
    artifacts.write_json("02_detect", f"page_{index:04d}_state.json", {
        "page": index,
        "width": detection.page_size[0],
        "height": detection.page_size[1],
        "detector": detection.blocks[0].detector if detection.blocks else "",
        "detected_lang": detection.detected_lang,
        "n_blocks_kept":     len(detection.blocks),
        "n_blocks_rejected": len(detection.rejected),
        "rejected_reasons":  [reason for _, reason in detection.rejected],
        "groups": [
            {
                "idx": i,
                "bbox": list(g.bbox),
                "text": g.text,
                "confidence": g.confidence,
                "source": g.source,
                "shape_kind": g.shape_kind,
                "rotation_deg": g.rotation_deg,
                "used_fallback": g.used_fallback,
                "n_text_masks": len(g.text_masks),
                "n_erase_masks": len(g.erase_masks),
            }
            for i, g in enumerate(groups)
        ],
    })

    detect_overlay = _draw_detect_overlay(image, detection)
    artifacts.write_image("02_detect", f"page_{index:04d}_detect.png", detect_overlay)

    group_overlay = _draw_group_overlay(image, groups)
    artifacts.write_image("03_group", f"page_{index:04d}_groups.png", group_overlay)


def _draw_detect_overlay(image: np.ndarray, detection: DetectionResult) -> np.ndarray:
    out = image.copy()
    for b in detection.blocks:
        cv2.rectangle(out, (b.bbox[0], b.bbox[1]), (b.bbox[2], b.bbox[3]),
                      (80, 200, 80), 2)
    for b, _reason in detection.rejected:
        cv2.rectangle(out, (b.bbox[0], b.bbox[1]), (b.bbox[2], b.bbox[3]),
                      _GREY, 1)
    return out


def _draw_group_overlay(image: np.ndarray, groups: tuple[BubbleGroup, ...]) -> np.ndarray:
    out = image.copy()
    for i, g in enumerate(groups):
        color = _MAGENTA if g.shape_kind == "burst" else _PALETTE[i % len(_PALETTE)]
        cv2.rectangle(out, (g.bbox[0], g.bbox[1]), (g.bbox[2], g.bbox[3]),
                      color, 3)
        text = g.text[:24].replace("\n", " ")
        _label(out, g.bbox[0], g.bbox[1], f"{i}: {text}", color)
    return out


def _label(image, x: int, y: int, text: str, color) -> None:
    y = max(14, int(y))
    x = max(0, int(x))
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale, thickness = 0.45, 1
    (tw, th), _ = cv2.getTextSize(text, font, scale, thickness)
    cv2.rectangle(image, (x, y - th - 4), (x + tw + 4, y + 2), color, -1)
    cv2.putText(image, text, (x + 2, y - 2), font, scale, (255, 255, 255),
                thickness, cv2.LINE_AA)
