"""Scan stage — PreparedChapter + VisionRuntime → ScanResult."""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from typoon.adapters.mask_store import BubbleMasks, MaskStore
from typoon.adapters.vision_runtime import VisionRuntime
from typoon.domain.prepared import PreparedChapter
from typoon.domain.scan import (
    BubbleGeometry,
    ScannedBubble,
    ScannedChapter,
    ScannedPage,
)
from typoon.runs.artifacts import ArtifactSink
from typoon.vision.draw import CYAN, GREEN, PALETTE, RED, YELLOW, label, rect
from typoon.vision.inspect import state_to_dict
from typoon.vision.types import PageScanState, VisualTextGroup


@dataclass(frozen=True)
class ScanResult:
    """Output of scan_chapter: typed chapter + pixel masks decoupled."""

    chapter: ScannedChapter
    masks:   MaskStore


def scan_chapter(
    prepared: PreparedChapter,
    runtime: VisionRuntime,
    *,
    artifacts: ArtifactSink | None = None,
) -> ScanResult:
    """Run vision pipeline on every prepared page.

    Returns ScanResult with immutable ScannedChapter and a MaskStore
    holding erase/text masks keyed by (page_index, bubble_idx).
    Masks are not part of the domain — they are consumed by render stage only.
    """
    scanned_pages: list[ScannedPage] = []
    masks = MaskStore()
    all_ocr: list[dict] = []

    for index in range(prepared.page_count):
        image = _load_rgb(prepared.page_path(index))
        state = runtime.scan_page_state(image)
        h, w = image.shape[:2]

        bubbles, page_masks = _extract_page(index, state)

        for sb, bm in zip(bubbles, page_masks):
            masks.put(sb.page_index, sb.idx, bm)

        scanned_pages.append(ScannedPage(
            index=index, width=w, height=h,
            bubbles=tuple(bubbles),
        ))

        if artifacts is not None:
            _write_page_artifacts(artifacts, index, image, state)
            all_ocr.append(_ocr_dict(index, bubbles))

    if artifacts is not None:
        artifacts.write_json("04_ocr", "ocr.json", {"pages": all_ocr})

    chapter = ScannedChapter(prepared=prepared, pages=tuple(scanned_pages))
    return ScanResult(chapter=chapter, masks=masks)


# ── Conversion ───────────────────────────────────────────────────────


def _extract_page(
    page_index: int,
    state: PageScanState,
) -> tuple[list[ScannedBubble], list[BubbleMasks]]:
    from typoon.vision.text_grouping import to_visual_text_groups
    groups = to_visual_text_groups(state)
    bubbles = []
    page_masks = []
    for i, g in enumerate(groups):
        bubbles.append(_to_scanned_bubble(i, page_index, g))
        page_masks.append(BubbleMasks(
            erase_masks=tuple(g.erase_masks),
            text_masks=tuple(g.text_masks),
        ))
    return bubbles, page_masks


def _to_scanned_bubble(i: int, page_index: int, g: VisualTextGroup) -> ScannedBubble:
    return ScannedBubble(
        idx=i,
        page_index=page_index,
        source_text=g.text,
        confidence=g.confidence,
        geometry=BubbleGeometry(
            polygon=g.render_polygon,
            fit_bbox=g.fit_bbox,
            erase_bbox=g.erase_bbox,
            text_bbox=g.text_bbox,
        ),
    )


# ── Artifacts ────────────────────────────────────────────────────────


def _write_page_artifacts(
    artifacts: ArtifactSink,
    index: int,
    image: np.ndarray,
    state: PageScanState,
) -> None:
    tag = f"page_{index:04d}"
    artifacts.write_image("02_detect", f"{tag}_boxes.png", _draw_boxes(image, state))
    artifacts.write_image("03_group", f"{tag}_groups.png", _draw_groups(image, state))
    artifacts.write_json("03_group", f"{tag}_groups.json", state_to_dict(index, tag, state))
    artifacts.write_image("04_ocr", f"{tag}_masks.png", _draw_mask_highlight(image, state))


def _ocr_dict(page_index: int, bubbles: list[ScannedBubble]) -> dict:
    return {
        "page": page_index,
        "bubbles": [
            {"idx": b.idx, "text": b.source_text, "confidence": b.confidence}
            for b in bubbles
        ],
    }


# ── Drawing ──────────────────────────────────────────────────────────


def _draw_boxes(image: np.ndarray, state: PageScanState) -> np.ndarray:
    out = image.copy()
    for unit in state.units:
        color = RED if unit.is_noise else CYAN
        rect(out, unit.bbox, color, 2)
        label(out, unit.bbox[0], unit.bbox[1], f"u{unit.idx} {unit.unit_ocr_conf:.2f}", color)
    return out


def _draw_groups(image: np.ndarray, state: PageScanState) -> np.ndarray:
    out = image.copy()
    for scope in state.scopes:
        rect(out, scope.bbox, YELLOW, 2)
        label(out, scope.bbox[0], scope.bbox[1], f"s{scope.idx} {scope.confidence:.2f}", YELLOW)
    for group in state.groups:
        color = GREEN if group.accepted else RED
        rect(out, group.fit_bbox, color, 3)
        text = (group.ocr_text or "")[:24].replace("\n", " ")
        label(out, group.fit_bbox[0], group.fit_bbox[1], f"g{group.idx} {group.ocr_conf:.2f} {text}", color)
    return out


def _draw_mask_highlight(image: np.ndarray, state: PageScanState) -> np.ndarray:
    out = image.copy()
    overlay = out.copy()
    accepted = [g for g in state.groups if g.accepted]
    for gi, group in enumerate(accepted):
        color = PALETTE[gi % len(PALETTE)]
        for unit_idx in group.unit_indices:
            mask = state.units[unit_idx].region.mask
            if mask is None:
                continue
            h, w = mask.image.shape[:2]
            y1, y2 = max(0, mask.y), min(out.shape[0], mask.y + h)
            x1, x2 = max(0, mask.x), min(out.shape[1], mask.x + w)
            if y2 <= y1 or x2 <= x1:
                continue
            crop = mask.image[y1 - mask.y:y2 - mask.y, x1 - mask.x:x2 - mask.x]
            overlay[y1:y2, x1:x2][crop > 0] = color
    cv2.addWeighted(overlay, 0.55, out, 0.45, 0, out)
    return out


def _load_rgb(path) -> np.ndarray:
    bgr = cv2.imread(str(path))
    if bgr is None:
        raise FileNotFoundError(f"Cannot read prepared page: {path}")
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
