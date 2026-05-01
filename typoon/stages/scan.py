"""Scan stage — PreparedChapter + VisionRuntime → ScanOutput."""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from typoon.adapters.mask_store import Masks, MaskStore
from typoon.adapters.vision_runtime import VisionRuntime
from typoon.domain import scan as scan_domain
from typoon.domain.prepared import Chapter as PreparedChapter
from typoon.runs.artifacts import ArtifactSink
from typoon.vision.draw import CYAN, GREEN, PALETTE, RED, YELLOW, label, rect
from typoon.vision.grouping import ScanState, export_groups
from typoon.vision.inspect import state_to_dict
from typoon.vision.types import DetectedGroup


@dataclass(frozen=True)
class ScanOutput:
    """Output of scan_chapter: immutable chapter + decoupled pixel masks."""

    chapter: scan_domain.Chapter
    masks:   MaskStore


def scan_chapter(
    prepared: PreparedChapter,
    runtime: VisionRuntime,
    *,
    artifacts: ArtifactSink | None = None,
) -> ScanOutput:
    """Run vision pipeline on every prepared page.

    Returns ScanOutput with an immutable scan.Chapter and a MaskStore
    holding erase/text masks keyed by (page_index, bubble_idx).
    """
    pages: list[scan_domain.Page] = []
    masks = MaskStore()
    all_ocr: list[dict] = []

    for index in range(prepared.page_count):
        image = _load_rgb(prepared.page_path(index))
        state = runtime.scan_page_state(image)
        h, w = image.shape[:2]

        bubbles, page_masks = _extract_page(index, state)

        for sb, bm in zip(bubbles, page_masks):
            masks.put(sb.page_index, sb.idx, bm)

        pages.append(scan_domain.Page(
            index=index, width=w, height=h,
            bubbles=tuple(bubbles),
        ))

        if artifacts is not None:
            _write_artifacts(artifacts, index, image, state)
            all_ocr.append({
                "page": index,
                "bubbles": [
                    {"idx": b.idx, "text": b.source_text, "confidence": b.confidence}
                    for b in bubbles
                ],
            })

    if artifacts is not None:
        artifacts.write_json("04_ocr", "ocr.json", {"pages": all_ocr})

    chapter = scan_domain.Chapter(prepared=prepared, pages=tuple(pages))
    return ScanOutput(chapter=chapter, masks=masks)


# ── Conversion ───────────────────────────────────────────────────────


def _extract_page(
    page_index: int,
    state: ScanState,
) -> tuple[list[scan_domain.Bubble], list[Masks]]:
    groups = export_groups(state)
    bubbles = [_to_bubble(i, page_index, g) for i, g in enumerate(groups)]
    page_masks = [
        Masks(erase_masks=tuple(g.erase_masks), text_masks=tuple(g.text_masks))
        for g in groups
    ]
    return bubbles, page_masks


def _to_bubble(i: int, page_index: int, g: DetectedGroup) -> scan_domain.Bubble:
    return scan_domain.Bubble(
        idx=i,
        page_index=page_index,
        source_text=g.text,
        confidence=g.confidence,
        box=scan_domain.Box(
            polygon=g.render_polygon,
            fit=g.fit_box,
            erase=g.erase_box,
            text=g.text_box,
        ),
    )


# ── Artifacts ────────────────────────────────────────────────────────


def _write_artifacts(artifacts: ArtifactSink, index: int, image: np.ndarray, state: ScanState) -> None:
    tag = f"page_{index:04d}"
    artifacts.write_image("02_detect", f"{tag}_boxes.png", _draw_boxes(image, state))
    artifacts.write_image("03_group",  f"{tag}_groups.png", _draw_groups(image, state))
    artifacts.write_json("03_group",   f"{tag}_groups.json", state_to_dict(index, tag, state))
    artifacts.write_image("04_ocr",    f"{tag}_masks.png",  _draw_mask_highlight(image, state))


# ── Drawing ──────────────────────────────────────────────────────────


def _draw_boxes(image: np.ndarray, state: ScanState) -> np.ndarray:
    out = image.copy()
    for u in state.units:
        color = RED if u.is_noise else CYAN
        rect(out, u.bbox, color, 2)
        label(out, u.bbox[0], u.bbox[1], f"u{u.idx} {u.confidence:.2f}", color)
    return out


def _draw_groups(image: np.ndarray, state: ScanState) -> np.ndarray:
    out = image.copy()
    for s in state.scopes:
        rect(out, s.bbox, YELLOW, 2)
        label(out, s.bbox[0], s.bbox[1], f"s{s.idx} {s.confidence:.2f}", YELLOW)
    for g in state.groups:
        color = GREEN if g.accepted else RED
        rect(out, g.fit_bbox, color, 3)
        text = (g.text or "")[:24].replace("\n", " ")
        label(out, g.fit_bbox[0], g.fit_bbox[1], f"g{g.idx} {g.confidence:.2f} {text}", color)
    return out


def _draw_mask_highlight(image: np.ndarray, state: ScanState) -> np.ndarray:
    out = image.copy()
    overlay = out.copy()
    for gi, g in enumerate([g for g in state.groups if g.accepted]):
        color = PALETTE[gi % len(PALETTE)]
        for ui in g.unit_indices:
            mask = state.units[ui].region.mask
            if mask is None:
                continue
            h, w = mask.image.shape[:2]
            y1, y2 = max(0, mask.y), min(out.shape[0], mask.y + h)
            x1, x2 = max(0, mask.x), min(out.shape[1], mask.x + w)
            if y2 > y1 and x2 > x1:
                crop = mask.image[y1 - mask.y:y2 - mask.y, x1 - mask.x:x2 - mask.x]
                overlay[y1:y2, x1:x2][crop > 0] = color
    cv2.addWeighted(overlay, 0.55, out, 0.45, 0, out)
    return out


def _load_rgb(path) -> np.ndarray:
    bgr = cv2.imread(str(path))
    if bgr is None:
        raise FileNotFoundError(f"Cannot read prepared page: {path}")
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


# Backward-compat alias
ScanResult = ScanOutput
