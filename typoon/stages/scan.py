"""Scan stage — vision pipeline over a PreparedChapter → list[Page]."""

from __future__ import annotations

import cv2
import numpy as np

from typoon.adapters.vision_runtime import VisionRuntime
from typoon.domain.bubble import Bubble, Page
from typoon.domain.prepared import PreparedChapter
from typoon.runs.artifacts import ArtifactSink
from typoon.vision.draw import CYAN, GREEN, PALETTE, RED, YELLOW, label, rect
from typoon.vision.inspect import state_to_dict
from typoon.vision.types import PageScanState, VisualTextGroup


def scan_chapter(
    chapter: PreparedChapter,
    runtime: VisionRuntime,
    *,
    artifacts: ArtifactSink | None = None,
) -> list[Page]:
    """Run vision pipeline on every prepared page and return scanned pages.

    Each returned Page has Bubble objects populated with source_text, masks,
    and render geometry. Pages with no detected text have an empty bubble list.
    """
    pages: list[Page] = []
    all_ocr: list[dict] = []

    for index in range(chapter.page_count):
        image = _load_rgb(chapter.page_path(index))
        state = runtime.scan_page_state(image)
        bubbles = _bubbles_from_state(index, state)

        if artifacts is not None:
            _write_page_artifacts(artifacts, index, image, state, runtime)
            all_ocr.append(_ocr_dict(index, bubbles))

        pages.append(Page(index=index, bubbles=bubbles))

    if artifacts is not None:
        artifacts.write_json("04_ocr", "ocr.json", {"pages": all_ocr})

    return pages


# ── Conversion ───────────────────────────────────────────────────────


def _bubbles_from_state(page_index: int, state: PageScanState) -> list[Bubble]:
    from typoon.vision.text_grouping import to_visual_text_groups
    groups = to_visual_text_groups(state)
    return [_to_bubble(i, page_index, g) for i, g in enumerate(groups)]


def _to_bubble(i: int, page_index: int, group: VisualTextGroup) -> Bubble:
    return Bubble(
        idx=i,
        page_index=page_index,
        polygon=group.render_polygon,
        erase_masks=group.erase_masks,
        text_masks=group.text_masks,
        source_text=group.text,
        ocr_confidence=group.confidence,
    )


# ── Artifacts ────────────────────────────────────────────────────────


def _write_page_artifacts(
    artifacts: ArtifactSink,
    index: int,
    image: np.ndarray,
    state: PageScanState,
    runtime: VisionRuntime,
) -> None:
    tag = f"page_{index:04d}"

    # 02_detect — unit boxes overlay
    artifacts.write_image("02_detect", f"{tag}_boxes.png", _draw_boxes(image, state))

    # 03_group — group polygons + scope boxes
    artifacts.write_image("03_group", f"{tag}_groups.png", _draw_groups(image, state))
    artifacts.write_json("03_group", f"{tag}_groups.json", state_to_dict(index, tag, state))

    # 04_ocr — erase mask highlight (no inpainting — that belongs to render stage)
    artifacts.write_image("04_ocr", f"{tag}_masks.png", _draw_mask_highlight(image, state))


def _ocr_dict(page_index: int, bubbles: list[Bubble]) -> dict:
    return {
        "page": page_index,
        "bubbles": [
            {"idx": b.idx, "text": b.source_text, "confidence": b.ocr_confidence}
            for b in bubbles
        ],
    }


# ── Drawing helpers ──────────────────────────────────────────────────


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
            y1 = max(0, mask.y)
            y2 = min(out.shape[0], mask.y + h)
            x1 = max(0, mask.x)
            x2 = min(out.shape[1], mask.x + w)
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
