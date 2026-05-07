"""Scan stage — PreparedChapter + VisionRuntime → ScanOutput."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from typoon.adapters.mask_store import BubbleMasks, MaskStore
from typoon.adapters.prepared_reader import PreparedReader
from typoon.adapters.vision_runtime import VisionRuntime
from typoon.domain import scan
from typoon.domain.prepared import Chapter
from typoon.domain.scan import BubbleGeometry, PageGeometry
from typoon.runs.artifacts import ArtifactSink
from typoon.runs.events import Hook, PageDone
from typoon.vision.grouping import ScanState, export_groups
from typoon.vision.types import DetectedGroup


@dataclass(frozen=True)
class ScanOutput:
    """Output of scan_chapter — pure data, no persistence logic."""
    chapter:  scan.Chapter
    masks:    MaskStore
    geometry: list[PageGeometry]

    def bubble_records(self) -> list[dict]:
        """Flat list for store.save_bubbles()."""
        return [
            {
                "page_index": b.page_index,
                "bubble_idx": b.idx,
                "source_text": b.source_text,
                "confidence": b.confidence,
            }
            for b in self.chapter.all_bubbles
        ]

    def geometry_records(self) -> list[dict]:
        """Page-shaped geometry for store.save_geometry()."""
        return [
            {
                "page_index": pg.page_index,
                "width":  pg.width,
                "height": pg.height,
                "bubbles": [
                    {
                        "bubble_idx": bg.bubble_idx,
                        "polygon":   bg.polygon,
                        "fit_box":   bg.fit_box,
                        "erase_box": bg.erase_box,
                        "text_box":  bg.text_box,
                    }
                    for bg in pg.bubbles
                ],
            }
            for pg in self.geometry
        ]


def scan_chapter(
    prepared: Chapter,
    reader: PreparedReader,
    runtime: VisionRuntime,
    *,
    chapter_id: int = 0,
    project_id: int = 0,
    hook: Hook | None = None,
    artifacts: ArtifactSink | None = None,
) -> ScanOutput:
    """Run vision pipeline on every prepared page. Returns ScanOutput."""
    pages:    list[scan.Page] = []
    geometry: list[PageGeometry]     = []
    masks     = MaskStore()
    all_ocr:  list[dict]             = []
    total = prepared.page_count

    for index in range(total):
        image = reader.read_rgb(index)
        state = runtime.scan_page_state(image)
        h, w  = image.shape[:2]

        bubbles, page_geom, page_masks = _extract_page(index, state, w, h)

        for sb, bm in zip(bubbles, page_masks):
            masks.put(sb.page_index, sb.idx, bm)

        pages.append(scan.Page(
            index=index, width=w, height=h,
            bubbles=tuple(bubbles),
        ))
        geometry.append(page_geom)

        if hook is not None:
            hook.on(PageDone(chapter_id=chapter_id, project_id=project_id, stage="scan", page_index=index, page_total=total))

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
        artifacts.write_json("04_ocr", "ocr_all_pages.json", all_ocr)

    return ScanOutput(
        chapter=scan.Chapter(prepared=prepared, pages=tuple(pages)),
        masks=masks,
        geometry=geometry,
    )


# ── Internal helpers ──────────────────────────────────────────────────


def _extract_page(
    index: int,
    state: ScanState,
    width: int,
    height: int,
) -> tuple[list[scan.Bubble], PageGeometry, list[BubbleMasks]]:
    groups = export_groups(state)
    bubbles:  list[scan.Bubble] = []
    geom_list: list[BubbleGeometry]    = []
    masks_out: list[BubbleMasks]       = []

    for i, g in enumerate(groups):
        b = scan.Bubble(
            idx=i,
            page_index=index,
            source_text=g.text,
            confidence=g.confidence,
            box=scan.Box(
                polygon=g.render_polygon,
                fit=g.fit_box,
                erase=g.erase_box,
                text=g.text_box,
            ),
        )
        bubbles.append(b)
        geom_list.append(BubbleGeometry(
            bubble_idx=i,
            polygon=g.render_polygon,
            fit_box=g.fit_box,
            erase_box=g.erase_box,
            text_box=g.text_box,
        ))
        masks_out.append(BubbleMasks(
            erase_masks=tuple(g.erase_masks),
            text_masks=tuple(g.text_masks),
        ))

    page_geom = PageGeometry(
        page_index=index,
        width=width,
        height=height,
        bubbles=tuple(geom_list),
    )
    return bubbles, page_geom, masks_out


# ── Artifact helpers ──────────────────────────────────────────────────


def _write_artifacts(
    artifacts: ArtifactSink,
    index: int,
    image: np.ndarray,
    state: ScanState,
) -> None:
    from typoon.vision.draw import PALETTE, RED, label, rect
    from typoon.vision.inspect import state_to_dict

    info = state_to_dict(index, "", state)
    artifacts.write_json("02_detect", f"page_{index:04d}_state.json", info)

    vis = image.copy()
    for i, g in enumerate(state.groups):
        color = PALETTE[i % len(PALETTE)] if g.accepted else RED
        if g.fit_bbox:
            rect(vis, g.fit_bbox, color, thickness=1)
        groups_info = info.get("groups") or []
        text = groups_info[i].get("text", "")[:20] if i < len(groups_info) else ""
        if g.fit_bbox:
            label(vis, g.fit_bbox[0], g.fit_bbox[1], text, color)
    artifacts.write_image("02_detect", f"page_{index:04d}_detect.png", vis)

    vis2 = image.copy()
    for i, g in enumerate(state.groups):
        if not g.accepted:
            continue
        rect(vis2, g.fit_bbox, PALETTE[i % len(PALETTE)], thickness=1)
    artifacts.write_image("03_group", f"page_{index:04d}_groups.png", vis2)
