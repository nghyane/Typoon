"""Engine — vision compute. Preprocess, erase, render.

Preprocess: stitch for scan → smart re-cut → lazy page provider.
Erase/render: load one logical page at a time. Stateless.
"""

from __future__ import annotations

import time
from itertools import accumulate
from pathlib import Path
from typing import Any

import numpy as np

from .app.events import Hook, ModelsUnloaded, PageErased, PageRendered, PageScanned
from .models import ModelHub
from .ports import ChapterSource
from .domain.bubble import Bubble, Page
from .vision.chapter_images import LazyPageProvider, StitchedStrip
from .vision.erase import Eraser
from .vision.types import VisualTextGroup
from .vision.visual_group import clip_group_to_slice

_NO_HOOK = Hook()
_MAX_PAGE_HEIGHT = 2500
_CUT_TARGET_HEIGHT = 1800
_CUT_MIN_HEIGHT = 800
_TEXT_CUT_PENALTY = 1_000_000.0
_SCOPE_CUT_PENALTY = 100_000.0
_FIT_CUT_PENALTY = 50_000.0
_DISTANCE_CUT_WEIGHT = 0.35
_EDGE_CUT_WEIGHT = 12.0
_CUT_BAND = 8


class Engine:
    """Vision compute: preprocess, erase, render. Stateless."""

    def __init__(
        self,
        scanner,
        eraser: Eraser,
        hub: ModelHub,
        *,
        bubble_scope_imgsz: int = 640,
    ) -> None:
        self.scanner = scanner
        self.eraser = eraser
        self._hub = hub
        self._bubble_scope_imgsz = bubble_scope_imgsz
        self._yolo_model = None

    @staticmethod
    def from_config(config=None, paths=None):
        from .config import load_config
        from .vision.scanner import create_scanner

        if config is None or paths is None:
            config, paths = load_config()
        hub = ModelHub(Path(config.models_dir))
        engine = Engine(
            scanner=create_scanner(hub=hub),
            eraser=Eraser(str(hub.dir)),
            hub=hub,
            bubble_scope_imgsz=config.bubble_scope_imgsz,
        )
        return engine, config, paths

    def unload_scan_models(self, hook: Hook = _NO_HOOK) -> None:
        self.scanner = None  # type: ignore[assignment]
        hook.on(ModelsUnloaded(stage="scan"))

    def unload_erase_models(self, hook: Hook = _NO_HOOK) -> None:
        self.eraser = None  # type: ignore[assignment]
        hook.on(ModelsUnloaded(stage="erase"))

    def ensure_scan_models(self) -> None:
        if self.scanner is not None:
            return
        from .vision.scanner import create_scanner
        self.scanner = create_scanner(hub=self._hub)

    def ensure_erase_models(self) -> None:
        if self.eraser is not None:
            return
        self.eraser = Eraser(str(self._hub.dir))

    def preprocess(
        self, source: ChapterSource, hook: Hook = _NO_HOOK,
    ) -> tuple[list[Page], LazyPageProvider]:
        """Stitch → scan full strip → smart re-cut → lazy page provider.

        Stitches all source pages into one continuous strip so bubbles
        at page boundaries are never split. Scans once on the full
        strip (detector tiles internally), computes logical pages that
        avoid cutting through bubbles, then releases the strip.
        """
        self.ensure_scan_models()

        # Phase 1: stitch all source pages into one strip
        t0 = time.time()
        raw_images = [source.load_page(i) for i in range(source.page_count())]
        strip = StitchedStrip.from_pages(raw_images)
        full_img = strip.image
        original_heights = strip.heights
        del raw_images

        # Phase 2: scan on the full stitched strip
        scanned = self._scan_page(full_img)
        t_scan = time.time() - t0

        # Phase 3: smart re-cut into logical pages
        logical = _split_strip(full_img, scanned, original_heights)
        page_ranges = [(y_start, y_end) for _, y_start, y_end in logical]
        provider = LazyPageProvider.build(source, original_heights, strip.width, page_ranges)
        del full_img
        strip.free()

        # Phase 4: map bubbles to logical pages
        pages: list[Page] = []
        for i, (page_bubbles, _, _) in enumerate(logical):
            page = Page(index=i)
            for sb in page_bubbles:
                page.bubbles.append(Bubble(
                    idx=len(page.bubbles),
                    page_index=i,
                    polygon=sb.render_polygon,
                    erase_masks=sb.erase_masks,
                    text_masks=sb.text_masks,
                    source_text=sb.text,
                    ocr_confidence=sb.confidence,
                ))
            pages.append(page)

        n_logical = len(pages)
        for i, page in enumerate(pages):
            hook.on(PageScanned(
                page=i, total=n_logical, bubbles=len(page.bubbles),
                det_ms=t_scan * 1000 / n_logical if i == 0 else 0, ocr_ms=0,
            ))

        return pages, provider

    def _get_yolo_model(self) -> Any | None:
        if self._yolo_model is None:
            from .vision.bubble_scope import load_yolo_model
            import sys
            if sys.platform == "darwin":
                path = self._hub.resolve("bubble-scope-yolov8m.mlpackage")
            else:
                path = self._hub.resolve("bubble-scope-yolov8m.pt")
            self._yolo_model = load_yolo_model(path)
        return self._yolo_model

    def scan_page(self, image: np.ndarray) -> list[VisualTextGroup]:
        """Scan a page into canonical visual text groups."""
        return self._scan_page(image)

    def _scan_page(self, image: np.ndarray) -> list[VisualTextGroup]:
        return self.scanner.scan(
            image,
            scope_model=self._get_yolo_model(),
            scope_imgsz=self._bubble_scope_imgsz,
        )

    def erase_and_render(
        self, pages: list[Page], images: LazyPageProvider, hook: Hook = _NO_HOOK,
    ) -> None:
        self.ensure_erase_models()
        import typoon_render

        total = len(pages)
        for page in pages:
            img = images.page(page.index)

            t0 = time.time()
            h, w = img.shape[:2]
            canvas = np.dstack([img, np.full((h, w), 255, dtype=np.uint8)])
            masks = [m for b in page.bubbles for m in b.erase_masks]
            self.eraser.erase(canvas, masks)
            erased = canvas[:, :, :3]
            del canvas
            hook.on(PageErased(page=page.index, total=total, ms=(time.time() - t0) * 1000))

            polygons = [b.polygon for b in page.bubbles]
            texts = [b.translated_text or "" for b in page.bubbles]
            result = typoon_render.render(img, erased, polygons, texts, img.shape[1])

            page.rendered = result.image
            for b, info in zip(page.bubbles, result.bubbles):
                b.font_size = info.font_size_px
                b.overflow = info.overflow

            hook.on(PageRendered(page=page.index, total=total))
            del erased

        images.free()

    def erase(
        self, pages: list[Page], images: LazyPageProvider, hook: Hook = _NO_HOOK,
    ) -> None:
        self.ensure_erase_models()
        total = len(pages)

        for page in pages:
            img = images.page(page.index)
            t0 = time.time()
            h, w = img.shape[:2]
            canvas = np.dstack([img, np.full((h, w), 255, dtype=np.uint8)])
            masks = [m for b in page.bubbles for m in b.erase_masks]
            self.eraser.erase(canvas, masks)
            page.erased = canvas[:, :, :3]
            del canvas
            hook.on(PageErased(page=page.index, total=total, ms=(time.time() - t0) * 1000))

        images.free()



def _split_strip(
    full_img: np.ndarray,
    scanned: list[VisualTextGroup],
    original_heights: list[int],
) -> list[tuple[list[VisualTextGroup], int, int]]:
    """Cut a stitched strip into logical pages that never cut bubbles.

    Uses original page boundaries as preferred cut points, then
    _choose_page_cuts for segments that are still too tall.
    """
    h, w = full_img.shape[:2]

    # Original page boundaries as candidate cuts
    orig_cuts = list(accumulate(original_heights[:-1]))
    safe_cuts = [c for c in orig_cuts if not _cuts_bubble(c, scanned)]

    # Build segments from safe original cuts
    boundaries = [0] + safe_cuts + [h]
    segments: list[tuple[int, int]] = list(zip(boundaries, boundaries[1:]))

    # Sub-split any segment still taller than _MAX_PAGE_HEIGHT
    final: list[tuple[int, int]] = []
    for y_start, y_end in segments:
        if y_end - y_start <= _MAX_PAGE_HEIGHT:
            final.append((y_start, y_end))
            continue
        seg_img = full_img[y_start:y_end]
        seg_bubbles = _bubbles_in_range(scanned, y_start, y_end, w)
        sub_cuts = _choose_page_cuts(seg_img, seg_bubbles)
        sub_bounds = [y_start] + [y_start + c for c in sub_cuts] + [y_end]
        final.extend(zip(sub_bounds, sub_bounds[1:]))

    # Clip bubbles per logical page
    out: list[tuple[list[VisualTextGroup], int, int]] = []
    for y_start, y_end in final:
        slice_bubbles = [
            sliced
            for sb in scanned
            if (sliced := clip_group_to_slice(sb, y_start, y_end, page_w=w)) is not None
        ]
        out.append((slice_bubbles, y_start, y_end))
    return out


def _cuts_bubble(y: int, scanned: list[VisualTextGroup]) -> bool:
    for sb in scanned:
        if _crosses(y, _fit_y_range(sb)):
            return True
    return False


def _bubbles_in_range(
    scanned: list[VisualTextGroup], y_start: int, y_end: int, page_w: int,
) -> list[VisualTextGroup]:
    """Return bubbles with coordinates clipped to segment-local."""
    return [
        sliced
        for sb in scanned
        if (sliced := clip_group_to_slice(sb, y_start, y_end, page_w)) is not None
    ]


def _choose_page_cuts(img: np.ndarray, scanned: list[VisualTextGroup]) -> list[int]:
    h = img.shape[0]
    edge_cost = _row_edge_cost(img)
    cuts: list[int] = []
    y = 0
    while y + _MAX_PAGE_HEIGHT < h:
        lo = y + _CUT_MIN_HEIGHT
        hi = min(y + _MAX_PAGE_HEIGHT, h)
        target = min(y + _CUT_TARGET_HEIGHT, hi)
        candidates = range(lo, hi + 1)
        cut = min(
            candidates,
            key=lambda cand: _cut_cost(cand, target, scanned, edge_cost),
        )
        cuts.append(cut)
        y = cut
    return cuts


def _cut_cost(
    y: int, target: int, scanned: list[VisualTextGroup], edge_cost: np.ndarray,
) -> float:
    cost = abs(y - target) * _DISTANCE_CUT_WEIGHT
    y0 = max(0, y - _CUT_BAND)
    y1 = min(len(edge_cost), y + _CUT_BAND + 1)
    if y1 > y0:
        cost += float(edge_cost[y0:y1].mean()) * _EDGE_CUT_WEIGHT

    for sb in scanned:
        if _crosses(y, _polygon_y_range(sb.render_polygon)):
            cost += _TEXT_CUT_PENALTY
        if _crosses(y, _fit_y_range(sb)):
            cost += _FIT_CUT_PENALTY
        if sb.scope_bbox is not None and _crosses(y, (float(sb.scope_bbox[1]), float(sb.scope_bbox[3]))):
            cost += _SCOPE_CUT_PENALTY
        if _crosses(y, (float(sb.erase_bbox[1]), float(sb.erase_bbox[3]))):
            cost += _FIT_CUT_PENALTY
    return cost


def _row_edge_cost(img: np.ndarray) -> np.ndarray:
    gray = img.mean(axis=2).astype(np.float32) if img.ndim == 3 else img.astype(np.float32)
    if gray.shape[0] <= 1:
        return np.zeros(gray.shape[0], dtype=np.float32)
    vertical_delta = np.abs(np.diff(gray, axis=0)).mean(axis=1)
    cost = np.concatenate([vertical_delta[:1], vertical_delta])
    max_cost = float(cost.max())
    return cost / max_cost if max_cost > 0 else cost


def _polygon_y_range(polygon: list[list[float]]) -> tuple[float, float]:
    return min(p[1] for p in polygon), max(p[1] for p in polygon)


def _fit_y_range(sb: VisualTextGroup) -> tuple[float, float]:
    ranges = [_polygon_y_range(sb.render_polygon)]
    for mask in sb.erase_masks:
        ranges.append((float(mask.y), float(mask.y + mask.image.shape[0])))
    return min(r[0] for r in ranges), max(r[1] for r in ranges)


def _crosses(y: int, y_range: tuple[float, float]) -> bool:
    return y_range[0] < y < y_range[1]
