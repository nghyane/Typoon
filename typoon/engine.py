"""Engine — vision compute. Preprocess, erase, render.

Preprocess: per-page scan → boundary overlap check → stitch into ChapterImages.
Erase/render: per-page from ChapterImages views. Stateless.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any

import numpy as np

from .app.events import Hook, ModelsUnloaded, PageErased, PageRendered, PageScanned
from .models import ModelHub
from .ports import ChapterSource
from .domain.bubble import Bubble, Page
from .vision.chapter_images import ChapterImages
from .vision.erase import Eraser
from .vision.types import TextMask
from .vision.visual_group import offset_group

_NO_HOOK = Hook()
_BOUNDARY_OVERLAP = 500  # px scanned from each side of page boundary
_MAX_PAGE_HEIGHT = 2500  # pages taller than this get split
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

    # ── Model lifecycle ──────────────────────────────────────────

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

    # ── Preprocess ───────────────────────────────────────────────

    def preprocess(
        self, source: ChapterSource, hook: Hook = _NO_HOOK,
    ) -> tuple[list[Page], ChapterImages]:
        """Per-page scan → boundary merge if needed → ChapterImages.

        Fast path (manga): per-page scan, no stitch. 7MB peak.
        Slow path (manhwa): stitch boundary pairs only when split bubbles detected.
        """
        self.ensure_scan_models()
        n = source.page_count()

        # Phase 1: per-page scan
        t0 = time.time()
        page_results: list[tuple[list, np.ndarray]] = []  # (scanned_bubbles, image)
        for i in range(n):
            img = source.load_page(i)
            scanned = self._scan_page(img)
            page_results.append((scanned, img))

        # Phase 2: scan boundary overlap zones to catch split bubbles
        page_results = self._scan_boundary_zones(page_results)

        t_scan = time.time() - t0

        # Phase 3: split long pages into logical pages (~1800px)
        logical = _split_long_pages(page_results)

        # Phase 4: build ChapterImages from logical pages
        page_images = [img for _, img in logical]
        images = ChapterImages.from_pages(page_images)

        # Phase 5: map bubbles to logical pages
        pages: list[Page] = []
        for i, (scanned, _) in enumerate(logical):
            page = Page(index=i)
            for sb in scanned:
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

        return pages, images

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

    def scan_page(self, image: np.ndarray):
        """Scan a page into canonical visual text groups."""
        return self._scan_page(image)

    def _scan_page(self, image: np.ndarray):
        return self.scanner.scan(
            image,
            scope_model=self._get_yolo_model(),
            scope_imgsz=self._bubble_scope_imgsz,
        )

    def _scan_boundary_zones(
        self, page_results: list[tuple[list, np.ndarray]],
    ) -> list[tuple[list, np.ndarray]]:
        """Scan overlap zones between adjacent pages to catch split bubbles.

        For each page pair, scans a small boundary zone (bottom of page N +
        top of page N+1). If a bubble crosses the boundary, it replaces
        the partial detections from per-page scanning.

        Cost: ~250ms per boundary zone (690×1000px). Only runs N-1 times.
        """
        import cv2

        n = len(page_results)
        if n < 2:
            return page_results

        for i in range(n - 1):
            scanned_a, img_a = page_results[i]
            scanned_b, img_b = page_results[i + 1]
            h_a, w_a = img_a.shape[:2]
            h_b, w_b = img_b.shape[:2]

            # Build boundary zone
            target_w = min(w_a, w_b)
            top_part = img_a[max(0, h_a - _BOUNDARY_OVERLAP):]
            bot_part = img_b[:min(_BOUNDARY_OVERLAP, h_b)]
            if top_part.shape[1] != target_w:
                top_part = cv2.resize(top_part, (target_w, top_part.shape[0]))
            if bot_part.shape[1] != target_w:
                bot_part = cv2.resize(bot_part, (target_w, bot_part.shape[0]))

            zone = np.concatenate([top_part, bot_part], axis=0)
            split_y = top_part.shape[0]  # boundary within zone
            zone_y_offset = h_a - top_part.shape[0]  # zone start in page A coords

            zone_bubbles = self._scan_page(zone)

            # Find bubbles that cross the boundary
            for sb in zone_bubbles:
                ys = [p[1] for p in sb.render_polygon]
                if min(ys) < split_y and max(ys) > split_y:
                    # Cross-boundary bubble — assign to page where center is
                    cy = (min(ys) + max(ys)) / 2
                    if cy < split_y:
                        # Assign to page A, shift to page-A coords
                        sb = offset_group(sb, zone_y_offset)
                        scanned_a = [
                            s for s in scanned_a
                            if not _overlaps_y(s.render_polygon, sb.render_polygon)
                        ]
                        scanned_a.append(sb)
                    else:
                        # Assign to page B, shift to page-B coords
                        sb = offset_group(sb, -split_y)
                        scanned_b = [
                            s for s in scanned_b
                            if not _overlaps_y(s.render_polygon, sb.render_polygon)
                        ]
                        scanned_b.append(sb)

            page_results[i] = (scanned_a, img_a)
            page_results[i + 1] = (scanned_b, img_b)

        return page_results

    # ── Erase + Render ───────────────────────────────────────────

    def erase_and_render(
        self, pages: list[Page], images: ChapterImages, hook: Hook = _NO_HOOK,
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

    # ── Erase only ───────────────────────────────────────────────

    def erase(
        self, pages: list[Page], images: ChapterImages, hook: Hook = _NO_HOOK,
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


# ── Helpers ──────────────────────────────────────────────────────────


def _split_long_pages(
    page_results: list[tuple[list, np.ndarray]],
) -> list[tuple[list, np.ndarray]]:
    """Split pages taller than _MAX_PAGE_HEIGHT into logical pages.

    Uses smart_split to find cut points that don't cut bubbles.
    Short pages pass through unchanged.
    """

    out: list[tuple[list, np.ndarray]] = []

    for scanned, img in page_results:
        h, w = img.shape[:2]
        if h <= _MAX_PAGE_HEIGHT:
            out.append((scanned, img))
            continue

        cuts = _choose_page_cuts(img, scanned)
        if not cuts:
            out.append((scanned, img))
            continue

        boundaries = [0] + cuts + [h]
        for y_start, y_end in zip(boundaries, boundaries[1:]):
            slice_img = img[y_start:y_end]
            slice_bubbles = [
                sliced
                for sb in scanned
                if (sliced := _slice_scanned_bubble(sb, y_start, y_end, page_w=w)) is not None
            ]
            out.append((slice_bubbles, slice_img))

    return out


def _choose_page_cuts(img: np.ndarray, scanned: list) -> list[int]:
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


def _cut_cost(y: int, target: int, scanned: list, edge_cost: np.ndarray) -> float:
    cost = abs(y - target) * _DISTANCE_CUT_WEIGHT
    y0 = max(0, y - _CUT_BAND)
    y1 = min(len(edge_cost), y + _CUT_BAND + 1)
    if y1 > y0:
        cost += float(edge_cost[y0:y1].mean()) * _EDGE_CUT_WEIGHT

    for sb in scanned:
        text_range = _polygon_y_range(sb.render_polygon)
        if _crosses(y, text_range):
            cost += _TEXT_CUT_PENALTY
        if _crosses(y, _fit_y_range(sb)):
            cost += _FIT_CUT_PENALTY
        scope = getattr(sb, "scope_bbox", None)
        if scope is not None and _crosses(y, (float(scope[1]), float(scope[3]))):
            cost += _SCOPE_CUT_PENALTY
        erase_bbox = getattr(sb, "erase_bbox", None)
        if erase_bbox is not None and _crosses(y, (float(erase_bbox[1]), float(erase_bbox[3]))):
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


def _fit_y_range(sb) -> tuple[float, float]:
    ranges = [_polygon_y_range(sb.render_polygon)]
    for mask in sb.erase_masks:
        ranges.append((float(mask.y), float(mask.y + mask.image.shape[0])))
    return min(r[0] for r in ranges), max(r[1] for r in ranges)


def _crosses(y: int, y_range: tuple[float, float]) -> bool:
    y1, y2 = y_range
    return y1 < y < y2


def _slice_scanned_bubble(sb, y_start: int, y_end: int, page_w: int):
    from .vision.visual_group import clip_group_to_slice
    return clip_group_to_slice(sb, y_start, y_end, page_w)


def _overlaps_y(poly_a: list[list[float]], poly_b: list[list[float]]) -> bool:
    """Check if two polygons overlap vertically (>50% of shorter range)."""
    a_min = min(p[1] for p in poly_a)
    a_max = max(p[1] for p in poly_a)
    b_min = min(p[1] for p in poly_b)
    b_max = max(p[1] for p in poly_b)
    overlap = max(0, min(a_max, b_max) - max(a_min, b_min))
    shorter = min(a_max - a_min, b_max - b_min)
    return overlap > shorter * 0.5 if shorter > 0 else False
