"""Engine — vision compute. Preprocess, erase, render.

Preprocess: per-page scan → boundary overlap check → stitch into ChapterImages.
Erase/render: per-page from ChapterImages views. Stateless.
"""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np

from .events import Hook, ModelsUnloaded, PageErased, PageRendered, PageScanned
from .models import ModelHub
from .ports import ChapterSource
from .types import Bubble, Page
from .vision.chapter_images import ChapterImages
from .vision.erase import Eraser
from .vision.types import TextMask

_NO_HOOK = Hook()
_BOUNDARY_OVERLAP = 500  # px scanned from each side of page boundary
_MAX_PAGE_HEIGHT = 2500  # pages taller than this get split


class Engine:
    """Vision compute: preprocess, erase, render. Stateless."""

    def __init__(self, scanner, eraser: Eraser) -> None:
        self.scanner = scanner
        self.eraser = eraser
        self._hub: ModelHub | None = None

    @staticmethod
    def from_config(config=None, paths=None):
        from .config import load_config
        from .vision.scanner import create_scanner

        if config is None or paths is None:
            config, paths = load_config()
        hub = ModelHub(Path(config.models_dir))
        engine = Engine(
            scanner=create_scanner(hub=hub),
            eraser=Eraser(config.models_dir),
        )
        engine._hub = hub
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
        if self._hub is None:
            raise RuntimeError("Cannot reload models: Engine not created via from_config()")
        from .vision.scanner import create_scanner
        self.scanner = create_scanner(hub=self._hub)

    def ensure_erase_models(self) -> None:
        if self.eraser is not None:
            return
        if self._hub is None:
            raise RuntimeError("Cannot reload models: Engine not created via from_config()")
        self.eraser = Eraser(self._hub._dir)

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
            scanned = self.scanner.scan(img)
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
                    polygon=sb.polygon,
                    masks=sb.masks,
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

            zone_bubbles = self.scanner.scan(zone)

            # Find bubbles that cross the boundary
            for sb in zone_bubbles:
                ys = [p[1] for p in sb.polygon]
                if min(ys) < split_y and max(ys) > split_y:
                    # Cross-boundary bubble — assign to page where center is
                    cy = (min(ys) + max(ys)) / 2
                    if cy < split_y:
                        # Assign to page A, shift to page-A coords
                        sb.polygon = [[p[0], p[1] + zone_y_offset] for p in sb.polygon]
                        sb.masks = [TextMask(x=m.x, y=m.y + zone_y_offset, image=m.image) for m in sb.masks]
                        # Remove any partial bubble near bottom of page A
                        scanned_a = [
                            s for s in scanned_a
                            if not _overlaps_y(s.polygon, sb.polygon)
                        ]
                        scanned_a.append(sb)
                    else:
                        # Assign to page B, shift to page-B coords
                        sb.polygon = [[p[0], p[1] - split_y] for p in sb.polygon]
                        sb.masks = [TextMask(x=m.x, y=m.y - split_y, image=m.image) for m in sb.masks]
                        scanned_b = [
                            s for s in scanned_b
                            if not _overlaps_y(s.polygon, sb.polygon)
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
            masks = [m for b in page.bubbles for m in b.masks]
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
            masks = [m for b in page.bubbles for m in b.masks]
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
    from .vision.paginate import smart_split

    out: list[tuple[list, np.ndarray]] = []

    for scanned, img in page_results:
        h = img.shape[0]
        if h <= _MAX_PAGE_HEIGHT:
            out.append((scanned, img))
            continue

        # Find cut points
        bubble_y_ranges = [
            (min(p[1] for p in sb.polygon), max(p[1] for p in sb.polygon))
            for sb in scanned
        ]
        cuts = smart_split(h, [h], bubble_y_ranges)
        if not cuts:
            out.append((scanned, img))
            continue

        # Slice image and distribute bubbles
        boundaries = [0] + cuts + [h]
        for si in range(len(boundaries) - 1):
            y_start = boundaries[si]
            y_end = boundaries[si + 1]
            slice_img = img[y_start:y_end]

            # Assign bubbles whose center falls in this slice
            slice_bubbles = []
            for sb in scanned:
                cy = (min(p[1] for p in sb.polygon) + max(p[1] for p in sb.polygon)) / 2
                if y_start <= cy < y_end:
                    # Shift to slice-local coordinates
                    sb_copy = type(sb)(
                        polygon=[[p[0], p[1] - y_start] for p in sb.polygon],
                        text=sb.text,
                        confidence=sb.confidence,
                        masks=[TextMask(x=m.x, y=m.y - y_start, image=m.image) for m in sb.masks],
                    )
                    slice_bubbles.append(sb_copy)

            out.append((slice_bubbles, slice_img))

    return out


def _overlaps_y(poly_a: list[list[float]], poly_b: list[list[float]]) -> bool:
    """Check if two polygons overlap vertically (>50% of shorter range)."""
    a_min = min(p[1] for p in poly_a)
    a_max = max(p[1] for p in poly_a)
    b_min = min(p[1] for p in poly_b)
    b_max = max(p[1] for p in poly_b)
    overlap = max(0, min(a_max, b_max) - max(a_min, b_min))
    shorter = min(a_max - a_min, b_max - b_min)
    return overlap > shorter * 0.5 if shorter > 0 else False
