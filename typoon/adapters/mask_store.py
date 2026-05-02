"""MaskStore — per-page pixel masks, and scan.npz geometry I/O.

masks/   — one npz per page, all bubbles for that page
scan.npz — bubble geometry serialized with numpy (see domain/scan.py for types)

Access pattern: render reads masks one page at a time → grouped by page.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from typoon.domain.scan import BubbleGeometry, PageGeometry
from typoon.vision.types import TextMask

if TYPE_CHECKING:
    from typoon.paths import ChapterPaths


# ── Pixel masks ───────────────────────────────────────────────────────


class BubbleMasks:
    """Erase and text masks for one bubble."""
    __slots__ = ("erase_masks", "text_masks")

    def __init__(self, erase_masks: tuple[TextMask, ...], text_masks: tuple[TextMask, ...]) -> None:
        self.erase_masks = erase_masks
        self.text_masks  = text_masks


class MaskStore:
    """In-memory masks for one chapter, grouped by page for efficient render access."""

    def __init__(self) -> None:
        self._pages: dict[int, dict[int, BubbleMasks]] = {}

    def put(self, page_index: int, bubble_idx: int, masks: BubbleMasks) -> None:
        self._pages.setdefault(page_index, {})[bubble_idx] = masks

    def get(self, page_index: int, bubble_idx: int) -> BubbleMasks | None:
        return self._pages.get(page_index, {}).get(bubble_idx)

    def __len__(self) -> int:
        return sum(len(b) for b in self._pages.values())

    def save(self, cp: "ChapterPaths") -> None:
        """Write one masks/<page>.npz per page."""
        cp.masks.mkdir(parents=True, exist_ok=True)
        for page_index, bubbles in self._pages.items():
            arrays: dict[str, np.ndarray] = {}
            meta_list = []
            for bubble_idx, bm in sorted(bubbles.items()):
                for i, m in enumerate(bm.erase_masks):
                    arrays[f"b{bubble_idx}_e{i}"]    = m.image
                    arrays[f"b{bubble_idx}_e{i}_xy"] = np.array([m.x, m.y], dtype=np.int32)
                for i, m in enumerate(bm.text_masks):
                    arrays[f"b{bubble_idx}_t{i}"]    = m.image
                    arrays[f"b{bubble_idx}_t{i}_xy"] = np.array([m.x, m.y], dtype=np.int32)
                meta_list.append(f"{bubble_idx},{len(bm.erase_masks)},{len(bm.text_masks)}")
            arrays["_meta"] = np.array(meta_list, dtype=object)
            np.savez_compressed(str(cp.mask(page_index)), **arrays)

    @classmethod
    def load(cls, cp: "ChapterPaths") -> "MaskStore":
        store = cls()
        if not cp.masks.exists():
            return store
        for mask_file in sorted(cp.masks.glob("????.npz")):
            page_index = int(mask_file.stem)
            _load_page_into(np.load(str(mask_file), allow_pickle=True), page_index, store)
        return store

    @classmethod
    def load_page(cls, cp: "ChapterPaths", page_index: int) -> dict[int, BubbleMasks]:
        """Load masks for one page — avoids loading the full chapter."""
        path = cp.mask(page_index)
        if not path.exists():
            return {}
        store = cls()
        _load_page_into(np.load(str(path), allow_pickle=True), page_index, store)
        return store._pages.get(page_index, {})


# ── Geometry I/O — scan.npz ───────────────────────────────────────────


def load_scan_geometry(cp: "ChapterPaths") -> list[PageGeometry]:
    """Load scan.npz → list[PageGeometry]."""
    data = np.load(str(cp.scan), allow_pickle=False)

    page_widths  = data["page_widths"].tolist()
    page_heights = data["page_heights"].tolist()
    bubble_page  = data["bubble_page"].tolist()
    bubble_idxs  = data["bubble_idx"].tolist()
    polygon_data = data["polygon_data"]
    polygon_off  = data["polygon_offsets"].tolist()
    polygon_len  = data["polygon_lengths"].tolist()
    fit_boxes    = data["fit_boxes"].tolist()
    erase_boxes  = data["erase_boxes"].tolist()
    text_boxes   = data["text_boxes"].tolist()

    pages_dict: dict[int, list[BubbleGeometry]] = {}
    for i, (pi, bi) in enumerate(zip(bubble_page, bubble_idxs)):
        off    = polygon_off[i]
        length = polygon_len[i]
        pages_dict.setdefault(pi, []).append(BubbleGeometry(
            bubble_idx=bi,
            polygon=polygon_data[off: off + length].reshape(-1, 2).tolist(),
            fit_box=fit_boxes[i],
            erase_box=erase_boxes[i],
            text_box=text_boxes[i],
        ))

    return [
        PageGeometry(
            page_index=pi,
            width=page_widths[pi],
            height=page_heights[pi],
            bubbles=tuple(pages_dict.get(pi, [])),
        )
        for pi in range(len(page_widths))
    ]


def save_scan_geometry(cp: "ChapterPaths", pages: list[PageGeometry]) -> None:
    """Save list[PageGeometry] → scan.npz."""
    page_widths  = np.array([p.width  for p in pages], dtype=np.int32)
    page_heights = np.array([p.height for p in pages], dtype=np.int32)

    bp, bi_list, pd, po, pl = [], [], [], [], []
    fit, erase, text = [], [], []
    offset = 0
    for pg in pages:
        for bg in pg.bubbles:
            flat = [c for pt in bg.polygon for c in pt]
            bp.append(pg.page_index)
            bi_list.append(bg.bubble_idx)
            pd.extend(flat)
            po.append(offset)
            pl.append(len(flat))
            fit.append(bg.fit_box)
            erase.append(bg.erase_box)
            text.append(bg.text_box)
            offset += len(flat)

    _empty4 = np.empty((0, 4), dtype=np.int32)
    np.savez_compressed(
        str(cp.scan),
        page_widths=page_widths,
        page_heights=page_heights,
        bubble_page=np.array(bp, dtype=np.int32),
        bubble_idx=np.array(bi_list, dtype=np.int32),
        polygon_data=np.array(pd, dtype=np.float32),
        polygon_offsets=np.array(po, dtype=np.int32),
        polygon_lengths=np.array(pl, dtype=np.int32),
        fit_boxes=np.array(fit, dtype=np.int32)   if fit   else _empty4,
        erase_boxes=np.array(erase, dtype=np.int32) if erase else _empty4,
        text_boxes=np.array(text, dtype=np.int32)  if text  else _empty4,
    )


# ── Internal ──────────────────────────────────────────────────────────


def _load_page_into(data, page_index: int, store: MaskStore) -> None:
    for entry in data["_meta"].tolist():
        bi, n_erase, n_text = (int(x) for x in entry.split(","))
        erase = tuple(
            TextMask(x=int(data[f"b{bi}_e{i}_xy"][0]), y=int(data[f"b{bi}_e{i}_xy"][1]),
                     image=data[f"b{bi}_e{i}"])
            for i in range(n_erase)
        )
        text = tuple(
            TextMask(x=int(data[f"b{bi}_t{i}_xy"][0]), y=int(data[f"b{bi}_t{i}_xy"][1]),
                     image=data[f"b{bi}_t{i}"])
            for i in range(n_text)
        )
        store.put(page_index, bi, BubbleMasks(erase_masks=erase, text_masks=text))
