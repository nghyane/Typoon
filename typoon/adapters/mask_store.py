"""MaskStore — per-page pixel masks, keyed by page_index.

scan.npz  → bubble geometry (polygon, boxes) for whole chapter
masks/    → one npz per page, all bubbles for that page

Access pattern: render reads one page at a time → group masks by page.
Reduces file opens from N_bubbles to N_pages.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

from typoon.vision.types import TextMask

if TYPE_CHECKING:
    from typoon.paths import ChapterPaths


@dataclass(frozen=True)
class BubbleMasks:
    """Erase and text masks for one bubble."""
    erase_masks: tuple[TextMask, ...]
    text_masks:  tuple[TextMask, ...]


# ── Scan geometry ─────────────────────────────────────────────────────


@dataclass(frozen=True)
class BubbleGeometry:
    """Geometry for one bubble, loaded from scan.npz."""
    bubble_idx: int
    polygon:    list[list[float]]
    fit_box:    list[int]
    erase_box:  list[int]
    text_box:   list[int]


@dataclass(frozen=True)
class PageGeometry:
    """All geometry for one page."""
    page_index: int
    width:      int
    height:     int
    bubbles:    tuple[BubbleGeometry, ...]


def load_scan_geometry(cp: "ChapterPaths") -> list[PageGeometry]:
    """Load scan.npz → list of PageGeometry, one per page."""
    data = np.load(str(cp.scan), allow_pickle=False)

    page_widths  = data["page_widths"].tolist()
    page_heights = data["page_heights"].tolist()

    bubble_page    = data["bubble_page"].tolist()
    bubble_idx_arr = data["bubble_idx"].tolist()
    polygon_data   = data["polygon_data"]
    polygon_off    = data["polygon_offsets"].tolist()
    polygon_len    = data["polygon_lengths"].tolist()
    fit_boxes      = data["fit_boxes"].tolist()
    erase_boxes    = data["erase_boxes"].tolist()
    text_boxes     = data["text_boxes"].tolist()

    pages_dict: dict[int, list[BubbleGeometry]] = {}
    for i, (pi, bi) in enumerate(zip(bubble_page, bubble_idx_arr)):
        off = polygon_off[i]
        length = polygon_len[i]
        polygon = polygon_data[off: off + length].reshape(-1, 2).tolist()
        pages_dict.setdefault(pi, []).append(BubbleGeometry(
            bubble_idx=bi,
            polygon=polygon,
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
    """Save list of PageGeometry → scan.npz."""
    page_widths  = np.array([p.width  for p in pages], dtype=np.int32)
    page_heights = np.array([p.height for p in pages], dtype=np.int32)

    bubble_page_list:  list[int] = []
    bubble_idx_list:   list[int] = []
    polygon_data_list: list[float] = []
    polygon_off_list:  list[int] = []
    polygon_len_list:  list[int] = []
    fit_list:   list[list[int]] = []
    erase_list: list[list[int]] = []
    text_list:  list[list[int]] = []

    offset = 0
    for pg in pages:
        for bg in pg.bubbles:
            flat = [coord for pt in bg.polygon for coord in pt]
            bubble_page_list.append(pg.page_index)
            bubble_idx_list.append(bg.bubble_idx)
            polygon_data_list.extend(flat)
            polygon_off_list.append(offset)
            polygon_len_list.append(len(flat))
            fit_list.append(bg.fit_box)
            erase_list.append(bg.erase_box)
            text_list.append(bg.text_box)
            offset += len(flat)

    np.savez_compressed(
        str(cp.scan),
        page_widths=page_widths,
        page_heights=page_heights,
        bubble_page=np.array(bubble_page_list, dtype=np.int32),
        bubble_idx=np.array(bubble_idx_list, dtype=np.int32),
        polygon_data=np.array(polygon_data_list, dtype=np.float32),
        polygon_offsets=np.array(polygon_off_list, dtype=np.int32),
        polygon_lengths=np.array(polygon_len_list, dtype=np.int32),
        fit_boxes=np.array(fit_list, dtype=np.int32) if fit_list else np.empty((0, 4), dtype=np.int32),
        erase_boxes=np.array(erase_list, dtype=np.int32) if erase_list else np.empty((0, 4), dtype=np.int32),
        text_boxes=np.array(text_list, dtype=np.int32) if text_list else np.empty((0, 4), dtype=np.int32),
    )


# ── Mask store ────────────────────────────────────────────────────────


class MaskStore:
    """In-memory masks for one chapter, grouped by page for efficient render access."""

    def __init__(self) -> None:
        # page_index → {bubble_idx → BubbleMasks}
        self._pages: dict[int, dict[int, BubbleMasks]] = {}

    def put(self, page_index: int, bubble_idx: int, masks: BubbleMasks) -> None:
        self._pages.setdefault(page_index, {})[bubble_idx] = masks

    def get(self, page_index: int, bubble_idx: int) -> BubbleMasks | None:
        return self._pages.get(page_index, {}).get(bubble_idx)

    def page_indices(self) -> list[int]:
        return sorted(self._pages)

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
                    arrays[f"b{bubble_idx}_e{i}"] = m.image
                    arrays[f"b{bubble_idx}_e{i}_xy"] = np.array([m.x, m.y], dtype=np.int32)
                for i, m in enumerate(bm.text_masks):
                    arrays[f"b{bubble_idx}_t{i}"] = m.image
                    arrays[f"b{bubble_idx}_t{i}_xy"] = np.array([m.x, m.y], dtype=np.int32)
                meta_list.append({
                    "bubble_idx": bubble_idx,
                    "n_erase": len(bm.erase_masks),
                    "n_text": len(bm.text_masks),
                })
            # store meta as object array
            arrays["_meta"] = np.array(
                [f"{m['bubble_idx']},{m['n_erase']},{m['n_text']}" for m in meta_list],
                dtype=object,
            )
            np.savez_compressed(str(cp.mask(page_index)), **arrays)

    @classmethod
    def load(cls, cp: "ChapterPaths") -> "MaskStore":
        """Load all per-page mask files."""
        store = cls()
        if not cp.masks.exists():
            return store
        for mask_file in sorted(cp.masks.glob("????.npz")):
            page_index = int(mask_file.stem)
            data = np.load(str(mask_file), allow_pickle=True)
            meta_raw = data["_meta"].tolist()
            for entry in meta_raw:
                bi, n_erase, n_text = (int(x) for x in entry.split(","))
                erase = tuple(
                    TextMask(
                        x=int(data[f"b{bi}_e{i}_xy"][0]),
                        y=int(data[f"b{bi}_e{i}_xy"][1]),
                        image=data[f"b{bi}_e{i}"],
                    )
                    for i in range(n_erase)
                )
                text = tuple(
                    TextMask(
                        x=int(data[f"b{bi}_t{i}_xy"][0]),
                        y=int(data[f"b{bi}_t{i}_xy"][1]),
                        image=data[f"b{bi}_t{i}"],
                    )
                    for i in range(n_text)
                )
                store.put(page_index, bi, BubbleMasks(erase_masks=erase, text_masks=text))
        return store

    @classmethod
    def load_page(cls, cp: "ChapterPaths", page_index: int) -> dict[int, BubbleMasks]:
        """Load masks for a single page only — for streaming render."""
        path = cp.mask(page_index)
        if not path.exists():
            return {}
        data = np.load(str(path), allow_pickle=True)
        meta_raw = data["_meta"].tolist()
        result: dict[int, BubbleMasks] = {}
        for entry in meta_raw:
            bi, n_erase, n_text = (int(x) for x in entry.split(","))
            erase = tuple(
                TextMask(
                    x=int(data[f"b{bi}_e{i}_xy"][0]),
                    y=int(data[f"b{bi}_e{i}_xy"][1]),
                    image=data[f"b{bi}_e{i}"],
                )
                for i in range(n_erase)
            )
            text = tuple(
                TextMask(
                    x=int(data[f"b{bi}_t{i}_xy"][0]),
                    y=int(data[f"b{bi}_t{i}_xy"][1]),
                    image=data[f"b{bi}_t{i}"],
                )
                for i in range(n_text)
            )
            result[bi] = BubbleMasks(erase_masks=erase, text_masks=text)
        return result
