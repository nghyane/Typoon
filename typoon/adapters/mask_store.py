"""MaskStore — in-memory pixel masks, packed/unpacked as a single npz blob.

Detection emits one or more `TextMask` per bubble (text mask + dilated erase
mask). Render needs them per page to drive the eraser. The store is in-memory
during scan, packed into a single `.npz` for transit between workers, and
unpacked back into memory at render time.

Geometry (polygon, fit/erase/text boxes) lives in the DB; this store carries
only pixel data.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from typoon.vision.types import TextMask


class BubbleMasks:
    """Erase and text masks for one bubble."""
    __slots__ = ("erase_masks", "text_masks")

    def __init__(self, erase_masks: tuple[TextMask, ...], text_masks: tuple[TextMask, ...]) -> None:
        self.erase_masks = erase_masks
        self.text_masks  = text_masks


class MaskStore:
    """In-memory masks for one chapter, grouped by page."""

    def __init__(self) -> None:
        self._pages: dict[int, dict[int, BubbleMasks]] = {}

    def put(self, page_index: int, bubble_idx: int, masks: BubbleMasks) -> None:
        self._pages.setdefault(page_index, {})[bubble_idx] = masks

    def get(self, page_index: int, bubble_idx: int) -> BubbleMasks | None:
        return self._pages.get(page_index, {}).get(bubble_idx)

    def page_masks(self, page_index: int) -> dict[int, BubbleMasks]:
        return self._pages.get(page_index, {})

    def __len__(self) -> int:
        return sum(len(b) for b in self._pages.values())

    def pack(self, path: Path) -> None:
        """Serialize all masks into a single compressed npz file."""
        arrays: dict[str, np.ndarray] = {}
        meta_lines: list[str] = []
        for page_index, bubbles in self._pages.items():
            for bubble_idx, bm in sorted(bubbles.items()):
                for i, m in enumerate(bm.erase_masks):
                    arrays[f"p{page_index}_b{bubble_idx}_e{i}"]    = m.image
                    arrays[f"p{page_index}_b{bubble_idx}_e{i}_xy"] = np.array([m.x, m.y], dtype=np.int32)
                for i, m in enumerate(bm.text_masks):
                    arrays[f"p{page_index}_b{bubble_idx}_t{i}"]    = m.image
                    arrays[f"p{page_index}_b{bubble_idx}_t{i}_xy"] = np.array([m.x, m.y], dtype=np.int32)
                meta_lines.append(f"{page_index},{bubble_idx},{len(bm.erase_masks)},{len(bm.text_masks)}")
        arrays["_meta"] = np.array(meta_lines, dtype=object)
        np.savez_compressed(str(path), **arrays)

    @classmethod
    def unpack(cls, path: Path) -> "MaskStore":
        store = cls()
        data = np.load(str(path), allow_pickle=True)
        for entry in data["_meta"].tolist():
            page_index, bubble_idx, n_erase, n_text = (int(x) for x in entry.split(","))
            erase = tuple(
                TextMask(
                    x=int(data[f"p{page_index}_b{bubble_idx}_e{i}_xy"][0]),
                    y=int(data[f"p{page_index}_b{bubble_idx}_e{i}_xy"][1]),
                    image=data[f"p{page_index}_b{bubble_idx}_e{i}"],
                )
                for i in range(n_erase)
            )
            text = tuple(
                TextMask(
                    x=int(data[f"p{page_index}_b{bubble_idx}_t{i}_xy"][0]),
                    y=int(data[f"p{page_index}_b{bubble_idx}_t{i}_xy"][1]),
                    image=data[f"p{page_index}_b{bubble_idx}_t{i}"],
                )
                for i in range(n_text)
            )
            store.put(page_index, bubble_idx, BubbleMasks(erase_masks=erase, text_masks=text))
        return store
