"""MaskStore — in-memory pixel masks, serialized as a compressed npz blob.

Scan stage produces masks, packs them for transit to blob storage.
Render stage unpacks and consumes. In-process use (no workers) never
calls pack/unpack — the store is passed directly.

text_masks == erase_masks: they are the same blob. The distinction was
legacy; both fields are kept for backward-compat with existing npz files
already stored in blob storage. New code only reads erase_masks.

Removed:
  - _bubble_masks (CTD UNet page-level mask) — no consumer after scan
  - put_bubble_mask / page_bubble_mask — dead API
  - pack/unpack encode both erase+text; they are identical so only
    erase is written/read; text slot reconstructed from erase on unpack.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from typoon.vision.contracts import TextMask


class BubbleMasks:
    """Erase masks for one bubble."""
    __slots__ = ("erase_masks", "text_masks")

    def __init__(
        self,
        erase_masks: tuple[TextMask, ...],
        text_masks:  tuple[TextMask, ...] | None = None,
    ) -> None:
        self.erase_masks = erase_masks
        # text_masks == erase_masks — kept for compat, not used by render
        self.text_masks  = text_masks if text_masks is not None else erase_masks


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
        """Serialize to compressed npz for cross-worker transit."""
        arrays: dict[str, np.ndarray] = {}
        meta_lines: list[str] = []
        for page_index, bubbles in self._pages.items():
            for bubble_idx, bm in sorted(bubbles.items()):
                for i, m in enumerate(bm.erase_masks):
                    arrays[f"p{page_index}_b{bubble_idx}_e{i}"]    = m.image
                    arrays[f"p{page_index}_b{bubble_idx}_e{i}_xy"] = np.array(
                        [m.x, m.y], dtype=np.int32,
                    )
                meta_lines.append(f"{page_index},{bubble_idx},{len(bm.erase_masks)}")
        arrays["_meta"] = np.array(meta_lines, dtype=object)
        np.savez_compressed(str(path), **arrays)

    @classmethod
    def unpack(cls, path: Path) -> "MaskStore":
        """Deserialize npz. Handles both old format (4-field meta) and new (3-field)."""
        store = cls()
        data  = np.load(str(path), allow_pickle=True)
        for entry in data["_meta"].tolist():
            parts = entry.split(",")
            page_index, bubble_idx, n_erase = int(parts[0]), int(parts[1]), int(parts[2])
            erase = tuple(
                TextMask(
                    x=int(data[f"p{page_index}_b{bubble_idx}_e{i}_xy"][0]),
                    y=int(data[f"p{page_index}_b{bubble_idx}_e{i}_xy"][1]),
                    image=data[f"p{page_index}_b{bubble_idx}_e{i}"],
                )
                for i in range(n_erase)
            )
            # Old format stored text masks separately; ignore them —
            # erase_masks == text_masks in new architecture.
            store.put(page_index, bubble_idx, BubbleMasks(erase_masks=erase))
        return store
