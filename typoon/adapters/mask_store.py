"""MaskStore — pixel-level mask data keyed by (page_index, bubble_idx).

Lives in adapter layer. Never imported by domain types.
Scan stage populates it; render stage consumes it.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from typoon.vision.types import TextMask

MaskKey = tuple[int, int]  # (page_index, bubble_idx)


@dataclass(frozen=True)
class Masks:
    """Erase and text masks for one bubble."""

    erase_masks: tuple[TextMask, ...]
    text_masks:  tuple[TextMask, ...]


class MaskStore:
    """In-memory mask store for one chapter scan."""

    def __init__(self) -> None:
        self._data: dict[MaskKey, Masks] = {}

    def put(self, page_index: int, bubble_idx: int, masks: Masks) -> None:
        self._data[(page_index, bubble_idx)] = masks

    def get(self, page_index: int, bubble_idx: int) -> Masks | None:
        return self._data.get((page_index, bubble_idx))

    def __len__(self) -> int:
        return len(self._data)

    def save(self, base_dir: Path) -> Path:
        """Serialize masks to <base_dir>/scan/masks/.

        Each bubble: one .npz file containing erase and text mask arrays
        plus x/y offsets as metadata.
        """
        masks_dir = Path(base_dir) / "scan" / "masks"
        masks_dir.mkdir(parents=True, exist_ok=True)

        index: list[dict] = []
        for (page_idx, bubble_idx), masks in self._data.items():
            stem = f"{page_idx:04d}_{bubble_idx:04d}"
            arrays: dict[str, np.ndarray] = {}
            meta: dict[str, list] = {"erase": [], "text": []}

            for i, m in enumerate(masks.erase_masks):
                arrays[f"erase_{i}"] = m.image
                meta["erase"].append({"x": m.x, "y": m.y})
            for i, m in enumerate(masks.text_masks):
                arrays[f"text_{i}"] = m.image
                meta["text"].append({"x": m.x, "y": m.y})

            np.savez_compressed(masks_dir / f"{stem}.npz", **arrays)
            index.append({
                "page": page_idx, "bubble": bubble_idx,
                "file": f"{stem}.npz", "meta": meta,
            })

        (masks_dir / "index.json").write_text(
            json.dumps(index, ensure_ascii=False, indent=2), "utf-8"
        )
        return masks_dir

    @classmethod
    def load(cls, base_dir: Path) -> "MaskStore":
        """Load from <base_dir>/scan/masks/."""
        masks_dir = Path(base_dir) / "scan" / "masks"
        index_path = masks_dir / "index.json"
        store = cls()
        if not index_path.exists():
            return store

        index = json.loads(index_path.read_text("utf-8"))
        for entry in index:
            npz = np.load(masks_dir / entry["file"])
            meta = entry["meta"]

            erase = tuple(
                TextMask(x=m["x"], y=m["y"], image=npz[f"erase_{i}"])
                for i, m in enumerate(meta["erase"])
            )
            text = tuple(
                TextMask(x=m["x"], y=m["y"], image=npz[f"text_{i}"])
                for i, m in enumerate(meta["text"])
            )
            store.put(entry["page"], entry["bubble"], Masks(erase_masks=erase, text_masks=text))

        return store
