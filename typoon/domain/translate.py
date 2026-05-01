"""Translate stage output types — pure Python, no LLM or adapter imports."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .scan import Box, Bubble as ScannedBubble, Chapter as ScannedChapter, Page as ScannedPage


@dataclass(frozen=True)
class Bubble:
    """One bubble after translation."""

    source:           ScannedBubble
    translation_key:  str
    translated_text:  str
    kind:             str   # "dialogue" | "sfx" | "skip"

    @property
    def idx(self) -> int:
        return self.source.idx

    @property
    def page_index(self) -> int:
        return self.source.page_index

    @property
    def source_text(self) -> str:
        return self.source.source_text


@dataclass(frozen=True)
class Page:
    """One page after translation."""

    source:  ScannedPage
    bubbles: tuple[Bubble, ...]

    @property
    def index(self) -> int:
        return self.source.index


@dataclass(frozen=True)
class Chapter:
    """Full translate output — typed boundary between translate and render stages."""

    scan:  ScannedChapter
    pages: tuple[Page, ...]

    @property
    def all_bubbles(self) -> list[Bubble]:
        return [b for p in self.pages for b in p.bubbles]

    def save(self, base_dir: Path) -> Path:
        """Serialize to <base_dir>/translate/manifest.json."""
        out_dir = Path(base_dir) / "translate"
        out_dir.mkdir(parents=True, exist_ok=True)

        bubbles_data: list[dict[str, Any]] = []
        for b in self.all_bubbles:
            bubbles_data.append({
                "idx":             b.idx,
                "page_index":      b.page_index,
                "translation_key": b.translation_key,
                "source_text":     b.source_text,
                "translated_text": b.translated_text,
                "kind":            b.kind,
                "confidence":      b.source.confidence,
                "box": {
                    "polygon": b.source.box.polygon,
                    "fit":     b.source.box.fit,
                    "erase":   b.source.box.erase,
                    "text":    b.source.box.text,
                },
            })

        manifest: dict[str, Any] = {"version": 1, "bubbles": bubbles_data}
        manifest_path = out_dir / "manifest.json"
        manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), "utf-8")
        return manifest_path

    @classmethod
    def load(cls, base_dir: Path) -> "Chapter":
        """Load from <base_dir>/translate/manifest.json.

        Requires the matching scan output to exist at <base_dir>/scan/.
        """
        scan = ScannedChapter.load(base_dir)
        manifest = json.loads(
            (Path(base_dir) / "translate" / "manifest.json").read_text("utf-8")
        )

        # Build lookup: (page_index, bubble_idx) → ScannedBubble
        scan_bubble: dict[tuple[int, int], ScannedBubble] = {
            (b.page_index, b.idx): b for b in scan.all_bubbles
        }

        by_page: dict[int, list[Bubble]] = {}
        for bd in manifest["bubbles"]:
            key = (bd["page_index"], bd["idx"])
            sb = scan_bubble.get(key)
            if sb is None:
                # Reconstruct from saved data if scan bubble not found
                sb = ScannedBubble(
                    idx=bd["idx"],
                    page_index=bd["page_index"],
                    source_text=bd["source_text"],
                    confidence=bd["confidence"],
                    box=Box(
                        polygon=bd["box"]["polygon"],
                        fit=bd["box"]["fit"],
                        erase=bd["box"]["erase"],
                        text=bd["box"]["text"],
                    ),
                )
            tb = Bubble(
                source=sb,
                translation_key=bd["translation_key"],
                translated_text=bd["translated_text"],
                kind=bd["kind"],
            )
            by_page.setdefault(bd["page_index"], []).append(tb)

        pages = tuple(
            Page(source=sp, bubbles=tuple(by_page.get(sp.index, [])))
            for sp in scan.pages
        )
        return cls(scan=scan, pages=pages)
