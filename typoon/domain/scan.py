"""Scan stage output types — pure Python, no numpy, no vision imports."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .prepared import Chapter as _PreparedChapter


@dataclass(frozen=True)
class Box:
    """Render and erase geometry for one bubble, in prepared-page coordinates."""

    polygon:  list[list[float]]  # render boundary [[x,y], ...]
    fit:      list[int]          # [x1,y1,x2,y2] tight fit around text
    erase:    list[int]          # [x1,y1,x2,y2] dilated for inpainting
    text:     list[int]          # [x1,y1,x2,y2] raw detection bbox


@dataclass(frozen=True)
class Bubble:
    """One accepted text group after detect → group → OCR."""

    idx:         int
    page_index:  int
    source_text: str
    confidence:  float
    box:         Box


@dataclass(frozen=True)
class Page:
    """One prepared page after scan stage."""

    index:   int
    width:   int
    height:  int
    bubbles: tuple[Bubble, ...]


@dataclass(frozen=True)
class Chapter:
    """Full scan output — typed boundary between scan and translate stages."""

    prepared: _PreparedChapter
    pages:    tuple[Page, ...]

    @property
    def page_count(self) -> int:
        return len(self.pages)

    @property
    def all_bubbles(self) -> list[Bubble]:
        return [b for p in self.pages for b in p.bubbles]

    def save(self, out_dir: Path) -> Path:
        """Serialize to <out_dir>/scan/manifest.json + pages/*.json."""
        out_dir = Path(out_dir) / "scan"
        pages_dir = out_dir / "pages"
        pages_dir.mkdir(parents=True, exist_ok=True)

        for page in self.pages:
            page_data = {
                "index": page.index,
                "width": page.width,
                "height": page.height,
                "bubbles": [
                    {
                        "idx": b.idx,
                        "page_index": b.page_index,
                        "source_text": b.source_text,
                        "confidence": b.confidence,
                        "box": {
                            "polygon": b.box.polygon,
                            "fit":     b.box.fit,
                            "erase":   b.box.erase,
                            "text":    b.box.text,
                        },
                    }
                    for b in page.bubbles
                ],
            }
            (pages_dir / f"{page.index:04d}.json").write_text(
                json.dumps(page_data, ensure_ascii=False, indent=2), "utf-8"
            )

        manifest: dict[str, Any] = {
            "version": 1,
            "prepared_root": str(self.prepared.root),
            "page_count": self.page_count,
            "bubble_count": len(self.all_bubbles),
        }
        manifest_path = out_dir / "manifest.json"
        manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), "utf-8")
        return manifest_path

    @classmethod
    def load(cls, base_dir: Path) -> "Chapter":
        """Load from <base_dir>/scan/."""
        from .prepared import Chapter as PreparedChapter
        scan_dir = Path(base_dir) / "scan"
        manifest = json.loads((scan_dir / "manifest.json").read_text("utf-8"))
        prepared = PreparedChapter.load(manifest["prepared_root"])
        pages_dir = scan_dir / "pages"
        pages: list[Page] = []
        for page_file in sorted(pages_dir.glob("*.json")):
            pd = json.loads(page_file.read_text("utf-8"))
            bubbles = tuple(
                Bubble(
                    idx=b["idx"],
                    page_index=b["page_index"],
                    source_text=b["source_text"],
                    confidence=b["confidence"],
                    box=Box(
                        polygon=b["box"]["polygon"],
                        fit=b["box"]["fit"],
                        erase=b["box"]["erase"],
                        text=b["box"]["text"],
                    ),
                )
                for b in pd["bubbles"]
            )
            pages.append(Page(index=pd["index"], width=pd["width"],
                              height=pd["height"], bubbles=bubbles))
        return cls(prepared=prepared, pages=tuple(pages))
