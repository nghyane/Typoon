"""Run Lens detector + lens_native grouper, return raw + grouped output."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from typoon.models import ModelHub
from typoon.vision._backends.comic_detr import load_session
from typoon.vision.contracts import BubbleGroup, DetectionResult
from typoon.vision.detectors.lens_blocks import LensBlocksDetector
from typoon.vision.groupers.lens_native import LensNativeGrouper


@dataclass(slots=True)
class ProbeResult:
    image:     np.ndarray
    detection: DetectionResult
    groups:    tuple[BubbleGroup, ...]


def run(
    image: np.ndarray, models_dir: Path, lang: str | None = None,
) -> ProbeResult:
    """Detector + grouper end-to-end in a fresh event loop."""
    async def _go():
        hub = ModelHub(models_dir)
        detector = LensBlocksDetector(comic_detr=load_session(hub.resolve_comic_detr()))
        detection = await detector.detect(image, lang=lang)
        grouper = LensNativeGrouper()
        groups = await grouper.group(image, detection, lang)
        return detection, groups

    detection, groups = asyncio.run(_go())
    return ProbeResult(image=image, detection=detection, groups=groups)


def to_json(result: ProbeResult) -> dict:
    """Serialise the full probe payload — Lens blocks + DETR regions + groups."""
    d = result.detection
    return {
        "page_size":     list(d.page_size),
        "detected_lang": d.detected_lang,
        "lens_blocks": [
            {
                "bbox":           list(b.bbox),
                "text":           b.text or "",
                "rotation_deg":   round(b.rotation_deg, 2),
                "text_direction": b.text_direction,
                "n_words":        len(b.words),
                "n_lines":        len(b.lines),
                "words": [
                    {"bbox": list(w.bbox), "text": w.text} for w in b.words
                ],
                "lines": [
                    {"bbox": list(l.bbox), "text": l.text} for l in b.lines
                ],
            }
            for b in d.blocks
        ],
        "lens_rejected": [
            {"bbox": list(b.bbox), "text": b.text or "", "reason": r}
            for (b, r) in d.rejected
        ],
        "comic_detr_regions": [
            {"cls": cls, "bbox": list(bbox), "conf": round(conf, 3)}
            for (cls, bbox, conf) in d.bubble_regions
        ],
        "groups": [
            {
                "bbox":           list(g.bbox),
                "polygon":        [list(p) for p in g.polygon],
                "text":           g.text,
                "shape_kind":     g.shape_kind,
                "text_direction": g.text_direction,
                "rotation_deg":   round(g.rotation_deg, 2),
                "typesetting": (
                    None if g.typesetting is None
                    else {
                        "font_size_px":       g.typesetting.font_size_px,
                        "line_count":         g.typesetting.line_count,
                        "avg_chars_per_line": round(g.typesetting.avg_chars_per_line, 2),
                    }
                ),
            }
            for g in result.groups
        ],
    }
