#!/usr/bin/env python3
"""Generate browser-sdk dev analysis artifact from the real scan pipeline.

The artifact is the browser SDK contract consumed by the dev reader:
ImagePixels -> PageAnalysis -> RenderPlan[] -> translation + overlay.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
import time
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image


ROOT = Path(__file__).resolve().parents[3]
DEFAULT_IMAGE = ROOT / "packages/browser-sdk/dev/public/sample-page.jpg"
DEFAULT_OUTPUT = ROOT / "packages/browser-sdk/dev/public/artifacts/sample-analysis.json"
DEFAULT_MODEL = ROOT / "workers/scan/container/comic-detr-v4s-int8.onnx"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", type=Path, default=DEFAULT_IMAGE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--lang", default="en")
    args = parser.parse_args()

    artifact = asyncio.run(generate(args.image, args.model, args.lang))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(artifact, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"wrote {args.output}")


async def generate(image_path: Path, model_path: Path, lang: str) -> dict[str, Any]:
    from typoon.vision._backends.comic_detr import load_session
    from typoon.vision.detectors.lens.detector import LensBlocksDetector
    from typoon.vision.groupers.lens_native import LensNativeGrouper

    image = np.array(Image.open(image_path).convert("RGB"))
    h, w = image.shape[:2]

    t0 = time.perf_counter()
    comic = load_session(str(model_path))
    detector = LensBlocksDetector(
        comic_detr=comic,
        endpoint=os.environ.get("LENS_ENDPOINT") or None,
        max_concurrent=15,
    )
    detection = await detector.detect(image, lang)
    detection = replace(detection, blocks=dedupe_text_blocks(list(detection.blocks)))
    t_detect = time.perf_counter()

    groups = await LensNativeGrouper().group(image, detection, lang)
    t_group = time.perf_counter()

    blocks = list(detection.blocks)
    meta = read_sample_meta(image_path)
    plans = [render_plan_to_wire(i, group, blocks, w, h) for i, group in enumerate(groups)]
    return {
        "version": 1,
        "image": {
            "src": "/sample-page.jpg",
            "width": w,
            "height": h,
            "source": meta.get("source", "MangaDex"),
            "chapterId": meta.get("chapter_id"),
            "pageIndex": meta.get("page_index"),
        },
        "analysis": {
            "pageIndex": 0,
            "pageSize": [w, h],
            "detectedLanguage": detection.detected_lang,
            "plans": plans,
            "timingMs": {
                "detect": round((t_detect - t0) * 1000),
                "group": round((t_group - t_detect) * 1000),
                "total": round((t_group - t0) * 1000),
            },
        },
    }


def read_sample_meta(image_path: Path) -> dict[str, Any]:
    meta_path = image_path.with_suffix(".mangadex.json")
    if not meta_path.exists():
        return {}
    try:
        return json.loads(meta_path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def render_plan_to_wire(idx: int, group, blocks: list[Any], w: int, h: int) -> dict[str, Any]:
    bbox = list(map(int, group.bbox))
    members = find_member_blocks(blocks, bbox)
    line_boxes = dedupe_boxes([entry["bbox"] for entry in line_entries(members)])
    text_boxes = line_boxes or dedupe_boxes([entry["bbox"] for entry in word_entries(members)]) or [bbox]
    class_ = classify_group(group)
    source_text = group.text
    return {
        "id": f"p0-r{idx}",
        "pageIndex": 0,
        "pageSize": [w, h],
        "sourceText": source_text,
        "drawable": polygon_to_wire(group.polygon) or bbox_to_polygon(bbox),
        "bbox": bbox,
        "textBoxes": text_boxes,
        "class": class_,
        "rotationDeg": float(group.rotation_deg),
        "confidence": float(group.confidence),
        "fontHint": font_hint_to_wire(group, line_count=len(line_boxes) or None, source_text=source_text),
    }


def dedupe_text_blocks(blocks: list[Any]) -> tuple[Any, ...]:
    kept: list[Any] = []
    for block in blocks:
        duplicate_idx = next(
            (i for i, existing in enumerate(kept) if same_text_region(block, existing)),
            None,
        )
        if duplicate_idx is None:
            kept.append(block)
        elif prefer_text_block(block, kept[duplicate_idx]):
            kept[duplicate_idx] = block
    return tuple(kept)


def same_text_region(a: Any, b: Any) -> bool:
    if bbox_iou(a.bbox, b.bbox) >= 0.90:
        return True
    a_lines = [line.bbox for line in getattr(a, "lines", []) or []]
    b_lines = [line.bbox for line in getattr(b, "lines", []) or []]
    if len(a_lines) == len(b_lines) and a_lines:
        overlaps = [
            bbox_iou(a_box, b_box)
            for a_box, b_box in zip(sort_boxes(a_lines), sort_boxes(b_lines))
        ]
        return min(overlaps) >= 0.70
    return False


def prefer_text_block(candidate: Any, current: Any) -> bool:
    candidate_score = text_block_quality(candidate)
    current_score = text_block_quality(current)
    return candidate_score > current_score


def text_block_quality(block: Any) -> tuple[int, int, float, int]:
    detector = getattr(block, "detector", "") or ""
    detector_priority = 1 if detector.endswith("/bubble") else 0
    line_count = len(getattr(block, "lines", []) or [])
    word_count = len(getattr(block, "words", []) or [])
    confidence = float(getattr(block, "confidence", 0.0) or 0.0)
    area = bbox_area(block.bbox)
    return (detector_priority, line_count + word_count, confidence, area)


def find_member_blocks(blocks: list[Any], bbox: list[int]) -> list[Any]:
    members = [block for block in blocks if center_inside(block.bbox, bbox)]
    if members:
        return members
    return [block for block in blocks if containment(block.bbox, bbox) >= 0.35]


def center_inside(inner: Any, outer: list[int]) -> bool:
    x1, y1, x2, y2 = list(map(float, inner))
    cx = (x1 + x2) / 2
    cy = (y1 + y2) / 2
    return outer[0] <= cx <= outer[2] and outer[1] <= cy <= outer[3]


def containment(inner: Any, outer: list[int]) -> float:
    x1, y1, x2, y2 = list(map(float, inner))
    ix1 = max(x1, outer[0])
    iy1 = max(y1, outer[1])
    ix2 = min(x2, outer[2])
    iy2 = min(y2, outer[3])
    if ix1 >= ix2 or iy1 >= iy2:
        return 0.0
    return ((ix2 - ix1) * (iy2 - iy1)) / max(1.0, (x2 - x1) * (y2 - y1))


def line_entries(blocks: list[Any]) -> list[dict[str, Any]]:
    entries: list[dict[str, Any]] = []
    for block in blocks:
        for line in getattr(block, "lines", []) or []:
            entries.append({"bbox": list(map(int, line.bbox)), "text": line.text})
    return entries


def word_entries(blocks: list[Any]) -> list[dict[str, Any]]:
    entries: list[dict[str, Any]] = []
    for block in blocks:
        for word in getattr(block, "words", []) or []:
            entries.append({"bbox": list(map(int, word.bbox)), "text": word.text})
    return entries


def dedupe_boxes(boxes: list[list[int]]) -> list[list[int]]:
    out: list[list[int]] = []
    for box in sorted(boxes, key=lambda b: (b[1], b[0], b[3], b[2])):
        if any(bbox_iou(box, kept) >= 0.55 for kept in out):
            continue
        out.append(box)
    return out


def bbox_iou(a: Any, b: Any) -> float:
    ix1 = max(a[0], b[0])
    iy1 = max(a[1], b[1])
    ix2 = min(a[2], b[2])
    iy2 = min(a[3], b[3])
    if ix1 >= ix2 or iy1 >= iy2:
        return 0.0
    inter = (ix2 - ix1) * (iy2 - iy1)
    area_a = bbox_area(a)
    area_b = bbox_area(b)
    return inter / (area_a + area_b - inter)


def bbox_area(box: Any) -> int:
    return max(1, int((box[2] - box[0]) * (box[3] - box[1])))


def sort_boxes(boxes: list[Any]) -> list[Any]:
    return sorted(boxes, key=lambda b: (b[1], b[0], b[3], b[2]))


def classify_group(group) -> str:
    try:
        from typoon.vision.groupers._classify import classify_block

        class B:
            rotation_deg = group.rotation_deg
            bbox = group.bbox
            text = group.text

        return classify_block(B(), group.text or "")
    except Exception:
        return "dialogue"


def font_hint_to_wire(group, line_count: int | None, source_text: str) -> dict[str, Any] | None:
    hint = group.typesetting
    if hint is None:
        return None
    clean_line_count = line_count or int(hint.line_count)
    char_count = sum(1 for ch in source_text if not ch.isspace())
    return {
        "sourceFontPx": int(hint.font_size_px),
        "sourceLineCount": clean_line_count,
        "sourceAvgCharsPerLine": float(char_count / clean_line_count) if clean_line_count else float(hint.avg_chars_per_line),
        "sourceDirection": "vertical" if getattr(group, "is_vertical", False) else "horizontal",
    }


def polygon_to_wire(poly) -> list[list[float]] | None:
    if poly is None:
        return None
    return [[float(x), float(y)] for x, y in poly]


def bbox_to_polygon(bbox: list[int]) -> list[list[float]]:
    x1, y1, x2, y2 = bbox
    return [[x1, y1], [x2, y1], [x2, y2], [x1, y2]]


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        raise
    except Exception as exc:
        print(f"generate-dev-analysis failed: {type(exc).__name__}: {exc}", file=sys.stderr)
        raise
