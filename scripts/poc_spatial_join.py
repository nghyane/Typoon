"""POC — comic_detr × Lens spatial join + gap recovery.

End-to-end proof of concept for the proposed pipeline:

  parallel:
    lens_blocks  = lens.detect(image)
    comic_dets   = comic_detr.detect(image)        # bubble + text_bubble + text_free

  phase 1 — text-region anchored groups:
    for each text_bubble / text_free region in comic_dets:
      lens_inside = lens_blocks where iou(b.bbox, region.bbox) >= IOU_INSIDE
      coverage = covered_area(lens_inside ∩ region) / area(region)
      if coverage < COVERAGE_THRESHOLD:
        recover = lens.process(preprocessed_crop(region))   # whiten dots + upscale
        lens_inside ∪= recovered
      parent_bubble = closest bubble enclosing region
      emit group(bbox=parent_bubble or region, text=concat(lens_inside.text))

  phase 2 — stray Lens blocks (comic_detr missed):
    for b in lens_blocks not assigned in phase 1:
      emit standalone group(b)

  phase 3 — bubble_mask:
    rasterize bubble class bboxes for render fit-expansion

Output: per-fixture diff vs current pipeline + key metrics:
  - total groups produced
  - text recovered vs missing
  - latency
  - artefact: debug-runs/<probe>/poc_spatial_join.png (visual overlay)
"""

from __future__ import annotations

import asyncio
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import cv2
import numpy as np
import onnxruntime as ort
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from typoon.vision.detectors.lens_blocks import LensBlocksDetector  # noqa: E402


# ─── Tunables ─────────────────────────────────────────────────────────────


IOU_INSIDE          = 0.30   # min IoU(lens, region) to count lens block as inside
COVERAGE_THRESHOLD  = 0.60   # below this → trigger gap recovery
WHITEN_MAX_AREA     = 400    # connected components smaller than this → whitened
UPSCALE_MIN_DIM     = 320    # min(h, w) target for gap recovery crops
COMIC_DETR_CONF     = 0.30
PAD_PX              = 8


# ─── Detector wrappers ────────────────────────────────────────────────────


def _load_comic_detr() -> ort.InferenceSession:
    from huggingface_hub import hf_hub_download
    path = hf_hub_download(
        "ogkalu/comic-text-and-bubble-detector",
        "detector-v4-s_int8.onnx",
    )
    return ort.InferenceSession(
        path, providers=["CoreMLExecutionProvider", "CPUExecutionProvider"],
    )


@dataclass(frozen=True)
class _ComicDet:
    cls:  Literal["bubble", "text_bubble", "text_free"]
    conf: float
    bbox: tuple[int, int, int, int]


def _comic_detect(sess, img: np.ndarray) -> list[_ComicDet]:
    h, w = img.shape[:2]
    resized = cv2.resize(img, (640, 640), interpolation=cv2.INTER_LINEAR)
    arr = resized.astype(np.float32).transpose(2, 0, 1)[None] / 255.0
    out = sess.run(
        None,
        {"images": arr, "orig_target_sizes": np.array([[w, h]], dtype=np.int64)},
    )
    labels, boxes, scores = out
    above = scores[0] > COMIC_DETR_CONF
    names = {0: "bubble", 1: "text_bubble", 2: "text_free"}
    out_list: list[_ComicDet] = []
    for i in np.where(above)[0]:
        x1, y1, x2, y2 = boxes[0][i].tolist()
        out_list.append(_ComicDet(
            cls=names[int(labels[0][i])],
            conf=float(scores[0][i]),
            bbox=(
                max(0, int(x1) - PAD_PX),
                max(0, int(y1) - PAD_PX),
                min(w, int(x2) + PAD_PX),
                min(h, int(y2) + PAD_PX),
            ),
        ))
    return out_list


# ─── Geometry helpers ─────────────────────────────────────────────────────


def _intersect_area(a, b) -> int:
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    if ix1 >= ix2 or iy1 >= iy2:
        return 0
    return (ix2 - ix1) * (iy2 - iy1)


def _area(b) -> int:
    return max(1, (b[2] - b[0]) * (b[3] - b[1]))


def _iou(a, b) -> float:
    inter = _intersect_area(a, b)
    if inter == 0:
        return 0.0
    return inter / (_area(a) + _area(b) - inter)


def _io_smaller(a, b) -> float:
    """Intersection over smaller — measures how much smaller is inside larger."""
    inter = _intersect_area(a, b)
    return inter / min(_area(a), _area(b))


def _contains(outer, inner) -> bool:
    return (outer[0] <= inner[0] and outer[1] <= inner[1]
            and outer[2] >= inner[2] and outer[3] >= inner[3])


# ─── Coverage analysis ────────────────────────────────────────────────────


def _coverage_in_region(lens_bboxes: list[tuple[int, int, int, int]],
                        region: tuple[int, int, int, int]) -> float:
    """Fraction of region area covered by union of lens bboxes."""
    rx1, ry1, rx2, ry2 = region
    rw = max(1, rx2 - rx1)
    rh = max(1, ry2 - ry1)
    if not lens_bboxes:
        return 0.0
    # Rasterize at low resolution to compute union area cheaply
    mask = np.zeros((rh, rw), dtype=np.uint8)
    for b in lens_bboxes:
        ix1 = max(0, b[0] - rx1)
        iy1 = max(0, b[1] - ry1)
        ix2 = min(rw, b[2] - rx1)
        iy2 = min(rh, b[3] - ry1)
        if ix2 > ix1 and iy2 > iy1:
            mask[iy1:iy2, ix1:ix2] = 255
    return float(mask.sum() / 255) / (rw * rh)


# ─── Gap recovery — preprocessed crop ─────────────────────────────────────


def _whiten_dots(img: np.ndarray, max_area: int = WHITEN_MAX_AREA) -> np.ndarray:
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    n, labels, stats, _ = cv2.connectedComponentsWithStats(binary, 8)
    out = img.copy()
    for i in range(1, n):
        if stats[i, cv2.CC_STAT_AREA] < max_area:
            out[labels == i] = [255, 255, 255]
    return out


def _upscale(img: np.ndarray, target_short: int = UPSCALE_MIN_DIM) -> np.ndarray:
    h, w = img.shape[:2]
    if min(h, w) >= target_short:
        return img
    scale = max(1, int(np.ceil(target_short / min(h, w))))
    return np.asarray(
        Image.fromarray(img).resize((w * scale, h * scale), Image.LANCZOS)
    )


async def _lens_ocr(api, crop: np.ndarray, lang: str) -> list[dict]:
    result = await api.process_image(
        crop, ocr_language=lang, output_format="detailed",
    )
    return result.get("detailed_blocks") or []


# ─── Bubble lookup ────────────────────────────────────────────────────────


def _find_parent_bubble(
    text_region: tuple[int, int, int, int],
    bubbles: list[_ComicDet],
) -> _ComicDet | None:
    """The smallest bubble that contains the text region (best fit)."""
    candidates = [b for b in bubbles if _contains(b.bbox, text_region)]
    if not candidates:
        # Fall back to highest-overlap bubble
        scored = [(b, _io_smaller(b.bbox, text_region)) for b in bubbles]
        scored = [(b, s) for b, s in scored if s > 0.5]
        if not scored:
            return None
        return max(scored, key=lambda t: t[1])[0]
    return min(candidates, key=lambda b: _area(b.bbox))


# ─── Pipeline orchestration ───────────────────────────────────────────────


@dataclass
class _Group:
    bbox:       tuple[int, int, int, int]
    text:       str
    source:     str   # "anchored" | "stray"
    region_cls: str   # "text_bubble" | "text_free" | "stray"
    coverage:   float  # initial Lens coverage of the region
    recovered:  bool   # did gap recovery fire?


async def _build_groups(
    image: np.ndarray,
    api,
    lens_blocks: list,
    comic_dets: list[_ComicDet],
    lang: str = "zh-Hans",
) -> tuple[list[_Group], list]:
    bubbles      = [d for d in comic_dets if d.cls == "bubble"]
    text_regions = [d for d in comic_dets if d.cls in ("text_bubble", "text_free")]

    assigned_lens_ids: set[int] = set()
    out_groups: list[_Group] = []

    for region in text_regions:
        rb = region.bbox
        inside = [
            (i, b) for i, b in enumerate(lens_blocks)
            if _io_smaller(b.bbox, rb) >= IOU_INSIDE
        ]
        for i, _ in inside:
            assigned_lens_ids.add(i)

        coverage = _coverage_in_region([b.bbox for _, b in inside], rb)

        # Gap recovery
        recovered = False
        if coverage < COVERAGE_THRESHOLD:
            crop = image[rb[1]:rb[3], rb[0]:rb[2]]
            crop = _whiten_dots(crop)
            crop = _upscale(crop)
            extra = await _lens_ocr(api, crop, lang)
            for p in extra:
                t = (p.get("text") or "").replace("\n", " ").strip()
                if not t:
                    continue
                # Avoid duplicating text we already have
                if any(t in (b.text or "") or (b.text or "") in t
                       for _, b in inside):
                    continue
                # Synthesize a pseudo Lens block (just text + bbox)
                inside.append((None, type("Pseudo", (), {
                    "text": t,
                    "bbox": rb,  # rough — we don't know inside-region geometry
                })()))
                recovered = True

        # Bubble lookup
        parent_bubble = _find_parent_bubble(rb, bubbles)
        group_bbox = parent_bubble.bbox if parent_bubble else rb
        # Reading-order: top-down for horizontal, right-to-left for vertical.
        # POC simplification: sort by y first then x descending (works for both).
        ordered = sorted(
            inside,
            key=lambda t: (t[1].bbox[1], -t[1].bbox[0]),
        )
        text = "\n".join((b.text or "").strip() for _, b in ordered if (b.text or "").strip())

        out_groups.append(_Group(
            bbox=group_bbox,
            text=text,
            source="anchored",
            region_cls=region.cls,
            coverage=coverage,
            recovered=recovered,
        ))

    # Phase 2: stray Lens blocks not matched to any region
    for i, b in enumerate(lens_blocks):
        if i in assigned_lens_ids:
            continue
        out_groups.append(_Group(
            bbox=b.bbox,
            text=(b.text or "").strip(),
            source="stray",
            region_cls="stray",
            coverage=0.0,
            recovered=False,
        ))

    bubble_mask_bboxes = [b.bbox for b in bubbles]
    return out_groups, bubble_mask_bboxes


# ─── Visualisation ────────────────────────────────────────────────────────


def _draw_overlay(
    img: np.ndarray,
    groups: list[_Group],
    bubble_bboxes: list[tuple[int, int, int, int]],
) -> np.ndarray:
    out = img.copy()
    # Bubble outlines (dashed yellow)
    for b in bubble_bboxes:
        cv2.rectangle(out, (b[0], b[1]), (b[2], b[3]), (255, 200, 0), 2)
    # Groups: anchored=green, stray=magenta, recovered=cyan
    for g in groups:
        if g.recovered:
            color = (0, 220, 220)
        elif g.source == "stray":
            color = (255, 0, 200)
        else:
            color = (0, 230, 0)
        cv2.rectangle(out, (g.bbox[0], g.bbox[1]), (g.bbox[2], g.bbox[3]), color, 3)
        label = f"{g.region_cls[:4]} cov={g.coverage:.2f}"
        if g.recovered:
            label += " RECOV"
        cv2.putText(
            out, label, (g.bbox[0], max(15, g.bbox[1] - 5)),
            cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2, cv2.LINE_AA,
        )
    return out


# ─── Driver ───────────────────────────────────────────────────────────────


async def _run_fixture(name: str, lens_det: LensBlocksDetector, comic_sess) -> None:
    src = ROOT / "debug-runs" / name / "source.png"
    if not src.exists():
        print(f"  skip: {src}")
        return
    img = np.asarray(Image.open(src).convert("RGB"))
    H, W = img.shape[:2]
    print(f"\n=== {name} ({W}x{H}) ===")

    # Parallel: Lens detect + comic_detr
    t0 = time.perf_counter()
    lens_task = asyncio.create_task(lens_det.detect(img, lang="zh-Hans"))
    comic_task = asyncio.create_task(asyncio.to_thread(_comic_detect, comic_sess, img))
    lens_res, comic_dets = await asyncio.gather(lens_task, comic_task)
    parallel_ms = (time.perf_counter() - t0) * 1000
    print(
        f"  parallel detect: {parallel_ms:.0f}ms  "
        f"lens={len(lens_res.blocks)}  "
        f"comic={len(comic_dets)} ("
        f"{sum(1 for d in comic_dets if d.cls == 'bubble')} bubble, "
        f"{sum(1 for d in comic_dets if d.cls == 'text_bubble')} text_bubble, "
        f"{sum(1 for d in comic_dets if d.cls == 'text_free')} text_free)"
    )

    # Spatial join + gap recovery
    api = await lens_det._get_api()  # noqa: SLF001
    t1 = time.perf_counter()
    groups, bubble_bboxes = await _build_groups(
        img, api, list(lens_res.blocks), comic_dets, lang="zh-Hans",
    )
    join_ms = (time.perf_counter() - t1) * 1000

    anchored  = sum(1 for g in groups if g.source == "anchored")
    stray     = sum(1 for g in groups if g.source == "stray")
    recovered = sum(1 for g in groups if g.recovered)
    print(
        f"  spatial join: {join_ms:.0f}ms  "
        f"groups={len(groups)} ({anchored} anchored + {stray} stray)  "
        f"recovered={recovered}"
    )

    # Print groups
    print("  groups:")
    for g in groups:
        flag = " RECOV" if g.recovered else ""
        print(
            f"    [{g.source[:3]} {g.region_cls[:4]:4s}] "
            f"bbox={g.bbox} cov={g.coverage:.2f}{flag}  "
            f"text={g.text[:50]!r}"
        )

    # Visual
    overlay = _draw_overlay(img, groups, bubble_bboxes)
    out_path = src.parent / "poc_spatial_join.png"
    Image.fromarray(overlay).save(out_path)
    print(f"  overlay → {out_path}")


async def main() -> None:
    lens_det = LensBlocksDetector()
    comic_sess = _load_comic_detr()
    for name in ["lens_bubble_probe", "lens_bubble_probe2", "lens_bubble_probe3"]:
        await _run_fixture(name, lens_det, comic_sess)


if __name__ == "__main__":
    asyncio.run(main())
