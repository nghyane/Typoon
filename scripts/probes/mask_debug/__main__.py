"""Mask debug probe — 3-layer visual audit.

Layer 1 — Lens masks (raw from _erase_masks_from_words):
  - Cyan rectangles / polygons: each TextMask drawn on page
  - Numbers correlate with BubbleGroup index

Layer 2 — CTD augment (masks after CTDMaskAugmenter.augment):
  - Green = CTD pixel mask (replaced)
  - Cyan = Lens mask kept (no CTD match)
  - Orange border = CTD region bbox (IoU match that triggered replacement)

Layer 3 — Eraser output:
  - Side-by-side: original page vs erased page
  - Erase mask composited in red before erasing so you see exactly what
    was erased where

Output layout:
  debug-runs/mask_debug/<stem>/
    layer1_lens_masks.png
    layer2_ctd_augment.png
    layer3_erase_before.png   ← page + red erase overlay
    layer3_erase_after.png    ← page after AOT/median eraser
    overview.png              ← 2×2 grid of layers 1-3 + diff
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


# ─── helpers ──────────────────────────────────────────────────────────────


def _load_rgb(path: Path) -> np.ndarray:
    img = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if img is None:
        raise FileNotFoundError(path)
    if img.ndim == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    if img.shape[2] == 4:
        img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def _save(path: Path, img: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    out = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    cv2.imwrite(str(path), out)
    print(f"  saved {path.relative_to(ROOT)}")


def _stamp_mask(target: np.ndarray, mask, value: int = 255) -> None:
    """Blit a TextMask onto a HxW uint8 accumulator."""
    H, W = target.shape[:2]
    mx, my = mask.x, mask.y
    mh, mw = mask.image.shape[:2]
    x1, y1 = max(0, mx), max(0, my)
    x2, y2 = min(W, mx + mw), min(H, my + mh)
    if x2 <= x1 or y2 <= y1:
        return
    sx1, sy1 = x1 - mx, y1 - my
    sx2, sy2 = sx1 + (x2 - x1), sy1 + (y2 - y1)
    sub = mask.image[sy1:sy2, sx1:sx2]
    target[y1:y2, x1:x2] = np.maximum(target[y1:y2, x1:x2], sub)


def _overlay(base: np.ndarray, mask_acc: np.ndarray, color: tuple, alpha: float = 0.55) -> np.ndarray:
    flat = np.zeros_like(base)
    flat[mask_acc > 0] = color
    out = cv2.addWeighted(base, 1 - alpha, flat, alpha, 0)
    return out.astype(np.uint8)


def _label(img: np.ndarray, text: str) -> np.ndarray:
    bar = max(22, img.shape[0] // 40)
    cv2.rectangle(img, (0, 0), (img.shape[1], bar), (32, 32, 32), -1)
    cv2.putText(img, text, (8, int(bar * 0.72)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)
    return img


def _grid_2x2(tl, tr, bl, br) -> np.ndarray:
    top = np.concatenate([tl, tr], axis=1)
    bot = np.concatenate([bl, br], axis=1)
    return np.concatenate([top, bot], axis=0)


def _resize_to(img: np.ndarray, h: int, w: int) -> np.ndarray:
    return cv2.resize(img, (w, h), interpolation=cv2.INTER_AREA)


# ─── Layer 1: raw groups + filter result ─────────────────────────────────


def render_layer1(image: np.ndarray, groups_raw, filter_result) -> np.ndarray:
    """All raw groups — kept (cyan) and rejected (red cross)."""
    H, W = image.shape[:2]
    out = image.copy()
    rejected_bboxes = {id(g) for g, _ in filter_result.rejected}

    for gi, g in enumerate(groups_raw):
        x1, y1, x2, y2 = g.bbox
        if id(g) in rejected_bboxes:
            col = (255, 60, 60)
            cv2.rectangle(out, (x1, y1), (x2, y2), col, 2)
            cv2.line(out, (x1, y1), (x2, y2), col, 1)
            cv2.line(out, (x2, y1), (x1, y2), col, 1)
        else:
            col = (0, 200, 255)
            cv2.rectangle(out, (x1, y1), (x2, y2), col, 1)
        cv2.putText(out, f"#{gi}", (x1+2, max(14, y1+12)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.35, col, 1, cv2.LINE_AA)

    _label(out, f"Layer 1: raw groups={len(groups_raw)}  kept={filter_result.n_kept}  rejected={filter_result.n_rejected}")
    return out


# ─── Layer 2: materialized masks ──────────────────────────────────────────


def render_layer2(image: np.ndarray, groups_with_masks) -> np.ndarray:
    """Materialized erase masks per kept group (magenta blob)."""
    H, W = image.shape[:2]
    acc = np.zeros((H, W), dtype=np.uint8)
    out = image.copy()

    colors = [(255,50,200),(0,200,255),(255,200,0),(0,255,128),(200,0,255),(255,128,0)]
    for gi, (g, masks, recipe) in enumerate(groups_with_masks):
        col = colors[gi % len(colors)]
        for m in masks:
            _stamp_mask(acc, m)
        x1, y1 = g.bbox[0], g.bbox[1]
        cv2.rectangle(out, (g.bbox[0], g.bbox[1]), (g.bbox[2], g.bbox[3]), col, 1)
        cv2.putText(out, f"#{gi} {recipe.strategy.value[:4]}", (x1+2, max(14, y1+12)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.32, col, 1, cv2.LINE_AA)

    fill = np.zeros_like(out)
    fill[acc > 0] = (255, 50, 200)
    out = cv2.addWeighted(out, 0.55, fill, 0.45, 0).astype(np.uint8)
    total = sum(len(m) for _, m, _ in groups_with_masks)
    _label(out, f"Layer 2: materialized masks  groups={len(groups_with_masks)}  masks={total}")
    return out


# ─── Layer 3: Eraser output ───────────────────────────────────────────────


def render_layer3_before(image: np.ndarray, groups_with_masks) -> np.ndarray:
    H, W = image.shape[:2]
    acc = np.zeros((H, W), dtype=np.uint8)
    for _, masks, _ in groups_with_masks:
        for em in masks:
            _stamp_mask(acc, em)
    out = image.copy()
    fill = np.zeros_like(out)
    fill[acc > 0] = (255, 50, 50)
    out = cv2.addWeighted(out, 0.55, fill, 0.45, 0).astype(np.uint8)
    cnts, _ = cv2.findContours(acc, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(out, cnts, -1, (200, 0, 0), 1)
    _label(out, "Layer 3a: Erase mask overlay (red = will be inpainted)")
    return out


async def render_layer3_after(image: np.ndarray, groups_with_masks, models_dir: Path) -> np.ndarray:
    from typoon.vision.contracts import TextMask
    from typoon.vision.erasers import AOTGANEraser, MedianEraser
    canvas = image.copy().astype(np.uint8)
    if canvas.shape[2] == 3:
        canvas = cv2.cvtColor(canvas, cv2.COLOR_RGB2RGBA)
    erase_masks = tuple(m for _, masks, _ in groups_with_masks for m in masks)
    aot_path = models_dir / "aot-inpaint.onnx"
    if aot_path.exists():
        eraser = AOTGANEraser(models_dir=models_dir)
    else:
        eraser = MedianEraser()
        print("  [warn] AOT model not found — using MedianEraser fallback")
    result = await eraser.erase(canvas, erase_masks)
    if result.shape[2] == 4:
        result = cv2.cvtColor(result, cv2.COLOR_RGBA2RGB)
    out = result.copy()
    _label(out, f"Layer 3b: Eraser output  masks={len(erase_masks)}  backend={eraser.name}")
    return out


# ─── Diff ────────────────────────────────────────────────────────────────


def render_diff(before: np.ndarray, after: np.ndarray) -> np.ndarray:
    """Absolute pixel diff amplified 4× — highlights residue or over-erase."""
    b = before.astype(np.int16)
    a = after.astype(np.int16)
    diff = np.abs(b - a).clip(0, 255).astype(np.uint8)
    # Amplify 4×
    diff_amp = (diff.astype(np.uint16) * 4).clip(0, 255).astype(np.uint8)
    # Colorize: R channel = original brighter, G channel = erased brighter
    vis = np.zeros_like(before)
    vis[..., 0] = np.where(b[..., 0] > a[..., 0], diff_amp[..., 0], 0)  # R = original bright
    vis[..., 1] = np.where(a[..., 0] > b[..., 0], diff_amp[..., 0], 0)  # G = erased bright
    vis[..., 2] = diff_amp[..., 2]
    _label(vis, "Diff (R=residue G=over-erase, 4× amplified)")
    return vis


# ─── Main ─────────────────────────────────────────────────────────────────


async def _run(image_path: Path, models_dir: Path, out_dir: Path, lang: str | None) -> None:
    from typoon.models import ModelHub
    from typoon.vision._backends.comic_detr import load_session
    from typoon.vision.detectors.lens.detector import LensBlocksDetector
    from typoon.vision.groupers.lens_native import LensNativeGrouper
    from typoon.vision.filters import default_filter
    from typoon.vision.masks import select_strategy, materialize
    from typoon.vision.masks.ctd_seg_runner import CtdSegRunner
    from typoon.domain.filter import ScoringContext

    print(f"loading {image_path.name}...")
    image = _load_rgb(image_path)
    H, W = image.shape[:2]
    print(f"  {W}\u00d7{H}")

    # 1. GROUP — geometry only, no masks
    hub = ModelHub(models_dir)
    detector = LensBlocksDetector(
        comic_detr=load_session(hub.resolve_comic_detr())
    )
    detection = await detector.detect(image, lang=lang)
    groups_raw = await LensNativeGrouper().group(image, detection, lang)
    print(f"  lens: {len(detection.blocks)} blocks → {len(groups_raw)} groups")

    # CTD seg-only for Japanese
    bubble_mask = None
    if lang and lang.startswith("ja"):
        ctd_onnx = models_dir / "ctd.onnx"
        if ctd_onnx.exists():
            bubble_mask = await asyncio.to_thread(CtdSegRunner(ctd_onnx).run, image)
            print(f"  ctd_seg: bubble_mask active={int((bubble_mask>0).sum())}px")

    # 2. FILTER
    ctx = ScoringContext(page_size=(W, H), page_groups=groups_raw, image=image, blocks=tuple(detection.blocks))
    filter_result = default_filter().evaluate(groups_raw, ctx)
    print(f"  filter: kept={filter_result.n_kept} rejected={filter_result.n_rejected}")
    for g, v in filter_result.rejected:
        print(f"    REJECTED [{v.signals[0].scorer}] {repr(g.text[:40])}")

    # 3. MASK — materialize per kept group
    blocks = list(detection.blocks)
    groups_with_masks = []
    for g in filter_result.kept:
        members = tuple(
            b for b in blocks
            if g.bbox[0] <= (b.bbox[0]+b.bbox[2])/2 <= g.bbox[2]
            and g.bbox[1] <= (b.bbox[1]+b.bbox[3])/2 <= g.bbox[3]
        )
        recipe = select_strategy(g, image, bubble_mask=bubble_mask)
        masks  = materialize(recipe, g, members, image, bubble_mask=bubble_mask)
        groups_with_masks.append((g, masks, recipe))

    print("rendering layers...")
    l1 = render_layer1(image, groups_raw, filter_result)
    l2 = render_layer2(image, groups_with_masks)
    l3_before = render_layer3_before(image, groups_with_masks)
    l3_after  = await render_layer3_after(image, groups_with_masks, models_dir)

    _save(out_dir / "layer1_raw_groups.png",     l1)
    _save(out_dir / "layer2_masks.png",          l2)
    _save(out_dir / "layer3a_erase_overlay.png", l3_before)
    _save(out_dir / "layer3b_eraser_output.png", l3_after)

    diff = render_diff(image, l3_after)
    _save(out_dir / "layer3c_diff.png", diff)

    target_h = min(H, 1200)
    target_w = int(W * target_h / H)
    def rs(img): return _resize_to(img, target_h, target_w)
    overview = _grid_2x2(rs(l1), rs(l2), rs(l3_before), rs(l3_after))
    _save(out_dir / "overview.png", overview)

    _save_group_details(out_dir / "groups", image, groups_with_masks)
    print(f"\nall artifacts → {out_dir.relative_to(ROOT)}")

def _save_group_details(
    out_dir: Path,
    image: np.ndarray,
    groups_with_masks: list,
) -> None:
    """Crop each kept group and show mask overlay."""
    H, W = image.shape[:2]
    out_dir.mkdir(parents=True, exist_ok=True)
    PAD = 20

    for gi, (g, masks, recipe) in enumerate(groups_with_masks):
        x1, y1, x2, y2 = g.bbox
        rx1 = max(0, x1 - PAD); ry1 = max(0, y1 - PAD)
        rx2 = min(W, x2 + PAD); ry2 = min(H, y2 + PAD)
        crop = image[ry1:ry2, rx1:rx2].copy()
        cH, cW = crop.shape[:2]
        if cH < 4 or cW < 4:
            continue

        def _stamp_to_crop(ms, color):
            acc = np.zeros((cH, cW), dtype=np.uint8)
            for m in ms:
                from typoon.vision.contracts import TextMask
                shifted = TextMask(x=m.x - rx1, y=m.y - ry1, image=m.image)
                _stamp_mask(acc, shifted)
            fill = np.zeros_like(crop)
            fill[acc > 0] = color
            return cv2.addWeighted(crop, 0.55, fill, 0.45, 0).astype(np.uint8)

        mask_view = _stamp_to_crop(masks, (255, 50, 200))
        _label(mask_view, f"#{gi} {recipe.strategy.value} ({len(masks)}m)")
        _save(out_dir / f"group_{gi:03d}.png", mask_view)

def main() -> None:
    ap = argparse.ArgumentParser(prog="mask_debug_probe")
    ap.add_argument("image", type=Path)
    ap.add_argument("--out", type=Path,
                    default=ROOT / "debug-runs" / "mask_debug")
    ap.add_argument("--models", type=Path, default=ROOT / "models")
    ap.add_argument("--lang",   type=str, default=None)
    args = ap.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(levelname).1s %(name)s: %(message)s",
    )

    stem = args.image.stem
    out_dir = args.out / stem
    out_dir.mkdir(parents=True, exist_ok=True)

    asyncio.run(_run(args.image, args.models, out_dir, args.lang))


if __name__ == "__main__":
    main()
