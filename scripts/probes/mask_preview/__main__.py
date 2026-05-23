"""Visual mask preview — full page overlay, no crops.

Vẽ trực tiếp lên full page:
  - Đỏ trong = vùng mask (sẽ bị inpaint xóa)
  - Viền xanh = bbox của từng group
  - Label: idx, origin, class, fill%

Usage:
    python -m scripts.probes.mask_preview <image> [image2 ...]
        [--out debug-runs/mask-preview/]
        [--lang ja]
        [--scale 1.0]      # resize output (0.5 = half)
"""
from __future__ import annotations

import argparse
import asyncio
import os
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))


def draw_mask_overlay(
    img:      np.ndarray,   # H×W×3 RGB
    mask:     np.ndarray,   # H×W uint8 (0 or 255)
    plan:     dict,
    scale:    float = 1.0,
) -> Image.Image:
    H, W = img.shape[:2]

    # --- base: semitransparent red on masked pixels ---
    base   = Image.fromarray(img)
    red    = Image.new("RGB", (W, H), (220, 40, 40))
    mask_img = Image.fromarray(mask)             # L mode, 0 or 255
    # alpha: 0 outside mask, 150 inside mask
    alpha  = mask_img.point(lambda p: 150 if p > 0 else 0)
    base.paste(red, mask=alpha)

    draw = ImageDraw.Draw(base, "RGBA")

    try:
        font_lbl = ImageFont.load_default(size=13)
        font_sm  = ImageFont.load_default(size=11)
    except Exception:
        font_lbl = ImageFont.load_default()
        font_sm  = font_lbl

    # color per origin
    ORIGIN_COLOR = {
        "ctd_unet":         (0, 200, 80),    # green
        "lens_obb":         (0, 180, 255),   # cyan
        "lens_aabb":        (0, 120, 255),   # blue
        "polygon_fallback": (255, 160, 0),   # orange
    }
    CLASS_COLOR = {
        "sfx":       (255, 220, 0),
        "dialogue":  (200, 200, 255),
        "narration": (255, 180, 200),
    }

    for g in plan["groups"]:
        x1, y1, x2, y2 = g["bbox"]
        origin  = g.get("origin", "?")
        cls     = g.get("class", "?")
        idx     = g["idx"]

        col_box = ORIGIN_COLOR.get(origin, (200, 200, 200))
        col_lbl = CLASS_COLOR.get(cls, (255, 255, 255))

        # Compute mask fill inside this bbox (clamped)
        bx0 = max(0, x1); by0 = max(0, y1)
        bx1 = min(W, x2); by1 = min(H, y2)
        if bx1 > bx0 and by1 > by0:
            roi       = mask[by0:by1, bx0:bx1]
            fill_pct  = int(100 * np.count_nonzero(roi) / roi.size)
        else:
            fill_pct = 0

        # Draw bbox rectangle
        draw.rectangle([x1, y1, x2, y2], outline=col_box, width=2)

        # Label background pill
        label = f"#{idx} {origin[:3].upper()} {cls[:3]} {fill_pct}%"
        tw, th = draw.textlength(label, font=font_lbl), 14
        lx = x1; ly = max(0, y1 - 16)
        draw.rectangle([lx, ly, lx + tw + 4, ly + th], fill=(*col_box, 180))
        draw.text((lx + 2, ly + 1), label, fill=(0, 0, 0), font=font_lbl)

    # Scale down if requested
    if scale != 1.0:
        nw, nh = int(W * scale), int(H * scale)
        base = base.resize((nw, nh), Image.LANCZOS)

    return base


async def process_page(
    img_path: Path,
    out_dir:  Path,
    lang:     str,
    scale:    float,
) -> Path:
    from typoon_inpaint.scan import build_plan_for_image
    from typoon_inpaint import decode_plan, rasterise_plan_mask
    from typoon_inpaint.artifact_sink import NullSink

    img_np = np.array(Image.open(img_path).convert("RGB"))
    H, W   = img_np.shape[:2]

    plan_bytes  = await build_plan_for_image(img_path, lang=lang, sink=NullSink())
    plan        = decode_plan(plan_bytes)
    mask_bytes  = rasterise_plan_mask(plan_bytes, img_path.read_bytes())
    mask_np     = np.frombuffer(bytes(mask_bytes), dtype=np.uint8).reshape(H, W)

    n_groups   = len(plan["groups"])
    mask_fill  = int(100 * np.count_nonzero(mask_np) / mask_np.size)

    print(f"  {img_path.name}: {W}×{H}  groups={n_groups}  "
          f"mask={mask_fill}%  page_kind={plan['page_kind']}")

    overlay  = draw_mask_overlay(img_np, mask_np, plan, scale=scale)
    out_path = out_dir / f"{img_path.stem}_mask.png"
    overlay.save(out_path, optimize=False)
    return out_path


async def main_async(args) -> None:
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    images = []
    for p in args.images:
        p = Path(p)
        if p.is_dir():
            exts = {".jpg", ".jpeg", ".png", ".webp"}
            images += sorted(x for x in p.iterdir() if x.suffix.lower() in exts)
        else:
            images.append(p)

    if not images:
        print("No images found.")
        return

    print(f"Processing {len(images)} page(s) → {out_dir}")
    sem = asyncio.Semaphore(3)

    async def one(p: Path) -> None:
        async with sem:
            try:
                out = await process_page(p, out_dir, args.lang, args.scale)
                print(f"  → {out}")
            except Exception as e:
                print(f"  ERROR {p.name}: {e}")

    await asyncio.gather(*[one(p) for p in images])
    print(f"\nDone. Open {out_dir}")


def main() -> None:
    import logging
    logging.basicConfig(level=logging.WARNING,
                        format="%(levelname).1s %(name)s: %(message)s")

    ap = argparse.ArgumentParser()
    ap.add_argument("images",  nargs="+")
    ap.add_argument("--out",   default="debug-runs/mask-preview")
    ap.add_argument("--lang",  default="ja")
    ap.add_argument("--scale", type=float, default=0.6,
                    help="Output scale factor (default 0.6)")
    args = ap.parse_args()
    asyncio.run(main_async(args))


if __name__ == "__main__":
    main()
