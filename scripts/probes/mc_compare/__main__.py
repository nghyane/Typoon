"""mc_compare — probe manga-cleaner inpaint tricks vs current AOT pipeline.

Variants:
  v0 baseline       AOT page-level (current default, downscaled to <=384)
  v1 mc_dilate      AOT page-level + mc-style mask (4%xW ellipse, 2 iter)
  v2 per_blob       AOT per-blob 384px tile (native res, no downscale)
  v3 per_blob_mc    v2 + mc-style dilation
  v4 telea          OpenCV TeLeA radius 8 (non-ML reference)
  v5 lama_page      LaMa page-level (resize page->512 fixed, base mask)
  v6 lama_per_blob  LaMa per-blob 512 tile (native res, base mask)
  v7 lama_per_blob_mc  v6 + mc-style dilation (manga-cleaner clone)
  overview          rebuild 3x3 grid from saved PNGs

Run ONE variant per process to avoid coremltools/CoreML leak observed on
Apple Silicon when calling predict() repeatedly across many tiles.

  python -m scripts.probes.mc_compare --variant v0
  python -m scripts.probes.mc_compare --variant v1
  ...
  python -m scripts.probes.mc_compare --variant overview

--backend onnx (default) forces ONNX Runtime even on macOS.
--backend coreml is opt-in (known to leak on long runs).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[3]
for pkg in ("typoon-core", "typoon-vision"):
    p = ROOT / "packages" / pkg
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))


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
    if img.ndim == 2:
        cv2.imwrite(str(path), img)
    else:
        cv2.imwrite(str(path), cv2.cvtColor(img, cv2.COLOR_RGB2BGR))


def _label(img: np.ndarray, text: str) -> np.ndarray:
    out = img.copy()
    bar = max(22, out.shape[0] // 40)
    cv2.rectangle(out, (0, 0), (out.shape[1], bar), (32, 32, 32), -1)
    cv2.putText(out, text, (8, int(bar * 0.72)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)
    return out


def _diff(before: np.ndarray, after: np.ndarray, amp: int = 4) -> np.ndarray:
    b = before.astype(np.int16)
    a = after.astype(np.int16)
    d = np.abs(b - a).clip(0, 255).astype(np.uint8)
    return (d.astype(np.uint16) * amp).clip(0, 255).astype(np.uint8)


def _resize(img: np.ndarray, h: int, w: int) -> np.ndarray:
    return cv2.resize(img, (w, h), interpolation=cv2.INTER_AREA)


def mc_dilate(mask: np.ndarray, page_w: int) -> np.ndarray:
    """4%xW ellipse, 2 iterations — manga-cleaner OCR mask expansion."""
    k = max(11, int(page_w * 0.04))
    if k % 2 == 0:
        k += 1
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
    return cv2.dilate(mask, kernel, iterations=2)


def v_telea(image: np.ndarray, mask: np.ndarray, radius: int = 8) -> np.ndarray:
    bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    out = cv2.inpaint(bgr, mask, radius, cv2.INPAINT_TELEA)
    return cv2.cvtColor(out, cv2.COLOR_BGR2RGB)


def v_aot_page(inp, image: np.ndarray, mask: np.ndarray) -> np.ndarray:
    return inp.inpaint(image, mask)


# ─── LaMa backend (Carve/LaMa-ONNX fp32, fixed 512x512) ───────────────


class LamaInpainter:
    """LaMa fp32 ONNX wrapper. Fixed 512x512 input. RGB uint8 in/out."""

    SIZE = 512

    def __init__(self, onnx_path: Path) -> None:
        import onnxruntime as ort
        so = ort.SessionOptions()
        so.log_severity_level = 3
        self._sess = ort.InferenceSession(
            str(onnx_path), sess_options=so, providers=["CPUExecutionProvider"]
        )

    def _forward_512(self, image_rgb: np.ndarray, mask: np.ndarray) -> np.ndarray:
        # Carve/LaMa-ONNX convention: image [0,1] in, output already [0,255].
        img = image_rgb.astype(np.float32).transpose(2, 0, 1)[None] / 255.0
        msk = (mask >= 127).astype(np.float32)[None, None]
        out = self._sess.run(None, {"image": img, "mask": msk})[0]
        return out[0].transpose(1, 2, 0).clip(0, 255).astype(np.uint8)

    def inpaint(self, image_rgb: np.ndarray, mask: np.ndarray) -> np.ndarray:
        h, w = image_rgb.shape[:2]
        if (h, w) == (self.SIZE, self.SIZE):
            res = self._forward_512(image_rgb, mask)
        elif max(h, w) <= self.SIZE:
            # pad to 512x512 with reflect
            pad_h, pad_w = self.SIZE - h, self.SIZE - w
            img_p  = cv2.copyMakeBorder(image_rgb, 0, pad_h, 0, pad_w, cv2.BORDER_REFLECT_101)
            mask_p = cv2.copyMakeBorder(mask,      0, pad_h, 0, pad_w, cv2.BORDER_REFLECT_101)
            res = self._forward_512(img_p, mask_p)[:h, :w]
        else:
            # resize whole image -> 512x512 -> back. used for page-level only.
            img_r  = cv2.resize(image_rgb, (self.SIZE, self.SIZE), interpolation=cv2.INTER_AREA)
            mask_r = cv2.resize(mask,      (self.SIZE, self.SIZE), interpolation=cv2.INTER_NEAREST)
            out_r  = self._forward_512(img_r, mask_r)
            res = cv2.resize(out_r, (w, h), interpolation=cv2.INTER_LINEAR)
        # paste only masked pixels back
        m3 = (mask >= 127)[:, :, None]
        return np.where(m3, res, image_rgb)


def v_lama_page(lama, image: np.ndarray, mask: np.ndarray) -> np.ndarray:
    return lama.inpaint(image, mask)


def v_lama_per_blob(
    lama: "LamaInpainter",
    image: np.ndarray,
    mask: np.ndarray,
    tile: int = 512,
    max_tiles: int = 40,
) -> np.ndarray:
    """Per-blob 512 tile with LaMa, manga-cleaner style.

    tile=512 matches LaMa fixed input -> no resize ever happens.
    """
    H, W = image.shape[:2]
    out = image.copy()
    n, _labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    blobs = [stats[i] for i in range(1, n) if stats[i, 4] > 5]
    blobs.sort(key=lambda s: s[4], reverse=True)

    covered = np.zeros_like(mask)
    n_tiles = 0
    for bx, by, bw, bh, _area in blobs:
        if n_tiles >= max_tiles:
            print(f"  lama_per_blob: hit max_tiles={max_tiles}, stopping")
            break
        if np.all(covered[by:by + bh, bx:bx + bw] == 255):
            continue
        cx, cy = bx + bw // 2, by + bh // 2
        x1 = max(0, cx - tile // 2)
        y1 = max(0, cy - tile // 2)
        x2 = min(W, x1 + tile)
        y2 = min(H, y1 + tile)
        x1 = max(0, x2 - tile)
        y1 = max(0, y2 - tile)
        tile_img  = out[y1:y2, x1:x2]
        tile_mask = mask[y1:y2, x1:x2]
        if not (tile_mask > 0).any():
            continue
        res = lama.inpaint(tile_img, tile_mask)
        m3 = (tile_mask >= 127)[:, :, None]
        out[y1:y2, x1:x2] = np.where(m3, res, tile_img)
        covered[y1:y2, x1:x2] = 255
        n_tiles += 1
    print(f"  lama_per_blob: {n_tiles} tiles ({tile}x{tile})")
    return out


def v_aot_per_blob(
    inp,
    image: np.ndarray,
    mask: np.ndarray,
    tile: int = 384,
    max_tiles: int = 40,
) -> np.ndarray:
    """Per-connected-component tile, manga-cleaner style."""
    H, W = image.shape[:2]
    out = image.copy()
    n, _labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    blobs = [stats[i] for i in range(1, n) if stats[i, 4] > 5]
    blobs.sort(key=lambda s: s[4], reverse=True)

    covered = np.zeros_like(mask)
    n_tiles = 0
    for bx, by, bw, bh, _area in blobs:
        if n_tiles >= max_tiles:
            print(f"  per_blob: hit max_tiles={max_tiles}, stopping early")
            break
        if np.all(covered[by:by + bh, bx:bx + bw] == 255):
            continue
        cx, cy = bx + bw // 2, by + bh // 2
        x1 = max(0, cx - tile // 2)
        y1 = max(0, cy - tile // 2)
        x2 = min(W, x1 + tile)
        y2 = min(H, y1 + tile)
        x1 = max(0, x2 - tile)
        y1 = max(0, y2 - tile)

        tile_img = out[y1:y2, x1:x2]
        tile_mask = mask[y1:y2, x1:x2]
        if not (tile_mask > 0).any():
            continue
        res = inp.inpaint(tile_img, tile_mask)
        m3 = (tile_mask >= 127)[:, :, None]
        out[y1:y2, x1:x2] = np.where(m3, res, tile_img)
        covered[y1:y2, x1:x2] = 255
        n_tiles += 1
    print(f"  per_blob: {n_tiles} tiles ({tile}x{tile})")
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--image", default="debug-runs/mask_debug/test_7/worker_results/original.png")
    ap.add_argument("--mask",  default="debug-runs/mask_debug/test_7/worker_results/mask.png")
    ap.add_argument("--out",   default=None)
    ap.add_argument("--models", default="models")
    ap.add_argument("--tile", type=int, default=384)
    ap.add_argument("--variant", choices=["v0","v1","v2","v3","v4","v5","v6","v7","overview"],
                    required=True, help="run one variant per process (memory safety)")
    ap.add_argument("--backend", choices=["onnx", "coreml"], default="onnx",
                    help="force AOT backend; coreml is known to leak on long runs")
    ap.add_argument("--lama", default="models/lama_fp32.onnx")
    ap.add_argument("--lama-tile", type=int, default=512)
    ap.add_argument("--max-tiles", type=int, default=40)
    args = ap.parse_args()

    image_path = (ROOT / args.image).resolve()
    mask_path  = (ROOT / args.mask).resolve()
    models_dir = (ROOT / args.models).resolve()
    stem = image_path.parent.parent.name
    out_dir = Path(args.out) if args.out else ROOT / "debug-runs" / "mc_compare" / stem

    print(f"image  : {image_path.relative_to(ROOT)}")
    print(f"mask   : {mask_path.relative_to(ROOT)}")
    print(f"out    : {out_dir.relative_to(ROOT)}")
    print(f"variant: {args.variant}  backend={args.backend}")

    image = _load_rgb(image_path)
    mask  = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
    if mask is None:
        raise FileNotFoundError(mask_path)
    H, W = image.shape[:2]
    print(f"page   : {W}x{H}  mask_pct={(mask>0).mean()*100:.2f}%")

    mask_mc = mc_dilate(mask, W)
    k = max(11, int(W * 0.04)) | 1
    print(f"mc_dilate: k={k}px  pct {(mask>0).mean()*100:.2f}% -> {(mask_mc>0).mean()*100:.2f}%")
    _save(out_dir / "mask_base.png", mask)
    _save(out_dir / "mask_mc.png",   mask_mc)

    def _build_aot():
        from typoon.vision._backends.aot import AOTInpainter
        inp = AOTInpainter(models_dir)
        if args.backend == "onnx":
            inp._use_coreml = False
        return inp

    if args.variant == "v0":
        print("  V0 baseline (AOT page)")
        out = v_aot_page(_build_aot(), image, mask)
        _save(out_dir / "v0_baseline.png", out)
        _save(out_dir / "diff_v0.png", _diff(image, out))

    elif args.variant == "v1":
        print("  V1 mc_dilate (AOT page + mc mask)")
        out = v_aot_page(_build_aot(), image, mask_mc)
        _save(out_dir / "v1_mc_dilate.png", out)
        _save(out_dir / "diff_v1.png", _diff(image, out))

    elif args.variant == "v2":
        print(f"  V2 per_blob (AOT {args.tile} tile, base mask)")
        out = v_aot_per_blob(_build_aot(), image, mask, tile=args.tile, max_tiles=args.max_tiles)
        _save(out_dir / "v2_per_blob.png", out)
        _save(out_dir / "diff_v2.png", _diff(image, out))

    elif args.variant == "v3":
        print(f"  V3 per_blob + mc_dilate ({args.tile} tile)")
        out = v_aot_per_blob(_build_aot(), image, mask_mc, tile=args.tile, max_tiles=args.max_tiles)
        _save(out_dir / "v3_per_blob_mc.png", out)
        _save(out_dir / "diff_v3.png", _diff(image, out))

    elif args.variant == "v4":
        print("  V4 telea (no model)")
        out = v_telea(image, mask, radius=8)
        _save(out_dir / "v4_telea.png", out)
        _save(out_dir / "diff_v4.png", _diff(image, out))

    elif args.variant == "v5":
        print(f"  V5 LaMa page-level ({args.lama})")
        lama = LamaInpainter(ROOT / args.lama)
        out = v_lama_page(lama, image, mask)
        _save(out_dir / "v5_lama_page.png", out)
        _save(out_dir / "diff_v5.png", _diff(image, out))

    elif args.variant == "v6":
        print(f"  V6 LaMa per-blob {args.lama_tile} (base mask)")
        lama = LamaInpainter(ROOT / args.lama)
        out = v_lama_per_blob(lama, image, mask, tile=args.lama_tile, max_tiles=args.max_tiles)
        _save(out_dir / "v6_lama_per_blob.png", out)
        _save(out_dir / "diff_v6.png", _diff(image, out))

    elif args.variant == "v7":
        print(f"  V7 LaMa per-blob {args.lama_tile} + mc-dilate (manga-cleaner clone)")
        lama = LamaInpainter(ROOT / args.lama)
        out = v_lama_per_blob(lama, image, mask_mc, tile=args.lama_tile, max_tiles=args.max_tiles)
        _save(out_dir / "v7_lama_per_blob_mc.png", out)
        _save(out_dir / "diff_v7.png", _diff(image, out))

    elif args.variant == "overview":
        def _load_or_blank(name: str) -> np.ndarray:
            p = out_dir / name
            if p.exists():
                return _load_rgb(p)
            blank = np.full_like(image, 64)
            return _label(blank, f"missing {name}")
        v0 = _load_or_blank("v0_baseline.png")
        v1 = _load_or_blank("v1_mc_dilate.png")
        v2 = _load_or_blank("v2_per_blob.png")
        v3 = _load_or_blank("v3_per_blob_mc.png")
        v4 = _load_or_blank("v4_telea.png")
        v5 = _load_or_blank("v5_lama_page.png")
        v6 = _load_or_blank("v6_lama_per_blob.png")
        v7 = _load_or_blank("v7_lama_per_blob_mc.png")
        target_h = min(H, 720)
        target_w = int(W * target_h / H)
        def rs(im, t):
            return _label(_resize(im, target_h, target_w), t)
        row1 = np.concatenate([
            rs(image, "input"),
            rs(v0,    "V0 AOT page (downscale)"),
            rs(v1,    "V1 AOT page + mc-dilate"),
        ], axis=1)
        row2 = np.concatenate([
            rs(v2,    f"V2 AOT per-blob {args.tile}"),
            rs(v3,    "V3 V2 + mc-dilate"),
            rs(v4,    "V4 TeLeA r=8"),
        ], axis=1)
        row3 = np.concatenate([
            rs(v5,    "V5 LaMa page (resize 512)"),
            rs(v6,    f"V6 LaMa per-blob {args.lama_tile}"),
            rs(v7,    "V7 V6 + mc-dilate (mc clone)"),
        ], axis=1)
        _save(out_dir / "overview.png", np.concatenate([row1, row2, row3], axis=0))

    print(f"\nartifacts -> {out_dir.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
