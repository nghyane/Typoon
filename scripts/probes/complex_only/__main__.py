"""complex_only — race AOT vs LaMa per-blob, but only on COMPLEX bubbles.

Uses Typoon's real `classify_masks` (luminance spread < 30 = uniform → TeLeA,
else complex → ML inpaint). Probe applies only the complex backend so we see
where the real difference lies; uniform bubbles are left untouched as input
context.

Variants (all per-blob, native res, paste only where mask==255):
  c0 input             original page (sanity)
  c1 aot_per_blob_384  AOT-GAN per-blob 384 native
  c2 lama_per_blob_512 LaMa  per-blob 512 native

Run ONE variant per process:
  python -m scripts.probes.complex_only --variant c1
  python -m scripts.probes.complex_only --variant c2
  python -m scripts.probes.complex_only --variant overview
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
    im = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if im.ndim == 2: im = cv2.cvtColor(im, cv2.COLOR_GRAY2BGR)
    if im.shape[2] == 4: im = cv2.cvtColor(im, cv2.COLOR_BGRA2BGR)
    return cv2.cvtColor(im, cv2.COLOR_BGR2RGB)


def _save(p: Path, im: np.ndarray) -> None:
    p.parent.mkdir(parents=True, exist_ok=True)
    if im.ndim == 2: cv2.imwrite(str(p), im)
    else: cv2.imwrite(str(p), cv2.cvtColor(im, cv2.COLOR_RGB2BGR))


def _label(im, t):
    out = im.copy()
    bar = max(28, out.shape[0]//30)
    cv2.rectangle(out, (0,0), (out.shape[1], bar), (24,24,24), -1)
    cv2.putText(out, t, (10, int(bar*0.72)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2, cv2.LINE_AA)
    return out


def _rs(im, h):
    w = int(im.shape[1] * h / im.shape[0])
    return cv2.resize(im, (w, h), interpolation=cv2.INTER_AREA)


# ─── split page mask into per-blob TextMask list, then classify ──────────


def split_mask_into_textmasks(page_mask: np.ndarray):
    """One TextMask per connected component, image cropped to bbox."""
    from typoon.vision.contracts import TextMask
    n, labels, stats, _ = cv2.connectedComponentsWithStats(page_mask, 8)
    masks = []
    for i in range(1, n):
        if stats[i, 4] < 50: continue
        x, y, w, h, _ = stats[i]
        crop = (labels[y:y+h, x:x+w] == i).astype(np.uint8) * 255
        masks.append(TextMask(x=int(x), y=int(y), image=crop))
    return masks


def build_page_mask_from_list(masks, page_w, page_h) -> np.ndarray:
    pm = np.zeros((page_h, page_w), dtype=np.uint8)
    for m in masks:
        mx, my = m.x, m.y
        mh, mw = m.image.shape[:2]
        x1, y1 = max(0, mx), max(0, my)
        x2, y2 = min(page_w, mx+mw), min(page_h, my+mh)
        if x2 <= x1 or y2 <= y1: continue
        sx, sy = x1-mx, y1-my
        pm[y1:y2, x1:x2] |= m.image[sy:sy+(y2-y1), sx:sx+(x2-x1)]
    return pm


# ─── per-blob inpaint loop ────────────────────────────────────────────────


def per_blob_inpaint(inpainter, image, page_mask, tile, max_tiles=40, label=""):
    H, W = image.shape[:2]
    out = image.copy()
    n, _l, stats, _ = cv2.connectedComponentsWithStats(page_mask, 8)
    blobs = [stats[i] for i in range(1,n) if stats[i,4] > 5]
    blobs.sort(key=lambda s: s[4], reverse=True)
    covered = np.zeros_like(page_mask)
    nt = 0
    for bx,by,bw,bh,_ in blobs:
        if nt >= max_tiles: break
        if np.all(covered[by:by+bh, bx:bx+bw] == 255): continue
        cx, cy = bx+bw//2, by+bh//2
        x1 = max(0, cx - tile//2); y1 = max(0, cy - tile//2)
        x2 = min(W, x1+tile);      y2 = min(H, y1+tile)
        x1 = max(0, x2-tile);      y1 = max(0, y2-tile)
        ti = out[y1:y2, x1:x2]
        tm = page_mask[y1:y2, x1:x2]
        if not (tm > 0).any(): continue
        res = inpainter.inpaint(ti, tm)
        m3 = (tm >= 127)[:,:,None]
        out[y1:y2, x1:x2] = np.where(m3, res, ti)
        covered[y1:y2, x1:x2] = 255
        nt += 1
    print(f"  {label}: {nt} tiles ({tile}x{tile})")
    return out


# ─── LaMa backend ─────────────────────────────────────────────────────────


class LamaInpainter:
    SIZE = 512
    def __init__(self, onnx_path: Path) -> None:
        import onnxruntime as ort
        so = ort.SessionOptions(); so.log_severity_level = 3
        self._sess = ort.InferenceSession(
            str(onnx_path), sess_options=so, providers=["CPUExecutionProvider"]
        )

    def _forward(self, img_rgb, mask):
        img = img_rgb.astype(np.float32).transpose(2,0,1)[None] / 255.0
        msk = (mask >= 127).astype(np.float32)[None,None]
        out = self._sess.run(None, {"image": img, "mask": msk})[0]
        return out[0].transpose(1,2,0).clip(0,255).astype(np.uint8)

    def inpaint(self, image_rgb, mask):
        h, w = image_rgb.shape[:2]
        if (h, w) == (self.SIZE, self.SIZE):
            res = self._forward(image_rgb, mask)
        elif max(h, w) <= self.SIZE:
            ph, pw = self.SIZE - h, self.SIZE - w
            ip = cv2.copyMakeBorder(image_rgb, 0, ph, 0, pw, cv2.BORDER_REFLECT_101)
            mp = cv2.copyMakeBorder(mask,      0, ph, 0, pw, cv2.BORDER_REFLECT_101)
            res = self._forward(ip, mp)[:h, :w]
        else:
            ir = cv2.resize(image_rgb, (self.SIZE,self.SIZE), interpolation=cv2.INTER_AREA)
            mr = cv2.resize(mask,      (self.SIZE,self.SIZE), interpolation=cv2.INTER_NEAREST)
            out_r = self._forward(ir, mr)
            res = cv2.resize(out_r, (w, h), interpolation=cv2.INTER_LINEAR)
        m3 = (mask >= 127)[:,:,None]
        return np.where(m3, res, image_rgb)


# ─── main ─────────────────────────────────────────────────────────────────


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--image", default="debug-runs/mask_debug/test_7/worker_results/original.png")
    ap.add_argument("--mask",  default="debug-runs/mask_debug/test_7/worker_results/mask.png")
    ap.add_argument("--out", default=None)
    ap.add_argument("--variant", choices=["c1","c2","c3","overview","split"], required=True)
    ap.add_argument("--models", default="models")
    ap.add_argument("--lama",   default="models/lama_fp32.onnx")
    ap.add_argument("--spread-threshold", type=int, default=30,
                    help="Typoon classify_masks threshold (default 30)")
    ap.add_argument("--max-tiles", type=int, default=20)
    args = ap.parse_args()

    img_path = ROOT / args.image
    msk_path = ROOT / args.mask
    stem = Path(args.image).parent.parent.name or "out"
    out_dir = Path(args.out) if args.out else ROOT / "debug-runs" / "complex_only" / stem

    image = _load_rgb(img_path)
    full_mask = cv2.imread(str(msk_path), cv2.IMREAD_GRAYSCALE)
    H, W = image.shape[:2]
    print(f"page : {W}x{H}")
    print(f"out  : {out_dir}")

    # 1. break full mask into per-blob TextMasks
    masks = split_mask_into_textmasks(full_mask)
    print(f"masks: {len(masks)} blobs (>=50px)")

    # 2. classify with Typoon's real rule
    from typoon.vision.erasers.routing import classify_masks
    canvas_rgba = cv2.cvtColor(image, cv2.COLOR_RGB2RGBA)
    uniform, complex_ = classify_masks(canvas_rgba, masks, args.spread_threshold)
    print(f"  uniform: {len(uniform)}  complex: {len(complex_)}  (threshold={args.spread_threshold})")
    for tag, ms in [("uniform", uniform), ("complex", complex_)]:
        for i, m in enumerate(ms):
            print(f"    {tag}[{i}] xy=({m.x},{m.y}) size={m.image.shape[1]}x{m.image.shape[0]}")

    # build masks
    page_complex = build_page_mask_from_list(complex_, W, H)
    page_uniform = build_page_mask_from_list(uniform, W, H)
    _save(out_dir / "mask_complex.png", page_complex)
    _save(out_dir / "mask_uniform.png", page_uniform)

    # overlay visualization
    vis = image.copy()
    flat_u = np.zeros_like(vis); flat_u[page_uniform > 0] = (60, 200, 60)
    flat_c = np.zeros_like(vis); flat_c[page_complex > 0] = (255, 80, 80)
    vis = cv2.addWeighted(vis, 0.6, flat_u, 0.4, 0)
    vis = cv2.addWeighted(vis, 1.0, flat_c, 0.4, 0)
    _save(out_dir / "split_overlay.png",
          _label(vis, f"green=uniform({len(uniform)})  red=complex({len(complex_)})  thr={args.spread_threshold}"))

    if args.variant == "split":
        print(f"\nartifacts -> {out_dir}")
        return

    if not (page_complex > 0).any():
        print("\nNO COMPLEX MASKS — try lowering --spread-threshold")
        return

    if args.variant == "c1":
        print(f"\nc1 AOT per-blob 384 (complex only)")
        from typoon.vision._backends.aot import AOTInpainter
        inp = AOTInpainter(ROOT / args.models)
        inp._use_coreml = False
        out = per_blob_inpaint(inp, image, page_complex, tile=384,
                               max_tiles=args.max_tiles, label="aot")
        _save(out_dir / "c1_aot_complex.png", out)

    elif args.variant == "c2":
        print(f"\nc2 LaMa per-blob 512 (complex only)")
        lama = LamaInpainter(ROOT / args.lama)
        out = per_blob_inpaint(lama, image, page_complex, tile=512,
                               max_tiles=args.max_tiles, label="lama")
        _save(out_dir / "c2_lama_complex.png", out)

    elif args.variant == "c3":
        print(f"\nc3 Legacy AOTGANEraser cluster path (complex only)")
        from typoon.vision._backends.aot import AOTInpainter
        from typoon.vision.erasers import (
            _cluster_masks, _cluster_crop, _blit_mask, _blend_inpainted_cluster,
            _INPAINT_MAX_DIM,
        )
        inp = AOTInpainter(ROOT / args.models)
        inp._use_coreml = False
        # canvas needs RGBA for _blend_inpainted_cluster
        canvas = cv2.cvtColor(image, cv2.COLOR_RGB2RGBA)
        clusters = _cluster_masks(list(complex_))
        print(f"  clusters: {len(clusters)}  (masks per cluster: "
              f"{[len(c) for c in clusters]})")
        for ci, cluster in enumerate(clusters):
            cx1, cy1, cx2, cy2 = _cluster_crop(cluster, W, H)
            cw, chh = cx2 - cx1, cy2 - cy1
            combined = np.zeros((chh, cw), dtype=np.uint8)
            for m in cluster:
                _blit_mask(combined, m, cx1, cy1)
            crop_rgb = canvas[cy1:cy2, cx1:cx2, :3].copy()
            ls = max(cw, chh)
            if ls > _INPAINT_MAX_DIM:
                scale = _INPAINT_MAX_DIM / ls
                nw, nh = round(cw*scale), round(chh*scale)
                inf_rgb  = cv2.resize(crop_rgb, (nw, nh))
                inf_mask = cv2.resize(combined, (nw, nh), interpolation=cv2.INTER_NEAREST)
                print(f"  cluster[{ci}] crop={cw}x{chh} downscaled to {nw}x{nh}")
            else:
                inf_rgb, inf_mask = crop_rgb, combined
                print(f"  cluster[{ci}] crop={cw}x{chh} native (no downscale)")
            res = inp.inpaint(inf_rgb, inf_mask)
            if ls > _INPAINT_MAX_DIM:
                res = cv2.resize(res, (cw, chh))
            _blend_inpainted_cluster(canvas, res, combined, cx1, cy1, cw, chh)
        out = cv2.cvtColor(canvas, cv2.COLOR_RGBA2RGB)
        _save(out_dir / "c3_aot_legacy_cluster.png", out)

    elif args.variant == "overview":
        def _load_or_blank(name):
            p = out_dir / name
            if p.exists(): return _load_rgb(p)
            return _label(np.full_like(image, 64), f"missing {name}")
        c1 = _load_or_blank("c1_aot_complex.png")
        c2 = _load_or_blank("c2_lama_complex.png")
        c3 = _load_or_blank("c3_aot_legacy_cluster.png")
        h = 900
        row = np.concatenate([
            _label(_rs(image, h), "INPUT"),
            _label(_rs(c1,    h), "c1 AOT per-blob 384 (hard-paste)"),
            _label(_rs(c2,    h), "c2 LaMa per-blob 512 (hard-paste)"),
            _label(_rs(c3,    h), "c3 AOT legacy (cluster+blend)"),
        ], axis=1)
        _save(out_dir / "overview.png", row)

    print(f"\nartifacts -> {out_dir}")


if __name__ == "__main__":
    main()
