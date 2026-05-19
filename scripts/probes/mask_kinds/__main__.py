"""mask_kinds — visualize 3 mask abstractions on one page.

Compares what gets sent to the inpainter:

  A. bubble interior (Typoon current materialize output)
  B. stroke from CTD det channel 0 (manga-cleaner equivalent)
  C. stroke + light dilate (B + 5% width ellipse, 1 iter — gentle mc style)

Output:
  debug-runs/mask_kinds/<stem>/
    a_bubble.png        green overlay
    b_stroke.png        red overlay
    c_stroke_dilate.png orange overlay
    overview.png        4-up
    a_pct.txt etc.      diagnostics

Run:
  python -m scripts.probes.mask_kinds --image <path>
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
    if img.ndim == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    if img.shape[2] == 4:
        img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def _save(p: Path, img: np.ndarray) -> None:
    p.parent.mkdir(parents=True, exist_ok=True)
    if img.ndim == 2:
        cv2.imwrite(str(p), img)
    else:
        cv2.imwrite(str(p), cv2.cvtColor(img, cv2.COLOR_RGB2BGR))


def _overlay(base: np.ndarray, mask: np.ndarray, color: tuple, alpha: float = 0.5) -> np.ndarray:
    flat = np.zeros_like(base); flat[mask > 0] = color
    return cv2.addWeighted(base, 1 - alpha, flat, alpha, 0).astype(np.uint8)


def _label(img, text):
    out = img.copy()
    bar = max(24, out.shape[0] // 40)
    cv2.rectangle(out, (0, 0), (out.shape[1], bar), (32, 32, 32), -1)
    cv2.putText(out, text, (8, int(bar * 0.72)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)
    return out


# ─── CTD outputs ───────────────────────────────────────────────────────────


def _run_ctd(image_rgb: np.ndarray, ctd_onnx: Path):
    """Run CTD once, return (text_mask, bubble_mask) both uint8 page-res.

    text_mask  = _decode_det_fused: DBNet sigmoid prob ∪ UNet seg, thr 0.30,
                 close(r=10) + dilate(r=3). Ôm stroke chữ chuẩn (đã có sẵn
                 trong _backends/ctd, không cần tự viết lại).
    bubble_mask = _decode_seg: UNet seg > 0.5, plain binary upsample.
    """
    import onnxruntime as ort
    from typoon.vision._backends.ctd import (
        _preprocess, _decode_det_fused, _decode_seg,
    )
    so = ort.SessionOptions(); so.log_severity_level = 3
    sess = ort.InferenceSession(
        str(ctd_onnx), sess_options=so,
        providers=["CPUExecutionProvider"],
    )
    H, W = image_rgb.shape[:2]
    inp, rw, rh = _preprocess(image_rgb)
    blk_out, seg_out, det_out = sess.run(None, {"images": inp})
    text_mask   = _decode_det_fused(det_out[0], seg_out[0, 0], W, H, rw, rh)
    bubble_mask = _decode_seg(seg_out[0, 0], W, H, rw, rh)
    return text_mask, bubble_mask


# ─── main ──────────────────────────────────────────────────────────────────


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--image", default="debug-runs/mask_debug/test_7/worker_results/original.png")
    ap.add_argument("--existing-mask",
                    default="debug-runs/mask_debug/test_7/worker_results/mask.png",
                    help="current Typoon materialize() output to compare")
    ap.add_argument("--ctd", default="models/ctd.onnx")
    ap.add_argument("--out", default=None)
    ap.add_argument("--thr", type=float, default=0.3, help="det threshold")
    args = ap.parse_args()

    img_path = ROOT / args.image
    ctd_path = ROOT / args.ctd
    stem = Path(args.image).parent.parent.name or "out"
    out_dir = Path(args.out) if args.out else ROOT / "debug-runs" / "mask_kinds" / stem

    print(f"image: {args.image}")
    print(f"ctd  : {args.ctd}")
    print(f"out  : {out_dir}")

    image = _load_rgb(img_path)
    H, W = image.shape[:2]
    print(f"page : {W}x{H}")

    # A. existing materialize output (Typoon CtdUNetStrategy: bubble interior)
    print("\nA. existing materialize (Typoon current)")
    a = cv2.imread(str(ROOT / args.existing_mask), cv2.IMREAD_GRAYSCALE)
    print(f"   pct: {(a>0).mean()*100:.2f}%")

    # CTD outputs from one ONNX run
    print("\nrunning CTD")
    import onnxruntime as ort
    from typoon.vision._backends.ctd import (
        _preprocess, _decode_det_fused, _decode_seg,
    )
    so = ort.SessionOptions(); so.log_severity_level = 3
    sess = ort.InferenceSession(
        str(ctd_path), sess_options=so, providers=["CPUExecutionProvider"],
    )
    inp, rw, rh = _preprocess(image)
    blk_out, seg_out, det_out = sess.run(None, {"images": inp})

    # B0. DBNet stroke pure (shrink-thresh sigmoid, no seg fuse)
    shrink, thresh = det_out[0, 0], det_out[0, 1]
    prob = 1.0 / (1.0 + np.exp(-50.0 * (shrink[:rh, :rw] - thresh[:rh, :rw])))
    prob_full = cv2.resize(prob, (W, H), interpolation=cv2.INTER_LINEAR)
    b = (prob_full > 0.3).astype(np.uint8) * 255
    print(f"B. DBNet stroke pure (no seg fuse, thr 0.30)             pct: {(b>0).mean()*100:.2f}%")

    # B1. fused (current Typoon _decode_det_fused)
    b_fused = _decode_det_fused(det_out[0], seg_out[0, 0], W, H, rw, rh)
    print(f"   fused (=current Typoon)                                pct: {(b_fused>0).mean()*100:.2f}%")

    # D. bubble seg
    d = _decode_seg(seg_out[0, 0], W, H, rw, rh)
    print(f"D. CTD bubble_mask (seg > 0.5)                            pct: {(d>0).mean()*100:.2f}%")

    # C. B + small extra halo
    k = max(5, int(W * 0.01)) | 1
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
    c = cv2.dilate(b, kernel, iterations=1)
    print(f"C. B + extra dilate (ellipse k={k}, 1 iter)               pct: {(c>0).mean()*100:.2f}%")

    # E. stroke clipped to bubble (recommended for inpaint input)
    e = cv2.bitwise_and(c, d)
    print(f"E. C ∩ D (stroke+dilate clipped to bubble)                pct: {(e>0).mean()*100:.2f}%")

    _save(out_dir / "b_fused_for_reference.png", b_fused)

    # save raw masks
    _save(out_dir / "a_existing_mask.png", a)
    _save(out_dir / "b_stroke_raw.png", b)
    _save(out_dir / "c_stroke_dilate.png", c)
    _save(out_dir / "d_bubble_seg.png", d)
    _save(out_dir / "e_stroke_clipped.png", e)

    # overlays
    A = _overlay(image, a, (0, 255, 0))
    B = _overlay(image, b, (255, 0, 0))
    C = _overlay(image, c, (255, 140, 0))
    D = _overlay(image, d, (200, 0, 200))
    E = _overlay(image, e, (255, 50, 200))

    _save(out_dir / "overlay_a.png", _label(A, f"A. Typoon current materialize  pct={(a>0).mean()*100:.1f}%"))
    _save(out_dir / "overlay_b.png", _label(B, f"B. CTD text_mask (fused stroke)  pct={(b>0).mean()*100:.1f}%"))
    _save(out_dir / "overlay_c.png", _label(C, f"C. B + extra dilate k={k}  pct={(c>0).mean()*100:.1f}%"))
    _save(out_dir / "overlay_d.png", _label(D, f"D. CTD bubble_mask (seg)  pct={(d>0).mean()*100:.1f}%"))
    _save(out_dir / "overlay_e.png", _label(E, f"E. C clipped to D (recommended)  pct={(e>0).mean()*100:.1f}%"))

    def rs(im, h=720):
        w = int(im.shape[1] * h / im.shape[0])
        return cv2.resize(im, (w, h), interpolation=cv2.INTER_AREA)
    row1 = np.concatenate([rs(_label(image, "input")), rs(_label(A, "A. Typoon current (bubble interior)")),
                           rs(_label(B, "B. CTD text_mask = stroke"))], axis=1)
    row2 = np.concatenate([rs(_label(C, "C. B + extra dilate")), rs(_label(D, "D. CTD bubble_mask")),
                           rs(_label(E, "E. C ∩ D (clip to bubble)"))], axis=1)
    _save(out_dir / "overview.png", np.concatenate([row1, row2], axis=0))

    print(f"\nartifacts -> {out_dir.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
