"""prod_scan_probe — run the real Typoon scan pipeline on arbitrary images
and probe inpaint backends on the resulting masks.

What this does (matches packages/typoon-stages/typoon/stages/scan.py exactly):

  1. build VisionRuntime from VisionPipelineSpec (preset='lens')
  2. for each input image:
       a. detector.detect      (Lens + comic-detr)
       b. grouper.group        (LensNativeGrouper)
       c. ctd_seg.run          (CTD UNet seg-only, ja only)
       d. default_filter()     (drop noise groups)
       e. _attach_masks        (select_strategy + materialize per group)
       f. classify_masks       (split masks → uniform vs complex)
       g. inpaint complex masks with each backend variant:
            v_aot_perblob_384    — AOT per-blob hard-paste
            v_lama_perblob_512   — LaMa per-blob hard-paste (needs lama_fp32.onnx)
            v_aot_legacy_cluster — legacy AOTGANEraser cluster+blend path
       h. write artifacts under debug-runs/prod_scan_probe/<stem>/

Note: this calls scan internals (`_attach_masks`) intentionally — it's the
only way to reproduce the prod mask without spinning up bunle archives.
Mask abstraction stays identical to prod scan_chapter.

Run:
  python -m scripts.probes.prod_scan_probe --image <path> [--image <path> ...]
  python -m scripts.probes.prod_scan_probe --variant overview
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import os
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[3]
for pkg in ("typoon-core", "typoon-vision", "typoon-stages"):
    p = ROOT / "packages" / pkg
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
logger = logging.getLogger("prod_scan_probe")


# ─── helpers ──────────────────────────────────────────────────────────────


def _load_rgb(path: Path) -> np.ndarray:
    im = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if im is None:
        raise FileNotFoundError(path)
    if im.ndim == 2:
        im = cv2.cvtColor(im, cv2.COLOR_GRAY2BGR)
    if im.shape[2] == 4:
        im = cv2.cvtColor(im, cv2.COLOR_BGRA2BGR)
    return cv2.cvtColor(im, cv2.COLOR_BGR2RGB)


def _save(p: Path, im: np.ndarray) -> None:
    p.parent.mkdir(parents=True, exist_ok=True)
    if im.ndim == 2:
        cv2.imwrite(str(p), im)
    else:
        cv2.imwrite(str(p), cv2.cvtColor(im, cv2.COLOR_RGB2BGR))


def _label(im: np.ndarray, text: str) -> np.ndarray:
    out = im.copy()
    bar = max(28, out.shape[0] // 30)
    cv2.rectangle(out, (0, 0), (out.shape[1], bar), (24, 24, 24), -1)
    cv2.putText(out, text, (10, int(bar * 0.72)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
    return out


def _rs(im: np.ndarray, h: int) -> np.ndarray:
    w = int(im.shape[1] * h / im.shape[0])
    return cv2.resize(im, (w, h), interpolation=cv2.INTER_AREA)


# ─── scan steps (mirrors stages.scan exactly) ─────────────────────────────


async def run_prod_scan(image: np.ndarray, source_lang: str, models_dir: Path):
    """Returns (groups_with_masks, bubble_mask) — same shape as scan_chapter."""
    from typoon.domain.filter import ScoringContext
    from typoon.vision.pipeline import VisionPipelineSpec
    from typoon.vision.runtime import build_vision_runtime
    from typoon.vision.filters import default_filter
    from typoon.stages.scan import _attach_masks, _seg_unet, _detect, _group

    lens_endpoint = os.environ.get("LENS_ENDPOINT")
    spec = VisionPipelineSpec.preset("lens")
    runtime = build_vision_runtime(
        spec, models_dir=models_dir,
        source_lang=source_lang,
        lens_endpoint=lens_endpoint,
    )

    h, w = image.shape[:2]
    logger.info("detect: lens (endpoint=%s)", "set" if lens_endpoint else "default")
    detection = await _detect(runtime, image, source_lang)
    logger.info("  blocks: %d", len(detection.blocks))

    raw_groups = await _group(runtime, image, detection, source_lang)
    logger.info("group: %d raw groups", len(raw_groups))

    bubble_mask = await _seg_unet(runtime, image)
    if bubble_mask is not None:
        logger.info("ctd_seg: bubble_mask active=%d px", int((bubble_mask > 0).sum()))

    ctx = ScoringContext(
        page_size=(w, h),
        page_groups=raw_groups,
        image=image,
        blocks=tuple(detection.blocks),
    )
    fr = default_filter().evaluate(raw_groups, ctx)
    logger.info("filter: kept=%d rejected=%d", fr.n_kept, fr.n_rejected)
    groups = fr.kept

    groups_with_masks = _attach_masks(
        groups, list(detection.blocks), image, bubble_mask
    )
    return groups_with_masks, bubble_mask


# ─── inpaint backends ─────────────────────────────────────────────────────


class LamaInpainter:
    SIZE = 512
    def __init__(self, onnx_path: Path) -> None:
        import onnxruntime as ort
        so = ort.SessionOptions(); so.log_severity_level = 3
        self._sess = ort.InferenceSession(
            str(onnx_path), sess_options=so, providers=["CPUExecutionProvider"],
        )

    def _forward(self, img_rgb, mask):
        img = img_rgb.astype(np.float32).transpose(2, 0, 1)[None] / 255.0
        msk = (mask >= 127).astype(np.float32)[None, None]
        out = self._sess.run(None, {"image": img, "mask": msk})[0]
        return out[0].transpose(1, 2, 0).clip(0, 255).astype(np.uint8)

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
            ir = cv2.resize(image_rgb, (self.SIZE, self.SIZE), interpolation=cv2.INTER_AREA)
            mr = cv2.resize(mask,      (self.SIZE, self.SIZE), interpolation=cv2.INTER_NEAREST)
            out_r = self._forward(ir, mr)
            res = cv2.resize(out_r, (w, h), interpolation=cv2.INTER_LINEAR)
        m3 = (mask >= 127)[:, :, None]
        return np.where(m3, res, image_rgb)


def per_blob_inpaint(inpainter, image, page_mask, tile, max_tiles=40, label=""):
    H, W = image.shape[:2]
    out = image.copy()
    n, _l, stats, _ = cv2.connectedComponentsWithStats(page_mask, 8)
    blobs = [stats[i] for i in range(1, n) if stats[i, 4] > 5]
    blobs.sort(key=lambda s: s[4], reverse=True)
    covered = np.zeros_like(page_mask)
    nt = 0
    for bx, by, bw, bh, _ in blobs:
        if nt >= max_tiles:
            break
        if np.all(covered[by:by+bh, bx:bx+bw] == 255):
            continue
        cx, cy = bx + bw // 2, by + bh // 2
        x1 = max(0, cx - tile // 2); y1 = max(0, cy - tile // 2)
        x2 = min(W, x1 + tile);      y2 = min(H, y1 + tile)
        x1 = max(0, x2 - tile);      y1 = max(0, y2 - tile)
        ti = out[y1:y2, x1:x2]
        tm = page_mask[y1:y2, x1:x2]
        if not (tm > 0).any():
            continue
        res = inpainter.inpaint(ti, tm)
        m3 = (tm >= 127)[:, :, None]
        out[y1:y2, x1:x2] = np.where(m3, res, ti)
        covered[y1:y2, x1:x2] = 255
        nt += 1
    logger.info("%s: %d tiles (%dx%d)", label, nt, tile, tile)
    return out


def legacy_cluster_inpaint(image_rgb: np.ndarray, complex_masks):
    """Drive legacy AOTGANEraser cluster+blend path on complex masks only."""
    from typoon.vision._backends.aot import AOTInpainter
    from typoon.vision.erasers import (
        _cluster_masks, _cluster_crop, _blit_mask,
        _blend_inpainted_cluster, _INPAINT_MAX_DIM,
    )
    inp = AOTInpainter(ROOT / "models")
    inp._use_coreml = False

    canvas = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2RGBA)
    H, W = image_rgb.shape[:2]
    clusters = _cluster_masks(list(complex_masks))
    logger.info("legacy: %d clusters", len(clusters))
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
            nw, nh = round(cw * scale), round(chh * scale)
            inf_rgb  = cv2.resize(crop_rgb, (nw, nh))
            inf_mask = cv2.resize(combined, (nw, nh), interpolation=cv2.INTER_NEAREST)
        else:
            inf_rgb, inf_mask = crop_rgb, combined
        res = inp.inpaint(inf_rgb, inf_mask)
        if ls > _INPAINT_MAX_DIM:
            res = cv2.resize(res, (cw, chh))
        _blend_inpainted_cluster(canvas, res, combined, cx1, cy1, cw, chh)
    return cv2.cvtColor(canvas, cv2.COLOR_RGBA2RGB)


def aot_perblob_inpaint(image_rgb: np.ndarray, page_mask: np.ndarray, max_tiles=40):
    from typoon.vision._backends.aot import AOTInpainter
    inp = AOTInpainter(ROOT / "models")
    inp._use_coreml = False
    return per_blob_inpaint(inp, image_rgb, page_mask, tile=384,
                            max_tiles=max_tiles, label="aot_perblob_384")


# ─── per-image driver ─────────────────────────────────────────────────────


def split_complex_uniform(canvas_rgba: np.ndarray, masks: list, threshold: int):
    from typoon.vision.erasers.routing import classify_masks
    return classify_masks(canvas_rgba, masks, threshold)


def build_page_mask(masks: list, W: int, H: int) -> np.ndarray:
    from typoon.vision.erasers.routing import build_page_mask as _bpm
    return _bpm(masks, W, H)


async def process_one(
    image_path: Path,
    variant: str,
    out_dir: Path,
    source_lang: str,
    models_dir: Path,
    spread_threshold: int,
    max_tiles: int,
):
    image = _load_rgb(image_path)
    H, W = image.shape[:2]
    logger.info("page %dx%d  variant=%s", W, H, variant)

    groups_with_masks, _bm = await run_prod_scan(image, source_lang, models_dir)
    all_masks = [m for _g, masks in groups_with_masks for m in masks]
    logger.info("scan: %d groups, %d TextMasks total",
                len(groups_with_masks), len(all_masks))

    if not all_masks:
        logger.warning("no masks produced — skipping")
        return

    canvas_rgba = cv2.cvtColor(image, cv2.COLOR_RGB2RGBA)
    uniform, complex_ = split_complex_uniform(canvas_rgba, all_masks, spread_threshold)
    logger.info("classify: uniform=%d  complex=%d  (threshold=%d)",
                len(uniform), len(complex_), spread_threshold)

    page_complex = build_page_mask(complex_, W, H)
    page_uniform = build_page_mask(uniform, W, H)
    _save(out_dir / "mask_complex.png", page_complex)
    _save(out_dir / "mask_uniform.png", page_uniform)

    # split overlay for reference
    vis = image.copy()
    flat_u = np.zeros_like(vis); flat_u[page_uniform > 0] = (60, 200, 60)
    flat_c = np.zeros_like(vis); flat_c[page_complex > 0] = (255, 80, 80)
    vis = cv2.addWeighted(vis, 0.6, flat_u, 0.4, 0)
    vis = cv2.addWeighted(vis, 1.0, flat_c, 0.4, 0)
    _save(out_dir / "split_overlay.png",
          _label(vis, f"green=uniform({len(uniform)})  "
                       f"red=complex({len(complex_)})  thr={spread_threshold}"))

    if not (page_complex > 0).any():
        logger.warning("no complex masks — set --spread-threshold lower")
        return

    if variant == "scan_only":
        return  # mask artifacts already written

    if variant == "aot_perblob":
        out = aot_perblob_inpaint(image, page_complex, max_tiles=max_tiles)
        _save(out_dir / "v_aot_perblob_384.png", out)

    elif variant == "lama_perblob":
        lama_path = ROOT / "models" / "lama_fp32.onnx"
        if not lama_path.exists():
            raise FileNotFoundError(
                f"missing {lama_path} — download from "
                "https://huggingface.co/Carve/LaMa-ONNX/resolve/main/lama_fp32.onnx"
            )
        lama = LamaInpainter(lama_path)
        out = per_blob_inpaint(lama, image, page_complex, tile=512,
                               max_tiles=max_tiles, label="lama_perblob_512")
        _save(out_dir / "v_lama_perblob_512.png", out)

    elif variant == "aot_legacy":
        out = legacy_cluster_inpaint(image, complex_)
        _save(out_dir / "v_aot_legacy_cluster.png", out)


def build_overview(out_dir: Path) -> None:
    """Assemble overview.png from variant outputs in a single image dir."""
    orig = None
    for cand in ("00_input.png",):
        p = out_dir / cand
        if p.exists():
            orig = _load_rgb(p); break
    if orig is None:
        # try parent's input symlink
        for p in out_dir.iterdir():
            if p.name.startswith("00_input"):
                orig = _load_rgb(p); break
    if orig is None:
        logger.error("no input found in %s — re-run a variant first", out_dir)
        return

    def _load_or_blank(name):
        p = out_dir / name
        if p.exists(): return _load_rgb(p)
        blank = np.full_like(orig, 64)
        return _label(blank, f"missing {name}")

    aot   = _load_or_blank("v_aot_perblob_384.png")
    lama  = _load_or_blank("v_lama_perblob_512.png")
    legacy = _load_or_blank("v_aot_legacy_cluster.png")

    h = 900
    row = np.concatenate([
        _label(_rs(orig,   h), "INPUT"),
        _label(_rs(aot,    h), "AOT per-blob 384 (hard-paste)"),
        _label(_rs(lama,   h), "LaMa per-blob 512 (hard-paste)"),
        _label(_rs(legacy, h), "AOT legacy cluster+blend"),
    ], axis=1)
    _save(out_dir / "overview.png", row)


# ─── main ─────────────────────────────────────────────────────────────────


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--image", action="append", default=[],
                    help="image path; repeat for multiple images")
    ap.add_argument("--variant",
                    choices=["scan_only", "aot_perblob", "lama_perblob",
                             "aot_legacy", "overview", "all"],
                    default="scan_only")
    ap.add_argument("--out-root", default="debug-runs/prod_scan_probe")
    ap.add_argument("--source-lang", default="ja",
                    help="source_lang hint for scan (ja triggers CTD seg)")
    ap.add_argument("--models", default="models")
    ap.add_argument("--spread-threshold", type=int, default=30)
    ap.add_argument("--max-tiles", type=int, default=20)
    args = ap.parse_args()

    if not args.image:
        ap.error("at least one --image required")

    models_dir = ROOT / args.models
    out_root = ROOT / args.out_root

    for img in args.image:
        img_path = ROOT / img if not Path(img).is_absolute() else Path(img)
        stem = img_path.stem
        out_dir = out_root / stem
        out_dir.mkdir(parents=True, exist_ok=True)

        # always cache original copy for overview convenience
        _save(out_dir / "00_input.png", _load_rgb(img_path))

        variants = (["aot_perblob", "lama_perblob", "aot_legacy"]
                    if args.variant == "all" else [args.variant])

        for v in variants:
            if v == "overview":
                build_overview(out_dir)
                continue
            asyncio.run(process_one(
                image_path=img_path,
                variant=v,
                out_dir=out_dir,
                source_lang=args.source_lang,
                models_dir=models_dir,
                spread_threshold=args.spread_threshold,
                max_tiles=args.max_tiles,
            ))

        if args.variant in ("all", "overview"):
            build_overview(out_dir)

        logger.info("done -> %s", out_dir)


if __name__ == "__main__":
    main()
