"""E2E chapter inpaint — scan + inpaint một thư mục ảnh local.

Usage:
    python -m scripts.probes.e2e_chapter <folder>
        [--model crates/inpaint/model.safetensors]
        [--out debug-runs/e2e-<name>/final]
        [--concurrency 4]
        [--lang ja]
        [--pages 0,1,2]   # subset; default = all
        [--no-aot]        # mask only, stub inpainter
"""
from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import sys
import time
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s %(levelname).1s %(name)s: %(message)s")
log = logging.getLogger("e2e_chapter")


async def scan_page(
    idx: int,
    img_path: Path,
    lang: str,
    sink,
) -> dict:
    """Scan one page → return {idx, plan_bytes, jpeg_bytes, scan_meta}."""
    from typoon_inpaint.scan import build_plan_for_image
    t0 = time.perf_counter()
    plan_bytes = await build_plan_for_image(
        img_path, page_index=idx, lang=lang,
        sink=sink.subdir(f"page_{idx:04d}/scan"),
    )
    elapsed = time.perf_counter() - t0
    log.info("scan page %04d  %.1fs  %d bytes plan", idx, elapsed, len(plan_bytes))
    return {
        "idx":        idx,
        "plan_bytes": plan_bytes,
        "jpeg_bytes": img_path.read_bytes(),
        "path":       img_path,
        "scan_ms":    round(elapsed * 1000),
    }


async def inpaint_page(
    rt,
    page: dict,
    out_dir: Path,
    sink,
    no_aot: bool,
) -> dict:
    """Inpaint one page → write PNG, return diagnostics."""
    idx = page["idx"]
    t0  = time.perf_counter()

    debug_dir = str(sink.subdir(f"page_{idx:04d}/inpaint").path)

    if no_aot:
        # Mask-only: just rasterise to verify mask pipeline
        from typoon_inpaint import rasterise_plan_mask
        mask = rasterise_plan_mask(
            page["plan_bytes"], page["jpeg_bytes"],
            debug_dir=debug_dir,
        )
        # Write mask as PNG
        from PIL import Image
        import numpy as np
        img = np.array(Image.open(page["path"]).convert("RGB"))
        H, W = img.shape[:2]
        mask_arr = np.frombuffer(bytes(mask), dtype=np.uint8).reshape(H, W)
        # Overlay: red tint on masked pixels
        overlay = img.copy()
        overlay[mask_arr > 0, 0] = 200
        overlay[mask_arr > 0, 1] = (overlay[mask_arr > 0, 1] * 0.4).astype(np.uint8)
        overlay[mask_arr > 0, 2] = (overlay[mask_arr > 0, 2] * 0.4).astype(np.uint8)
        out_path = out_dir / f"{idx:04d}_mask.png"
        Image.fromarray(overlay).save(out_path)
        png_bytes = out_path.read_bytes()
    else:
        # inpaint_page là sync (Rust releases GIL) — wrap qua to_thread
        # để không block event loop trong khi Candle inference chạy.
        png_bytes = await asyncio.to_thread(
            rt.inpaint_page,
            page["jpeg_bytes"], page["plan_bytes"],
            debug_dir,
        )
        out_path = out_dir / f"{idx:04d}.png"
        out_path.write_bytes(bytes(png_bytes))

        # Also write mask overlay on original for visual verification
        from typoon_inpaint import rasterise_plan_mask
        from PIL import Image
        import numpy as np
        mask    = rasterise_plan_mask(page["plan_bytes"], page["jpeg_bytes"])
        img     = np.array(Image.open(page["path"]).convert("RGB"))
        H, W    = img.shape[:2]
        mask_a  = np.frombuffer(bytes(mask), dtype=np.uint8).reshape(H, W)
        overlay = img.copy()
        overlay[mask_a > 0, 0] = 220
        overlay[mask_a > 0, 1] = (overlay[mask_a > 0, 1].astype(int) * 0.3).astype(np.uint8)
        overlay[mask_a > 0, 2] = (overlay[mask_a > 0, 2].astype(int) * 0.3).astype(np.uint8)
        Image.fromarray(overlay).save(out_dir / f"{idx:04d}_overlay.png")

    elapsed = time.perf_counter() - t0

    # Quick diagnostics from plan
    from typoon_inpaint import decode_plan
    plan = decode_plan(page["plan_bytes"])
    kinds   = Counter(g["kind"]  for g in plan["groups"])
    classes = Counter(g["class"] for g in plan["groups"])

    log.info(
        "inpaint page %04d  %.1fs  groups=%d  kinds=%s  out=%s",
        idx, elapsed, len(plan["groups"]),
        dict(kinds), out_path.name,
    )
    return {
        "idx":        idx,
        "inpaint_ms": round(elapsed * 1000),
        "scan_ms":    page["scan_ms"],
        "n_groups":   len(plan["groups"]),
        "page_kind":  plan["page_kind"],
        "kinds":      dict(kinds),
        "classes":    dict(classes),
        "out":        str(out_path),
    }


async def run(args) -> None:
    from typoon_inpaint.artifact_sink import FileArtifactSink

    folder   = Path(args.folder)
    run_id   = f"e2e-{folder.name}"
    run_dir  = ROOT / "debug-runs" / run_id
    out_dir  = Path(args.out) if args.out else run_dir / "final"
    out_dir.mkdir(parents=True, exist_ok=True)
    sink = FileArtifactSink(run_dir)

    # Collect pages
    exts   = {".jpg", ".jpeg", ".png", ".webp"}
    all_imgs = sorted(p for p in folder.iterdir() if p.suffix.lower() in exts)
    if args.pages:
        indices = [int(x) for x in args.pages.split(",")]
        imgs    = [all_imgs[i] for i in indices if i < len(all_imgs)]
    else:
        imgs = all_imgs

    log.info("chapter: %s  pages=%d  concurrency=%d  no_aot=%s",
             folder.name, len(imgs), args.concurrency, args.no_aot)

    # Init inpainter once
    rt = None
    if not args.no_aot:
        from typoon_inpaint import InpaintRuntime
        model = args.model or str(ROOT / "crates/inpaint/model.safetensors")
        log.info("loading model: %s", model)
        rt = InpaintRuntime(str(model))
        log.info("model ready")

    sem      = asyncio.Semaphore(args.concurrency)
    t_total  = time.perf_counter()
    results  = []
    failures = []

    async def process(i: int, img_path: Path) -> None:
        async with sem:
            try:
                page  = await scan_page(i, img_path, args.lang, sink)
                diag  = await inpaint_page(rt, page, out_dir, sink, args.no_aot)
                results.append(diag)
            except Exception as e:
                log.exception("page %04d FAILED: %s", i, e)
                failures.append({"idx": i, "error": str(e)})

    await asyncio.gather(*[process(i, p) for i, p in enumerate(imgs)])

    wall_s = time.perf_counter() - t_total
    results.sort(key=lambda r: r["idx"])

    # Summary
    total_groups   = sum(r["n_groups"] for r in results)
    all_kinds      = Counter()
    all_classes    = Counter()
    all_page_kinds = Counter()
    for r in results:
        all_kinds.update(r["kinds"])
        all_classes.update(r["classes"])
        all_page_kinds[r["page_kind"]] += 1

    summary = {
        "run_id":        run_id,
        "pages":         len(imgs),
        "success":       len(results),
        "failures":      len(failures),
        "wall_s":        round(wall_s, 1),
        "scan_ms_avg":   round(sum(r["scan_ms"]    for r in results) / max(len(results),1)),
        "inpaint_ms_avg":round(sum(r["inpaint_ms"] for r in results) / max(len(results),1)),
        "total_groups":  total_groups,
        "kinds":         dict(all_kinds),
        "classes":       dict(all_classes),
        "page_kinds":    dict(all_page_kinds),
        "output_dir":    str(out_dir),
        "failures":      failures,
    }
    (run_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    (run_dir / "pages.json").write_text(json.dumps(results, indent=2))

    print("\n" + "="*60)
    print(f"  Chapter E2E — {run_id}")
    print("="*60)
    print(f"  pages     : {summary['pages']} ({summary['success']} OK, {summary['failures']} failed)")
    print(f"  wall      : {wall_s:.1f}s  (~{wall_s/max(len(imgs),1):.1f}s/page)")
    print(f"  scan avg  : {summary['scan_ms_avg']}ms")
    print(f"  inpaint avg: {summary['inpaint_ms_avg']}ms")
    print(f"  groups    : {total_groups} total")
    print(f"  kinds     : {dict(all_kinds)}")
    print(f"  classes   : {dict(all_classes)}")
    print(f"  page_kinds: {dict(all_page_kinds)}")
    print(f"  output    : {out_dir}")
    print(f"  artifacts : {run_dir}")
    print("="*60)

    if failures:
        print(f"\n  FAILURES:")
        for f in failures:
            print(f"    page {f['idx']:04d}: {f['error']}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("folder", type=str)
    ap.add_argument("--model",       default=None)
    ap.add_argument("--out",         default=None)
    ap.add_argument("--concurrency", type=int, default=4)
    ap.add_argument("--lang",        default="ja")
    ap.add_argument("--pages",       default=None,
                    help="Comma-separated page indices, e.g. 0,1,2")
    ap.add_argument("--no-aot",      action="store_true",
                    help="Skip AOT inference — emit mask overlay only")
    args = ap.parse_args()
    asyncio.run(run(args))


if __name__ == "__main__":
    main()
