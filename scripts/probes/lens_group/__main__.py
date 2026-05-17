"""Probe CLI entry."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from . import panel_container, panel_detr, panel_lens, panel_masks  # noqa: E402
from .compose import grid_2x2  # noqa: E402
from .io_utils import load_rgb, save_json, save_png  # noqa: E402
from .probe import run, to_json  # noqa: E402
from .trace import print_trace  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(prog="lens_group_probe")
    ap.add_argument("image", type=Path)
    ap.add_argument(
        "--out", type=Path,
        default=ROOT / "debug-runs" / "lens_group_probe",
    )
    ap.add_argument("--models", type=Path, default=ROOT / "models")
    ap.add_argument("--lang",   type=str, default=None,
                    help="Optional source lang hint (e.g. ja, zh-Hans)")
    args = ap.parse_args()

    import logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(levelname).1s %(name)s: %(message)s",
    )

    out_dir = args.out
    out_dir.mkdir(parents=True, exist_ok=True)

    img = load_rgb(args.image)
    save_png(out_dir / "source.png", img)
    print(f"loaded {args.image.name}: {img.shape[1]}x{img.shape[0]}")

    result = run(img, args.models, lang=args.lang)
    print(
        f"  lens kept={len(result.detection.blocks)} "
        f"regions={len(result.detection.bubble_regions)} "
        f"groups={len(result.groups)}"
    )

    save_json(out_dir / "raw.json", to_json(result))
    print_trace(result.detection)

    blocks   = list(result.detection.blocks)
    rejected = list(result.detection.rejected)
    regions  = result.detection.bubble_regions
    groups   = result.groups
    overview = grid_2x2(
        tl=panel_lens.render(img, blocks, rejected),
        tr=panel_detr.render(img, regions, blocks),
        bl=panel_container.render(img, groups),
        br=panel_masks.render(img, groups, regions),
    )
    save_png(out_dir / "overview.png", overview)
    print(f"artifacts → {out_dir}")


if __name__ == "__main__":
    main()
