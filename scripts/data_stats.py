#!/usr/bin/env python3
"""
Training data statistics for MI-GAN distillation.

Reports: image count, resolution distribution, grayscale vs color ratio,
file sizes, and per-series breakdown.

Usage:
    python3 scripts/data_stats.py data/training
    python3 scripts/data_stats.py data/training --verify   # also remove corrupt files
"""

import argparse
import os
import sys
from collections import defaultdict
from pathlib import Path

from PIL import Image

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tiff"}


def is_grayscale(img: Image.Image, sample_size: int = 100) -> bool:
    """Fast grayscale detection by sampling pixels."""
    if img.mode == "L":
        return True
    if img.mode != "RGB":
        img = img.convert("RGB")

    w, h = img.size
    import random
    random.seed(42)
    for _ in range(sample_size):
        x, y = random.randint(0, w - 1), random.randint(0, h - 1)
        r, g, b = img.getpixel((x, y))
        if abs(r - g) > 10 or abs(r - b) > 10 or abs(g - b) > 10:
            return False
    return True


def analyze_directory(root: str, verify: bool = False) -> dict:
    """Analyze all images under root."""
    stats = {
        "total": 0,
        "grayscale": 0,
        "color": 0,
        "corrupt": 0,
        "total_bytes": 0,
        "widths": [],
        "heights": [],
        "series": defaultdict(lambda: {"count": 0, "grayscale": 0, "color": 0, "bytes": 0}),
    }

    corrupt_files = []

    for dirpath, _, filenames in os.walk(root):
        for fname in sorted(filenames):
            if Path(fname).suffix.lower() not in IMAGE_EXTENSIONS:
                continue

            fpath = os.path.join(dirpath, fname)
            fsize = os.path.getsize(fpath)

            # Determine series name (2 levels up from file: series/chapter/file)
            rel = os.path.relpath(fpath, root)
            parts = rel.split(os.sep)
            if len(parts) >= 3:
                series_key = f"{parts[0]}/{parts[1]}"  # type/series
            elif len(parts) >= 2:
                series_key = parts[0]
            else:
                series_key = "root"

            try:
                img = Image.open(fpath)
                img.verify()
                # Re-open after verify (verify closes the file)
                img = Image.open(fpath)
                w, h = img.size

                gray = is_grayscale(img)

                stats["total"] += 1
                stats["total_bytes"] += fsize
                stats["widths"].append(w)
                stats["heights"].append(h)

                s = stats["series"][series_key]
                s["count"] += 1
                s["bytes"] += fsize

                if gray:
                    stats["grayscale"] += 1
                    s["grayscale"] += 1
                else:
                    stats["color"] += 1
                    s["color"] += 1

            except Exception as e:
                stats["corrupt"] += 1
                corrupt_files.append(fpath)
                if verify:
                    os.remove(fpath)
                    print(f"  ✗ Removed corrupt: {fpath} ({e})")

    if corrupt_files and not verify:
        print(f"\n⚠ Found {len(corrupt_files)} corrupt files (run with --verify to remove)")

    return stats


def print_stats(stats: dict, root: str):
    """Pretty-print statistics."""
    total = stats["total"]
    if total == 0:
        print(f"No images found in {root}")
        return

    size_mb = stats["total_bytes"] / 1024 / 1024

    print(f"\n{'='*60}")
    print(f"  Training Data Statistics: {root}")
    print(f"{'='*60}")
    print(f"  Total images:    {total:,}")
    print(f"  Total size:      {size_mb:,.0f} MB ({size_mb/1024:.1f} GB)")
    print(f"  Corrupt files:   {stats['corrupt']}")
    print()

    # Grayscale vs Color
    g_pct = stats["grayscale"] / total * 100
    c_pct = stats["color"] / total * 100
    print(f"  Grayscale (B/W): {stats['grayscale']:,} ({g_pct:.1f}%)")
    print(f"  Color:           {stats['color']:,} ({c_pct:.1f}%)")
    print()

    # Resolution distribution
    import statistics
    widths = stats["widths"]
    heights = stats["heights"]
    print(f"  Resolution distribution:")
    print(f"    Width:  min={min(widths)}, median={int(statistics.median(widths))}, max={max(widths)}")
    print(f"    Height: min={min(heights)}, median={int(statistics.median(heights))}, max={max(heights)}")

    # Bucket resolutions
    buckets = defaultdict(int)
    for w, h in zip(widths, heights):
        short = min(w, h)
        if short < 512:
            buckets["< 512px"] += 1
        elif short < 1024:
            buckets["512-1023px"] += 1
        elif short < 2048:
            buckets["1024-2047px"] += 1
        else:
            buckets[">= 2048px"] += 1

    print(f"    Size buckets (shorter side):")
    for bucket in ["< 512px", "512-1023px", "1024-2047px", ">= 2048px"]:
        count = buckets.get(bucket, 0)
        pct = count / total * 100
        bar = "█" * int(pct / 2)
        print(f"      {bucket:>12s}: {count:>6,} ({pct:5.1f}%) {bar}")

    # Per-series breakdown
    print(f"\n  Per-series breakdown:")
    print(f"    {'Series':<40s} {'Count':>7s} {'B/W':>6s} {'Color':>6s} {'Size':>8s}")
    print(f"    {'─'*40} {'─'*7} {'─'*6} {'─'*6} {'─'*8}")
    for series_key in sorted(stats["series"]):
        s = stats["series"][series_key]
        smb = s["bytes"] / 1024 / 1024
        print(f"    {series_key:<40s} {s['count']:>7,} {s['grayscale']:>6,} {s['color']:>6,} {smb:>7.0f}M")

    print(f"{'='*60}")


def main():
    p = argparse.ArgumentParser(description="Training data statistics")
    p.add_argument("directory", help="Root directory to analyze")
    p.add_argument("--verify", action="store_true",
                   help="Remove corrupt files")
    args = p.parse_args()

    if not os.path.isdir(args.directory):
        print(f"Error: {args.directory} is not a directory")
        sys.exit(1)

    print(f"Analyzing {args.directory}...")
    stats = analyze_directory(args.directory, verify=args.verify)
    print_stats(stats, args.directory)


if __name__ == "__main__":
    main()
