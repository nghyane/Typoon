"""CLI helpers — image discovery, chapter parsing, path utilities."""

from __future__ import annotations

import re
from pathlib import Path

_IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tiff"}

from ..paths import _ch_label as ch_label  # noqa: E402, F401


def has_images(path: Path) -> bool:
    return any(f.suffix.lower() in _IMAGE_EXTS for f in path.iterdir() if f.is_file())


def has_chapter_subdirs(path: Path) -> bool:
    return any(d.is_dir() and has_images(d) for d in path.iterdir())


def parse_chapter_num(name: str) -> float:
    m = re.search(r"(\d+(?:\.\d+)?)", name)
    return float(m.group(1)) if m else 0


def discover_chapters(path: Path) -> list[Path]:
    dirs = [d for d in path.iterdir() if d.is_dir() and has_images(d)]
    return sorted(dirs, key=lambda d: parse_chapter_num(d.name))


def is_url(s: str) -> bool:
    return s.startswith("http://") or s.startswith("https://")
