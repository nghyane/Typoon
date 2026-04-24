"""App path resolution — no config data, no loading logic."""

from __future__ import annotations

import hashlib
import os
import re as _re
import time as _time
from pathlib import Path


def home() -> Path:
    """Fixed app data directory. Never depends on CWD."""
    return Path(os.environ.get("TYPOON_HOME", "~/.typoon")).expanduser()


class Paths:
    """All app paths resolved from home."""

    def __init__(self, root: Path | None = None) -> None:
        self.root = (root or home()).resolve()

    @property
    def config_file(self) -> Path: return self.root / "config.toml"
    @property
    def db(self) -> Path: return self.root / "typoon.db"
    @property
    def models(self) -> Path: return self.root / "models"
    @property
    def cache(self) -> Path: return self.root / "cache"
    @property
    def output(self) -> Path: return self.root / "output"
    @property
    def projects(self) -> Path: return self.root / "projects"

    def ensure(self) -> None:
        """Create all directories."""
        for d in (self.root, self.cache, self.output, self.projects):
            d.mkdir(parents=True, exist_ok=True)


# ── Slug / chapter labels ─────────────────────────────────────────


def _slugify(title: str, url: str = "") -> str:
    """Filesystem-safe slug with optional collision-resistant hash."""
    base = title.lower().strip()
    base = _re.sub(r"[^\w\s-]", "", base)
    base = _re.sub(r"[\s]+", "-", base).strip("-")
    if not base:
        base = f"unnamed-{int(_time.time())}"
    if url:
        h = hashlib.md5(url.encode()).hexdigest()[:6]
        return f"{base}-{h}"
    return base


def _ch_label(ch: float) -> str:
    """Chapter directory label, e.g. ch001 or ch0001.5."""
    return f"ch{int(ch):03d}" if ch == int(ch) else f"ch{ch:06.1f}"


# ── Per-project workspace ─────────────────────────────────────────


class ProjectPaths:
    """Isolated source cache and output dirs for one project."""

    def __init__(self, projects_root: Path, title: str, source_url: str = "") -> None:
        self._slug = _slugify(title, source_url)
        self.root = projects_root / self._slug
        self.source = self.root / "source"
        self.output_dir = self.root / "output"

    @property
    def slug(self) -> str:
        return self._slug

    def chapter_source(self, ch: float) -> Path:
        return self.source / _ch_label(ch)

    def chapter_output(self, ch: float) -> Path:
        return self.output_dir / _ch_label(ch)

    def ensure(self) -> None:
        for d in (self.root, self.source, self.output_dir):
            d.mkdir(parents=True, exist_ok=True)
