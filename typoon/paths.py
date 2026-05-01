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
    def projects(self) -> Path: return self.root / "projects"

    def ensure(self) -> None:
        for d in (self.root, self.projects):
            d.mkdir(parents=True, exist_ok=True)


class ChapterPaths:
    """All paths for one chapter — derived from project slug + chapter index."""

    def __init__(self, projects_root: Path, slug: str, idx: float) -> None:
        self.root     = projects_root / slug / _ch_label(idx)
        self.pages    = self.root / "pages"
        self.manifest = self.root / "manifest.json"
        self.scan     = self.root / "scan.json"
        self.masks    = self.root / "masks"
        self.translate = self.root / "translate.json"
        self.render   = self.root / "render"

    def ensure(self) -> None:
        for d in (self.pages, self.masks, self.render):
            d.mkdir(parents=True, exist_ok=True)

    @property
    def is_prepared(self) -> bool:
        return self.manifest.exists()

    @property
    def is_scanned(self) -> bool:
        return self.scan.exists()

    @property
    def is_translated(self) -> bool:
        return self.translate.exists()

    @property
    def is_rendered(self) -> bool:
        return self.render.exists() and any(self.render.iterdir())


class ProjectPaths:
    """Paths for one project."""

    def __init__(self, projects_root: Path, slug: str) -> None:
        self.root = projects_root / slug
        self.slug = slug

    def chapter(self, idx: float) -> ChapterPaths:
        return ChapterPaths(self.root.parent, self.slug, idx)

    def ensure(self) -> None:
        self.root.mkdir(parents=True, exist_ok=True)


# ── Slug / chapter labels ─────────────────────────────────────────


def slugify(title: str, url: str = "") -> str:
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
    return f"ch{int(ch):03d}" if ch == int(ch) else f"ch{ch:06.1f}"


def ch_label(ch: float) -> str:
    return _ch_label(ch)


# ── Backward compat ───────────────────────────────────────────────
_slugify = slugify
