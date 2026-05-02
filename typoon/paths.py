"""App path resolution."""

from __future__ import annotations

import hashlib
import os
import re as _re
import time as _time
from dataclasses import dataclass
from pathlib import Path


_HOME: Path | None = None


def home() -> Path:
    global _HOME
    if _HOME is None:
        _HOME = Path(os.environ.get("TYPOON_HOME", "~/.typoon")).expanduser()
    return _HOME


def slugify(title: str, url: str = "") -> str:
    base = _re.sub(r"[\s]+", "-", _re.sub(r"[^\w\s-]", "", title.lower().strip())).strip("-")
    if not base:
        base = f"unnamed-{int(_time.time())}"
    return f"{base}-{hashlib.md5(url.encode()).hexdigest()[:6]}" if url else base


@dataclass(frozen=True)
class ChapterPaths:
    """All file paths for one chapter. Keyed by DB chapter_id (int)."""

    projects_root: Path
    slug:          str
    chapter_id:    int          # DB primary key — never float, never ambiguous

    @property
    def root(self) -> Path:
        return self.projects_root / self.slug / str(self.chapter_id)

    # ── Directories ───────────────────────────────────────────────

    @property
    def pages(self) -> Path:
        return self.root / "pages"

    @property
    def masks(self) -> Path:
        return self.root / "masks"

    @property
    def render(self) -> Path:
        return self.root / "render"

    # ── Files ─────────────────────────────────────────────────────

    @property
    def scan(self) -> Path:
        """scan.npz — bubble geometry for the whole chapter."""
        return self.root / "scan.npz"

    def page(self, index: int) -> Path:
        return self.pages / f"{index:04d}.png"

    def mask(self, page_index: int) -> Path:
        """Per-page mask file — all bubbles for that page."""
        return self.masks / f"{page_index:04d}.npz"

    def rendered(self, index: int) -> Path:
        return self.render / f"{index:04d}.png"

    # ── Stage done — derived from data presence, never stored ─────

    @property
    def is_prepared(self) -> bool:
        return self.pages.exists() and any(self.pages.iterdir())

    @property
    def is_scanned(self) -> bool:
        return self.scan.exists()

    @property
    def is_rendered(self) -> bool:
        return self.render.exists() and any(self.render.iterdir())

    def ensure(self) -> None:
        for d in (self.pages, self.masks, self.render):
            d.mkdir(parents=True, exist_ok=True)


@dataclass(frozen=True)
class ProjectPaths:
    projects_root: Path
    slug:          str

    @property
    def root(self) -> Path:
        return self.projects_root / self.slug

    def chapter(self, chapter_id: int) -> ChapterPaths:
        return ChapterPaths(self.projects_root, self.slug, chapter_id)

    def ensure(self) -> None:
        self.root.mkdir(parents=True, exist_ok=True)


@dataclass(frozen=True)
class Paths:
    root: Path = None  # type: ignore[assignment]

    def __post_init__(self) -> None:
        object.__setattr__(self, "root", (self.root or home()).resolve())

    @property
    def config_file(self) -> Path: return self.root / "config.toml"

    @property
    def db(self) -> Path: return self.root / "typoon.db"

    @property
    def models(self) -> Path: return self.root / "models"

    @property
    def projects(self) -> Path: return self.root / "projects"

    @property
    def cache(self) -> Path: return self.root / "cache"

    def ensure(self) -> None:
        for d in (self.root, self.projects):
            d.mkdir(parents=True, exist_ok=True)
