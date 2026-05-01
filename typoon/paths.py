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
    """Fixed app data directory. Cached after first call."""
    global _HOME
    if _HOME is None:
        _HOME = Path(os.environ.get("TYPOON_HOME", "~/.typoon")).expanduser()
    return _HOME


def slugify(title: str, url: str = "") -> str:
    """Filesystem-safe slug. Appends URL hash when provided to avoid collisions."""
    base = _re.sub(r"[\s]+", "-", _re.sub(r"[^\w\s-]", "", title.lower().strip())).strip("-")
    if not base:
        base = f"unnamed-{int(_time.time())}"
    return f"{base}-{hashlib.md5(url.encode()).hexdigest()[:6]}" if url else base


def ch_label(idx: float) -> str:
    """Chapter directory label: 1.0 → 'ch001', 1.5 → 'ch0001.5'."""
    return f"ch{int(idx):03d}" if idx == int(idx) else f"ch{idx:06.1f}"


@dataclass(frozen=True)
class ChapterPaths:
    """All file paths for one chapter. Derived — never stored in DB."""

    projects_root: Path
    slug:          str
    idx:           float

    @property
    def root(self) -> Path:
        return self.projects_root / self.slug / ch_label(self.idx)

    @property
    def pages(self) -> Path:
        return self.root / "pages"

    @property
    def manifest(self) -> Path:
        return self.root / "manifest.json"

    @property
    def scan(self) -> Path:
        return self.root / "scan.json"

    @property
    def masks(self) -> Path:
        return self.root / "masks"

    @property
    def translate(self) -> Path:
        return self.root / "translate.json"

    @property
    def render(self) -> Path:
        return self.root / "render"

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


@dataclass(frozen=True)
class ProjectPaths:
    """Paths for one project."""

    projects_root: Path
    slug:          str

    @property
    def root(self) -> Path:
        return self.projects_root / self.slug

    def chapter(self, idx: float) -> ChapterPaths:
        return ChapterPaths(self.projects_root, self.slug, idx)

    def ensure(self) -> None:
        self.root.mkdir(parents=True, exist_ok=True)


@dataclass(frozen=True)
class Paths:
    """All app paths resolved from home directory."""

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

    def ensure(self) -> None:
        for d in (self.root, self.projects):
            d.mkdir(parents=True, exist_ok=True)
