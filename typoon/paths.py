"""App path resolution.

Chapter pixel and mask data live in the artifact store under deterministic
keys. The filesystem layout under `~/.typoon/` carries only:

  config.toml        — app config
  typoon.db          — SQLite knowledge store
  models/            — model weights
  artifacts/         — LocalArtifactStore root for prepared.bnl / render.bnl / masks.npz
  exports/<slug>/    — user-facing output (PDF / zip / WebP) produced by `typoon export`
  projects/<slug>/   — per-project metadata (cover image, etc.); no per-chapter dirs
  cache/             — transient

`ChapterPaths` is intentionally not exposed: chapter-level filesystem
scoping is dead. Workers use `ArtifactStore` keys built by
`adapters.chapter_archive` instead.
"""

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
class ProjectPaths:
    projects_root: Path
    slug:          str

    @property
    def root(self) -> Path:
        return self.projects_root / self.slug

    @property
    def cover(self) -> Path:
        return self.root / "cover.jpg"

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
    def artifacts(self) -> Path: return self.root / "artifacts"

    @property
    def exports(self) -> Path: return self.root / "exports"

    @property
    def cache(self) -> Path: return self.root / "cache"

    def ensure(self) -> None:
        for d in (self.root, self.projects, self.artifacts, self.exports):
            d.mkdir(parents=True, exist_ok=True)
