"""App path resolution.

Chapter pixel and mask data live in BlobStore / ArtifactStore backends
under deterministic keys. The DB is Postgres, reached via
`DATABASE_URL`, so no DB file lives in `~/.typoon/`. Filesystem layout:

  config.toml        — app config
  models/            — model weights
  artifacts/         — LocalBlobStore / LocalArtifactStore root.
                       Single-host: holds prepared.bnl, masks.npz, and
                       render.bnl (served via /files mount).
                       Multi-host: only the storage role's API host
                       fills this with pipeline blobs; render.bnl
                       lives on the configured public store (HF/R2/...).
  exports/<slug>/    — user-facing output (PDF / zip / WebP) produced
                       by `typoon export`
  projects/<slug>/   — per-project metadata (cover image, etc.);
                       no per-chapter dirs
  cache/             — transient

`ChapterPaths` is intentionally not exposed: chapter-level filesystem
scoping is dead. Workers use BlobStore keys built by
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
