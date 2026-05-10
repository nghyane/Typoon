"""Unzip uploaded chapter archives into a flat folder of pages.

The browser (web SPA + extension) ships chapters as a single store-mode
zip via the multipart inbox path. This module turns that zip into a
flat folder of `NNNN.<ext>` files in reading order. The prepare worker
then hands that folder to `prepare_chapter_to_archive`.

Folder ingest is HTTP-only: the legacy `typoon add /folder/` shortcut
was removed when the multipart inbox flow became the single ingest
path.

This module never touches the network. The Discord bot / browser
extension owns scraping; this is the only path data enters the engine
via the API.
"""

from __future__ import annotations

import io
import logging
import re
import zipfile
from pathlib import Path

from .constants import IMAGE_EXTS

logger = logging.getLogger(__name__)


class UnpackError(ValueError):
    """Raised when an upload doesn't contain any usable images."""


def unpack_zip(data: bytes, dest: Path) -> int:
    """Extract image entries to dest/. Returns count of pages written.

    Order: natural sort of in-archive paths (so 'page-2' < 'page-10').
    Hidden files (`__MACOSX`, leading dot) are ignored.
    """
    dest.mkdir(parents=True, exist_ok=True)
    try:
        zf = zipfile.ZipFile(io.BytesIO(data))
    except zipfile.BadZipFile as e:
        raise UnpackError(f"Not a valid zip/cbz: {e}") from e

    entries: list[tuple[str, zipfile.ZipInfo]] = []
    for info in zf.infolist():
        if info.is_dir():
            continue
        name = info.filename
        if "__MACOSX" in name or name.rsplit("/", 1)[-1].startswith("."):
            continue
        suffix = "." + name.rsplit(".", 1)[-1].lower() if "." in name else ""
        if suffix not in IMAGE_EXTS:
            continue
        entries.append((name, info))

    entries.sort(key=lambda x: _natural_key(x[0]))
    if not entries:
        raise UnpackError("Archive contained no images")

    for i, (orig_name, info) in enumerate(entries):
        suffix = "." + orig_name.rsplit(".", 1)[-1].lower()
        out = dest / f"{i + 1:04d}{suffix}"
        with zf.open(info) as src, out.open("wb") as dst:
            dst.write(src.read())
    return len(entries)


# ── Helpers ───────────────────────────────────────────────────────────


_NATURAL_SPLIT = re.compile(r"(\d+)")


def _natural_key(name: str) -> tuple:
    """Sort key for natural alphanumeric ordering ('p2' < 'p10')."""
    return tuple(
        int(part) if part.isdigit() else part.lower()
        for part in _NATURAL_SPLIT.split(name)
    )
