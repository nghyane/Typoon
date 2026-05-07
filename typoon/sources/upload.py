"""Unpack user uploads into a flat folder of pages.

Three input shapes are accepted by the API upload endpoint:

  application/zip  / .cbz / .cbr  → unzip image files
  application/pdf                  → render each page to WEBP at 200 DPI
  image/* (multiple files)         → write to disk in the order given

All variants converge on the same output: a temp folder containing
NNNN.webp (or original extension) named in reading order. The caller
hands that folder to `Projects.ingest_chapter(...)`.

This module never touches the network. The Discord bot / browser
extension owns scraping; this is the only path data enters the engine.
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


# ── Detection ─────────────────────────────────────────────────────────


def detect_kind(filename: str, content_type: str | None) -> str:
    """Return one of: 'pdf', 'zip', 'image', or raise UnpackError.

    Filename is lowercased before suffix check; CBZ/CBR are zip
    in disguise. Content-Type is consulted as a hint when the filename
    doesn't carry a useful extension (e.g. some browsers strip them).
    """
    name = filename.lower()
    if name.endswith(".pdf") or content_type == "application/pdf":
        return "pdf"
    if name.endswith((".zip", ".cbz", ".cbr")) or content_type in (
        "application/zip", "application/x-cbr", "application/x-cbz",
    ):
        return "zip"
    suffix = "." + name.rsplit(".", 1)[-1] if "." in name else ""
    if suffix in IMAGE_EXTS or (content_type and content_type.startswith("image/")):
        return "image"
    raise UnpackError(f"Unsupported file type: {filename} ({content_type})")


# ── Unpackers ─────────────────────────────────────────────────────────


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


def unpack_pdf(data: bytes, dest: Path, *, dpi: int = 200) -> int:
    """Render every PDF page to a WEBP file. Returns page count.

    Default 200 DPI is a quality/size sweet spot for Kindle-sourced
    manga PDFs (text legible, files ~600KB/page). Lossless WEBP keeps
    downstream pipeline assumptions intact.
    """
    try:
        import pypdfium2 as pdfium  # noqa: PLC0415
    except ModuleNotFoundError as e:
        raise UnpackError(
            "PDF support requires pypdfium2. "
            "Install with: pip install pypdfium2",
        ) from e

    dest.mkdir(parents=True, exist_ok=True)
    try:
        pdf = pdfium.PdfDocument(data)
    except pdfium.PdfiumError as e:
        raise UnpackError(f"Not a valid PDF (or DRM-protected): {e}") from e

    n = len(pdf)
    if n == 0:
        raise UnpackError("PDF has no pages")

    scale = dpi / 72.0
    for i in range(n):
        page = pdf[i]
        bitmap = page.render(scale=scale)
        pil = bitmap.to_pil()
        out = dest / f"{i + 1:04d}.webp"
        # lossless to keep downstream prepare/scan deterministic
        pil.save(out, format="WEBP", lossless=True, quality=100)
    return n


def write_image_files(
    files: list[tuple[str, bytes]], dest: Path,
) -> int:
    """Write a list of (filename, bytes) to dest/ in the given order.

    Caller is responsible for sort order; the API endpoint sorts by
    the original filename so user-supplied numbering is honoured.
    """
    dest.mkdir(parents=True, exist_ok=True)
    n = 0
    for i, (name, data) in enumerate(files):
        suffix = "." + name.rsplit(".", 1)[-1].lower() if "." in name else ".webp"
        if suffix not in IMAGE_EXTS:
            continue
        out = dest / f"{i + 1:04d}{suffix}"
        out.write_bytes(data)
        n += 1
    if n == 0:
        raise UnpackError("No image files in upload")
    return n


# ── Helpers ───────────────────────────────────────────────────────────


_NATURAL_SPLIT = re.compile(r"(\d+)")


def _natural_key(name: str) -> tuple:
    """Sort key for natural alphanumeric ordering ('p2' < 'p10')."""
    return tuple(
        int(part) if part.isdigit() else part.lower()
        for part in _NATURAL_SPLIT.split(name)
    )
