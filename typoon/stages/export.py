"""Export rendered chapters to user-facing files (PDF, zip, WebP).

The export stage downloads `render.bnl` from the artifact store, decodes
each page, and produces the requested format(s) into a destination
directory. A `manifest.json` records what was produced so subsequent
`typoon export` calls can skip already-fresh outputs.

Outputs are user-facing artifacts; they live under `~/.typoon/exports/`
by default and never collide with the internal `artifacts/` namespace.
"""

from __future__ import annotations

import io
import json
import tempfile
import time
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import bunle
import img2pdf
from PIL import Image

from typoon.adapters.storage_registry import StorageRegistry

ExportFormat = Literal["pdf", "zip", "webp"]
_MANIFEST_NAME = "manifest.json"


@dataclass(frozen=True)
class ExportedChapter:
    chapter_idx: float
    formats: dict[str, Path]   # format → produced file/dir path


@dataclass(frozen=True)
class ChapterRef:
    chapter_id: int
    chapter_idx: float
    archive_backend: str           # which store the archive is in
    archive_locator: str           # opaque locator within that backend
    rendered_at: str | None = None  # opaque marker (DB updated_at, mtime, …)


async def export_chapters(
    *,
    slug: str,
    chapters: list[ChapterRef],
    formats: list[ExportFormat],
    stores: StorageRegistry,
    dest_dir: Path,
    force: bool = False,
) -> list[ExportedChapter]:
    """Export each chapter into `dest_dir/<slug>/` (or directly into `dest_dir`
    when the caller already passed a per-project directory).

    Re-running is idempotent: chapters whose manifest entry is fresh are
    skipped unless `force=True`.
    """
    dest_dir.mkdir(parents=True, exist_ok=True)
    manifest = _load_manifest(dest_dir)
    results: list[ExportedChapter] = []

    for ref in chapters:
        rendered_at = ref.rendered_at or ""
        prior = manifest.get(_chapter_key(ref))
        if not force and prior and _is_fresh(prior, rendered_at, formats):
            results.append(ExportedChapter(
                chapter_idx=ref.chapter_idx,
                formats={fmt: dest_dir / prior["formats"][fmt] for fmt in formats},
            ))
            continue

        with tempfile.TemporaryDirectory() as tmp:
            local_archive = Path(tmp) / "render.bnl"
            reader = stores.reader(ref.archive_backend)
            await reader.get(ref.archive_locator, local_archive)
            produced = _render_outputs(local_archive, dest_dir, slug, ref.chapter_idx, formats)

        manifest[_chapter_key(ref)] = {
            "chapter_id": ref.chapter_id,
            "chapter_idx": ref.chapter_idx,
            "rendered_at": rendered_at,
            "exported_at": _now_iso(),
            "formats": {fmt: str(p.relative_to(dest_dir)) for fmt, p in produced.items()},
        }
        results.append(ExportedChapter(chapter_idx=ref.chapter_idx, formats=produced))

    _save_manifest(dest_dir, manifest)
    return results


# ── Output writers ────────────────────────────────────────────────────


def _render_outputs(
    archive_path: Path,
    dest_dir: Path,
    slug: str,
    chapter_idx: float,
    formats: list[ExportFormat],
) -> dict[str, Path]:
    out: dict[str, Path] = {}
    with bunle.Reader(str(archive_path)) as reader:
        page_bytes = [reader.page(i) for i in range(reader.page_count)]

    base = f"ch{chapter_idx:.4g}"
    if "pdf" in formats:
        out["pdf"] = _write_pdf(dest_dir / f"{base}.pdf", page_bytes)
    if "zip" in formats:
        out["zip"] = _write_zip(dest_dir / f"{base}.zip", page_bytes)
    if "webp" in formats:
        out["webp"] = _write_webp_dir(dest_dir / base, page_bytes)
    return out


def _write_pdf(path: Path, page_bytes: list[bytes]) -> Path:
    # Re-encode each page as JPEG q=95 with no chroma subsampling, then
    # let img2pdf passthrough the JPEG bytes verbatim into the PDF.
    jpeg_pages: list[bytes] = []
    for raw in page_bytes:
        with Image.open(io.BytesIO(raw)) as img:
            buf = io.BytesIO()
            img.convert("RGB").save(buf, format="JPEG", quality=95, subsampling=0)
            jpeg_pages.append(buf.getvalue())
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as f:
        f.write(img2pdf.convert(jpeg_pages))
    return path


def _write_zip(path: Path, page_bytes: list[bytes]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_STORED) as zf:
        for i, raw in enumerate(page_bytes):
            zf.writestr(f"{i:04d}.webp", raw)
    return path


def _write_webp_dir(out_dir: Path, page_bytes: list[bytes]) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    for i, raw in enumerate(page_bytes):
        (out_dir / f"{i:04d}.webp").write_bytes(raw)
    return out_dir


# ── Manifest ──────────────────────────────────────────────────────────


def _load_manifest(dest_dir: Path) -> dict:
    path = dest_dir / _MANIFEST_NAME
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text())
    except json.JSONDecodeError:
        return {}


def _save_manifest(dest_dir: Path, manifest: dict) -> None:
    path = dest_dir / _MANIFEST_NAME
    path.write_text(json.dumps(manifest, indent=2, sort_keys=True))


def _chapter_key(ref: ChapterRef) -> str:
    return f"{ref.chapter_idx:.4g}"


def _is_fresh(entry: dict, rendered_at: str, formats: list[str]) -> bool:
    if entry.get("rendered_at", "") != rendered_at:
        return False
    have = set(entry.get("formats", {}).keys())
    return set(formats).issubset(have)


def _now_iso() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%S")
