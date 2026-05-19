"""Chapter domain loaders — assemble domain objects from DB + storage.

Single source of truth per data type:
  PreparedReader    — Bunle archive at ArtifactStore (random-access pixels)
  PreparedChapter   — metadata view derived from the reader's index
  scan.Chapter      — DB geometry + DB bubble text + reader-derived prepared
  translate.Chapter — same as scan, plus DB translations

NFC normalisation is enforced here for `translated_text` so render
never sees decomposed Vietnamese (NFD), which the embedded font cannot
render properly. Some LLM responses and stored edits arrive in NFD;
catching them at the load boundary keeps the render path simple.
"""

from __future__ import annotations

import unicodedata
from pathlib import Path

from typoon.adapters.blob_store import BlobStore
from typoon.adapters.prepared_reader import PreparedReader
from typoon.domain import scan, translate
from typoon.domain.prepared import Chapter as PreparedChapter
from typoon.domain.scan import BubbleGeometry, PageGeometry
from typoon.storage.store import Store


async def open_prepared_reader(
    store: BlobStore,
    prepared_key: str,
    workdir: Path,
) -> PreparedReader:
    """Download a prepared archive from the store and open it.

    Caller owns workdir lifecycle and reader.close().
    """
    workdir.mkdir(parents=True, exist_ok=True)
    local = workdir / "prepared.bnl"
    await store.get(prepared_key, local)
    return PreparedReader.open(local)


async def load_scanned(
    reader: PreparedReader,
    db: Store,
    chapter_id: int,
) -> scan.Chapter:
    page_geoms = await _load_page_geometry(db, chapter_id)
    bubbles_db = await db.get_bubbles(chapter_id)
    return _build_scanned(reader.chapter(), bubbles_db, page_geoms)


async def load_translated_with_geometry(
    reader: PreparedReader,
    db: Store,
    chapter_id: int,
    draft_id: int,
    *,
    translation_id: int | None = None,
) -> tuple[translate.Chapter, dict[int, PageGeometry]]:
    """Assemble a translate.Chapter from chapter-level scan output +
    draft-level translation bubbles, with optional sparse edits
    overlaid from a per-user translation row.

    Geometry and source bubbles come from the chapter (shared across
    every translation). Translated text comes from the draft. Edits,
    when provided, override draft bubbles by (page_index, bubble_idx).
    """
    page_geoms = await _load_page_geometry(db, chapter_id)
    bubbles_db = await db.get_bubbles(chapter_id)
    draft_bubbles = await db.get_draft_bubbles(draft_id)

    # Index draft text + (optionally) edits by (page, idx) so the
    # per-page assembly below is a dict lookup, not a scan.
    text_by_pos: dict[tuple[int, int], dict] = {
        (b["page_index"], b["bubble_idx"]): b for b in draft_bubbles
    }
    if translation_id is not None:
        for e in await db.get_translation_edits(translation_id):
            key = (e["page_index"], e["bubble_idx"])
            if key in text_by_pos:
                # Override the draft text; keep its kind.
                text_by_pos[key] = {
                    **text_by_pos[key],
                    "translated_text": e["edited_text"],
                }
            else:
                # Edit on a bubble the draft skipped — surface as a
                # synthetic "dialogue" override.
                text_by_pos[key] = {
                    "page_index": e["page_index"],
                    "bubble_idx": e["bubble_idx"],
                    "translated_text": e["edited_text"],
                    "kind": "dialogue",
                }

    scanned = _build_scanned(reader.chapter(), bubbles_db, page_geoms)

    pages = tuple(
        translate.Page(
            source=sp,
            bubbles=tuple(
                translate.Bubble(
                    source=sb,
                    translation_key=f"p{sb.page_index}_b{sb.idx}",
                    translated_text=_normalize_for_render(
                        text_by_pos.get(
                            (sb.page_index, sb.idx), {},
                        ).get("translated_text", "")
                    ),
                    kind=text_by_pos.get(
                        (sb.page_index, sb.idx), {},
                    ).get("kind", "skip"),
                )
                for sb in sp.bubbles
            ),
        )
        for sp in scanned.pages
    )
    return translate.Chapter(scan=scanned, pages=pages), page_geoms


def _normalize_for_render(text: str) -> str:
    """Force NFC so combining diacritics merge into precomposed glyphs.

    LLM providers (and human translation edits) can emit Vietnamese in
    decomposed form (NFD): `nghĩa` arrives as `n g h i \\u0303 a`. The
    embedded render font has glyphs only for precomposed codepoints, so
    combining marks render as blank advances → visible as phantom
    spaces in the output. NFC normalisation is a no-op on text that's
    already precomposed.

    Applied at the loader boundary (not at translate-stage save) so
    historical bubbles in the DB also get the fix on read.
    """
    if not text:
        return text
    return unicodedata.normalize("NFC", text)


async def _load_page_geometry(db: Store, chapter_id: int) -> dict[int, PageGeometry]:
    rows = await db.get_geometry(chapter_id)
    return {
        p["page_index"]: PageGeometry(
            page_index=p["page_index"],
            width=p["width"],
            height=p["height"],
            bubbles=tuple(
                BubbleGeometry(
                    bubble_idx=b["bubble_idx"],
                    polygon=b["polygon"],
                    rotation_deg=b.get("rotation_deg", 0.0),
                    src_font_size_px=b.get("src_font_size_px", 0),
                    src_line_count=b.get("src_line_count", 0),
                    src_avg_chars_per_line=b.get("src_avg_chars_per_line", 0.0),
                    text_direction=b.get("text_direction", "horizontal"),
                )
                for b in p["bubbles"]
            ),
        )
        for p in rows
    }


def _build_scanned(
    prepared: PreparedChapter,
    bubbles_db: list[dict],
    page_geoms: dict[int, PageGeometry],
) -> scan.Chapter:
    geom_idx = {
        (pg.page_index, bg.bubble_idx): bg
        for pg in page_geoms.values()
        for bg in pg.bubbles
    }

    pages_bubbles: dict[int, list[scan.Bubble]] = {}
    for bd in bubbles_db:
        pi, bi = bd["page_index"], bd["bubble_idx"]
        bg = geom_idx.get((pi, bi))
        if bg is None:
            continue
        pages_bubbles.setdefault(pi, []).append(scan.Bubble(
            idx=bi,
            page_index=pi,
            source_text=bd["source_text"],
            confidence=bd["confidence"],
            polygon=bg.polygon,
            shape_kind=bd.get("shape_kind", "dialogue"),
            rotation_deg=bd.get("rotation_deg", 0.0),
            src_font_size_px=bd.get("src_font_size_px", 0),
            src_line_count=bd.get("src_line_count", 0),
            src_avg_chars_per_line=bd.get("src_avg_chars_per_line", 0.0),
            text_direction=bd.get("text_direction", "horizontal"),
        ))

    pages = tuple(
        scan.Page(
            index=pi,
            width=page_geoms[pi].width,
            height=page_geoms[pi].height,
            bubbles=tuple(pages_bubbles.get(pi, [])),
        )
        for pi in sorted(page_geoms)
    )
    return scan.Chapter(prepared=prepared, pages=pages)
