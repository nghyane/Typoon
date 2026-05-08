"""Chapter domain loaders — assemble domain objects from DB + storage.

Single source of truth per data type:
  PreparedReader    — Bunle archive at ArtifactStore (random-access pixels)
  PreparedChapter   — metadata view derived from the reader's index
  scan.Chapter      — DB geometry + DB bubble text + reader-derived prepared
  translate.Chapter — same as scan, plus DB translations
"""

from __future__ import annotations

from pathlib import Path

from typoon.adapters.artifact_store import ArtifactStore
from typoon.adapters.prepared_reader import PreparedReader
from typoon.domain import scan, translate
from typoon.domain.prepared import Chapter as PreparedChapter
from typoon.domain.scan import BubbleGeometry, PageGeometry
from typoon.storage import Store


async def open_prepared_reader(
    store: ArtifactStore,
    prepared_key: str,
    workdir: Path,
) -> PreparedReader:
    """Download a prepared archive from the store and open it.

    Caller owns workdir lifecycle and reader.close().
    """
    workdir.mkdir(parents=True, exist_ok=True)
    local = workdir / "prepared.bnl"
    await store.get_file(prepared_key, local)
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
) -> tuple[translate.Chapter, dict[int, PageGeometry]]:
    page_geoms   = await _load_page_geometry(db, chapter_id)
    bubbles_db   = await db.get_bubbles(chapter_id)
    translations = await db.get_translations(chapter_id)

    scanned = _build_scanned(reader.chapter(), bubbles_db, page_geoms)

    pages = tuple(
        translate.Page(
            source=sp,
            bubbles=tuple(
                translate.Bubble(
                    source=sb,
                    translation_key=f"p{sb.page_index}_b{sb.idx}",
                    translated_text=translations.get((sb.page_index, sb.idx), {}).get("translated_text", ""),
                    kind=translations.get((sb.page_index, sb.idx), {}).get("kind", "skip"),
                )
                for sb in sp.bubbles
            ),
        )
        for sp in scanned.pages
    )
    return translate.Chapter(scan=scanned, pages=pages), page_geoms


# ── Internal ──────────────────────────────────────────────────────────


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
                    fit_box=b["fit_box"],
                    erase_box=b["erase_box"],
                    text_box=b["text_box"],
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
            box=scan.Box(
                polygon=bg.polygon,
                fit=bg.fit_box,
                erase=bg.erase_box,
                text=bg.text_box,
            ),
            shape_kind=bd.get("shape_kind", "dialogue"),
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
