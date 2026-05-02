"""Chapter domain loaders — assemble domain objects from DB + filesystem.

Single source of truth per data type:
  PreparedChapter   — filesystem (pages/ directory)
  ScannedChapter    — filesystem (scan.npz geometry) + DB (bubble text)
  TranslatedChapter — filesystem (scan.npz) + DB (bubbles + translations)
"""

from __future__ import annotations

from typoon.adapters.mask_store import PageGeometry, load_scan_geometry
from typoon.domain.prepared import Chapter as PreparedChapter
from typoon.domain.scan import Box, Bubble, Chapter as ScannedChapter, Page as ScannedPage
from typoon.domain.translate import (
    Bubble as TranslatedBubble,
    Chapter as TranslatedChapter,
    Page as TranslatedPage,
)
from typoon.paths import ChapterPaths
from typoon.storage.store import Store


def load_prepared(cp: ChapterPaths) -> PreparedChapter:
    return PreparedChapter.from_paths(cp)


async def load_scanned(cp: ChapterPaths, db: Store, chapter_id: int) -> ScannedChapter:
    prepared   = load_prepared(cp)
    bubbles_db = await db.get_bubbles(chapter_id)
    page_geoms = {pg.page_index: pg for pg in load_scan_geometry(cp)}
    return _build_scanned(prepared, bubbles_db, page_geoms)


async def load_translated_with_geometry(
    cp: ChapterPaths,
    db: Store,
    chapter_id: int,
) -> tuple[TranslatedChapter, dict[int, PageGeometry]]:
    """Load TranslatedChapter and geometry in one pass — scan.npz read once."""
    page_geoms   = {pg.page_index: pg for pg in load_scan_geometry(cp)}
    bubbles_db   = await db.get_bubbles(chapter_id)
    translations = await db.get_translations(chapter_id)
    prepared     = load_prepared(cp)

    scanned = _build_scanned(prepared, bubbles_db, page_geoms)

    pages = tuple(
        TranslatedPage(
            source=sp,
            bubbles=tuple(
                TranslatedBubble(
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
    return TranslatedChapter(scan=scanned, pages=pages), page_geoms


# ── Internal ──────────────────────────────────────────────────────────


def _build_scanned(
    prepared: PreparedChapter,
    bubbles_db: list[dict],
    page_geoms: dict[int, PageGeometry],
) -> ScannedChapter:
    geom_idx = {
        (pg.page_index, bg.bubble_idx): bg
        for pg in page_geoms.values()
        for bg in pg.bubbles
    }

    pages_bubbles: dict[int, list[Bubble]] = {}
    for bd in bubbles_db:
        pi, bi = bd["page_index"], bd["bubble_idx"]
        bg = geom_idx.get((pi, bi))
        if bg is None:
            continue
        pages_bubbles.setdefault(pi, []).append(Bubble(
            idx=bi,
            page_index=pi,
            source_text=bd["source_text"],
            confidence=bd["confidence"],
            box=Box(
                polygon=bg.polygon,
                fit=bg.fit_box,
                erase=bg.erase_box,
                text=bg.text_box,
            ),
        ))

    pages = tuple(
        ScannedPage(
            index=pi,
            width=page_geoms[pi].width,
            height=page_geoms[pi].height,
            bubbles=tuple(pages_bubbles.get(pi, [])),
        )
        for pi in sorted(page_geoms)
    )
    return ScannedChapter(prepared=prepared, pages=pages)
