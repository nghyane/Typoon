"""Render stage — TranslatedChapter + masks → RenderedChapter on disk.

Async, page-parallel. Erase + render run in asyncio.to_thread (GIL-free
backends), gated by runtime.erase_gate. JPEG q=92 encoding matches the
prepare encoder so bunle stores byte-identical.
"""

from __future__ import annotations

import asyncio
import logging
from pathlib import Path

import cv2
import numpy as np

from typoon.adapters.mask_store import MaskStore
from typoon.adapters.prepared_reader import PreparedReader
from typoon.vision.runtime import VisionRuntime
from typoon.domain import render, translate
from typoon.domain.scan import PageGeometry
from typoon.runs.artifacts import ArtifactSink
from typoon.runs.events import Hook, PageDone


__all__ = ["render_chapter"]

logger = logging.getLogger(__name__)


async def render_chapter(
    translated: translate.Chapter,
    out_dir: Path,
    reader: PreparedReader,
    runtime: VisionRuntime,
    page_geoms: dict[int, PageGeometry],
    masks: MaskStore,
    *,
    chapter_id:  int = 0,
    target_kind: str = "draft",
    target_id:   int = 0,
    hook: Hook | None = None,
    artifacts: ArtifactSink | None = None,
    skip_pages: frozenset[int] = frozenset(),
) -> render.Chapter:
    """Erase source text, render translations, write JPEG pages.

    Pages render concurrently up to runtime.runtime.erase_gate. Output
    indices renumber contiguously to skip dropped pages.
    """
    out_dir.mkdir(parents=True, exist_ok=True)

    # Determine output index per source page (skip-aware).
    keep_pages = [tp for tp in translated.pages if tp.index not in skip_pages]
    out_indices = {tp.index: i for i, tp in enumerate(keep_pages)}

    rendered: list[render.Page | None] = [None] * len(translated.pages)
    total = len(translated.pages)

    async def render_one(slot: int, tp: translate.Page) -> None:
        if tp.index in skip_pages:
            if artifacts is not None:
                image = await asyncio.to_thread(reader.read_rgb, tp.index)
                await asyncio.to_thread(
                    artifacts.write_image, "07_render",
                    f"{tp.index:04d}_dropped.png", image,
                )
            if hook is not None:
                hook.on(_page_done(
                    chapter_id=chapter_id, target_kind=target_kind,
                    target_id=target_id, page_index=tp.index, page_total=total,
                ))
            return

        page_out = await _render_one_page(
            tp, reader, runtime, page_geoms, masks,
            out_dir / f"{out_indices[tp.index]:04d}.jpg",
            artifacts,
        )
        rendered[slot] = page_out

        if hook is not None:
            hook.on(_page_done(
                chapter_id=chapter_id, target_kind=target_kind,
                target_id=target_id, page_index=tp.index, page_total=total,
            ))

    async with asyncio.TaskGroup() as tg:
        for slot, tp in enumerate(translated.pages):
            tg.create_task(render_one(slot, tp))

    return render.Chapter(
        source=translated,
        pages=tuple(p for p in rendered if p is not None),
    )


async def _render_one_page(
    tp: translate.Page,
    reader: PreparedReader,
    runtime: VisionRuntime,
    page_geoms: dict[int, PageGeometry],
    masks: MaskStore,
    out_path: Path,
    artifacts: ArtifactSink | None,
) -> render.Page:
    import typoon_render

    original = await asyncio.to_thread(reader.read_rgb, tp.index)
    canvas = _to_rgba(original)
    # `original` is just the source for `canvas`; we no longer pass it
    # separately to the renderer. Drop the reference so numpy can reclaim
    # the buffer if the GC runs before render_page completes.
    del original
    page_masks = masks.page_masks(tp.index)

    # Single predicate drives BOTH erase and re-render: LLM marks
    # logo/credit/page-number/watermark bubbles as `kind="skip"`; their
    # pixels must stay untouched and no Vietnamese text is laid over
    # them. Keeping one predicate in one place prevents the two paths
    # from drifting (which caused the earlier "logo erased but no text"
    # regression).
    renderable_bubbles = [tb for tb in tp.bubbles if _is_renderable(tb)]

    canvas_h, canvas_w = canvas.shape[:2]
    erase_masks = tuple(
        m
        for tb in renderable_bubbles
        for bm in [page_masks.get(tb.idx)]
        if bm is not None
        for m in bm.erase_masks
        if _mask_in_bounds(m, canvas_w, canvas_h)
    )
    if erase_masks:
        async with runtime.erase_gate:
            await runtime.eraser.erase(canvas, erase_masks)

    # Inpaint output IS the render input. The eraser mutated `canvas`
    # in place; the renderer's border-inset detector reads bubble
    # outlines from the same RGBA buffer (inpaint preserves them).
    #
    # Dump the inpainted intermediate before render writes glyphs on top
    # — the only visual signal for AOT erase quality (text-bleed,
    # over-erase onto art, fallback colour). Production worker passes
    # `artifacts=None`, so no overhead in the hot path.
    if artifacts is not None:
        await asyncio.to_thread(
            artifacts.write_image, "07_render",
            f"{tp.index:04d}_inpainted.png", canvas,
        )

    pg_geom = page_geoms.get(tp.index)
    geom_by_idx = {bg.bubble_idx: bg for bg in pg_geom.bubbles} if pg_geom else {}

    active_triples = [
        (tb, geom_by_idx[tb.idx], tb.translated_text)
        for tb in renderable_bubbles
        if tb.translated_text.strip()
        and tb.idx in geom_by_idx
    ]
    active   = [t[0] for t in active_triples]
    polygons = [t[1].polygon for t in active_triples]
    texts    = [t[2] for t in active_triples]
    hints    = [_geom_to_hint(t[1]) for t in active_triples]

    # Polygon already carries the drawable area (Lens word union +
    # padding); no border scan, no bubble-mask expansion needed.
    result = await asyncio.to_thread(
        typoon_render.typoon_render.render,
        canvas, polygons, texts, canvas.shape[1], hints,
    )

    active_info = {tb.idx: rb for tb, rb in zip(active, result.bubbles)}
    rendered_bubbles = tuple(
        render.Bubble(
            source=tb,
            font_size=active_info[tb.idx].font_size_px if tb.idx in active_info else 0,
            overflow=active_info[tb.idx].overflow if tb.idx in active_info else False,
        )
        for tb in tp.bubbles
    )

    await asyncio.to_thread(_write_jpeg, out_path, result.image)
    if artifacts is not None:
        await asyncio.to_thread(
            artifacts.write_image, "07_render",
            f"{tp.index:04d}_rendered.png", result.image,
        )

    return render.Page(source=tp, bubbles=rendered_bubbles)


# ─── Helpers ──────────────────────────────────────────────────────────────


def _page_done(
    *,
    chapter_id:  int,
    target_kind: str,
    target_id:   int,
    page_index:  int,
    page_total:  int,
) -> PageDone:
    if target_kind == "draft":
        return PageDone(
            chapter_id=chapter_id, draft_id=target_id, stage="render",
            page_index=page_index, page_total=page_total,
        )
    return PageDone(
        chapter_id=chapter_id, translation_id=target_id, stage="render",
        page_index=page_index, page_total=page_total,
    )


def _to_rgba(image: np.ndarray) -> np.ndarray:
    h, w = image.shape[:2]
    return np.dstack([image, np.full((h, w), 255, dtype=np.uint8)])


def _mask_in_bounds(m, canvas_w: int, canvas_h: int) -> bool:
    """True when the mask has at least 1 pixel inside the canvas.

    Masks that land entirely outside the canvas are silently dropped —
    they produce no visible effect and would cause index errors in the
    eraser's background sampler if passed through.
    """
    mh, mw = m.image.shape[:2]
    return m.x < canvas_w and m.y < canvas_h and m.x + mw > 0 and m.y + mh > 0


def _is_renderable(tb: translate.Bubble) -> bool:
    """True if a bubble should be erased + re-rendered.

    The render path has two consumer sites (erase mask gather +
    active_triples build) that must agree on which bubbles to touch.
    Centralising the predicate prevents the "erase but don't render"
    regression where logos got wiped to grey but had no Vietnamese
    text replacing them.

    Currently the only skip signal is `kind == "skip"`, set by:
      - `noise.is_auto_skip` upstream (URL / handle / page-counter /
        platform brand patterns)
      - `page._parse_blocks` defense-in-depth when the LLM emits a
        block for a deterministically-noisy bubble
      - LLM itself emitting `@@ KEY skip` (story-level decision)

    Add new skip categories by extending this predicate, not by adding
    parallel filters at the call sites.
    """
    return tb.kind != "skip"


def _geom_to_hint(geom):
    """Convert BubbleGeometry → typoon_render.TypesettingHint (or None).

    Returns None when the detector did not surface line geometry
    (`src_font_size_px == 0`), letting render's fit fall back to pure
    binary search.

    Passes `text_direction` so Rust can skip hint refinement for
    vertical-source (Japanese tategaki) bubbles.
    """
    if geom.src_font_size_px <= 0 or geom.src_line_count <= 0:
        return None
    import typoon_render
    return typoon_render.typoon_render.TypesettingHint(
        font_size_px=geom.src_font_size_px,
        line_count=geom.src_line_count,
        avg_chars_per_line=geom.src_avg_chars_per_line,
        text_direction=geom.text_direction,
    )


def _write_jpeg(path: Path, rgba: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    # Renderer outputs RGBA; JPEG is RGB-only. Drop alpha at the
    # encoder boundary, not at the renderer — keeps the rest of the
    # pipeline single-format.
    bgr = cv2.cvtColor(rgba, cv2.COLOR_RGBA2BGR)
    if not cv2.imwrite(
        str(path), bgr,
        [cv2.IMWRITE_JPEG_QUALITY, 92, cv2.IMWRITE_JPEG_OPTIMIZE, 1],
    ):
        raise RuntimeError(f"Failed to write JPEG: {path}")
