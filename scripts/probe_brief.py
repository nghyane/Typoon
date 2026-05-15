"""Probe: test new context agent format (GLOSSARY / ADDRESS / BRIEF / voice)
against a real chapter storyboard via live API.

Usage:
    python scripts/probe_brief.py [CHAPTER_ID]

Defaults to chapter 14 (manhua, zh→vi, color). Writes debug artifacts
under debug-runs/probe_brief_<chapter_id>/ so you can inspect:
  - storyboard_system.txt  (full system prompt sent to context agent)
  - 05_brief/storyboard_*.jpg
  - 05_brief/reply_*.txt   (raw agent reply)
  - brief.json             (parsed ChapterBrief)
  - sample_translator_context.txt (brief_slice for first window)
"""

from __future__ import annotations

import asyncio
import json
import os
import sys
import tempfile
from pathlib import Path

from dotenv import load_dotenv
load_dotenv()

sys.path.insert(0, str(Path(__file__).parent.parent))

import asyncpg

from typoon.config import load_config
from typoon.adapters.loader import open_prepared_reader
from typoon.adapters.chapter_archive import prepared_key
from typoon.adapters.storage_registry import build_storage
from typoon.runs.artifacts import FileArtifactSink
from typoon.runs.events import LLMCall, LLMResponse
from typoon.stages.brief import build_chapter_brief, brief_slice
from typoon.stages.keys import assign_keys
from typoon.stages import prompt as prompt_mod
from typoon.providers import make_translation_provider, make_vision_provider


class _PrintHook:
    def on(self, event):
        if isinstance(event, LLMCall):
            print(f"  → LLM call: {event.agent}")
        elif isinstance(event, LLMResponse):
            print(f"  ← done: {event.agent} ({event.ms:.0f}ms)")


async def _load_chapter(chapter_id: int, store, tmp: Path):
    """Load chapter metadata, open PreparedReader, build keyed bubbles."""
    conn = await asyncpg.connect(os.environ["DATABASE_URL"])
    try:
        ch = await conn.fetchrow(
            """
            SELECT c.id, c.source_lang, d.target_lang
            FROM chapters c
            LEFT JOIN translation_drafts d ON d.chapter_id = c.id
            WHERE c.id = $1
            ORDER BY d.created_at DESC
            LIMIT 1
            """,
            chapter_id,
        )
        if ch is None:
            raise SystemExit(f"Chapter {chapter_id} not found")

        pages_rows = await conn.fetch(
            "SELECT page_index, width, height FROM page_geometry "
            "WHERE chapter_id=$1 ORDER BY page_index",
            chapter_id,
        )
        bubbles_rows = await conn.fetch(
            "SELECT page_index, bubble_idx, source_text, shape_kind, confidence "
            "FROM bubbles WHERE chapter_id=$1 ORDER BY page_index, bubble_idx",
            chapter_id,
        )
        geom_rows = await conn.fetch(
            "SELECT page_index, bubble_idx, polygon, fit_box, erase_box, text_box "
            "FROM bubble_geometry WHERE chapter_id=$1 "
            "ORDER BY page_index, bubble_idx",
            chapter_id,
        )
    finally:
        await conn.close()

    source_lang = ch["source_lang"] or "zh"
    target_lang = ch["target_lang"] or "vi"

    # Build geometry lookup
    import json as _json
    def _parse(v):
        return _json.loads(v) if isinstance(v, str) else (list(v) if v else None)

    geom_map: dict[tuple[int, int], dict] = {}
    for g in geom_rows:
        geom_map[(g["page_index"], g["bubble_idx"])] = {
            "polygon":  _parse(g["polygon"]) or [],
            "fit":      _parse(g["fit_box"]) or [0, 0, 10, 10],
            "erase":    _parse(g["erase_box"]) or [0, 0, 10, 10],
            "text":     _parse(g["text_box"]) or [0, 0, 10, 10],
        }

    from typoon.domain.scan import Box, Bubble, Chapter as ScanChapter, Page

    page_map: dict[int, dict] = {r["page_index"]: r for r in pages_rows}
    bubbles_by_page: dict[int, list[Bubble]] = {}
    for b in bubbles_rows:
        pi  = b["page_index"]
        geo = geom_map.get((pi, b["bubble_idx"]), {
            "polygon": [], "fit": [0, 0, 10, 10],
            "erase": [0, 0, 10, 10], "text": [0, 0, 10, 10],
        })
        bubble = Bubble(
            idx=b["bubble_idx"],
            page_index=pi,
            source_text=b["source_text"] or "",
            confidence=float(b["confidence"] or 0.9),
            box=Box(
                polygon=geo["polygon"],
                fit=geo["fit"],
                erase=geo["erase"],
                text=geo["text"],
            ),
            shape_kind=b["shape_kind"] or "dialogue",
        )
        bubbles_by_page.setdefault(pi, []).append(bubble)

    reader = await open_prepared_reader(store, prepared_key(chapter_id), tmp)
    chapter = reader.chapter(source=str(chapter_id))

    pages = tuple(
        Page(
            index=pi,
            width=page_map[pi]["width"],
            height=page_map[pi]["height"],
            bubbles=tuple(bubbles_by_page.get(pi, [])),
        )
        for pi in sorted(page_map)
    )
    scanned = ScanChapter(prepared=chapter, pages=pages)
    keyed   = assign_keys(scanned.all_bubbles, chapter_id=chapter_id)

    return reader, chapter, keyed, source_lang, target_lang


async def probe(chapter_id: int) -> None:
    cfg, paths = load_config()
    registry = build_storage(cfg, paths)

    run_dir = Path("debug-runs") / f"probe_brief_{chapter_id}"
    run_dir.mkdir(parents=True, exist_ok=True)
    artifacts = FileArtifactSink(run_dir.parent, f"probe_brief_{chapter_id}")

    print(f"\n=== probe_brief  chapter={chapter_id} ===\n")

    with tempfile.TemporaryDirectory() as tmp:
        reader, chapter, keyed, source_lang, target_lang = await _load_chapter(
            chapter_id, registry.pipeline, Path(tmp)
        )

        print(f"  source={source_lang}  target={target_lang}  is_color={chapter.is_color}")
        print(f"  pages={chapter.page_count}  bubbles={len(keyed)}\n")

        # Write the storyboard system prompt for inspection
        system_preview = prompt_mod.STORYBOARD_SYSTEM.format(
            source_lang_name=prompt_mod.lang_name(source_lang),
            target_lang_name=prompt_mod.lang_name(target_lang),
            is_color=chapter.is_color,
            target_agent_policy=prompt_mod.load_target_agent_policy(target_lang),
        )
        (run_dir / "storyboard_system.txt").write_text(system_preview, encoding="utf-8")
        print(f"  storyboard system prompt → storyboard_system.txt ({len(system_preview)} chars)\n")

        vision_provider      = make_vision_provider(cfg)
        translation_provider = make_translation_provider(cfg)

        # Build a minimal ctx — brief pass only needs vision_provider,
        # source/target_lang, chapter_id, and hook. Store/draft/owner are
        # not used by build_chapter_brief, so pass zeros.
        from typoon.adapters.ctx import TranslateCtx
        from typoon.runs.events import Hook as _Hook
        ctx = TranslateCtx(
            translation_provider=translation_provider,
            vision_provider=vision_provider,
            store=None,           # not used by brief pass
            chapter_id=chapter_id,
            draft_id=0,
            chapter_position=0,
            material_id=0,
            owner_id=0,
            source_lang=source_lang,
            target_lang=target_lang,
            hook=_PrintHook(),
        )

        print("Running context agent (vision pass)...")
        brief = await build_chapter_brief(ctx, reader, keyed, artifacts=artifacts)

        # Write parsed brief
        brief_json = brief.to_dict()
        (run_dir / "brief.json").write_text(
            json.dumps(brief_json, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

        print(f"\n{'='*60}")
        print("PARSED BRIEF")
        print(f"{'='*60}\n")

        print(f"brief_prose ({len(brief.brief_prose)} chars):")
        for line in brief.brief_prose.splitlines():
            print(f"  {line}")

        print(f"\nglossary ({len(brief.glossary)} entries):")
        for src, tgt in brief.glossary.items():
            mark = "" if src != tgt else "  ← IDENTITY (bad)"
            print(f"  {src!r} → {tgt!r}{mark}")

        print(f"\naddress_pairs ({len(brief.address_pairs)} pairs):")
        for (sp, li), pair in brief.address_pairs.items():
            print(f"  {sp} → {li}: {pair}")

        print(f"\ncharacters ({len(brief.characters)}):")
        for c in brief.characters:
            tname = f" → {c.target_name!r}" if c.target_name and c.target_name != c.name else ""
            print(f"  {c.name!r}{tname} ({c.gender}) voice={c.voice!r}")

        print(f"\nnoise_keys: {len(brief.noise_keys)}")

        print(f"\nkey_notes (first 8):")
        for k, v in list(brief.key_notes.items())[:8]:
            print(f"  {k}: {v}")

        # Sample translator context
        if keyed:
            first_keys = [bk.key for bk in keyed[:12]]
            sample_ctx = brief_slice(brief, page_indices=set(), keys=first_keys)
            (run_dir / "sample_translator_context.txt").write_text(
                sample_ctx, encoding="utf-8"
            )
            print(f"\n{'='*60}")
            print("SAMPLE TRANSLATOR CONTEXT (first 12 bubbles)")
            print(f"{'='*60}\n")
            print(sample_ctx)

        print(f"\nArtifacts → {run_dir}/")
        reader.close()


if __name__ == "__main__":
    chapter_id = int(sys.argv[1]) if len(sys.argv) > 1 else 14
    asyncio.run(probe(chapter_id))
