"""Single-page full-pipeline probe.

Wires the production stage functions against a one-page in-memory reader
so a single PNG can be exercised end-to-end without DB or worker.

Stages:
    1. scan        — detect + group + (no recognizer; lens ships text)
    2. translate   — brief + per-window LLM (real provider call)
    3. render      — AOT erase + typoon_render text rendering

Run:
    python -m scripts.probes.full_page <image>
        [--out debug-runs/<name>]
        [--source-lang ja] [--target-lang vi]
        [--no-translate]  # stub translations (echo source) — no LLM calls
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import sys
import time
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

# Make repo root importable when invoked via `python -m scripts.probes.full_page`.
ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from typoon.adapters.ctx import TranslateCtx  # noqa: E402
from typoon.config import load_config  # noqa: E402
from typoon.domain import translate as t_dom  # noqa: E402
from typoon.providers import (  # noqa: E402
    make_translation_provider, make_vision_provider,
)
from typoon.runs.artifacts import FileArtifactSink  # noqa: E402
from typoon.runs.events import LoggingHook  # noqa: E402
from typoon.stages.render import render_chapter  # noqa: E402
from typoon.stages.scan import scan_chapter  # noqa: E402
from typoon.stages.translate import _normalize_for_render  # noqa: E402
from typoon.stages.translate import translate_chapter  # noqa: E402
from typoon.vision.pipeline import PRESETS  # noqa: E402
from typoon.vision.runtime import build_vision_runtime  # noqa: E402

from .reader import SinglePageReader  # noqa: E402


def _stub_translate(scanned, *, target_lang: str):
    """Echo source text → translated_text. No LLM call."""
    from typoon.domain.brief import ChapterBrief
    from typoon.stages.keys import assign_keys
    from typoon.stages.noise import is_auto_skip

    keyed = assign_keys(scanned.all_bubbles, chapter_id=0)
    key_at = {(bk.bubble.page_index, bk.bubble.idx): bk.key for bk in keyed}

    pages = []
    for sp in scanned.pages:
        bubbles = []
        for sb in sp.bubbles:
            key = key_at.get((sb.page_index, sb.idx), f"p{sb.page_index}_b{sb.idx}")
            kind = "skip" if is_auto_skip(sb.source_text) else "dialogue"
            text = "" if kind == "skip" else _normalize_for_render(
                f"[{target_lang}] {sb.source_text}"
            )
            bubbles.append(t_dom.Bubble(
                source=sb,
                translation_key=key,
                translated_text=text,
                kind=kind,
            ))
        pages.append(t_dom.Page(source=sp, bubbles=tuple(bubbles)))

    return t_dom.Chapter(scan=scanned, pages=tuple(pages)), ChapterBrief()


async def main(args: argparse.Namespace) -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname).1s %(name)s: %(message)s",
    )
    log = logging.getLogger("full_page")

    image_path = args.image.resolve()
    if not image_path.exists():
        print(f"image not found: {image_path}", file=sys.stderr)
        return 2

    config, paths = load_config()
    models_dir = (args.models or paths.models).resolve()

    sink = FileArtifactSink(
        Path("debug-runs"),
        args.out_name,
        clean=True,
    )
    log.info("artifacts → %s", sink.root)

    reader = SinglePageReader.from_path(image_path)
    sink.write_image("01_prepare", "source.png", reader.read_rgb(0))
    log.info("loaded %s: %dx%d", image_path.name,
             reader._image.shape[1], reader._image.shape[0])

    # Vision runtime: production `lens` preset (the only active stack).
    runtime = build_vision_runtime(
        PRESETS["lens"],
        models_dir=models_dir,
        source_lang=args.source_lang,
        lens_endpoint=config.lens_endpoint or None,
    )

    # ─── 1. SCAN ────────────────────────────────────────────────────────
    t0 = time.monotonic()
    prepared = reader.chapter(source=str(image_path))
    scanned = await scan_chapter(
        prepared, reader, runtime,
        source_lang=args.source_lang,
        chapter_id=0,
        artifacts=sink,
    )
    n_bubbles = sum(len(p.bubbles) for p in scanned.chapter.pages)
    log.info("scan: %d bubbles in %.2fs (detected_lang=%s)",
             n_bubbles, time.monotonic() - t0, scanned.detected_lang)

    if n_bubbles == 0:
        log.warning("no bubbles detected; skipping translate + render")
        return 0

    # ─── 2. TRANSLATE ───────────────────────────────────────────────────
    t0 = time.monotonic()
    if args.no_translate:
        log.info("translate: STUBBED (echo source)")
        translated, brief = _stub_translate(scanned.chapter, target_lang=args.target_lang)
    else:
        ctx = TranslateCtx(
            translation_provider=make_translation_provider(config),
            vision_provider=make_vision_provider(config),
            store=None,  # unused by brief/translate/page stages
            chapter_id=0,
            draft_id=0,
            chapter_position=0,
            material_id=0,
            owner_id=0,
            source_lang=args.source_lang or scanned.detected_lang or "ja",
            target_lang=args.target_lang,
            hook=LoggingHook(),
        )
        translated, brief = await translate_chapter(
            scanned.chapter, reader, ctx, artifacts=sink,
        )

    n_translated = sum(
        1 for p in translated.pages for b in p.bubbles if b.translated_text
    )
    n_skip = sum(1 for p in translated.pages for b in p.bubbles if b.kind == "skip")
    log.info("translate: %d translated, %d skipped, %d noise_keys, brief=%dch in %.2fs",
             n_translated, n_skip, len(brief.noise_keys),
             len(brief.brief_prose), time.monotonic() - t0)

    # ─── 3. RENDER ──────────────────────────────────────────────────────
    t0 = time.monotonic()
    page_geoms = {pg.page_index: pg for pg in scanned.geometry}
    final_dir = sink.root / "final"
    final_dir.mkdir(parents=True, exist_ok=True)
    await render_chapter(
        translated,
        final_dir,
        reader,
        runtime,
        page_geoms,
        scanned.masks,
        chapter_id=0,
        target_kind="draft",
        target_id=0,
        hook=LoggingHook(),
        artifacts=sink,
    )
    log.info("render: wrote %s in %.2fs", final_dir, time.monotonic() - t0)
    log.info("done → %s", sink.root)
    return 0


def cli() -> None:
    ap = argparse.ArgumentParser(prog="full_page_probe")
    ap.add_argument("image", type=Path, help="input page image (PNG/JPG/WEBP, RGB)")
    ap.add_argument(
        "--out", dest="out_name", type=str, default="full_page_probe",
        help="run-id directory name under debug-runs/ (default: full_page_probe)",
    )
    ap.add_argument("--models", type=Path, default=None,
                    help="models dir (default: paths.models from config)")
    ap.add_argument("--source-lang", type=str, default=None,
                    help="hint for Lens OCR (e.g. ja, zh-Hans, ko)")
    ap.add_argument("--target-lang", type=str, default="vi",
                    help="translation target (default: vi)")
    ap.add_argument("--no-translate", action="store_true",
                    help="stub translations (echo source) — skip LLM calls")
    args = ap.parse_args()
    sys.exit(asyncio.run(main(args)))


if __name__ == "__main__":
    cli()
