"""Speaker probe — single chapter only (no mixing). Tests if the 100%-unknown
result at 3x3 was caused by mixed-chapter confusion vs visual readability.

Runs against ch001 (5 pages, 2x3 grid) and ch002 (4 pages, 2x2 grid)
separately. Each chapter has its own character continuity, so the model
should be able to assign speakers when it's not confused by chapter
boundaries.
"""

from __future__ import annotations

import asyncio
import base64
import json
import logging
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from speaker_probe_3x3 import (
    SYSTEM, WebpPreparedReader, build_grid, build_user, OUT
)

from typoon.adapters.vision_runtime import VisionRuntime
from typoon.config import load_config
from typoon.llm.ir import ContentPart, Message
from typoon.providers import make_vision_provider
from typoon.stages.keys import assign_keys
from typoon.stages.scan import scan_chapter

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
log = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parents[1]
CH001 = sorted((ROOT / "tests" / "fixtures" / "sample_chapters" / "ch001").glob("*.webp"))
CH002 = sorted((ROOT / "tests" / "fixtures" / "sample_chapters" / "ch002").glob("*.webp"))


async def probe_chapter(
    name: str, pages: list[Path], cols: int, rows: int,
    cell_w: int, cell_h: int, runtime, config,
):
    log.info("=== %s: %d pages, grid %dx%d, cell %dx%d ===",
             name, len(pages), cols, rows, cell_w, cell_h)
    reader = WebpPreparedReader(pages)
    prepared = reader.chapter(name)
    t0 = time.monotonic()
    out = scan_chapter(prepared, reader, runtime, source_lang="en")
    scan_t = time.monotonic() - t0
    log.info("scan: %.1fs, %d bubbles", scan_t, len(out.chapter.all_bubbles))

    keyed = assign_keys(out.chapter.all_bubbles, chapter_id=1)
    per_page: list[list[dict]] = [[] for _ in range(reader.page_count)]
    flat: list[dict] = []
    for bk in keyed:
        b = bk.bubble
        e = {"key": bk.key, "page": b.page_index, "text": b.source_text,
             "shape_kind": b.shape_kind, "polygon": b.box.polygon}
        per_page[b.page_index].append(e)
        flat.append(e)

    sb = build_grid(pages, per_page, cols=cols, rows=rows,
                    cell_w=cell_w, cell_h=cell_h, max_edge=2048, label_size=28)
    sb_path = OUT / f"speaker_probe_{name}.jpg"
    sb.save(sb_path, quality=88, optimize=True)
    log.info("storyboard: %s %d KB", sb.size, sb_path.stat().st_size // 1024)

    provider = make_vision_provider(config)
    data_uri = "data:image/jpeg;base64," + base64.b64encode(sb_path.read_bytes()).decode()
    msgs = [Message.system(SYSTEM), Message.user_parts([
        ContentPart.of_text(build_user(flat)), ContentPart.of_image(data_uri),
    ])]
    t0 = time.monotonic()
    resp = await provider.call(msgs, [])
    elapsed = time.monotonic() - t0
    text = resp.text or ""

    assigned: dict[str, str] = {}
    for line in text.splitlines():
        line = line.strip()
        if not line.startswith("@@ "):
            continue
        parts = line[3:].split(" ", 1)
        if len(parts) == 2:
            key, speaker = parts
            if any(b["key"] == key for b in flat):
                assigned[key] = speaker.strip()

    unknowns = sum(1 for v in assigned.values() if v.lower() == "unknown")
    log.info("vision: %.1fs, %d chars, %d/%d assigned, %d unknown",
             elapsed, len(text), len(assigned), len(flat), unknowns)

    lines = [
        f"# Speaker probe — {name}",
        "",
        f"- pages: {len(pages)}, grid: {cols}x{rows}, cell: {cell_w}x{cell_h}",
        f"- storyboard: {sb.size[0]}x{sb.size[1]}, {sb_path.stat().st_size // 1024} KB",
        f"- scan: {scan_t:.1f}s, vision: {elapsed:.1f}s",
        f"- assigned: {len(assigned)}/{len(flat)}, unknown: {unknowns}",
        "",
        "| key | page | shape | text | speaker |",
        "|---|---|---|---|---|",
    ]
    for b in flat:
        sp = assigned.get(b["key"], "—")
        tx = b["text"].replace("|", "\\|").replace("\n", " ")[:60]
        lines.append(f"| `{b['key']}` | {b['page']} | {b['shape_kind']} | {tx!r} | {sp} |")
    lines.append("\n## Raw\n```\n" + text + "\n```")
    (OUT / f"speaker_probe_{name}.md").write_text("\n".join(lines), "utf-8")


async def main() -> None:
    config, paths = load_config()
    runtime = VisionRuntime.from_config(config, paths, source_lang="en")[0]
    # ch001: 5 pages → 2 rows × 3 cols (one empty slot is fine)
    await probe_chapter("ch001", CH001, cols=3, rows=2, cell_w=800, cell_h=1100,
                        runtime=runtime, config=config)
    # ch002: 4 pages → 2x2 standard
    await probe_chapter("ch002", CH002, cols=2, rows=2, cell_w=900, cell_h=1200,
                        runtime=runtime, config=config)


if __name__ == "__main__":
    asyncio.run(main())
