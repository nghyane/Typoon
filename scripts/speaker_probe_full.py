"""Full speaker probe — real chapter (Chainsaw Man ch.1, page 5-13).

9 consecutive pages, same chapter (consistent characters), tests if the
prior 3x3 100%-unknown was caused by:
  (a) cognitive load — model gives up at high bubble count
  (b) cross-chapter character confusion — fixed here
  (c) visual readability at scale-down

Compares 2x2 (4 pages, 5.6s baseline) vs 3x3 (9 pages, same chapter).

Output: debug-runs/storyboard_proto/probe_full_*.md
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
CHAP = ROOT / "cache" / "probe_chapter"
# Pages 5..13 — past the cover/title, into actual dialogue. 9 consecutive.
SLICE = list(sorted(CHAP.glob("*.png")))[5:14]


async def probe(
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
                    cell_w=cell_w, cell_h=cell_h, max_edge=2048, label_size=26)
    sb_path = OUT / f"probe_full_{name}.jpg"
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
    speakers = {v for v in assigned.values() if v.lower() not in ("unknown", "sfx", "narrator")}
    log.info("vision: %.1fs, %d chars, %d/%d assigned, %d unknown, %d distinct named",
             elapsed, len(text), len(assigned), len(flat), unknowns, len(speakers))

    lines = [
        f"# Probe — {name}",
        "",
        f"- pages: {len(pages)} (real chapter, consecutive)",
        f"- grid: {cols}x{rows}, cell: {cell_w}x{cell_h}",
        f"- storyboard: {sb.size[0]}x{sb.size[1]}, {sb_path.stat().st_size // 1024} KB",
        f"- scan: {scan_t:.1f}s, vision: {elapsed:.1f}s",
        f"- assigned: {len(assigned)}/{len(flat)}, unknown: {unknowns}, distinct named speakers: {len(speakers)}",
        f"- named speakers: {sorted(speakers)}",
        "",
        "| key | page | shape | text | speaker |",
        "|---|---|---|---|---|",
    ]
    for b in flat:
        sp = assigned.get(b["key"], "—")
        tx = b["text"].replace("|", "\\|").replace("\n", " ")[:50]
        lines.append(f"| `{b['key']}` | {b['page']} | {b['shape_kind']} | {tx!r} | {sp} |")
    lines.append("\n## Raw\n```\n" + text + "\n```")
    (OUT / f"probe_full_{name}.md").write_text("\n".join(lines), "utf-8")
    return {
        "name": name, "pages": len(pages), "bubbles": len(flat),
        "scan_s": scan_t, "vision_s": elapsed,
        "assigned": len(assigned), "unknown": unknowns,
        "named_speakers": len(speakers),
    }


async def main() -> None:
    config, paths = load_config()
    runtime = VisionRuntime.from_config(config, paths, source_lang="en")[0]

    # 2x2 baseline: 4 consecutive pages.
    r1 = await probe("4page_2x2", SLICE[:4], cols=2, rows=2,
                     cell_w=900, cell_h=1200, runtime=runtime, config=config)

    # 3x3 stress: 9 consecutive pages.
    r2 = await probe("9page_3x3", SLICE[:9], cols=3, rows=3,
                     cell_w=700, cell_h=950, runtime=runtime, config=config)

    print("\n=== SUMMARY ===")
    for r in (r1, r2):
        print(f"{r['name']}: {r['pages']}p, {r['bubbles']}b → "
              f"{r['assigned'] - r['unknown']}/{r['bubbles']} named "
              f"({r['named_speakers']} distinct), "
              f"vision {r['vision_s']:.1f}s")


if __name__ == "__main__":
    asyncio.run(main())
