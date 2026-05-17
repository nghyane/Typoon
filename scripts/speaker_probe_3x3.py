"""Probe: speaker assignment at higher page density (3x3 = 9 pages).

Compare to the 2x2 baseline (4 pages, 692 KB storyboard, 5.6s, 26 bubbles).
The question: does the model still parse key labels and assign speakers
accurately when each page is scaled down ~2.25x more?

Uses tests/fixtures/sample_chapters/ch001 (5 pages) + ch002 (4 pages) = 9
pages. Chapters are mixed, so cross-chapter speaker continuity is NOT
tested — that's fine; we're probing visual readability, not narrative
coherence.

Output: debug-runs/storyboard_proto/speaker_probe_3x3.md
"""

from __future__ import annotations

import asyncio
import base64
import json
import logging
import time
from pathlib import Path
from typing import cast

from PIL import Image, ImageDraw, ImageFont

from typoon.adapters.vision_runtime import VisionRuntime
from typoon.config import load_config
from typoon.domain.prepared import Chapter, Page as PreparedPage
from typoon.llm.ir import ContentPart, Message
from typoon.providers import make_vision_provider
from typoon.stages.keys import assign_keys
from typoon.stages.scan import scan_chapter

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")
log = logging.getLogger("speaker_probe_3x3")

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "debug-runs" / "storyboard_proto"
OUT.mkdir(parents=True, exist_ok=True)

CH001 = ROOT / "tests" / "fixtures" / "sample_chapters" / "ch001"
CH002 = ROOT / "tests" / "fixtures" / "sample_chapters" / "ch002"

PAGES: list[Path] = (
    sorted(CH001.glob("*.webp")) +
    sorted(CH002.glob("*.webp"))
)
assert len(PAGES) == 9, f"expected 9 pages, got {len(PAGES)}"


class WebpPreparedReader:
    def __init__(self, paths: list[Path]):
        import numpy as np

        self._images = [np.array(Image.open(p).convert("RGB")) for p in paths]
        self._pages = tuple(
            PreparedPage(index=i, width=img.shape[1], height=img.shape[0])
            for i, img in enumerate(self._images)
        )

    @property
    def page_count(self) -> int:
        return len(self._images)

    def chapter(self, source: str = "") -> Chapter:
        return Chapter(source=source, pages=self._pages)

    def read_rgb(self, index: int):
        return self._images[index]

    def close(self) -> None:
        pass

    def __enter__(self):
        return self

    def __exit__(self, *_):
        pass


def _font(size: int) -> ImageFont.ImageFont:
    for cand in [
        "/System/Library/Fonts/Supplemental/Arial Bold.ttf",
        "/System/Library/Fonts/Helvetica.ttc",
    ]:
        if Path(cand).exists():
            try:
                return cast(ImageFont.ImageFont, ImageFont.truetype(cand, size))
            except OSError:
                pass
    return ImageFont.load_default()


def build_grid(
    page_paths: list[Path],
    bubbles_per_page: list[list[dict]],
    cols: int,
    rows: int,
    cell_w: int,
    cell_h: int,
    max_edge: int = 2048,
    label_size: int = 26,
) -> Image.Image:
    assert len(page_paths) <= cols * rows

    cells: list[Image.Image] = []
    for i, (path, bubbles) in enumerate(zip(page_paths, bubbles_per_page)):
        raw = Image.open(path).convert("RGB")
        scale = min(cell_w / raw.width, cell_h / raw.height, 1.0)
        new_w, new_h = int(raw.width * scale), int(raw.height * scale)
        scaled = raw.resize((new_w, new_h), Image.LANCZOS)

        draw = ImageDraw.Draw(scaled, "RGBA")
        font = _font(label_size)
        for b in bubbles:
            poly = b["polygon"]
            xs = [p[0] * scale for p in poly]
            ys = [p[1] * scale for p in poly]
            cx, cy = sum(xs) / len(xs), sum(ys) / len(ys)
            label = b["key"][:4]
            bbox = draw.textbbox((0, 0), label, font=font)
            tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
            pad = 3
            box = (cx - tw / 2 - pad, cy - th / 2 - pad, cx + tw / 2 + pad, cy + th / 2 + pad)
            draw.rectangle(box, fill=(255, 80, 80, 235))
            draw.text((cx - tw / 2, cy - th / 2), label, fill=(255, 255, 255), font=font)

        banner_h = 30
        labelled = Image.new("RGB", (scaled.width, scaled.height + banner_h), (32, 32, 32))
        labelled.paste(scaled, (0, banner_h))
        bdraw = ImageDraw.Draw(labelled)
        bdraw.text((6, 3), f"page {i}", fill=(240, 240, 240), font=_font(20))
        cells.append(labelled)

    # Lay out grid.
    pad = 6
    row_heights: list[int] = []
    col_widths: list[int] = [0] * cols
    for r in range(rows):
        row_cells = cells[r * cols:(r + 1) * cols]
        if not row_cells:
            break
        row_heights.append(max(c.height for c in row_cells))
        for c_idx, cell in enumerate(row_cells):
            col_widths[c_idx] = max(col_widths[c_idx], cell.width)

    canvas_w = sum(col_widths) + pad * (cols + 1)
    canvas_h = sum(row_heights) + pad * (len(row_heights) + 1)
    canvas = Image.new("RGB", (canvas_w, canvas_h), (20, 20, 20))

    y = pad
    for r, row_h in enumerate(row_heights):
        x = pad
        for c_idx in range(cols):
            idx = r * cols + c_idx
            if idx >= len(cells):
                break
            canvas.paste(cells[idx], (x, y))
            x += col_widths[c_idx] + pad
        y += row_h + pad

    longest = max(canvas.size)
    if longest > max_edge:
        s = max_edge / longest
        canvas = canvas.resize((int(canvas.width * s), int(canvas.height * s)), Image.LANCZOS)
    return canvas


SYSTEM = """\
You are a comic vision assistant. The user gives you a manga storyboard image
(multiple consecutive pages laid out in a grid), and a list of speech bubbles
with their OCR text and short keys (e.g. "ABC1"). Each bubble in the image is
overlaid with its 4-character key on a red label at the bubble center.

For each bubble key, assign a speaker.

Rules:
- "speaker" is the character speaking that line.
- If the bubble is narration (caption box, no tail) → "narrator".
- If the bubble is SFX → "sfx".
- If you cannot determine from visual evidence → "unknown".
- Do NOT guess. Wrong guesses cost the translator more than no guess.

Reply with one line per key, no preamble:
@@ KEY speaker_descriptor
"""


def build_user(bubbles_flat: list[dict]) -> str:
    lines = ["Bubbles to label:"]
    for b in bubbles_flat:
        text = b["text"].replace("\n", " ")[:80] or "(empty)"
        lines.append(f"@@ {b['key']} page={b['page']} kind={b['shape_kind']} text={text!r}")
    return "\n".join(lines)


async def main() -> None:
    config, paths = load_config()
    runtime = VisionRuntime.from_config(config, paths, source_lang="en")[0]

    reader = WebpPreparedReader(PAGES)
    prepared = reader.chapter("mix")
    log.info("scanning %d pages…", reader.page_count)
    t0 = time.monotonic()
    out = scan_chapter(prepared, reader, runtime, source_lang="en")
    scan_elapsed = time.monotonic() - t0
    log.info("scan done in %.1fs, %d bubbles", scan_elapsed, len(out.chapter.all_bubbles))

    if not out.chapter.all_bubbles:
        log.error("no bubbles; abort")
        return

    keyed = assign_keys(out.chapter.all_bubbles, chapter_id=1)
    per_page: list[list[dict]] = [[] for _ in range(reader.page_count)]
    flat: list[dict] = []
    for bk in keyed:
        b = bk.bubble
        entry = {
            "key": bk.key,
            "page": b.page_index,
            "text": b.source_text,
            "shape_kind": b.shape_kind,
            "polygon": b.polygon,
        }
        per_page[b.page_index].append(entry)
        flat.append(entry)

    # 3x3 layout @ 700x950 per cell — slightly larger cell budget than the
    # 2x2 probe (900x1200) compensates for the smaller per-cell footprint
    # after the canvas-level max_edge=2048 cap.
    sb = build_grid(PAGES, per_page, cols=3, rows=3, cell_w=700, cell_h=950)
    sb_path = OUT / "speaker_probe_3x3_storyboard.jpg"
    sb.save(sb_path, quality=88, optimize=True)
    log.info("storyboard: %s  %s  %d KB", sb_path, sb.size, sb_path.stat().st_size // 1024)

    inventory = OUT / "speaker_probe_3x3_bubbles.json"
    inventory.write_text(json.dumps(flat, indent=2, ensure_ascii=False), encoding="utf-8")

    provider = make_vision_provider(config)
    data_uri = "data:image/jpeg;base64," + base64.b64encode(sb_path.read_bytes()).decode()
    user = build_user(flat)
    messages = [
        Message.system(SYSTEM),
        Message.user_parts([ContentPart.of_text(user), ContentPart.of_image(data_uri)]),
    ]

    log.info("calling vision (%d bubbles)…", len(flat))
    t0 = time.monotonic()
    resp = await provider.call(messages, [])
    elapsed = time.monotonic() - t0
    text = resp.text or "(empty)"
    log.info("vision done in %.1fs, %d chars", elapsed, len(text))

    assigned: dict[str, str] = {}
    for line in text.splitlines():
        line = line.strip()
        if not line.startswith("@@ "):
            continue
        parts = line[3:].split(" ", 1)
        if len(parts) != 2:
            continue
        key, speaker = parts
        if any(b["key"] == key for b in flat):
            assigned[key] = speaker.strip()

    coverage = len(assigned) / len(flat) * 100
    unknowns = sum(1 for v in assigned.values() if v.lower() == "unknown")
    nameless_pct = unknowns / len(flat) * 100 if flat else 0
    log.info("coverage: %d/%d (%.0f%%), unknowns: %d (%.0f%%)",
             len(assigned), len(flat), coverage, unknowns, nameless_pct)

    lines = [
        "# Speaker probe — 3x3 (9 pages, mixed chapter)",
        "",
        f"- model: {config.vision_agent.model}",
        f"- pages: {reader.page_count}",
        f"- bubbles total: {len(flat)}",
        f"- storyboard: {sb.size[0]}x{sb.size[1]}, {sb_path.stat().st_size // 1024} KB",
        f"- scan time: {scan_elapsed:.1f}s",
        f"- vision call: {elapsed:.1f}s",
        f"- response length: {len(text)} chars",
        f"- assignments parsed: {len(assigned)} / {len(flat)} ({coverage:.0f}%)",
        f"- unknowns: {unknowns} ({nameless_pct:.0f}%)",
        "",
        "## Bubbles vs assignments",
        "",
        "| key | page | shape | text (truncated) | model speaker |",
        "|---|---|---|---|---|",
    ]
    for b in flat:
        sp = assigned.get(b["key"], "—")
        tx = b["text"].replace("|", "\\|").replace("\n", " ")[:60]
        lines.append(f"| `{b['key']}` | {b['page']} | {b['shape_kind']} | {tx!r} | {sp} |")
    lines.append("")
    lines.append("## Raw model response")
    lines.append("")
    lines.append("```")
    lines.append(text)
    lines.append("```")

    report = OUT / "speaker_probe_3x3.md"
    report.write_text("\n".join(lines), encoding="utf-8")
    log.info("wrote %s", report)


if __name__ == "__main__":
    asyncio.run(main())
