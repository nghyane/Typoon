"""Probe: real chapter scan → storyboard + bubble overlay → speaker assignment.

Pipeline:
  1. Build PreparedChapter from cache/benchmark_visual/ (5 pages already prepared).
  2. Run scan_chapter to get real bubble keys, text, polygons, shape_kind.
  3. Build 4-page storyboard with numbered key labels overlaid on each bubble.
  4. Ask vision model to assign speaker per key.

Output: debug-runs/storyboard_proto/speaker_probe.md
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

from typoon.adapters.prepared_reader import PreparedReader
from typoon.adapters.vision_runtime import VisionRuntime
from typoon.config import load_config
from typoon.domain.prepared import Chapter, Page as PreparedPage
from typoon.llm.ir import ContentPart, Message
from typoon.providers import make_vision_provider
from typoon.stages.keys import assign_keys
from typoon.stages.scan import scan_chapter

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")
log = logging.getLogger("speaker_probe")

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "cache" / "benchmark_visual"
OUT = ROOT / "debug-runs" / "storyboard_proto"
OUT.mkdir(parents=True, exist_ok=True)

PAGES = [SRC / f"p0{i}_original.jpg" for i in range(4)]


class JpegPreparedReader:
    def __init__(self, paths: list[Path]):
        import numpy as np
        from PIL import Image as _Image

        self._images = [np.array(_Image.open(p).convert("RGB")) for p in paths]
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


def _short_label(key: str) -> str:
    """First 4 chars of the bubble key — readable on the storyboard."""
    return key[:4]


def build_labelled_storyboard(
    page_paths: list[Path],
    bubbles_per_page: list[list[dict]],
    cell_w: int = 900,
    cell_h: int = 1200,
    max_edge: int = 2048,
) -> Image.Image:
    """2×2 grid of pages with key labels overlaid on each bubble polygon."""
    cells: list[Image.Image] = []
    for i, (path, bubbles) in enumerate(zip(page_paths, bubbles_per_page)):
        raw = Image.open(path).convert("RGB")
        # Scale to fit cell.
        scale = min(cell_w / raw.width, cell_h / raw.height, 1.0)
        new_w, new_h = int(raw.width * scale), int(raw.height * scale)
        scaled = raw.resize((new_w, new_h), Image.LANCZOS)

        # Draw labels at scaled polygon centers.
        draw = ImageDraw.Draw(scaled, "RGBA")
        font = _font(28)
        for b in bubbles:
            poly = b["polygon"]
            xs = [p[0] * scale for p in poly]
            ys = [p[1] * scale for p in poly]
            cx, cy = sum(xs) / len(xs), sum(ys) / len(ys)
            label = _short_label(b["key"])
            tw, th = draw.textbbox((0, 0), label, font=font)[2:]
            pad = 4
            box = (cx - tw / 2 - pad, cy - th / 2 - pad, cx + tw / 2 + pad, cy + th / 2 + pad)
            draw.rectangle(box, fill=(255, 80, 80, 230))
            draw.text((cx - tw / 2, cy - th / 2), label, fill=(255, 255, 255), font=font)

        # Page banner.
        banner_h = 36
        labelled = Image.new("RGB", (scaled.width, scaled.height + banner_h), (32, 32, 32))
        labelled.paste(scaled, (0, banner_h))
        bdraw = ImageDraw.Draw(labelled)
        bdraw.text((8, 4), f"page {i}", fill=(240, 240, 240), font=_font(22))
        cells.append(labelled)

    # 2×2 layout.
    pad = 8
    row_top_h = max(c.height for c in cells[:2])
    row_bot_h = max(c.height for c in cells[2:4])
    col_l_w = max(cells[0].width, cells[2].width)
    col_r_w = max(cells[1].width, cells[3].width)

    canvas = Image.new(
        "RGB",
        (col_l_w + col_r_w + pad * 3, row_top_h + row_bot_h + pad * 3),
        (20, 20, 20),
    )
    positions = [
        (pad, pad),
        (col_l_w + pad * 2, pad),
        (pad, row_top_h + pad * 2),
        (col_l_w + pad * 2, row_top_h + pad * 2),
    ]
    for cell, pos in zip(cells, positions):
        canvas.paste(cell, pos)

    # Cap longest edge.
    longest = max(canvas.size)
    if longest > max_edge:
        s = max_edge / longest
        canvas = canvas.resize((int(canvas.width * s), int(canvas.height * s)), Image.LANCZOS)
    return canvas


SYSTEM = """\
You are a comic vision assistant. The user gives you a manga storyboard image
(4 consecutive pages, 2x2 grid), and a list of speech bubbles with their OCR
text and short keys (e.g. "ABC1"). Each bubble in the image is overlaid with
its 4-character key on a red label at the bubble center.

Your job: for each bubble key, assign a speaker.

Rules:
- "speaker" is the character speaking that line. Use a stable descriptor or
  candidate name (e.g. "white-haired girl", "dark-haired elf", "narrator").
- If the bubble is narration (caption box, no tail, story voiceover) → speaker = "narrator".
- If the bubble is SFX (sound effect, no speaker) → speaker = "sfx".
- If you cannot determine the speaker from visual evidence (no tail visible,
  multiple candidates, partial panel) → speaker = "unknown".
- Do NOT guess. Mark "unknown" liberally — wrong guesses cost the translator
  more than no guess.

Reply in this exact line format, one line per key:
@@ KEY speaker_descriptor
(That's it — no preamble, no explanation, no markdown.)
"""


def build_user(bubbles_flat: list[dict]) -> str:
    lines = ["Bubbles to label (one line per key):"]
    for b in bubbles_flat:
        text = b["text"].replace("\n", " ")[:80] or "(empty)"
        lines.append(f"@@ {b['key']} page={b['page']} kind={b['shape_kind']} text={text!r}")
    return "\n".join(lines)


async def main() -> None:
    config, paths = load_config()
    log.info("loading vision runtime…")
    runtime = VisionRuntime.from_config(config, paths, source_lang="en")[0]

    reader = JpegPreparedReader(PAGES)
    prepared = reader.chapter("benchmark")
    log.info("scanning %d pages…", reader.page_count)
    t0 = time.monotonic()
    out = scan_chapter(prepared, reader, runtime, source_lang="en")
    log.info("scan done in %.1fs, %d bubbles", time.monotonic() - t0, len(out.chapter.all_bubbles))

    if not out.chapter.all_bubbles:
        log.error("no bubbles detected; cannot probe")
        return

    keyed = assign_keys(out.chapter.all_bubbles, chapter_id=1)
    key_by_loc = {(bk.bubble.page_index, bk.bubble.idx): bk.key for bk in keyed}

    # Build per-page bubble list (with polygons) for storyboard overlay.
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

    # Only first 4 pages used in 2x2 storyboard.
    sb = build_labelled_storyboard(PAGES, per_page[:4])
    sb_path = OUT / "speaker_probe_storyboard.jpg"
    sb.save(sb_path, quality=88, optimize=True)
    log.info("storyboard: %s  %s  %d KB", sb_path, sb.size, sb_path.stat().st_size // 1024)

    # Dump bubble inventory for reference.
    inventory_path = OUT / "speaker_probe_bubbles.json"
    inventory_path.write_text(json.dumps(flat, indent=2, ensure_ascii=False), encoding="utf-8")

    # Vision call.
    provider = make_vision_provider(config)
    data_uri = "data:image/jpeg;base64," + base64.b64encode(sb_path.read_bytes()).decode()
    flat_visible = [f for f in flat if f["page"] < 4]
    user = build_user(flat_visible)
    messages = [
        Message.system(SYSTEM),
        Message.user_parts([ContentPart.of_text(user), ContentPart.of_image(data_uri)]),
    ]

    log.info("calling vision (%d bubbles)…", len(flat_visible))
    t0 = time.monotonic()
    resp = await provider.call(messages, [])
    elapsed = time.monotonic() - t0
    text = resp.text or "(empty)"
    log.info("vision done in %.1fs, %d chars", elapsed, len(text))

    # Parse model response.
    assigned: dict[str, str] = {}
    for line in text.splitlines():
        line = line.strip()
        if not line.startswith("@@ "):
            continue
        parts = line[3:].split(" ", 1)
        if len(parts) != 2:
            continue
        key, speaker = parts
        if any(b["key"] == key for b in flat_visible):
            assigned[key] = speaker.strip()

    # Report.
    report_lines = [
        "# Speaker assignment probe (real scan + labelled storyboard)",
        "",
        f"- model: {config.vision_agent.model}",
        f"- pages: {reader.page_count}",
        f"- bubbles total: {len(flat)}",
        f"- bubbles in storyboard (first 4 pages): {len(flat_visible)}",
        f"- scan time: {time.monotonic() - t0:.1f}s (this segment)",
        f"- vision call: {elapsed:.1f}s",
        f"- model response length: {len(text)} chars",
        f"- assignments parsed: {len(assigned)}",
        "",
        "## Bubbles vs assignments",
        "",
        "| key | page | shape | text (truncated) | model speaker |",
        "|---|---|---|---|---|",
    ]
    for b in flat_visible:
        sp = assigned.get(b["key"], "—")
        tx = b["text"].replace("|", "\\|").replace("\n", " ")[:60]
        report_lines.append(f"| `{b['key']}` | {b['page']} | {b['shape_kind']} | {tx!r} | {sp} |")
    report_lines.append("")
    report_lines.append("## Raw model response")
    report_lines.append("")
    report_lines.append("```")
    report_lines.append(text)
    report_lines.append("```")

    report_path = OUT / "speaker_probe.md"
    report_path.write_text("\n".join(report_lines), encoding="utf-8")
    log.info("wrote %s", report_path)


if __name__ == "__main__":
    asyncio.run(main())
