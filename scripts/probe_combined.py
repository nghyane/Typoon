"""Combined-context probe — single LLM call produces everything the
current multi-turn context agent emits, with vision evidence.

Input: storyboard image + bubble list (key + OCR text + page + shape).
Output: structured line-sentinel sections covering:
  - characters discovered (name, gender, role) — material memory seed
  - speaker per bubble key — replaces key_notes
  - noise keys — replaces mark_noise tool
  - chapter style/register — replaces style_notes + summary
  - address rules — replaces address agent submission (only when speaker
    pairs are confidently observed)

Tests on Chainsaw Man ch.1 page 5-13 (same data as speaker_probe_full).
Single call, no tools, no retries.

Output: debug-runs/storyboard_proto/probe_combined.md
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
from speaker_probe_3x3 import WebpPreparedReader, build_grid, OUT

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
SLICE = list(sorted(CHAP.glob("*.png")))[5:14]  # 9 consecutive pages


SYSTEM = """\
You are a comic translation context agent with vision. Your job: read a
storyboard image plus its OCR bubble list, and emit ALL the context the
downstream translator needs, in a single structured reply.

You see a grid of consecutive pages from one chapter. Each speech bubble
is overlaid with its 4-character key on a red label.

DO NOT translate. DO NOT guess. If something is not clearly visible or
inferable, omit it or mark it unknown. Conservative is correct — wrong
guesses cost the translator more than missing data.

Reply format (one section per heading, sections in this exact order):

@@@ CHARACTERS
@@ name=NAME gender=male|female|unknown role=short_role_or_descriptor
(One line per distinct character that speaks or is named in this chapter.
Use a stable descriptor when no name is visible, e.g. "white-haired girl".)

@@@ SPEAKERS
@@ KEY speaker_name
(One line per bubble key. speaker_name is one of:
  - a NAME from CHARACTERS section above
  - "narrator" for caption boxes / internal monologue
  - "sfx" for sound-effect bubbles
  - "unknown" when you cannot tell — USE THIS LIBERALLY)

@@@ NOISE
@@ KEY
(One line per bubble that is platform chrome, watermark, page counter,
scanlation credit, "do not mirror" notice, etc. — bubbles that should
be skipped entirely by the translator. Omit section if none.)

@@@ STYLE
@@ register=formal|casual|action|comedy|drama
@@ mood=short_phrase
@@ note=short_translator_guidance
(2-4 lines describing the dominant tone the translator should preserve.)

@@@ ADDRESS
@@ speaker=NAME listener=NAME self=PRONOUN other=PRONOUN note=context
(Optional. Only emit when you can confidently see a recurring pair with
clear power/intimacy dynamic. Omit section if no confident pairs.
Pronouns must be in the target language: vi.)

No preamble, no markdown headers, no commentary. Just the @@@ sections.
"""


def build_user(bubbles: list[dict], target_lang: str) -> str:
    lines = [
        f"Target language: {target_lang}",
        "",
        "Bubble list (one line per bubble):",
    ]
    for b in bubbles:
        text = b["text"].replace("\n", " ")[:80] or "(empty)"
        lines.append(f"@@ {b['key']} page={b['page']} kind={b['shape_kind']} text={text!r}")
    return "\n".join(lines)


# ── Parser for the structured response ───────────────────────────────


def parse_response(text: str) -> dict:
    """Split the model reply into named sections, parse each line.

    The format uses '@@@ SECTION' as a section header and '@@ ...' for
    body lines. Anything not under a known section is dropped.
    """
    sections: dict[str, list[str]] = {}
    current: str | None = None
    for raw in text.splitlines():
        line = raw.rstrip()
        if line.startswith("@@@ "):
            current = line[4:].strip().upper()
            sections.setdefault(current, [])
        elif line.startswith("@@ ") and current is not None:
            sections[current].append(line[3:].strip())

    out: dict = {
        "characters": [],
        "speakers": {},
        "noise": set(),
        "style": {},
        "address": [],
    }

    # Characters: key=value pairs.
    for body in sections.get("CHARACTERS", []):
        ch = _kv(body)
        if "name" in ch:
            out["characters"].append({
                "name":   ch["name"],
                "gender": ch.get("gender", "unknown"),
                "role":   ch.get("role", ""),
            })

    # Speakers: "KEY name".
    for body in sections.get("SPEAKERS", []):
        parts = body.split(" ", 1)
        if len(parts) == 2:
            out["speakers"][parts[0]] = parts[1].strip()

    # Noise: "KEY" alone.
    for body in sections.get("NOISE", []):
        key = body.split()[0] if body else ""
        if key:
            out["noise"].add(key)

    # Style: key=value pairs.
    for body in sections.get("STYLE", []):
        kv = _kv(body)
        out["style"].update(kv)

    # Address: key=value pairs.
    for body in sections.get("ADDRESS", []):
        kv = _kv(body)
        if "speaker" in kv and "listener" in kv:
            out["address"].append(kv)

    return out


def _kv(line: str) -> dict:
    """Parse 'a=b c=d "with spaces"' into a dict. Tolerates simple quoting."""
    out: dict[str, str] = {}
    i = 0
    while i < len(line):
        if line[i].isspace():
            i += 1
            continue
        eq = line.find("=", i)
        if eq == -1:
            break
        key = line[i:eq].strip()
        i = eq + 1
        if i < len(line) and line[i] == '"':
            end = line.find('"', i + 1)
            if end == -1:
                val = line[i + 1:]
                i = len(line)
            else:
                val = line[i + 1:end]
                i = end + 1
        else:
            end = i
            while end < len(line) and not line[end].isspace():
                end += 1
            val = line[i:end]
            i = end
        if key:
            out[key] = val
    return out


# ── Driver ───────────────────────────────────────────────────────────


async def main() -> None:
    config, paths = load_config()
    runtime = VisionRuntime.from_config(config, paths, source_lang="en")[0]

    reader = WebpPreparedReader(SLICE)
    prepared = reader.chapter("chainsaw")
    log.info("scanning %d pages…", reader.page_count)
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

    sb = build_grid(SLICE, per_page, cols=3, rows=3,
                    cell_w=700, cell_h=950, max_edge=2048, label_size=26)
    sb_path = OUT / "probe_combined.jpg"
    sb.save(sb_path, quality=88, optimize=True)
    log.info("storyboard: %s %d KB", sb.size, sb_path.stat().st_size // 1024)

    provider = make_vision_provider(config)
    data_uri = "data:image/jpeg;base64," + base64.b64encode(sb_path.read_bytes()).decode()
    user = build_user(flat, target_lang="vi")
    msgs = [Message.system(SYSTEM), Message.user_parts([
        ContentPart.of_text(user), ContentPart.of_image(data_uri),
    ])]

    log.info("calling vision (%d bubbles, target=vi)…", len(flat))
    t0 = time.monotonic()
    resp = await provider.call(msgs, [])
    elapsed = time.monotonic() - t0
    text = resp.text or ""
    log.info("vision: %.1fs, %d chars", elapsed, len(text))

    parsed = parse_response(text)
    speakers = parsed["speakers"]
    unknowns = sum(1 for v in speakers.values() if v.lower() == "unknown")
    named = {v for v in speakers.values() if v.lower() not in ("unknown", "sfx", "narrator")}

    log.info("characters: %d, speakers: %d/%d (%d unknown, %d distinct named), "
             "noise: %d, address rules: %d",
             len(parsed["characters"]),
             len(speakers), len(flat), unknowns, len(named),
             len(parsed["noise"]), len(parsed["address"]))

    report = [
        "# Combined-context probe (single vision call)",
        "",
        f"- model: {config.vision_agent.model}",
        f"- pages: {reader.page_count}, bubbles: {len(flat)}",
        f"- storyboard: {sb.size[0]}x{sb.size[1]}, {sb_path.stat().st_size // 1024} KB",
        f"- scan: {scan_t:.1f}s, vision: {elapsed:.1f}s, response: {len(text)} chars",
        "",
        "## Parsed",
        "",
        "### Characters",
        "",
    ]
    for c in parsed["characters"]:
        report.append(f"- **{c['name']}** (gender: {c['gender']}, role: {c['role'] or '—'})")
    report.append("")
    report.append("### Speakers")
    report.append("")
    report.append("| key | page | shape | text | speaker |")
    report.append("|---|---|---|---|---|")
    for b in flat:
        sp = speakers.get(b["key"], "—")
        marker = " 🔇" if b["key"] in parsed["noise"] else ""
        tx = b["text"].replace("|", "\\|").replace("\n", " ")[:50]
        report.append(f"| `{b['key']}` | {b['page']} | {b['shape_kind']} | {tx!r} | {sp}{marker} |")
    report.append("")
    report.append(f"### Noise ({len(parsed['noise'])})")
    report.append("")
    if parsed["noise"]:
        report.append("- " + ", ".join(f"`{k}`" for k in sorted(parsed["noise"])))
    else:
        report.append("(none)")
    report.append("")
    report.append("### Style")
    report.append("")
    for k, v in parsed["style"].items():
        report.append(f"- **{k}**: {v}")
    report.append("")
    report.append(f"### Address rules ({len(parsed['address'])})")
    report.append("")
    for a in parsed["address"]:
        report.append(f"- `{a.get('speaker')}` → `{a.get('listener')}`: "
                      f"self={a.get('self', '?')}, other={a.get('other', '?')} "
                      f"({a.get('note', '')})")
    report.append("")
    report.append("## Raw model reply")
    report.append("")
    report.append("```")
    report.append(text)
    report.append("```")

    out_path = OUT / "probe_combined.md"
    out_path.write_text("\n".join(report), "utf-8")
    log.info("wrote %s", out_path)

    print("\n=== SUMMARY ===")
    print(f"vision: {elapsed:.1f}s")
    print(f"characters discovered: {[c['name'] for c in parsed['characters']]}")
    print(f"speaker coverage: {len(speakers) - unknowns}/{len(flat)} "
          f"named ({len(named)} distinct), {unknowns} unknown")
    print(f"noise: {len(parsed['noise'])} keys")
    print(f"style: {parsed['style']}")
    print(f"address rules: {len(parsed['address'])}")


if __name__ == "__main__":
    asyncio.run(main())
