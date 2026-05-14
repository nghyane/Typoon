"""scan_context — derive ChapterBrief from one vision pass over storyboards.

A single vision call per storyboard chunk produces every translator-facing
field:

  PreparedChapter + scan → storyboard image(s) → 1 vision call/storyboard
    → parse line-sentinel sections → merge across chunks → ChapterBrief

No tools, no retries, no submit_chapter_brief reject path. If the vision
call fails (provider outage, rate limit after retries), we degrade
gracefully to an empty ChapterBrief — the translator falls back to
neutral phrasing rather than failing the chapter.

Wire format mirrors translate (`@@@ SECTION` + `@@ KEY value` lines) so a
single parser style is used across the project.
"""

from __future__ import annotations

import asyncio
import logging
import re
import time
from dataclasses import dataclass

from typoon.adapters.ctx import TranslateCtx
from typoon.adapters.prepared_reader import PreparedReader
from typoon.domain.scan import BubbleKey
from typoon.llm.errors import OperatorActionRequired, UpstreamUnavailable
from typoon.llm.ir import ContentPart, Message
from typoon.runs.artifacts import ArtifactSink
from typoon.runs.events import LLMCall, LLMResponse
from typoon.stages.brief import Character, ChapterBrief
from typoon.stages.storyboard import (
    build_storyboard,
    chunk_pages,
    encode_jpeg,
)
from typoon.stages import prompt

logger = logging.getLogger(__name__)

# Mirror the translate stage's noise heuristic: a key that the
# deterministic auto-skip layer catches is also marked as noise in the
# brief so render-time skip_pages logic stays in sync.
from typoon.stages.page import _is_auto_skip


_SECTION_RE = re.compile(r"^@@@\s+(\w+)\s*$")
_LINE_RE = re.compile(r"^@@\s+(.*)$")


@dataclass(frozen=True)
class _ChunkResult:
    characters: list[Character]
    speakers:   dict[str, str]
    noise:      set[str]
    style:      list[str]


async def build_chapter_context(
    ctx: TranslateCtx,
    reader: PreparedReader,
    keyed: list[BubbleKey],
    *,
    artifacts: ArtifactSink | None = None,
) -> ChapterBrief:
    """Run the vision context pass; return a ChapterBrief.

    Provider errors propagate (the worker will retry the chapter). All
    other failures (parse errors, missing sections, partial replies)
    degrade silently — the resulting brief just carries less data,
    which the translator handles via neutral fallback.
    """
    if not keyed:
        return ChapterBrief()

    # Fold deterministic noise first — the agent never needs to see
    # bubbles that are trivially watermarks / page counters.
    deterministic_noise: set[str] = {
        bk.key for bk in keyed if _is_auto_skip(bk.bubble.source_text)
    }
    visible_keyed = [bk for bk in keyed if bk.key not in deterministic_noise]

    if not visible_keyed:
        return ChapterBrief(
            noise_keys=deterministic_noise,
            noise_pages=_full_noise_pages(keyed, deterministic_noise),
        )

    page_count = reader.page_count
    chunks = chunk_pages(page_count)

    # Fan out one vision call per storyboard, bounded by the provider
    # semaphore.
    tasks = [
        _process_chunk(ctx, reader, keyed, pages, chunk_idx=i,
                       total_chunks=len(chunks), artifacts=artifacts)
        for i, pages in enumerate(chunks)
    ]
    results = await asyncio.gather(*tasks, return_exceptions=True)

    merged_chars: list[Character] = []
    merged_speakers: dict[str, str] = {}
    merged_noise: set[str] = set(deterministic_noise)
    merged_style: list[str] = []
    seen_char_names: set[str] = set()

    for r in results:
        if isinstance(r, BaseException):
            # OperatorActionRequired / UpstreamUnavailable bubble up;
            # parse-level failures are absorbed (returned as empty chunk).
            if isinstance(r, (OperatorActionRequired, UpstreamUnavailable)):
                raise r
            logger.warning("scan_context chunk failed: %s", r)
            continue
        for c in r.characters:
            key = c.name.casefold()
            if key in seen_char_names:
                continue
            seen_char_names.add(key)
            merged_chars.append(c)
        merged_speakers.update(r.speakers)
        merged_noise.update(r.noise)
        # Style lines: dedupe by content, preserve order of first sighting.
        for line in r.style:
            if line not in merged_style:
                merged_style.append(line)

    noise_pages = _full_noise_pages(keyed, merged_noise)

    # Translator's key_notes carries speaker hint per bubble: derived
    # from speakers map, with narrator/sfx/unknown collapsed to a
    # one-word note the translator prompt is already tuned to read.
    key_notes: dict[str, str] = {}
    for k, sp in merged_speakers.items():
        s = sp.strip()
        if not s or s.lower() == "unknown":
            continue
        key_notes[k] = f"Speaker: {s}"

    glossary: dict[str, str] = {}
    for c in merged_chars:
        # Source-side name as-detected; target left blank because the
        # vision pass works in source language. Translator decides VI
        # rendering using glossary + style + speaker hint together.
        glossary[c.name] = c.name

    return ChapterBrief(
        glossary=glossary,
        style_notes=merged_style,
        key_notes=key_notes,
        characters=merged_chars,
        noise_keys=merged_noise,
        noise_pages=noise_pages,
    )


async def _process_chunk(
    ctx: TranslateCtx,
    reader: PreparedReader,
    keyed: list[BubbleKey],
    pages: range,
    *,
    chunk_idx: int,
    total_chunks: int,
    artifacts: ArtifactSink | None,
) -> _ChunkResult:
    """Build storyboard for one page range, run vision, parse reply."""
    storyboard = build_storyboard(reader, keyed, pages)
    jpeg = encode_jpeg(storyboard.image)

    if artifacts is not None:
        artifacts.write_bytes(
            "05_context",
            f"storyboard_{chunk_idx:02d}_p{pages.start}-{pages.stop - 1}.jpg",
            jpeg,
        )

    bubbles = [bk for bk in keyed if bk.bubble.page_index in pages]
    user = _build_user(bubbles, ctx.target_lang)
    system = prompt.STORYBOARD_SYSTEM.format(target_lang=ctx.target_lang)

    import base64
    data_uri = "data:image/jpeg;base64," + base64.b64encode(jpeg).decode()
    messages = [
        Message.system(system),
        Message.user_parts([
            ContentPart.of_text(user),
            ContentPart.of_image(data_uri),
        ]),
    ]

    agent = f"scan_context c{chunk_idx + 1}/{total_chunks}"
    ctx.hook.on(LLMCall(agent=agent, turn=1))
    t0 = time.monotonic()
    resp = await ctx.vision_provider.call(messages, [])
    ms = (time.monotonic() - t0) * 1000
    ctx.hook.on(LLMResponse(agent=agent, turn=1, tool_calls=0, ms=ms))

    text = resp.text or ""
    if artifacts is not None:
        artifacts.write_bytes(
            "05_context",
            f"reply_{chunk_idx:02d}.txt",
            text.encode("utf-8"),
        )

    return _parse_reply(text, valid_keys={bk.key for bk in bubbles})


def _build_user(bubbles: list[BubbleKey], target_lang: str) -> str:
    lines = [
        f"Target language: {target_lang}",
        "",
        "Bubble list (one line per bubble):",
    ]
    for bk in bubbles:
        b = bk.bubble
        text = (b.source_text or "").replace("\n", " ")[:80] or "(empty)"
        lines.append(
            f"@@ {bk.key} page={b.page_index} kind={b.shape_kind} text={text!r}"
        )
    return "\n".join(lines)


def _parse_reply(text: str, *, valid_keys: set[str]) -> _ChunkResult:
    """Parse the line-sentinel reply into a chunk result.

    Tolerant: unknown sections are ignored, malformed lines are skipped,
    keys not in `valid_keys` are dropped to prevent the model from
    inventing references.
    """
    # Strip any reasoning prelude the model might emit.
    if "</think>" in text:
        text = text[text.rfind("</think>") + len("</think>"):]

    sections: dict[str, list[str]] = {}
    current: str | None = None
    for raw in text.splitlines():
        line = raw.rstrip()
        m = _SECTION_RE.match(line)
        if m:
            current = m.group(1).upper()
            sections.setdefault(current, [])
            continue
        m = _LINE_RE.match(line)
        if m and current is not None:
            sections[current].append(m.group(1).strip())

    characters: list[Character] = []
    for body in sections.get("CHARACTERS", []):
        kv = _parse_kv(body)
        name = kv.get("name", "").strip()
        if not name:
            continue
        characters.append(Character(
            name=name,
            gender=kv.get("gender", "unknown").strip().lower() or "unknown",
            role=kv.get("role", "").strip(),
        ))

    speakers: dict[str, str] = {}
    for body in sections.get("SPEAKERS", []):
        parts = body.split(None, 1)
        if len(parts) != 2:
            continue
        key, speaker = parts[0].strip(), parts[1].strip()
        if key not in valid_keys:
            continue
        speakers[key] = speaker

    noise: set[str] = set()
    for body in sections.get("NOISE", []):
        first = body.split()[0] if body else ""
        if first in valid_keys:
            noise.add(first)

    style: list[str] = []
    for body in sections.get("STYLE", []):
        kv = _parse_kv(body)
        if not kv:
            # Plain line — keep as-is
            if body:
                style.append(body)
            continue
        for k, v in kv.items():
            v = v.strip()
            if v:
                style.append(f"{k}: {v}")

    return _ChunkResult(
        characters=characters,
        speakers=speakers,
        noise=noise,
        style=style,
    )


def _parse_kv(line: str) -> dict[str, str]:
    """Parse `key=value key="value with spaces" ...` into a dict.

    Quoted values capture everything up to the matching closing quote
    (no escape sequences — the prompt forbids embedded quotes in values).
    Unquoted values are read up to the next whitespace.
    """
    out: dict[str, str] = {}
    i, n = 0, len(line)
    while i < n:
        # Skip whitespace
        while i < n and line[i].isspace():
            i += 1
        if i >= n:
            break
        # Read key
        eq = line.find("=", i)
        if eq == -1:
            break
        key = line[i:eq].strip()
        i = eq + 1
        if i < n and line[i] == '"':
            end = line.find('"', i + 1)
            if end == -1:
                val = line[i + 1:]
                i = n
            else:
                val = line[i + 1:end]
                i = end + 1
        else:
            end = i
            while end < n and not line[end].isspace():
                end += 1
            val = line[i:end]
            i = end
        if key:
            out[key] = val
    return out


def _full_noise_pages(keyed: list[BubbleKey], noise_keys: set[str]) -> set[int]:
    """A page becomes noise-page only when every bubble on it is noise.

    Translates render-time `skip_pages` for full-page credits/ads. We
    never escalate a page to noise based on partial coverage — that
    risks dropping legitimate story pages.
    """
    by_page: dict[int, list[str]] = {}
    for bk in keyed:
        by_page.setdefault(bk.bubble.page_index, []).append(bk.key)
    noise_pages: set[int] = set()
    for page, keys in by_page.items():
        if keys and all(k in noise_keys for k in keys):
            noise_pages.add(page)
    return noise_pages
