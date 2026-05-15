"""Brief stage — derive ChapterBrief from one vision pass over storyboards.

One vision call per storyboard chunk produces every translator-facing
field plus the noise classification that gates the translator pipeline:

  PreparedChapter + scan → storyboard image(s) → 1 vision call/chunk
    → parse line-sentinel sections → merge → ChapterBrief

This stage does two jobs the translator depends on:

1. Context extraction — characters, speakers, style, glossary.
2. Noise classification — every bubble that is non-diegetic (watermarks,
   site credits, foreign-script overlays, page chrome, URLs) is added
   to `noise_keys`. The translate stage filters these out BEFORE the
   per-bubble LLM call, so by the time a bubble reaches the translator
   it is guaranteed to be real translatable content.

If the vision call fails (provider outage, rate limit after retries), the
stage degrades to an empty ChapterBrief — translator falls back to neutral
phrasing rather than failing the chapter. Per-chunk parse failures are
absorbed; only OperatorActionRequired / UpstreamUnavailable propagate.

Wire format mirrors translate (`@@@ SECTION` + `@@ KEY value` lines).
"""

from __future__ import annotations

import asyncio
import base64
import logging
import re
import shlex
import time
from collections.abc import Iterable
from dataclasses import dataclass

from typoon.adapters.ctx import TranslateCtx
from typoon.adapters.prepared_reader import PreparedReader
from typoon.domain.brief import Character, ChapterBrief
from typoon.domain.scan import BubbleKey
from typoon.llm.errors import OperatorActionRequired, UpstreamUnavailable
from typoon.llm.ir import ContentPart, Message
from typoon.runs.artifacts import ArtifactSink
from typoon.runs.events import LLMCall, LLMResponse
from typoon.stages import prompt
from typoon.stages.noise import is_auto_skip
from typoon.stages.storyboard import build_storyboard, chunk_pages, encode_jpeg


logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Wire format
# ---------------------------------------------------------------------------

_SECTION_RE = re.compile(r"^@@@\s+(\w+)\s*$")
_LINE_RE = re.compile(r"^@@\s+(.*)$")


# ---------------------------------------------------------------------------
# Script detection
# ---------------------------------------------------------------------------

# Cheap regex-only detection of writing systems we have concrete failure
# cases for. Used to flag bubbles whose script does not match the chapter
# source language so the vision agent has an explicit signal that
# something is non-diegetic (e.g. Han-script watermarks bleeding into a
# Spanish chapter). Anything not in this table collapses to "latin".
_SCRIPT_RANGES: tuple[tuple[str, re.Pattern[str]], ...] = (
    ("han",      re.compile(r"[\u3400-\u9fff\uf900-\ufaff]")),
    ("kana",     re.compile(r"[\u3040-\u30ff]")),
    ("hangul",   re.compile(r"[\uac00-\ud7af\u1100-\u11ff]")),
    ("cyrillic", re.compile(r"[\u0400-\u04ff]")),
    ("arabic",   re.compile(r"[\u0600-\u06ff]")),
)

# Languages whose primary script matches a tag above. Anything not
# listed defaults to latin.
_LANG_PRIMARY_SCRIPT: dict[str, str] = {
    "zh": "han",
    "ja": "kana",   # ja text mixes kana+han; either counts as native
    "ko": "hangul",
    "ru": "cyrillic",
    "ar": "arabic",
}


def _detect_scripts(text: str) -> list[str]:
    """Return scripts present in `text`, or `['latin']` for empty/symbol-only.

    May return multiple (e.g. mixed han+latin). Latin is the safe default
    so empty bubbles don't trigger script-mismatch heuristics.
    """
    found = [name for name, pat in _SCRIPT_RANGES if pat.search(text)]
    return found or ["latin"]


def _is_foreign_script(scripts: list[str], source_lang: str) -> bool:
    """True if every detected script is foreign to `source_lang`.

    Mixed-script bubbles (han + latin, kana + latin) are NOT foreign:
    latin is permissive everywhere and we don't want to flag loanwords,
    publisher marks, or punctuation as suspicious.
    """
    if "latin" in scripts:
        return False
    native = _LANG_PRIMARY_SCRIPT.get(source_lang, "latin")
    return all(s != native for s in scripts)


# ---------------------------------------------------------------------------
# Stage entry point
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class _ChunkResult:
    characters:    list[Character]
    speakers:      dict[str, str]   # key → "SPEAKER_NAME [-> LISTENER_NAME]"
    noise:         set[str]
    style:         list[str]
    glossary:      dict[str, str]           # source token → target rendering
    address_pairs: dict[tuple[str, str], str]  # (speaker, listener) → pair
    brief_prose:   str


async def build_chapter_brief(
    ctx: TranslateCtx,
    reader: PreparedReader,
    keyed: list[BubbleKey],
    *,
    artifacts: ArtifactSink | None = None,
) -> ChapterBrief:
    """Run the vision pass; return a ChapterBrief.

    Provider errors propagate (the worker will retry the chapter). All
    other failures (parse errors, missing sections, partial replies)
    degrade silently — the resulting brief just carries less data, which
    the translator handles via neutral fallback.
    """
    if not keyed:
        return ChapterBrief()

    # Fold deterministic noise first — the agent never needs to see
    # bubbles that are trivially watermarks / page counters.
    deterministic_noise = {
        bk.key for bk in keyed if is_auto_skip(bk.bubble.source_text)
    }
    if all(bk.key in deterministic_noise for bk in keyed):
        return ChapterBrief(
            noise_keys=deterministic_noise,
            noise_pages=_full_noise_pages(keyed, deterministic_noise),
        )

    chunks = chunk_pages(reader.page_count)
    results = await asyncio.gather(
        *[
            _process_chunk(
                ctx, reader, keyed, pages,
                chunk_idx=i, total_chunks=len(chunks),
                artifacts=artifacts,
            )
            for i, pages in enumerate(chunks)
        ],
        return_exceptions=True,
    )

    return _merge_chunks(keyed, deterministic_noise, results)


def _merge_chunks(
    keyed: list[BubbleKey],
    deterministic_noise: set[str],
    results: Iterable[_ChunkResult | BaseException],
) -> ChapterBrief:
    """Combine per-chunk results into a single ChapterBrief."""
    characters:    list[Character]             = []
    speakers:      dict[str, str]              = {}
    noise:         set[str]                    = set(deterministic_noise)
    style:         list[str]                   = []
    glossary:      dict[str, str]              = {}
    address_pairs: dict[tuple[str, str], str]  = {}
    brief_parts:   list[str]                   = []
    seen_names:    set[str]                    = set()

    for r in results:
        if isinstance(r, BaseException):
            if isinstance(r, (OperatorActionRequired, UpstreamUnavailable)):
                raise r
            logger.warning("brief chunk failed: %s", r)
            continue
        for c in r.characters:
            key = c.name.casefold()
            if key in seen_names:
                continue
            seen_names.add(key)
            characters.append(c)
        speakers.update(r.speakers)
        noise.update(r.noise)
        for line in r.style:
            if line not in style:
                style.append(line)
        # Glossary: first chunk wins per token (deterministic across retries).
        for src, tgt in r.glossary.items():
            glossary.setdefault(src, tgt)
        # Address pairs: first chunk wins per pair.
        for pair_key, pair_val in r.address_pairs.items():
            address_pairs.setdefault(pair_key, pair_val)
        if r.brief_prose:
            brief_parts.append(r.brief_prose)

    # key_notes: "Speaker: NAME" or "Speaker: NAME -> LISTENER" per bubble.
    # Skip narrator/sfx/unknown — translator handles these via neutral fallback.
    key_notes: dict[str, str] = {}
    for k, raw in speakers.items():
        # raw is "SPEAKER [-> LISTENER]" or just "SPEAKER"
        name_part = raw.strip()
        if not name_part:
            continue
        low = name_part.lower()
        if low in ("narrator", "sfx", "unknown"):
            continue
        # Resolve source name → target display name for the hint text.
        if "->" in name_part:
            sp_raw, _, li_raw = name_part.partition("->")
            sp = sp_raw.strip()
            li = li_raw.strip()
            key_notes[k] = f"Speaker: {sp} → {li}"
        else:
            key_notes[k] = f"Speaker: {name_part}"

    # Combine brief prose across chunks (each chunk covers a page range).
    brief_prose = "\n\n".join(brief_parts)

    return ChapterBrief(
        brief_prose=brief_prose,
        glossary=glossary,
        address_pairs=address_pairs,
        style_notes=style,
        key_notes=key_notes,
        characters=characters,
        noise_keys=noise,
        noise_pages=_full_noise_pages(keyed, noise),
    )


# ---------------------------------------------------------------------------
# Per-chunk vision call
# ---------------------------------------------------------------------------


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
    bubbles = [bk for bk in keyed if bk.bubble.page_index in pages]

    system = prompt.STORYBOARD_SYSTEM.format(
        source_lang_name=prompt.lang_name(ctx.source_lang),
        target_lang_name=prompt.lang_name(ctx.target_lang),
        is_color=reader.chapter().is_color,
        target_agent_policy=prompt.load_target_agent_policy(ctx.target_lang),
    )
    user = _build_brief_prompt(bubbles, ctx.source_lang, ctx.target_lang)

    data_uri = "data:image/jpeg;base64," + base64.b64encode(jpeg).decode()
    messages = [
        Message.system(system),
        Message.user_parts([
            ContentPart.of_text(user),
            ContentPart.of_image(data_uri),
        ]),
    ]

    agent = f"brief c{chunk_idx + 1}/{total_chunks}"
    ctx.hook.on(LLMCall(agent=agent, turn=1))
    t0 = time.monotonic()
    resp = await ctx.vision_provider.call(messages, [])
    ms = (time.monotonic() - t0) * 1000
    ctx.hook.on(LLMResponse(agent=agent, turn=1, tool_calls=0, ms=ms))

    text = resp.text or ""

    if artifacts is not None:
        _record_chunk(artifacts, chunk_idx, pages, jpeg, text)

    return _parse_reply(text, valid_keys={bk.key for bk in bubbles})


def _build_brief_prompt(
    bubbles: list[BubbleKey],
    source_lang: str,
    target_lang: str,
) -> str:
    """Build the user message for one storyboard chunk.

    Each bubble line includes a `script=` / `foreign=` annotation so the
    vision agent has an explicit signal that a bubble's writing system
    does not match the chapter source language — a strong predictor of
    non-diegetic content (watermarks, site credits, foreign overlays).
    """
    header = [
        f"Source language: {source_lang}",
        f"Target language: {target_lang}",
        "",
        "Bubble list (one line per bubble). `script` lists the writing",
        "systems detected in the OCR text; `foreign=1` means the script",
        "does not match the chapter source language and the bubble is a",
        "strong NOISE candidate (watermark, site credit, foreign overlay).",
        "",
    ]
    rows = [_format_bubble_row(bk, source_lang) for bk in bubbles]
    return "\n".join([*header, *rows])


def _format_bubble_row(bk: BubbleKey, source_lang: str) -> str:
    b = bk.bubble
    raw = b.source_text or ""
    preview = raw.replace("\n", " ")[:80] or "(empty)"
    scripts = _detect_scripts(raw)
    foreign = 1 if _is_foreign_script(scripts, source_lang) else 0
    return (
        f"@@ {bk.key} page={b.page_index} kind={b.shape_kind} "
        f"script={'+'.join(scripts)} foreign={foreign} text={preview!r}"
    )


def _record_chunk(
    artifacts: ArtifactSink,
    chunk_idx: int,
    pages: range,
    jpeg: bytes,
    reply: str,
) -> None:
    artifacts.write_bytes(
        "05_brief",
        f"storyboard_{chunk_idx:02d}_p{pages.start}-{pages.stop - 1}.jpg",
        jpeg,
    )
    artifacts.write_bytes(
        "05_brief",
        f"reply_{chunk_idx:02d}.txt",
        reply.encode("utf-8"),
    )


# ---------------------------------------------------------------------------
# Reply parsing
# ---------------------------------------------------------------------------


def _parse_reply(text: str, *, valid_keys: set[str]) -> _ChunkResult:
    """Parse a section-sentinel reply into a `_ChunkResult`.

    Tolerant: unknown sections are ignored, malformed lines skipped,
    keys not in `valid_keys` dropped to prevent the model from inventing
    references.
    """
    sections = _split_sections(_strip_think(text))
    return _ChunkResult(
        characters=_parse_characters(sections.get("CHARACTERS", [])),
        speakers=_parse_speakers(sections.get("SPEAKERS", []), valid_keys),
        noise=_parse_noise(sections.get("NOISE", []), valid_keys),
        style=_parse_style(sections.get("STYLE", [])),
        glossary=_parse_glossary(sections.get("GLOSSARY", [])),
        address_pairs=_parse_address(sections.get("ADDRESS", [])),
        brief_prose=_parse_brief(sections.get("BRIEF", [])),
    )


def _strip_think(text: str) -> str:
    """Drop any `<think>...</think>` reasoning prelude the model emits."""
    if "</think>" in text:
        return text[text.rfind("</think>") + len("</think>"):]
    return text


def _split_sections(text: str) -> dict[str, list[str]]:
    """Group body lines under their preceding `@@@ SECTION` header.

    For most sections, only `@@ ...` lines are collected (structured data).
    For the BRIEF section, ALL non-empty lines are collected as free prose —
    the agent writes that section as natural-language text without `@@` prefixes.
    """
    PROSE_SECTIONS = {"BRIEF"}
    sections: dict[str, list[str]] = {}
    current: str | None = None
    for raw in text.splitlines():
        line = raw.rstrip()
        if m := _SECTION_RE.match(line):
            current = m.group(1).upper()
            sections.setdefault(current, [])
            continue
        if current is None:
            continue
        if current in PROSE_SECTIONS:
            # Collect every non-empty line as prose.
            if line.strip():
                # Strip leading @@ prefix if agent mistakenly adds it.
                content = _LINE_RE.match(line)
                sections[current].append(content.group(1).strip() if content else line.strip())
        elif m := _LINE_RE.match(line):
            sections[current].append(m.group(1).strip())
    return sections


def _parse_characters(bodies: list[str]) -> list[Character]:
    out: list[Character] = []
    for body in bodies:
        kv = _parse_kv(body)
        name = kv.get("name", "").strip()
        if not name:
            continue
        out.append(Character(
            name=name,
            target_name=kv.get("target", "").strip(),
            gender=(kv.get("gender", "").strip().lower() or "unknown"),
            role=kv.get("role", "").strip(),
            voice=kv.get("voice", "").strip(),
        ))
    return out


def _parse_speakers(bodies: list[str], valid_keys: set[str]) -> dict[str, str]:
    """Parse SPEAKERS lines: `KEY SPEAKER_NAME [-> LISTENER_NAME]`.

    Returns raw string value per key so _merge_chunks can build key_notes
    with full speaker→listener info when available.
    """
    out: dict[str, str] = {}
    for body in bodies:
        # Format: KEY REST (where REST may be "Name" or "Name -> Other")
        parts = body.split(None, 1)
        if len(parts) != 2:
            continue
        key, rest = parts[0].strip(), parts[1].strip()
        if key in valid_keys:
            out[key] = rest
    return out


def _parse_glossary(bodies: list[str]) -> dict[str, str]:
    """Parse GLOSSARY lines: `SOURCE_TOKEN = TARGET_RENDERING`.

    Strips surrounding quotes (the model sometimes emits `"token" = "value"`).
    Skips: identity-map entries, URLs/domains, known platform brand tokens.
    """
    import re
    _url_re      = re.compile(r'\.(com|net|org|io|app|id)(/|$)', re.I)
    _platform_re = re.compile(
        r'包子|baozi|快看|kuaikan|sfacg|baozimh|快快看|漫画.*网|小说.*网|patreon|discord|ko.fi',
        re.I,
    )
    out: dict[str, str] = {}
    for body in bodies:
        if "=" not in body:
            continue
        src, _, tgt = body.partition("=")
        src = src.strip().strip('"\'')
        tgt = tgt.strip().strip('"\'')
        if not src or not tgt:
            continue
        if src == tgt:
            continue
        if _url_re.search(src):
            continue
        if _platform_re.search(src):
            continue
        out[src] = tgt
    return out


def _parse_address(bodies: list[str]) -> dict[tuple[str, str], str]:
    """Parse ADDRESS lines: `SPEAKER → LISTENER: PAIR`.

    Tolerates both ASCII `->` and Unicode `→` as the arrow.
    Drops pairs where speaker or listener is "unknown" — those carry
    no usable information for the translator.
    """
    out: dict[tuple[str, str], str] = {}
    for body in bodies:
        body = body.replace("→", "->")
        if "->" not in body or ":" not in body:
            continue
        speaker_listener, _, pair = body.rpartition(":")
        speaker, _, listener = speaker_listener.partition("->")
        speaker  = speaker.strip()
        listener = listener.strip()
        pair     = pair.strip()
        if not speaker or not listener or not pair:
            continue
        if speaker.lower() == "unknown" or listener.lower() == "unknown":
            continue
        out[(speaker, listener)] = pair
    return out


def _parse_brief(bodies: list[str]) -> str:
    """BRIEF is free prose — join all non-@@ lines under the section."""
    return "\n".join(bodies).strip()


def _parse_noise(bodies: list[str], valid_keys: set[str]) -> set[str]:
    out: set[str] = set()
    for body in bodies:
        first = body.split(maxsplit=1)[0] if body else ""
        if first in valid_keys:
            out.add(first)
    return out


def _parse_style(bodies: list[str]) -> list[str]:
    out: list[str] = []
    for body in bodies:
        kv = _parse_kv(body)
        if kv:
            for k, v in kv.items():
                v = v.strip()
                if v:
                    out.append(f"{k}: {v}")
        elif body:
            out.append(body)
    return out


def _parse_kv(line: str) -> dict[str, str]:
    """Parse `key=value key="value with spaces" ...` into a dict.

    Built on `shlex` for posix-style quoting. Unterminated quotes yield
    an empty dict rather than partial garbage — the prompt forbids them
    and we'd rather drop the line than guess.
    """
    if not line.strip():
        return {}
    lexer = shlex.shlex(line, posix=True)
    lexer.whitespace_split = True
    lexer.commenters = ""
    out: dict[str, str] = {}
    try:
        for tok in lexer:
            if "=" not in tok:
                continue
            key, _, value = tok.partition("=")
            if key:
                out[key] = value
    except ValueError:
        return {}
    return out


# ---------------------------------------------------------------------------
# Helpers for callers
# ---------------------------------------------------------------------------


def _full_noise_pages(keyed: list[BubbleKey], noise_keys: set[str]) -> set[int]:
    """Pages where every bubble is noise. Used for render-time skip.

    Partial coverage never escalates a page to noise — that risks
    dropping legitimate story pages.
    """
    by_page: dict[int, list[str]] = {}
    for bk in keyed:
        by_page.setdefault(bk.bubble.page_index, []).append(bk.key)
    return {
        page for page, keys in by_page.items()
        if keys and all(k in noise_keys for k in keys)
    }


def brief_slice(
    brief: ChapterBrief,
    page_indices: set[int],
    keys: list[str],
) -> str:
    """Render the subset of brief data relevant to one translation window.

    Injects into the translator's user message. Order matters — most
    important context first:
      1. BRIEF prose (tradition, genre, pacing, fallback register)
      2. Glossary (resolved name/term renderings)
      3. Address table (confirmed xưng hô pairs)
      4. Characters (voice descriptors)
      5. Speaker hints (per-bubble speaker → listener)
    """
    _ = page_indices  # reserved for per-page filtering

    parts: list[str] = []

    if brief.brief_prose:
        parts.append(f"## Chapter brief\n{brief.brief_prose}")

    if brief.glossary:
        parts.append("## Glossary\n" + "\n".join(
            f"- {src} → {tgt}" for src, tgt in brief.glossary.items()
        ))

    if brief.address_pairs:
        parts.append("## Address\n" + "\n".join(
            f"- {sp} → {li}: {pair}"
            for (sp, li), pair in brief.address_pairs.items()
        ))

    if brief.characters:
        parts.append("## Characters\n" + "\n".join(
            _render_character(c) for c in brief.characters
        ))

    # Legacy style_notes: only emit when no brief_prose (old chapters).
    if brief.style_notes and not brief.brief_prose:
        parts.append("## Style\n" + "\n".join(f"- {n}" for n in brief.style_notes))

    relevant_notes = [
        f"#{k}: {brief.key_notes[k]}"
        for k in keys if k in brief.key_notes
    ]
    if relevant_notes:
        parts.append("## Speaker hints\n" + "\n".join(relevant_notes))

    return "\n\n".join(parts) if parts else "(none)"


def _render_character(c: Character) -> str:
    display = c.display_name
    gender  = c.gender if c.gender and c.gender != "unknown" else "?"
    line    = f"- {display} ({gender})"
    if c.role:
        line += f": {c.role}"
    if c.voice:
        line += f" — voice: {c.voice}"
    return line
