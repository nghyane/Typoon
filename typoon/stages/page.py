"""Window translation — translate a batch of bubbles in one LLM call.

Wire format is line-based, sentinel-prefixed blocks. Input uses `>>>`,
output uses `@@` — the asymmetry is deliberate: it prevents the model
from "mirroring" an input header back as a fake output header, which
was the dominant failure mode of the previous symmetric format.

Input we send to the model::

    >>> KEY page=3 active w=280 h=60 lines=2
    source line 1
    source line 2
    >>> KEY2 page=3
    context-only source

Output we parse back::

    @@ KEY dialogue
    translated text
    @@ KEY2 sfx
    RẦM
    @@ KEY3 skip

`@@ <7-char KEY> <dialogue|sfx|skip>` at column 0 cannot collide with bubble
text. Keys are opaque to the model — generated from chapter_id + bubble
position by `stages.keys.assign_keys`. `skip` is the translator's escape
hatch for chrome that leaked past upstream noise filters (see page.md
"Embedded chrome").
"""

from __future__ import annotations

import re
import time
from collections.abc import Iterator
from dataclasses import dataclass
from typing import NamedTuple

from typoon.adapters.ctx import TranslateCtx
from typoon.domain.brief import ChapterBrief
from typoon.domain.scan import BubbleKey
from typoon.llm.ir import Message
from typoon.runs.artifacts import ArtifactSink
from typoon.runs.events import LLMCall, LLMResponse
from typoon.stages import prompt
from typoon.stages.brief import brief_slice
from typoon.stages.noise import is_auto_skip


_CONTEXT_SIZE = 20

# Strict header at column 0. Output sentinel is `@@` (distinct from input
# sentinel `>>>` so the model cannot accidentally mirror an input header
# back as a fake output header). Key is exactly 7 chars of [A-Z0-9] —
# matches assign_keys output (keys.py: 7 chars from a 32-char alphabet).
# Kind is a closed whitelist; we accept any case and lowercase it on parse.
# `skip` is allowed so the model can drop chrome bubbles that leaked
# past the upstream noise filter (see page.md "Embedded chrome").
_HEADER_RE = re.compile(r"^@@ ([A-Z0-9]{7}) (dialogue|sfx|skip)\s*$", re.IGNORECASE)


@dataclass(slots=True)
class TranslationOp:
    key:  str
    kind: str   # dialogue | sfx | skip
    text: str = ""


class _ParsedBlock(NamedTuple):
    """A `@@ KEY kind` block extracted from the model reply, pre-validation."""
    key:  str
    kind: str
    text: str


async def translate_window(
    ctx: TranslateCtx,
    brief: ChapterBrief,
    window_keys: list[str],
    all_keyed: list[BubbleKey],
    *,
    window_num: int = 0,
    total_windows: int = 0,
    artifacts: ArtifactSink | None = None,
    window_tag: str | None = None,
) -> list[TranslationOp]:
    """Translate one window of bubbles in a single LLM call.

    Returns ops for whichever active keys the model successfully emitted.
    May return fewer ops than active keys; the caller (translate_chapter)
    is responsible for collecting missing keys across windows and issuing
    a single combined retry. Provider errors propagate.

    If `artifacts` is provided, writes the exact prompt sent, the raw
    response, and a parsed summary under `06_translate/` keyed by
    `window_tag` (defaults to `w{window_num:02d}`). This is the primary
    debug signal when keys go missing — read response.txt to see what
    the model actually emitted.
    """
    key_map = {bk.key: bk for bk in all_keyed}

    auto_skipped, active = _partition_window(window_keys, key_map)
    if not active:
        return auto_skipped

    context_keys = _context_window(all_keyed, active)
    context_block = brief_slice(
        brief,
        page_indices={key_map[k].page_index for k in context_keys},
        keys=list(active),
    )

    system = prompt.PAGE_SYSTEM.format(
        source_lang_name=prompt.lang_name(ctx.source_lang),
        target_lang_name=prompt.lang_name(ctx.target_lang),
        source_policy=prompt.load_source_policy(ctx.source_lang),
        target_policy=prompt.load_target_policy(ctx.target_lang),
    )
    user = _build_window_prompt(context_block, context_keys, active, key_map)

    agent = f"translate w{window_num + 1}/{total_windows}"
    ctx.hook.on(LLMCall(agent=agent, turn=1))
    t0 = time.monotonic()
    resp = await ctx.translation_provider.call(
        [Message.system(system), Message.user_text(user)], [],
    )
    ms = (time.monotonic() - t0) * 1000

    response_text = resp.text or ""
    ops = _parse_translation_reply(response_text, active, key_map)
    ctx.hook.on(LLMResponse(agent=agent, turn=1, tool_calls=len(ops), ms=ms))

    if artifacts is not None:
        _record_window(
            artifacts,
            tag=window_tag or f"w{window_num:02d}",
            window_num=window_num,
            total_windows=total_windows,
            system=system,
            user=user,
            response=response_text,
            active=active,
            auto_skipped=auto_skipped,
            ops=ops,
            latency_ms=ms,
        )

    return auto_skipped + ops


# ---------------------------------------------------------------------------
# Window assembly
# ---------------------------------------------------------------------------


def _partition_window(
    window_keys: list[str],
    key_map: dict[str, BubbleKey],
) -> tuple[list[TranslationOp], set[str]]:
    """Split a window's keys into (deterministic-skip ops, active set)."""
    auto_skipped: list[TranslationOp] = []
    active: set[str] = set()
    for key in window_keys:
        if is_auto_skip(key_map[key].source_text):
            auto_skipped.append(TranslationOp(key=key, kind="skip"))
        else:
            active.add(key)
    return auto_skipped, active


def _context_window(
    all_keyed: list[BubbleKey],
    active: set[str],
) -> list[str]:
    """Return the contiguous key range covering `active` plus ±_CONTEXT_SIZE neighbours."""
    ordered = sorted(all_keyed, key=lambda bk: (bk.page_index, bk.idx))
    all_keys = [bk.key for bk in ordered]
    positions = [i for i, k in enumerate(all_keys) if k in active]
    lo = max(0, positions[0] - _CONTEXT_SIZE)
    hi = min(len(all_keys), positions[-1] + _CONTEXT_SIZE + 1)
    return all_keys[lo:hi]


def _build_window_prompt(
    context_block: str,
    context_keys: list[str],
    active: set[str],
    key_map: dict[str, BubbleKey],
) -> str:
    """Render the user message for one translation window.

    Each active bubble header includes `w=`, `h=`, `lines=` so the
    translator LLM can make informed line-break decisions.
    """
    blocks = []
    for key in context_keys:
        bk = key_map[key]
        flag = " active" if key in active else ""
        dims = _bubble_dims(bk)
        blocks.append(f">>> {key} page={bk.page_index}{flag}{dims}\n{bk.source_text}")
    return f"{context_block}\n\n" + "\n".join(blocks)


def _bubble_dims(bk: BubbleKey) -> str:
    """Return ` w=NNN h=NNN lines=N` string for active bubble headers.

    Derived from the bubble's fit_box [x, y, w, h] in prepared-page
    pixels. `lines` is estimated from height / (font_size * line_height)
    using the source font size hint when available, falling back to
    height / 28 (≈ median comic font at typical resolution).
    """
    fit = bk.bubble.box.fit  # [x, y, w, h]
    w, h = fit[2], fit[3]
    if w <= 0 or h <= 0:
        return ""
    # Source font hint → estimate lines; fallback to h/28.
    src_font = bk.bubble.src_font_size_px or 0
    if src_font > 0:
        line_h = src_font * 1.22
        lines = max(1, round(h / line_h))
    else:
        lines = max(1, round(h / 28))
    return f" w={w} h={h} lines={lines}"


# ---------------------------------------------------------------------------
# Reply parsing
# ---------------------------------------------------------------------------


def _strip_envelope(text: str) -> str:
    """Strip reasoning prelude (<think>...</think>) and ```fence``` wrappers."""
    if "</think>" in text:
        text = text[text.rfind("</think>") + len("</think>"):]
    text = text.strip()
    if text.startswith("```"):
        first_nl = text.find("\n")
        if first_nl != -1:
            text = text[first_nl + 1:]
        if text.endswith("```"):
            text = text[:-3]
    return text


def _iter_raw_blocks(text: str) -> Iterator[_ParsedBlock]:
    """Yield each `@@ KEY kind\\nbody...` block found in `text`.

    A block runs from one header line up to the next header line or EOF.
    Body whitespace is stripped. No semantic validation — caller decides
    whether to keep the block. Yields blocks in source order.
    """
    text = _strip_envelope(text)
    key: str | None = None
    kind: str = "dialogue"
    body: list[str] = []

    for line in text.splitlines():
        m = _HEADER_RE.match(line)
        if m:
            if key is not None:
                yield _ParsedBlock(key, kind, "\n".join(body).strip())
            key = m.group(1)
            # IGNORECASE on kind → normalize so downstream sees canonical form.
            kind = m.group(2).lower()
            body = []
        elif key is not None:
            body.append(line)

    if key is not None:
        yield _ParsedBlock(key, kind, "\n".join(body).strip())


def _parse_translation_reply(
    text: str,
    active: set[str],
    key_map: dict[str, BubbleKey],
) -> list[TranslationOp]:
    """Parse the model reply into TranslationOps for `active` keys only.

    Tolerant of preamble/postamble. Drops unknown keys, inactive keys,
    duplicates, and empty bodies. Auto-skip bubbles are forced to
    `kind="skip"` regardless of what the model emitted — that decision
    is owned by the brief stage and the deterministic noise filter, not
    by the translator LLM.
    """
    ops: list[TranslationOp] = []
    seen: set[str] = set()

    for block in _iter_raw_blocks(text):
        if block.key not in key_map or block.key not in active or block.key in seen:
            continue
        seen.add(block.key)
        if is_auto_skip(key_map[block.key].source_text):
            ops.append(TranslationOp(key=block.key, kind="skip"))
            continue
        if block.kind == "skip":
            # Model declared the bubble is leaked chrome — honor it,
            # body is ignored. This is how the translator drops noise
            # that the upstream brief/scan filters missed (e.g. brand
            # names glued onto OCR output as their own bubble).
            ops.append(TranslationOp(key=block.key, kind="skip"))
            continue
        if not block.text:
            # Empty body for a real bubble — treat as missing, not skip.
            # Caller's retry pass gets a second chance.
            seen.discard(block.key)
            continue
        ops.append(TranslationOp(key=block.key, kind=block.kind, text=block.text))

    return ops


# ---------------------------------------------------------------------------
# Artifacts
# ---------------------------------------------------------------------------


def _record_window(
    artifacts: ArtifactSink,
    *,
    tag: str,
    window_num: int,
    total_windows: int,
    system: str,
    user: str,
    response: str,
    active: set[str],
    auto_skipped: list[TranslationOp],
    ops: list[TranslationOp],
    latency_ms: float,
) -> None:
    """Persist per-window debug artifacts under `06_translate/`.

    The diagnostic flow when a key goes missing:
      1. Open `unresolved.json` / `missing_after_pass1.json` — which keys, which sources.
      2. Open the matching `{tag}_response.txt` — what the model actually said.
      3. Open `{tag}_parsed.json` — what we parsed out of it, by key.
    """
    emitted = {op.key for op in ops}
    missing = sorted(k for k in active if k not in emitted)

    artifacts.write_bytes("06_translate", f"{tag}_system.txt", system.encode("utf-8"))
    artifacts.write_bytes("06_translate", f"{tag}_prompt.txt", user.encode("utf-8"))
    artifacts.write_bytes("06_translate", f"{tag}_response.txt", response.encode("utf-8"))
    artifacts.write_json("06_translate", f"{tag}_parsed.json", {
        "window_num":     window_num,
        "total_windows":  total_windows,
        "active_keys":    sorted(active),
        "auto_skipped":   [op.key for op in auto_skipped],
        "emitted":        [
            {"key": op.key, "kind": op.kind, "chars": len(op.text)}
            for op in ops
        ],
        "missing":        missing,
        "response_chars": len(response),
        "latency_ms":     round(latency_ms, 1),
    })
