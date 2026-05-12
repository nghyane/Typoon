"""Window translation — translate a batch of bubbles in one LLM call.

Wire format (both directions) is line-based, sentinel-prefixed blocks::

    @@ KEY page=3 active
    source line 1
    source line 2
    @@ KEY2 page=3
    context-only source

The model replies with the same sentinel shape, kind in place of page/active::

    @@ KEY dialogue
    translated text
    @@ KEY2 sfx
    RẦM

The format is chosen specifically because the sentinel `@@ <UPPERCASE_KEY>` at
column 0 cannot collide with bubble text and cannot be produced by tag
mirroring — failure modes that plagued the previous XML format.
"""

from __future__ import annotations

import re
import time
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

from typoon.adapters.ctx import TranslateCtx
from typoon.stages.brief import ChapterBrief, brief_slice
from typoon.domain.scan import BubbleKey
from typoon.llm.ir import Message
from typoon.runs.events import LLMCall, LLMResponse

from . import prompt

_CONTEXT_SIZE = 20
_VALID_KINDS = ("dialogue", "sfx")
_OCR_NOISE_RE = re.compile(r"^[\W_\d]+$")
_NOISE_TERMS_PATH = Path(__file__).with_name("skills_data") / "noise_terms.txt"

# Strict header at column 0. Key is uppercase alnum (matches assign_keys output);
# kind is a closed whitelist. Anything else on the line → not a header.
_HEADER_RE = re.compile(r"^@@ ([A-Z0-9]+) (dialogue|sfx)\s*$")


@lru_cache(maxsize=1)
def _noise_term_patterns() -> tuple[re.Pattern[str], ...]:
    """Load and compile deterministic noise regexes once.

    Patterns live in skills_data/noise_terms.txt — one regex per line,
    matched case-insensitively against stripped bubble text. Comment
    lines start with '#'.
    """
    if not _NOISE_TERMS_PATH.exists():
        return ()
    out: list[re.Pattern[str]] = []
    for raw in _NOISE_TERMS_PATH.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        try:
            out.append(re.compile(line, re.IGNORECASE))
        except re.error:
            continue
    return tuple(out)


@dataclass(slots=True)
class TranslationOp:
    key:  str
    kind: str   # dialogue | sfx | skip
    text: str = ""


async def translate_window(
    ctx: TranslateCtx,
    brief: ChapterBrief,
    window_keys: list[str],
    all_keyed: list[BubbleKey],
    *,
    window_num: int = 0,
    total_windows: int = 0,
) -> list[TranslationOp]:
    """Translate one window of bubbles in a single LLM call.

    Returns ops for whichever active keys the model successfully emitted.
    May return fewer ops than active keys; the caller (translate_chapter)
    is responsible for collecting missing keys across windows and issuing
    a single combined retry. Provider errors propagate.
    """
    key_map = {bk.key: bk for bk in all_keyed}

    auto_skipped = [
        TranslationOp(key=k, kind="skip")
        for k in window_keys if _is_auto_skip(key_map[k].source_text)
    ]
    active = {k for k in window_keys if not _is_auto_skip(key_map[k].source_text)}

    if not active:
        return auto_skipped

    ordered = sorted(all_keyed, key=lambda bk: (bk.page_index, bk.idx))
    all_keys = [bk.key for bk in ordered]

    active_positions = [i for i, k in enumerate(all_keys) if k in active]
    lo = max(0, active_positions[0] - _CONTEXT_SIZE)
    hi = min(len(all_keys), active_positions[-1] + _CONTEXT_SIZE + 1)
    context_keys = all_keys[lo:hi]

    page_indices = {key_map[k].page_index for k in context_keys}
    context_block = brief_slice(brief, page_indices, list(active))

    system = prompt.PAGE_SYSTEM.format(
        source_lang=ctx.source_lang,
        target_lang=ctx.target_lang,
        source_policy=prompt.load_source_policy(ctx.source_lang),
        target_policy=prompt.load_target_policy(ctx.target_lang),
    )
    user = _build_user(context_block, context_keys, active, key_map)

    agent = f"translate w{window_num+1}/{total_windows}"
    ctx.hook.on(LLMCall(agent=agent, turn=1))
    t0 = time.monotonic()
    resp = await ctx.translation_provider.call(
        [Message.system(system), Message.user_text(user)], [],
    )
    ms = (time.monotonic() - t0) * 1000

    ops = _parse_blocks(resp.text or "", active, key_map)
    ctx.hook.on(LLMResponse(agent=agent, turn=1, tool_calls=len(ops), ms=ms))

    return auto_skipped + ops


def _build_user(
    context_block: str,
    context_keys: list[str],
    active: set[str],
    key_map: dict[str, BubbleKey],
) -> str:
    blocks: list[str] = []
    for key in context_keys:
        bk = key_map[key]
        flag = " active" if key in active else ""
        blocks.append(f"@@ {key} page={bk.page_index}{flag}\n{bk.source_text}")
    return f"{context_block}\n\n" + "\n".join(blocks)


def _is_auto_skip(text: str) -> bool:
    s = text.strip()
    if not s:
        return True
    if s.isdigit() or (len(s) == 1 and s.isalpha()):
        return True
    if _OCR_NOISE_RE.fullmatch(s) and not any(ch in s for ch in ":/%$¥₩€£"):
        return True
    for pat in _noise_term_patterns():
        if pat.fullmatch(s):
            return True
    return False


def _parse_blocks(
    text: str,
    active: set[str],
    key_map: dict[str, BubbleKey],
) -> list[TranslationOp]:
    """Parse line-sentinel translation blocks. Tolerant of preamble/postamble.

    A block starts on a line matching `^@@ <KEY> <kind>$` and ends at the
    next such header or end of text. Body lines are joined with `\n`,
    stripped of surrounding whitespace. Unknown/duplicate keys are
    dropped. Auto-skip bubbles are forced to kind="skip" regardless of
    what the model emitted.
    """
    if "</think>" in text:
        text = text[text.rfind("</think>") + len("</think>"):]

    # Strip ```...``` code fences if model wrapped its reply.
    text = text.strip()
    if text.startswith("```"):
        first_nl = text.find("\n")
        if first_nl != -1:
            text = text[first_nl + 1:]
        if text.endswith("```"):
            text = text[:-3]

    ops: list[TranslationOp] = []
    seen: set[str] = set()
    current_key: str | None = None
    current_kind: str = "dialogue"
    body: list[str] = []

    def flush() -> None:
        nonlocal current_key
        if current_key is None:
            return
        key, kind = current_key, current_kind
        translated = "\n".join(body).strip()
        current_key = None
        body.clear()

        if key not in key_map or key not in active or key in seen:
            return
        # Auto-skip rubble (single digits/symbols) regardless of what the LLM
        # decided. The LLM cannot output kind="skip" itself — that decision is
        # owned by the context agent (brief.noise_keys) and the deterministic
        # _is_auto_skip filter.
        if _is_auto_skip(key_map[key].source_text):
            seen.add(key)
            ops.append(TranslationOp(key=key, kind="skip"))
            return
        if not translated:
            return
        seen.add(key)
        ops.append(TranslationOp(key=key, kind=kind, text=translated))

    for line in text.splitlines():
        m = _HEADER_RE.match(line)
        if m:
            flush()
            current_key = m.group(1)
            current_kind = m.group(2)
        elif current_key is not None:
            body.append(line)
    flush()

    return ops
