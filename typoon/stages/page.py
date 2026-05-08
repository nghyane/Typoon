"""Window translation — translate a batch of bubbles in one LLM call."""

from __future__ import annotations

import re
import time
import xml.etree.ElementTree as ET
from dataclasses import dataclass

from typoon.adapters.ctx import TranslateCtx
from typoon.stages.brief import ChapterBrief, brief_slice
from typoon.domain.scan import BubbleKey
from typoon.llm.conversation import ConversationBuffer
from typoon.runs.events import LLMCall, LLMResponse

from . import prompt

_RETRIES = 1
_CONTEXT_SIZE = 20
_VALID_KINDS = {"dialogue", "sfx"}
_OCR_NOISE_RE = re.compile(r"^[\W_\d]+$")


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
    """Translate one window of bubbles. Raises on unrecoverable failure."""
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
    buf = ConversationBuffer(system, user)

    remaining = set(active)
    ops: list[TranslationOp] = []

    for attempt in range(_RETRIES + 1):
        ctx.hook.on(LLMCall(agent=f"translate w{window_num+1}/{total_windows}", turn=attempt + 1))
        t0 = time.monotonic()
        resp = await ctx.translation_provider.call(buf.messages(), [])
        ms = (time.monotonic() - t0) * 1000

        parsed = _parse_xml(resp.text or "", remaining, key_map)
        ctx.hook.on(LLMResponse(
            agent=f"translate w{window_num+1}/{total_windows}",
            turn=attempt + 1,
            tool_calls=len(parsed),
            ms=ms,
        ))

        for op in parsed:
            ops.append(op)
            remaining.discard(op.key)

        if not remaining:
            return auto_skipped + ops

        if attempt < _RETRIES:
            buf.append_assistant(resp.text or "")
            buf.append_user(
                f"Missing ids: {', '.join(sorted(remaining))}\n"
                f"Reply with a <translations> block for ONLY these missing ids."
            )

    raise RuntimeError(
        f"translate_window w{window_num+1}: incomplete after {_RETRIES+1} attempts. "
        f"Missing: {', '.join(sorted(remaining))}"
    )


def _build_user(
    context_block: str,
    context_keys: list[str],
    active: set[str],
    key_map: dict[str, BubbleKey],
) -> str:
    lines = []
    for key in context_keys:
        bk = key_map[key]
        active_attr = ' active="true"' if key in active else ""
        lines.append(f'<bubble key="{key}" page="{bk.page_index}"{active_attr}>{bk.source_text}</bubble>')
    return f"{context_block}\n\n" + "\n".join(lines)


def _is_auto_skip(text: str) -> bool:
    s = text.strip()
    if not s:
        return True
    if s.isdigit() or (len(s) == 1 and s.isalpha()):
        return True
    if _OCR_NOISE_RE.fullmatch(s) and not any(ch in s for ch in ":/%$¥₩€£"):
        return True
    return False


def _parse_xml(
    text: str,
    active: set[str],
    key_map: dict[str, BubbleKey],
) -> list[TranslationOp]:
    if "</think>" in text:
        text = text[text.rfind("</think>") + len("</think>"):].strip()

    start = text.find("<translations>")
    end = text.find("</translations>")
    if start == -1 or end == -1:
        return []

    try:
        root = ET.fromstring(text[start: end + len("</translations>")])
    except ET.ParseError:
        return []

    ops: list[TranslationOp] = []
    seen: set[str] = set()

    for t in root.findall("t"):
        key = t.attrib.get("id", "").strip().lstrip("#")
        kind = t.attrib.get("kind", "dialogue").strip().lower()
        translated = (t.text or "").strip()

        if not key or key not in key_map or key not in active or key in seen:
            continue
        if kind not in _VALID_KINDS:
            kind = "dialogue"
        # Auto-skip rubble (single digits/symbols) regardless of what the LLM
        # decided. The LLM cannot output kind="skip" itself — that decision is
        # owned by the context agent (brief.noise_keys) and the deterministic
        # _is_auto_skip filter.
        if _is_auto_skip(key_map[key].source_text):
            kind, translated = "skip", ""
        if kind != "skip" and not translated:
            continue

        seen.add(key)
        ops.append(TranslationOp(key=key, kind=kind, text=translated if kind != "skip" else ""))

    return ops
