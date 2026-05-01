"""Page/window translation — context window, XML with id, parallel windows."""

from __future__ import annotations

import time
import xml.etree.ElementTree as ET
from dataclasses import dataclass

from typoon.adapters.session import Session
from typoon.domain.scan import Bubble as ScannedBubble
from typoon.llm.ir import Message
from typoon.runs.events import LLMCall, LLMResponse

from . import prompt
from .brief import ChapterBrief, brief_slice

_RETRIES = 2
_CONTEXT_SIZE = 20   # bubbles before/after active window for context
_VALID_KINDS = {"dialogue", "sfx", "skip"}


@dataclass(slots=True)
class TranslationOp:
    key: str
    kind: str   # dialogue | sfx | skip
    text: str = ""


async def translate_window(
    session: Session,
    *,
    brief: ChapterBrief,
    window_keys: list[str],
    key_map: dict[str, ScannedBubble],
    window_num: int = 0,
    total_windows: int = 0,
) -> tuple[list[TranslationOp], int]:
    """Translate a window of keys in one call with surrounding context."""
    # Auto-skip before sending to LLM — no need to waste tokens
    auto_skipped = [
        TranslationOp(key=k, kind="skip", text="")
        for k in window_keys
        if _is_auto_skip(key_map[k].source_text)
    ]
    active = set(k for k in window_keys if not _is_auto_skip(key_map[k].source_text))

    if not active:
        return auto_skipped, 0

    # All bubbles in reading order for context window
    all_ordered = sorted(key_map.items(), key=lambda kv: (kv[1].page_index, kv[1].idx))
    all_keys_ordered = [k for k, _ in all_ordered]

    # Find index range of active keys, expand by _CONTEXT_SIZE each side
    active_indices = [i for i, k in enumerate(all_keys_ordered) if k in active]
    if not active_indices:
        return [], 0

    lo = max(0, active_indices[0] - _CONTEXT_SIZE)
    hi = min(len(all_keys_ordered), active_indices[-1] + _CONTEXT_SIZE + 1)
    context_keys = all_keys_ordered[lo:hi]

    # Build brief slice covering all page indices in context
    page_indices = {key_map[k].page_index for k in context_keys}
    ctx = brief_slice(brief, page_indices, window_keys)

    system = prompt.PAGE_SYSTEM.format(
        source_lang=session.source_lang,
        target_lang=session.target_lang,
        source_policy=prompt.load_policy(f"source_{session.source_lang}.md"),
        target_policy=prompt.load_policy(f"target_{session.target_lang}.md"),
    )

    turns = 0
    remaining = set(window_keys)
    all_ops: list[TranslationOp] = []

    for attempt in range(_RETRIES + 1):
        turns += 1
        user = _build_user(ctx, context_keys, remaining, key_map)
        messages = [Message.system(system), Message.user_text(user)]

        t0 = time.monotonic()
        session.hook.on(LLMCall(agent="translate", turn=window_num + 1))
        resp = await session.provider.call(messages, [])
        ms = (time.monotonic() - t0) * 1000
        text = resp.text or ""

        ops, errors = _parse_xml(text, remaining, key_map)
        resolved = len([op for op in ops if op.key in remaining])
        session.hook.on(LLMResponse(
            agent=f"translate w{window_num + 1}/{total_windows}",
            turn=attempt + 1,
            tool_calls=resolved,
            ms=ms,
        ))

        for op in ops:
            all_ops.append(op)
            remaining.discard(op.key)

        if not remaining:
            return auto_skipped + all_ops, turns

        if attempt < _RETRIES:
            messages.append(Message.assistant(text=text if text.strip() else "…"))
            messages.append(Message.user_text(
                f"Missing ids in your response: {', '.join(sorted(remaining))}\n"
                f"Reply with a <translations> block for ONLY these missing ids."
            ))

    raise RuntimeError(
        f"translate_window incomplete after {turns} turns. "
        f"Missing: {', '.join(sorted(remaining))}"
    )



# ── Prompt builders ───────────────────────────────────────────────────


def _build_user(
    ctx: str,
    context_keys: list[str],
    active: set[str],
    key_map: dict[str, ScannedBubble],
) -> str:
    lines = []
    for key in context_keys:
        b = key_map[key]
        marker = ">>> " if key in active else "    "
        lines.append(f"{marker}[p{b.page_index}] #{key} {b.source_text}")
    annotated = "\n".join(lines)
    return f"{ctx}\n\n{annotated}"


def _is_auto_skip(source_text: str) -> bool:
    """Standalone numbers are always skip — no translation needed."""
    return source_text.strip().isdigit()


# ── XML parsing ───────────────────────────────────────────────────────


def _parse_xml(
    text: str,
    active: set[str],
    key_map: dict[str, ScannedBubble],
) -> tuple[list[TranslationOp], list[str]]:
    """Parse <translations><t id=... kind=...>text</t></translations>."""
    errors: list[str] = []

    # Strip thinking tags
    if "</think>" in text:
        text = text[text.rfind("</think>") + len("</think>"):].strip()

    start = text.find("<translations>")
    end = text.find("</translations>")
    if start == -1 or end == -1:
        return [], ["No <translations> block found."]

    try:
        root = ET.fromstring(text[start: end + len("</translations>")])
    except ET.ParseError as e:
        return [], [f"XML parse error: {e}"]

    ops: list[TranslationOp] = []
    seen: set[str] = set()

    for t in root.findall("t"):
        id_ = t.attrib.get("id", "").strip()
        kind = t.attrib.get("kind", "dialogue").strip().lower()
        translated = (t.text or "").strip()

        if not id_:
            errors.append("t element missing id attribute")
            continue
        if id_ not in key_map:
            errors.append(f"#{id_}: unknown id")
            continue
        if id_ not in active:
            continue  # context key echoed back — ignore
        if id_ in seen:
            errors.append(f"#{id_}: duplicate")
            continue
        if kind not in _VALID_KINDS:
            kind = "dialogue"
        # Override: standalone numbers / single non-alpha chars are always skip
        if _is_auto_skip(key_map[id_].source_text):
            kind = "skip"
            translated = ""
        if kind != "skip" and not translated:
            errors.append(f"#{id_}: empty translation")
            continue

        seen.add(id_)
        ops.append(TranslationOp(
            key=id_,
            kind=kind,
            text=translated if kind != "skip" else "",
        ))

    missing = active - seen
    if missing:
        errors.append(f"Missing ids: {', '.join(sorted(missing))}")

    return ops, errors
