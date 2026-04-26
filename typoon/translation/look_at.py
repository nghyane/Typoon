"""Page-level visual clarification subagent."""

from __future__ import annotations

import re
import time

from typoon.app.events import LLMCall, LLMResponse, PipelineError
from typoon.domain.bubble import Session
from typoon.llm.ir import ContentPart, Message

from . import prompt
from .tools.view_page import encode_page_jpeg

_NOTE_RE = re.compile(r"^\s*#?([A-Z2-9]{6,8})\s*:\s*(.+)$")


async def look_at_page(
    session: Session,
    *,
    page_index: int,
    keys: list[str],
    query: str,
    source_by_key: dict[str, str],
    turn: int,
) -> dict[str, str]:
    if session.source is None or not hasattr(session.source, "load_page"):
        return {}
    related = "\n".join(f"#{k}: {source_by_key.get(k, '')}" for k in keys)
    user = prompt.LOOKAT_USER.format(page_index=page_index, query=query, related_text=related)
    parts = [ContentPart.of_text(user)]
    try:
        img = session.source.load_page(page_index)
        parts.append(ContentPart.of_image(encode_page_jpeg(img)))
    except Exception:
        return {}

    hook = session.hook
    hook.on(LLMCall(agent="translate/lookat", turn=turn))
    t0 = time.monotonic()
    try:
        resp = await session.context_provider.call(
            [Message.system(prompt.LOOKAT_SYSTEM), Message.user_parts(parts)], tools=[],
        )
    except Exception as e:
        hook.on(PipelineError(stage="translate/lookat", error=e))
        return {}
    hook.on(LLMResponse(agent="translate/lookat", turn=turn, tool_calls=0, ms=(time.monotonic() - t0) * 1000))
    allowed = set(keys)
    notes: dict[str, str] = {}
    for line in (resp.text or "").splitlines():
        m = _NOTE_RE.match(line)
        if m and m.group(1) in allowed:
            notes[m.group(1)] = m.group(2).strip()
    return notes
