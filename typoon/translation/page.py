"""Page/window translation worker."""

from __future__ import annotations

import time

from typoon.app.events import LLMCall, LLMResponse, ToolResult
from typoon.domain.bubble import Bubble, Session
from typoon.llm.ir import Message

from . import prompt
from .brief import ChapterBrief, brief_slice
from .protocol import TranslationOp, parse_response, validate_ops
from .tools.submit import submit_translations


async def translate_window(
    session: Session,
    *,
    brief: ChapterBrief,
    bubbles: list[Bubble],
    key_map: dict[str, Bubble],
    look_notes: dict[str, str],
    feedback: dict[str, str] | None,
    turn: int,
) -> tuple[list[TranslationOp], list[TranslationOp], dict[str, str]]:
    keys = [b.translation_key or "" for b in bubbles]
    active = set(keys)
    page_indices = {b.page_index for b in bubbles}
    system = prompt.PAGE_SYSTEM.format(
        source_lang=session.source_lang,
        target_lang=session.target_lang,
        source_policy=prompt.load_policy(f"source_{session.source_lang}.md"),
        target_policy=prompt.load_policy(f"target_{session.target_lang}.md"),
    )
    user = prompt.PAGE_USER.format(
        brief_slice=brief_slice(brief, page_indices, keys),
        feedback=_format_feedback(feedback or {}),
        keys="\n".join(f"#{b.translation_key} {b.source_text}" for b in bubbles),
    )
    hook = session.hook
    hook.on(LLMCall(agent="translate/page", turn=turn))
    t0 = time.monotonic()
    resp = await session.provider.call(
        [Message.system(system), Message.user_text(user)],
        tools=[submit_translations.definition],
    )
    hook.on(LLMResponse(
        agent="translate/page", turn=turn,
        tool_calls=len(resp.tool_calls or []), ms=(time.monotonic() - t0) * 1000,
    ))
    ops = parse_response(resp)
    result = validate_ops(ops, active=active, key_map=key_map, look_notes=look_notes)
    hook.on(ToolResult(
        agent="translate/page", turn=turn, tool="validate",
        result=f"ok={len(result.accepted)} need_look={len(result.need_look)} invalid={len(result.invalid)}",
    ))
    return result.accepted, result.need_look, result.invalid


def _format_feedback(feedback: dict[str, str]) -> str:
    if not feedback:
        return "(none)"
    return "\n".join(f"#{k}: {v}" for k, v in feedback.items())
