"""Chapter context stage."""

from __future__ import annotations

import time

from typoon.app.events import LLMCall, LLMResponse, PipelineError
from typoon.domain.bubble import Page, Session
from typoon.llm.ir import Message

from . import prompt
from .brief import ChapterBrief, chapter_text


async def build_chapter_brief(pages: list[Page], session: Session) -> tuple[ChapterBrief, int]:
    system = prompt.CONTEXT_SYSTEM.format(
        source_lang=session.source_lang,
        target_lang=session.target_lang,
        source_policy=prompt.load_policy(f"source_{session.source_lang}.md"),
        target_policy=prompt.load_policy(f"target_{session.target_lang}.md"),
    )
    user = prompt.CONTEXT_USER.format(
        prior_context=session.prior_context or "(none)",
        glossary_block=_glossary_block(session.glossary),
        chapter_text=chapter_text(pages),
    )
    hook = session.hook
    hook.on(LLMCall(agent="translate/context", turn=1))
    t0 = time.monotonic()
    try:
        resp = await session.context_provider.call([Message.system(system), Message.user_text(user)], tools=[])
    except Exception as e:
        hook.on(PipelineError(stage="translate/context", error=e))
        raise
    hook.on(LLMResponse(agent="translate/context", turn=1, tool_calls=0, ms=(time.monotonic() - t0) * 1000))
    return ChapterBrief.from_json(resp.text or "{}"), 1


def format_prior_context(recent_briefs: list[dict], search_hits: list[str]) -> str:
    parts: list[str] = []
    for rec in recent_briefs:
        chapter = rec.get("chapter")
        summary = rec.get("summary") or ""
        terms = rec.get("terms_text") or ""
        facts = rec.get("facts_text") or ""
        rules = rec.get("rules_text") or ""
        body = "\n".join(x for x in [summary, terms, facts, rules] if x).strip()
        if body:
            parts.append(f"[Ch{chapter} brief]\n{body}")
    if search_hits:
        parts.append("Search hits:\n" + "\n".join(search_hits))
    return "\n\n".join(parts) if parts else "(none)"


def _glossary_block(glossary: dict[str, str]) -> str:
    if not glossary:
        return "(none)"
    return "\n".join(f"- {k} => {v}" for k, v in glossary.items())
