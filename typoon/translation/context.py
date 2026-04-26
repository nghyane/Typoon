"""Chapter context stage — uses submit_chapter_brief tool with text fallback."""

from __future__ import annotations

import time

from typoon.app.events import LLMCall, LLMResponse, PipelineError
from typoon.domain.bubble import Page, Session
from typoon.llm.ir import Message

from . import prompt
from .brief import ChapterBrief, LookRequest, chapter_text
from .tools.brief import ChapterBriefArgs, submit_chapter_brief


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
        resp = await session.context_provider.call(
            [Message.system(system), Message.user_text(user)],
            tools=[submit_chapter_brief.definition],
        )
    except Exception as e:
        hook.on(PipelineError(stage="translate/context", error=e))
        raise
    ms = (time.monotonic() - t0) * 1000
    n_tools = len(resp.tool_calls) if resp.tool_calls else 0
    hook.on(LLMResponse(agent="translate/context", turn=1, tool_calls=n_tools, ms=ms))

    brief = _parse_tool(resp) or ChapterBrief.from_json(resp.text or "{}")
    if not brief.summary and not brief.glossary and not brief.page_notes:
        hook.on(PipelineError(stage="translate/context", error=RuntimeError("empty chapter brief")))
    return brief, 1


def _parse_tool(resp) -> ChapterBrief | None:
    for tc in resp.tool_calls or []:
        if tc.name != "submit_chapter_brief":
            continue
        try:
            args = ChapterBriefArgs.model_validate_json(tc.arguments)
        except Exception:
            continue
        return ChapterBrief(
            summary=args.summary,
            facts=args.facts,
            glossary={g.source: g.target for g in args.glossary},
            style_rules=args.style_rules,
            pronoun_rules=args.pronoun_rules,
            page_notes={pn.page: pn.note for pn in args.page_notes},
            key_notes={kn.key: kn.note for kn in args.key_notes},
            look_requests=[
                LookRequest(page_index=lr.page_index, keys=lr.keys, query=lr.query)
                for lr in args.look_requests
            ],
        )
    return None


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
