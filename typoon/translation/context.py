"""Chapter context stage — agent loop with look_at + submit_chapter_brief tools."""

from __future__ import annotations

import time

from typoon.app.events import LLMCall, LLMResponse, PipelineError, ToolResult
from typoon.domain.bubble import Page, Session
from typoon.llm.ir import ContentPart, Message, ToolResponse

from . import prompt
from .brief import ChapterBrief, chapter_text
from .look_at import look_at_page
from .tools.brief import ChapterBriefArgs, submit_chapter_brief
from .tools.look_at_tool import LookAtArgs, look_at as look_at_tool

_MAX_CONTEXT_TURNS = 5


async def build_chapter_brief(
    pages: list[Page], session: Session, key_map: dict,
) -> tuple[ChapterBrief, int]:
    system = prompt.CONTEXT_SYSTEM.format(
        source_lang=session.source_lang,
        target_lang=session.target_lang,
        source_policy=prompt.load_policy(f"source_{session.source_lang}.md"),
        target_policy=prompt.load_policy(f"target_{session.target_lang}.md"),
    )
    user_text = prompt.CONTEXT_USER.format(
        prior_context=session.prior_context or "(none)",
        glossary_block=_glossary_block(session.glossary),
        chapter_text=chapter_text(pages),
    )
    tools = [look_at_tool.definition, submit_chapter_brief.definition]
    messages = [Message.system(system), Message.user_text(user_text)]
    hook = session.hook

    for turn in range(1, _MAX_CONTEXT_TURNS + 1):
        hook.on(LLMCall(agent="translate/context", turn=turn))
        t0 = time.monotonic()
        resp = await session.context_provider.call(messages, tools=tools)
        n_tools = len(resp.tool_calls) if resp.tool_calls else 0
        hook.on(LLMResponse(
            agent="translate/context", turn=turn,
            tool_calls=n_tools, ms=(time.monotonic() - t0) * 1000,
        ))

        if not resp.tool_calls:
            raise RuntimeError(
                f"Context model did not call any tool. Text: {(resp.text or '')[:200]}"
            )

        messages.append(Message.assistant(text=resp.text, tool_calls=resp.tool_calls))

        brief: ChapterBrief | None = None
        for tc in resp.tool_calls:
            if tc.name == "submit_chapter_brief":
                brief = _parse_brief(tc)
                messages.append(Message.tool_result_text(tc.id, "ok"))
                hook.on(ToolResult(
                    agent="translate/context", turn=turn,
                    tool="submit_chapter_brief", result="accepted",
                ))
            elif tc.name == "look_at":
                result_text = await _execute_look_at(tc, session, key_map, turn)
                messages.append(Message.tool_result_text(tc.id, result_text))
                hook.on(ToolResult(
                    agent="translate/context", turn=turn,
                    tool="look_at", result=result_text[:80],
                ))
            else:
                messages.append(Message.tool_result_text(tc.id, f"Unknown tool: {tc.name}"))

        if brief is not None:
            return brief, turn

    raise RuntimeError("Context agent did not submit brief within turn limit")


def _parse_brief(tc) -> ChapterBrief:
    args = ChapterBriefArgs.model_validate_json(tc.arguments)
    return ChapterBrief(
        summary=args.summary,
        facts=args.facts,
        glossary={g.source: g.target for g in args.glossary},
        style_rules=args.rules,
        pronoun_rules=[],
        page_notes={pn.page: pn.note for pn in args.page_notes},
        key_notes={bn.key: bn.note for bn in args.bubble_notes},
    )


async def _execute_look_at(tc, session: Session, key_map: dict, turn: int) -> str:
    try:
        args = LookAtArgs.model_validate_json(tc.arguments)
    except Exception as e:
        return f"Error parsing look_at args: {e}"
    source_by_key = {k: key_map[k].source_text for k in args.keys if k in key_map}
    notes = await look_at_page(
        session,
        page_index=args.page,
        keys=args.keys,
        query=args.query,
        source_by_key=source_by_key,
        turn=turn,
    )
    if not notes:
        return "No visual notes returned."
    return "\n".join(f"#{k}: {v}" for k, v in notes.items())


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
