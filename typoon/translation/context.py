"""Chapter context agent — implements Agent protocol."""

from __future__ import annotations

from typoon.domain.bubble import Bubble, Page, Session
from typoon.llm.ir import Message, ToolCallMsg, ToolDef, ToolResponse

from . import prompt
from .brief import ChapterBrief, LookRequest, chapter_text
from .look_at import look_at_page
from .tools.brief import ChapterBriefArgs, submit_chapter_brief
from .tools.look_at_tool import LookAtArgs, look_at as look_at_tool


class ContextAgent:
    """Reads chapter text, optionally calls LookAt, then submits ChapterBrief."""

    def __init__(self, pages: list[Page], session: Session, key_map: dict[str, Bubble]) -> None:
        self._session = session
        self._key_map = key_map
        self._brief: ChapterBrief | None = None
        self._pages = pages

    def name(self) -> str:
        return "translate/context"

    def system_prompt(self) -> str:
        return prompt.CONTEXT_SYSTEM.format(
            source_lang=self._session.source_lang,
            target_lang=self._session.target_lang,
            source_policy=prompt.load_policy(f"source_{self._session.source_lang}.md"),
            target_policy=prompt.load_policy(f"target_{self._session.target_lang}.md"),
        )

    def user_message(self) -> Message:
        text = prompt.CONTEXT_USER.format(
            prior_context=self._session.prior_context or "(none)",
            glossary_block=_glossary_block(self._session.glossary),
            chapter_text=chapter_text(self._pages),
        )
        return Message.user_text(text)

    def tools(self) -> list[ToolDef]:
        return [look_at_tool.definition, submit_chapter_brief.definition]

    async def dispatch(self, call: ToolCallMsg) -> ToolResponse:
        if call.name == "submit_chapter_brief":
            return self._handle_brief(call)
        if call.name == "look_at":
            return await self._handle_look_at(call)
        return ToolResponse(f"Unknown tool: {call.name}")

    def on_text(self, text: str | None) -> None:
        pass

    def is_done(self) -> bool:
        return self._brief is not None

    def retry_prompt(self) -> str | None:
        if self._brief is None:
            return "You must call submit_chapter_brief to complete the analysis."
        return None

    def into_output(self) -> ChapterBrief | None:
        return self._brief

    def _handle_brief(self, call: ToolCallMsg) -> ToolResponse:
        try:
            args = ChapterBriefArgs.model_validate_json(call.arguments)
        except Exception as e:
            return ToolResponse(f"Invalid brief: {e}")
        self._brief = ChapterBrief(
            summary=args.summary,
            facts=args.facts,
            glossary={g.source: g.target for g in args.glossary},
            style_rules=args.rules,
            pronoun_rules=[],
            page_notes={pn.page: pn.note for pn in args.page_notes},
            key_notes={bn.key: bn.note for bn in args.bubble_notes},
        )
        return ToolResponse("ok")

    async def _handle_look_at(self, call: ToolCallMsg) -> ToolResponse:
        try:
            args = LookAtArgs.model_validate_json(call.arguments)
        except Exception as e:
            return ToolResponse(f"Error: {e}")
        source_by_key = {k: self._key_map[k].source_text for k in args.keys if k in self._key_map}
        notes = await look_at_page(
            self._session,
            page_index=args.page,
            keys=args.keys,
            query=args.query,
            source_by_key=source_by_key,
            turn=0,
        )
        if not notes:
            return ToolResponse("No visual notes returned.")
        return ToolResponse("\n".join(f"#{k}: {v}" for k, v in notes.items()))


async def build_chapter_brief(
    pages: list[Page], session: Session, key_map: dict[str, Bubble],
) -> tuple[ChapterBrief, int]:
    from typoon.llm.agent import run as agent_run
    agent = ContextAgent(pages, session, key_map)
    result = await agent_run(session.context_provider, agent, hook=session.hook)
    if result.error:
        raise result.error
    if result.output is None:
        raise RuntimeError("Context agent did not submit chapter brief")
    return result.output, result.turns


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
