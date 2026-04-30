"""Chapter context agent — implements Agent protocol."""

from __future__ import annotations

from typoon.adapters.session import Session
from typoon.domain.bubble import Bubble, Page
from typoon.llm.ir import Message, ToolCallMsg, ToolDef, ToolResponse

from . import prompt
from .brief import ChapterBrief, chapter_text
from .look_at import look_at
from .tools.brief import ChapterBriefArgs, submit_chapter_brief
from .tools.look_at import LookAtArgs, look_at as look_at_tool
from .tools.search_knowledge import SearchKnowledgeArgs, SearchScope, search_knowledge


class ContextAgent:
    """Analyzes chapter text, queries knowledge/images, submits ChapterBrief."""

    def __init__(self, pages: list[Page], session: Session, key_map: dict[str, Bubble]) -> None:
        self._session = session
        self._key_map = key_map
        self._brief: ChapterBrief | None = None
        self._pages = pages
        self._text_retries = 0
        self._tool_calls = 0
        self._last_text = ""

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
        return Message.user_text(prompt.CONTEXT_USER.format(
            chapter_text=chapter_text(self._pages),
        ))

    def tools(self) -> list[ToolDef]:
        return [search_knowledge.definition, look_at_tool.definition, submit_chapter_brief.definition]

    async def dispatch(self, call: ToolCallMsg) -> ToolResponse:
        match call.name:
            case "submit_chapter_brief":
                return self._handle_brief(call)
            case "look_at":
                return await self._handle_look_at(call)
            case "search_knowledge":
                return await self._handle_search(call)
            case _:
                return ToolResponse(f"Unknown tool: {call.name}")

    def on_text(self, text: str | None) -> None:
        self._text_retries += 1
        self._last_text = text or ""

    def is_done(self) -> bool:
        return self._brief is not None

    def retry_prompt(self) -> str | None:
        if self._brief is not None:
            return None
        if self._text_retries >= 2:
            return None  # stop loop, caller will raise
        return "Do not respond with text. You must call one of the provided tools."

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
            rules=args.rules,
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
        notes = await look_at(
            self._session, pages=args.pages, keys=args.keys,
            query=args.query, source_by_key=source_by_key,
            key_map=self._key_map,
        )
        if not notes:
            return ToolResponse("No visual notes returned.")
        return ToolResponse("\n".join(f"#{k}: {v}" for k, v in notes.items()))

    async def _handle_search(self, call: ToolCallMsg) -> ToolResponse:
        try:
            args = SearchKnowledgeArgs.model_validate_json(call.arguments)
        except Exception as e:
            return ToolResponse(f"Error: {e}")
        store = self._session.store
        results: list[str] = []
        scope = args.scope
        if scope in (SearchScope.all, SearchScope.glossary):
            for entry in await store.glossary_search(self._session.project_id, args.query):
                results.append(f"{entry['source_term']} => {entry['target_term']}")
        if scope in (SearchScope.all, SearchScope.briefs):
            hits = await store.search_briefs(self._session.project_id, [args.query], limit=5)
            results.extend(hits)
        if scope in (SearchScope.all, SearchScope.translations):
            hits = await store.search_context(self._session.project_id, [args.query], scope="translations", limit=5)
            results.extend(hits)
        return ToolResponse("\n".join(results) if results else "No results found.")


async def build_chapter_brief(
    pages: list[Page], session: Session, key_map: dict[str, Bubble],
) -> tuple[ChapterBrief, int]:
    from typoon.llm.agent import run as agent_run
    agent = ContextAgent(pages, session, key_map)
    result = await agent_run(session.context_provider, agent, hook=session.hook, max_turns=20)
    if result.error:
        raise result.error
    if result.output is None:
        raise RuntimeError(
            f"Context agent did not submit chapter brief. "
            f"Last model text: {agent._last_text[:300]}"
        )
    return result.output, result.turns
