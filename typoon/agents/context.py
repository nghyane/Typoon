"""Chapter context agent — Agent protocol + tools."""

from __future__ import annotations

from typoon.adapters.session import Session
from typoon.domain.prepared import Chapter
from typoon.domain.scan import Bubble as ScannedBubble
from typoon.llm.ir import Message, ToolCallMsg, ToolDef, ToolResponse

from . import prompt
from .brief import AddressRule, ChapterBrief, chapter_text
from .look_at import look_at
from .tools.brief import ChapterBriefArgs, submit_chapter_brief
from .tools.look_at import LookAtArgs, look_at as look_at_tool
from .tools.search_knowledge import SearchKnowledgeArgs, SearchScope, search_knowledge


class ContextAgent:
    """Analyzes chapter text, uses tools, submits ChapterBrief via tool call."""

    def __init__(
        self,
        session: Session,
        prepared: Chapter,
        key_map: dict[str, ScannedBubble],
        context_snapshot: str,
    ) -> None:
        self._session = session
        self._prepared = prepared
        self._key_map = key_map
        self._context_snapshot = context_snapshot
        self._brief: ChapterBrief | None = None
        self._text_retries = 0
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
            context_snapshot=self._context_snapshot,
            chapter_text=chapter_text(self._key_map),
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
        if self._text_retries >= 3:
            return None
        return (
            "You must call submit_chapter_brief now. "
            "Do not write any text. Call the tool directly with your analysis."
        )

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
            address=[
                AddressRule(
                    speaker=a.speaker,
                    listener=a.listener,
                    self_ref=a.self_ref,
                    other_ref=a.other_ref,
                    note=a.note,
                )
                for a in args.address
            ],
            style_notes=args.style_notes,
            page_notes={pn.page: pn.note for pn in args.page_notes},
            key_notes={bn.key: bn.note for bn in args.bubble_notes},
        )
        return ToolResponse("ok")

    async def _handle_look_at(self, call: ToolCallMsg) -> ToolResponse:
        try:
            args = LookAtArgs.model_validate_json(call.arguments)
        except Exception as e:
            return ToolResponse(f"Error: {e}")
        notes = await look_at(
            self._session, self._prepared,
            pages=args.pages, keys=args.keys,
            query=args.query, key_map=self._key_map,
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
    session: Session,
    prepared: Chapter,
    key_map: dict[str, ScannedBubble],
) -> tuple[ChapterBrief, int]:
    from typoon.llm.agent import run as agent_run

    context_snapshot = await _load_context_snapshot(session)
    agent = ContextAgent(session, prepared, key_map, context_snapshot)
    result = await agent_run(session.context_provider, agent, hook=session.hook, max_turns=20)
    if result.error:
        raise result.error
    if result.output is None:
        raise RuntimeError(
            f"Context agent did not submit chapter brief. "
            f"Last model text: {agent._last_text[:300]}"
        )
    return result.output, result.turns


async def _load_context_snapshot(session: Session) -> str:
    store = session.store
    parts: list[str] = []

    glossary = await store.get_glossary(session.project_id) if hasattr(store, "get_glossary") else {}
    if not glossary:
        glossary = session.glossary or {}
    if glossary:
        parts.append(
            f"## Glossary\n{len(glossary)} terms available. "
            f"Call search_knowledge(scope=glossary, query=<term>) to look up."
        )
    else:
        parts.append("## Glossary\nEmpty.")

    recent = await store.get_recent_chapter_briefs(
        session.project_id, before_chapter=session.chapter, limit=10
    )
    if recent:
        lines = [
            f"  Ch{r.get('chapter','?')}: {(r.get('brief') or {}).get('summary','')[:120]}"
            for r in recent
        ]
        parts.append(
            "## Prior chapters (briefs available)\n" + "\n".join(lines) +
            "\nCall search_knowledge(scope=briefs, query=<topic>) for details."
        )
    else:
        parts.append("## Prior chapters\nNone.")

    return "\n\n".join(parts)
