"""Context agent — searches project DB to answer questions about prior chapters.

Sub-agent called by the translation agent via get_context tool.
"""

from __future__ import annotations

import json

from pydantic import BaseModel, Field

from typoon.llm.ir import Message, Provider, ToolCallMsg, ToolDef, ToolResponse
from typoon.llm.agent import run as agent_run
from typoon.llm.tool_dec import tool
from typoon.ports import Store

SYSTEM = """\
You are a context retrieval sub-agent for a manga translation project.

The database contains:
- translations: source text → translated text for every bubble in past chapters.
- chapter notes: character introductions, relationships, plot events, settings.

Rules:
- Call search() ONCE with ALL relevant queries in a single message.
- After reviewing results, respond with a concise text answer (no tool calls).
- Plain text only. Answer directly with facts from the database.
- Do not speculate or add information not found in search results."""


class SearchArgs(BaseModel):
    queries: list[str] = Field(description='Short keyword queries, e.g. ["Max", "Joy", "team leader"]')
    scope: str = Field(default="all", description="Search scope: all, translations, notes")


@tool(strict=True)
async def search(args: SearchArgs) -> str:
    """Batch search across translations and/or notes."""
    raise NotImplementedError("dispatch handles this")


class ReadChapterArgs(BaseModel):
    chapter_index: int = Field(description="Chapter number")


@tool(strict=True)
async def read_chapter(args: ReadChapterArgs) -> str:
    """Get all translations for a specific chapter. Use only if search points to a chapter you need."""
    raise NotImplementedError("dispatch handles this")


async def ask(
    provider: Provider,
    store: Store,
    project_id: int,
    question: str,
) -> str:
    """Run context agent. Returns answer string."""
    agent = _Agent(store=store, project_id=project_id, question=question)
    result = await agent_run(provider, agent)
    return result.output


class _Agent:
    def __init__(self, store: Store, project_id: int, question: str) -> None:
        self._store = store
        self._project_id = project_id
        self._question = question
        self._answer: str | None = None

    def name(self) -> str:
        return "context"

    def system_prompt(self) -> str:
        return SYSTEM

    def user_message(self) -> Message:
        return Message.user_text(self._question)

    def tools(self) -> list[ToolDef]:
        return [search.definition, read_chapter.definition]

    async def dispatch(self, call: ToolCallMsg) -> ToolResponse:
        try:
            match call.name:
                case "search":
                    args = SearchArgs.model_validate_json(call.arguments)
                    hits = await self._store.search_context(
                        self._project_id, args.queries, args.scope, limit=12,
                    )
                    if not hits:
                        return ToolResponse("No results found.")
                    return ToolResponse(f"Found {len(hits)} results:\n" + "\n".join(f"  {h}" for h in hits))

                case "read_chapter":
                    args = ReadChapterArgs.model_validate_json(call.arguments)
                    pairs = await self._store.get_chapter_pairs(self._project_id, args.chapter_index)
                    if not pairs:
                        return ToolResponse("No translations found for this chapter.")
                    lines = [f"  {src} → {tgt}" for src, tgt in pairs]
                    return ToolResponse(f"Chapter {args.chapter_index} ({len(pairs)} translations):\n" + "\n".join(lines))

                case _:
                    return ToolResponse(f"Unknown tool: {call.name}")
        except Exception as e:
            return ToolResponse(f"Error: {e}")

    def on_text(self, text: str | None) -> None:
        self._answer = text

    def is_done(self) -> bool:
        return self._answer is not None

    def retry_prompt(self) -> str | None:
        return None

    def into_output(self) -> str:
        return self._answer or ""
