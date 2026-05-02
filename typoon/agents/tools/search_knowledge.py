"""search_knowledge tool factory."""

from __future__ import annotations

from enum import Enum

from pydantic import BaseModel, Field

from typoon.llm.ir import ToolResponse
from typoon.llm.tool import Tool, tool


class SearchScope(str, Enum):
    all = "all"
    glossary = "glossary"
    briefs = "briefs"
    translations = "translations"


class SearchKnowledgeArgs(BaseModel):
    queries: list[str] = Field(description="One or more search queries to look up in a single call")
    scope: SearchScope = Field(default=SearchScope.all, description="What to search")


@tool
async def search_knowledge(args: SearchKnowledgeArgs, ctx) -> ToolResponse:
    """Look up glossary, previous chapter briefs, or past translations. Pass multiple queries at once."""
    results: list[str] = []
    for query in args.queries:
        if args.scope in (SearchScope.all, SearchScope.glossary):
            for entry in await ctx.store.glossary_search(ctx.project_id, query):
                results.append(f"{entry['source_term']} => {entry['target_term']}")
        if args.scope in (SearchScope.all, SearchScope.briefs):
            results.extend(await ctx.store.search_briefs(
                ctx.project_id, [query], limit=5, before_chapter=ctx.chapter,
            ))
        if args.scope in (SearchScope.all, SearchScope.translations):
            results.extend(await ctx.store.search_context(
                ctx.project_id, [query], scope="translations", limit=5,
            ))
    return ToolResponse("\n".join(results) if results else "No results found.")


def make_search_knowledge(ctx) -> Tool:
    return search_knowledge(ctx=ctx)
