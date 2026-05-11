"""search_knowledge tool factory — slimmed for v5.

Phase B scope: glossary lookup only. Brief / past-translation search
relied on per-project FTS that doesn't exist in the material schema
(briefs bind to drafts now). Restoring those scopes is a polish-slice
task; the agent loses one shortcut but the rest of context-building
works.
"""

from __future__ import annotations

from enum import Enum

from pydantic import BaseModel, Field

from typoon.llm.ir import ToolResponse
from typoon.llm.tool import Tool, tool


class SearchScope(str, Enum):
    all = "all"
    glossary = "glossary"


class SearchKnowledgeArgs(BaseModel):
    queries: list[str] = Field(
        description="One or more search queries to look up in a single call"
    )
    scope: SearchScope = Field(
        default=SearchScope.all, description="What to search",
    )


@tool
async def search_knowledge(args: SearchKnowledgeArgs, ctx) -> ToolResponse:
    """Look up glossary terms for the current language pair. Pass
    multiple queries at once for a single tool call."""
    if args.scope not in (SearchScope.all, SearchScope.glossary):
        return ToolResponse("Only glossary scope is currently supported.")

    # Use the resolved glossary from list_user_glossary; filtering
    # client-side is fine because user glossaries are small (≤100s).
    user_terms = await ctx.store.list_user_glossary(
        ctx.owner_id,
        source_lang=ctx.source_lang,
        target_lang=ctx.target_lang,
    )
    haystack = {t["source_term"]: t["target_term"] for t in user_terms}

    results: list[str] = []
    for query in args.queries:
        q = query.lower()
        for src, tgt in haystack.items():
            if q in src.lower() or q in tgt.lower():
                results.append(f"{src} => {tgt}")

    return ToolResponse("\n".join(results) if results else "No results found.")


def make_search_knowledge(ctx) -> Tool:
    return search_knowledge(ctx=ctx)
