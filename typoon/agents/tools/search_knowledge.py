"""search_knowledge — context agent tool to query stored knowledge."""

from __future__ import annotations

from enum import Enum

from pydantic import BaseModel, Field

from typoon.llm.tool_dec import tool


class SearchScope(str, Enum):
    all = "all"
    glossary = "glossary"
    briefs = "briefs"
    translations = "translations"


class SearchKnowledgeArgs(BaseModel):
    query: str = Field(description="Search query for stored knowledge")
    scope: SearchScope = Field(default=SearchScope.all, description="What to search")


@tool(strict=True)
async def search_knowledge(args: SearchKnowledgeArgs) -> str:
    """Look up glossary, previous chapter briefs, or past translations."""
    raise NotImplementedError("dispatch handles this")
