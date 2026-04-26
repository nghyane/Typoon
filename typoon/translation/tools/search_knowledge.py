"""search_knowledge — context agent tool to query stored knowledge."""

from __future__ import annotations

from pydantic import BaseModel, Field

from typoon.llm.tool_dec import tool


class SearchKnowledgeArgs(BaseModel):
    query: str = Field(description="Search query for stored knowledge")
    scope: str = Field(
        default="all",
        description="Search scope: glossary, briefs, translations, or all",
    )


@tool(strict=True)
async def search_knowledge(args: SearchKnowledgeArgs) -> str:
    """Search stored project knowledge.

    Use to look up:
    - character names, terms, relationships from previous chapters
    - glossary entries for consistent translation
    - how a term was translated before

    Call before submit_chapter_brief if prior context in the prompt
    is insufficient.
    """
    raise NotImplementedError("dispatch handles this")
