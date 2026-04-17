from __future__ import annotations

from pydantic import BaseModel, Field

from typoon.llm.tool_dec import tool


class SearchGlossaryArgs(BaseModel):
    query: str = Field(description="Source-language term to look up")


@tool(strict=True)
async def search_glossary(args: SearchGlossaryArgs) -> str:
    """Search the persistent glossary for canonical translations.

    Only for terms NOT already listed in the user message glossary.
    Batch multiple calls in one message.
    If no match found, translate naturally and continue.

    When to use: proper nouns or recurring terms likely to appear again.
    When NOT to use: ordinary words, one-off phrases, or terms already in glossary.
    """
    raise NotImplementedError("dispatch handles this")
