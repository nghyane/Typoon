from __future__ import annotations

from pydantic import BaseModel, Field

from typoon.llm.tool_dec import tool


class GetContextArgs(BaseModel):
    question: str = Field(description="Specific question about the project's translation history")


@tool(strict=True)
async def get_context(args: GetContextArgs) -> str:
    """Search previous chapters for translation history and chapter notes.

    A sub-agent searches the project database which contains:
    - Prior translations: source → translated text for every bubble in past chapters.
    - Chapter notes: character introductions, relationships, plot events, settings.

    Ask a specific question mentioning names, terms, or relationships you need clarified.

    When to use: encountering a character/term/relationship from earlier chapters.
    When NOT to use: first chapter of a project (no prior context exists).
    """
    raise NotImplementedError("dispatch handles this")
