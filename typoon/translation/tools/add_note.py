from __future__ import annotations

from pydantic import BaseModel, Field

from typoon.llm.tool_dec import tool


class AddNoteArgs(BaseModel):
    note_type: str = Field(description="One of: character, relationship, event, setting")
    content: str = Field(description="Concise factual note")


@tool(strict=True)
async def add_note(args: AddNoteArgs) -> str:
    """Record a chapter-specific note for future reference.

    Types:
    - character: new character introduction (name, role, traits)
    - relationship: connection between characters (X is Y's brother)
    - event: significant plot event
    - setting: location or time period change
    """
    raise NotImplementedError("dispatch handles this")
