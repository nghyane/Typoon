from __future__ import annotations

from pydantic import BaseModel, Field

from typoon.llm.tool_dec import tool


class UpsertGlossaryArgs(BaseModel):
    source_term: str = Field(description="Term in source language")
    target_term: str = Field(description="Canonical translation")
    notes: str = Field(default="", description="Optional context (e.g. 'character name', 'ability')")


@tool(strict=True)
async def upsert_glossary(args: UpsertGlossaryArgs) -> str:
    """Add or update a glossary entry for a recurring term.

    Use for: character names, place names, abilities, titles, organizations.
    Do NOT add ordinary vocabulary.
    """
    raise NotImplementedError("dispatch handles this")
