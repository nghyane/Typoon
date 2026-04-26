"""submit_chapter_brief — structured context output tool."""

from __future__ import annotations

from pydantic import BaseModel, Field

from typoon.llm.tool_dec import tool


class PageNote(BaseModel):
    page: int = Field(description="Page index (integer)")
    note: str = Field(description="Short context note for this page")


class KeyNote(BaseModel):
    key: str = Field(description="Opaque bubble key")
    note: str = Field(description="Short context note for this key")


class LookRequestItem(BaseModel):
    page_index: int = Field(description="Page index to inspect")
    keys: list[str] = Field(description="Keys on that page needing visual help")
    query: str = Field(description="What to clarify visually")


class ChapterBriefArgs(BaseModel):
    summary: str = Field(description="One-paragraph chapter summary")
    facts: list[str] = Field(default_factory=list, description="Notable plot/character facts")
    glossary: list[KeyNote] = Field(default_factory=list, description="Source term -> target term pairs")
    style_rules: list[str] = Field(default_factory=list, description="Translation style constraints")
    pronoun_rules: list[str] = Field(default_factory=list, description="Speaker pronoun/address rules")
    page_notes: list[PageNote] = Field(default_factory=list, description="Per-page situation notes")
    key_notes: list[KeyNote] = Field(default_factory=list, description="Per-key context notes")
    look_requests: list[LookRequestItem] = Field(default_factory=list, description="Pages/keys needing visual clarification")


@tool(strict=True)
async def submit_chapter_brief(args: ChapterBriefArgs) -> str:
    """Submit the chapter analysis brief.

    Rules:
    - Use exact opaque keys from the chapter text.
    - Do not invent keys.
    - page indices must be integers matching the [pN] labels.
    - glossary entries use key=source_term, note=target_term.
    """
    raise NotImplementedError("dispatch handles this")
