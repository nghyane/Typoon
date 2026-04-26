"""submit_chapter_brief — structured context output tool."""

from __future__ import annotations

from pydantic import BaseModel, Field

from typoon.llm.tool_dec import tool


class GlossaryEntry(BaseModel):
    source: str = Field(description="Source-language term")
    target: str = Field(description="Target-language translation")


class PageNote(BaseModel):
    page: int = Field(description="Page index integer matching [pN] labels")
    note: str = Field(description="Short situation note for this page")


class BubbleNote(BaseModel):
    key: str = Field(description="Opaque bubble key from the chapter text")
    note: str = Field(description="Context note for this bubble")


class LookRequest(BaseModel):
    page: int = Field(description="Page index to inspect visually")
    keys: list[str] = Field(description="Bubble keys on that page needing visual help")
    query: str = Field(description="What to clarify from the page image")


class ChapterBriefArgs(BaseModel):
    summary: str = Field(description="One-paragraph chapter summary")
    facts: list[str] = Field(default_factory=list, description="Notable plot/character facts")
    glossary: list[GlossaryEntry] = Field(default_factory=list, description="New terms found in this chapter")
    rules: list[str] = Field(default_factory=list, description="Translation style and pronoun/address rules")
    page_notes: list[PageNote] = Field(default_factory=list, description="Per-page situation notes")
    bubble_notes: list[BubbleNote] = Field(default_factory=list, description="Per-bubble context notes")
    look_requests: list[LookRequest] = Field(default_factory=list, description="Pages needing visual clarification before translation")


@tool(strict=True)
async def submit_chapter_brief(args: ChapterBriefArgs) -> str:
    """Submit the chapter analysis brief.

    Rules:
    - Use exact opaque keys from the chapter text. Do not invent keys.
    - Page indices must be integers matching the [pN] labels.
    - Glossary: source is the original term, target is the translation.
    - Rules: include both style constraints and pronoun/address rules.
    - Use look_requests only when visual context is truly needed.
    """
    raise NotImplementedError("dispatch handles this")
