"""submit_chapter_brief — structured context output tool."""

from __future__ import annotations

from pydantic import BaseModel, Field

from typoon.llm.tool_dec import tool


class GlossaryEntry(BaseModel):
    source: str = Field(description="Source-language term (name, title, place, special word)")
    target: str = Field(description="Target-language translation to use consistently")


class AddressEntry(BaseModel):
    speaker: str = Field(description="Character name or role who is speaking, e.g. 'elf', 'Saran', 'narrator'")
    listener: str = Field(description="Character name or role being addressed, or '*' for general")
    self_ref: str = Field(description="Pronoun speaker uses for themselves in target language, e.g. 'tôi', 'ta', 'em', 'anh'")
    other_ref: str = Field(description="Pronoun/address speaker uses for listener in target language, e.g. 'cô', 'anh', 'cậu', 'mày'")
    note: str = Field(default="", description="Register/tone note, e.g. 'formal', 'hostile', 'intimate'")


class PageNote(BaseModel):
    page: int = Field(description="Page index integer matching [pN] labels")
    note: str = Field(description="Short situation note for this page")


class BubbleNote(BaseModel):
    key: str = Field(description="Opaque bubble key from the chapter text")
    note: str = Field(description="Context note for this bubble")


class ChapterBriefArgs(BaseModel):
    summary: str = Field(description="One-paragraph chapter summary")
    facts: list[str] = Field(default_factory=list, description="Notable plot/character facts")
    glossary: list[GlossaryEntry] = Field(
        default_factory=list,
        description="Names, titles, and special terms found in this chapter with consistent translations"
    )
    address: list[AddressEntry] = Field(
        default_factory=list,
        description=(
            "Xưng hô (pronoun/address) binding for each speaker→listener pair. "
            "MUST cover all character pairs that appear in dialogue. "
            "These are binding decisions — page translator will follow them exactly."
        )
    )
    style_notes: list[str] = Field(
        default_factory=list,
        description="Tone and style notes: register, capitalization, punctuation style, SFX handling"
    )
    page_notes: list[PageNote] = Field(default_factory=list, description="Per-page situation notes")
    bubble_notes: list[BubbleNote] = Field(default_factory=list, description="Per-bubble context notes")


@tool()
async def submit_chapter_brief(args: ChapterBriefArgs) -> str:
    """Submit the chapter analysis brief after gathering all needed context."""
    raise NotImplementedError("dispatch handles this")
