"""submit_chapter_brief tool factory."""

from __future__ import annotations

from pydantic import BaseModel, Field

from typoon.agents.brief import AddressRule, ChapterBrief
from typoon.llm.ir import ToolResponse
from typoon.llm.tool import Tool, tool


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
    key: str = Field(description="The bubble's hash key — copy exactly from after the # marker in the chapter text (e.g. '[p3] #R7SDXKG text' → key='R7SDXKG'). Do NOT use the page number [pN].")
    note: str = Field(description="Context note for this bubble — include speaker identity if known (e.g. 'Speaker: Malorie, addressing Brandi')")


class ChapterBriefArgs(BaseModel):
    summary: str = Field(description="One-paragraph chapter summary")
    facts: list[str] = Field(default_factory=list, description="Plot events and character relationship facts that affect meaning but NOT translation decisions — e.g. 'A just discovered B is the enemy', 'C is gravely injured'. Do NOT put language/register/tone decisions here.")
    glossary: list[GlossaryEntry] = Field(
        default_factory=list,
        description="Names, titles, and special terms found in this chapter with consistent translations"
    )
    address: list[AddressEntry] = Field(
        default_factory=list,
        description=(
            "Sparse xưng hô (pronoun/address) binding for confirmed recurring or "
            "translation-critical speaker→listener pairs only. Submit exactly one "
            "non-conflicting rule per speaker→listener pair. Do not add self-address rules. "
            "Do not guess one-off or unclear pairs; put uncertain speaker/listener guesses "
            "in bubble_notes instead. These are binding decisions — page translator will "
            "follow them exactly."
        )
    )
    style_notes: list[str] = Field(
        default_factory=list,
        description="Translation decisions only: loaded skill decisions, register, capitalization, punctuation, SFX, recurring speech patterns — e.g. 'Applied system-terms: translate readable skill names into Vietnamese unless glossary locks source form', 'Character X speaks bluntly, no honorifics', 'Narrator uses formal tone'. Do NOT put plot events or relationship facts here."
    )
    page_notes: list[PageNote] = Field(default_factory=list, description="Per-page situation notes")
    bubble_notes: list[BubbleNote] = Field(default_factory=list, description="Per-bubble context notes")


@tool
async def submit_chapter_brief(args: ChapterBriefArgs, on_submit) -> ToolResponse:
    """Submit the chapter analysis brief after gathering all needed context."""
    return await on_submit(args)


def make_submit_chapter_brief(on_submit) -> Tool:
    return submit_chapter_brief(on_submit=on_submit)
