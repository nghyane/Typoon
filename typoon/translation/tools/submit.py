"""submit_translations — keyed batch translation submission tool."""

from __future__ import annotations

from enum import Enum

from pydantic import BaseModel, Field

from typoon.llm.tool_dec import tool


class TextKind(str, Enum):
    dialogue = "dialogue"
    narration = "narration"
    sfx = "sfx"
    noise = "noise"
    meta = "meta"


class TranslationEdit(BaseModel):
    key: str = Field(description="Opaque bubble key exactly as given")
    kind: TextKind = Field(description="What this text is: dialogue, narration, sfx, noise, or meta")
    text: str = Field(default="", description="Translation in target language; empty for noise/meta")


class SubmitArgs(BaseModel):
    items: list[TranslationEdit] = Field(description="Keyed translation operations")


@tool(strict=True)
async def submit_translations(args: SubmitArgs) -> str:
    """Submit translations for the active keys in this page window."""
    raise NotImplementedError("dispatch handles this")
