"""submit_translations — keyed batch translation submission tool."""

from __future__ import annotations

from enum import Enum

from pydantic import BaseModel, Field

from typoon.llm.tool_dec import tool


class TextKind(str, Enum):
    dialogue = "dialogue"
    sfx = "sfx"
    skip = "skip"


class TranslationEdit(BaseModel):
    key: str = Field(description="Opaque bubble key exactly as given")
    kind: TextKind = Field(description="dialogue: speech/narration/thought/signs, sfx: sound effects, skip: noise/credits/URLs")
    text: str = Field(default="", description="Translation; empty for skip")


class SubmitArgs(BaseModel):
    items: list[TranslationEdit] = Field(description="Keyed translation operations")


@tool(strict=True)
async def submit_translations(args: SubmitArgs) -> str:
    """Submit translations for the active keys in this page window."""
    raise NotImplementedError("dispatch handles this")
