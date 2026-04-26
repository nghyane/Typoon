"""submit_translations — keyed batch translation submission tool."""

from __future__ import annotations

from enum import Enum

from pydantic import BaseModel, Field

from typoon.llm.tool_dec import tool


class TranslationStatus(str, Enum):
    ok = "ok"
    skip = "skip"


class TranslationEdit(BaseModel):
    key: str = Field(description="Opaque bubble key exactly as given")
    status: TranslationStatus = Field(description="ok: final translation, skip: do not render")
    text: str = Field(default="", description="Translated text for ok; empty for skip")


class SubmitArgs(BaseModel):
    items: list[TranslationEdit] = Field(description="Keyed translation operations")


@tool(strict=True)
async def submit_translations(args: SubmitArgs) -> str:
    """Submit translations for a batch of bubbles.

    Rules:
    - Use each opaque key exactly as given.
    - status=ok requires non-empty translated text.
    - status=skip uses empty text for text that should not render.
    """
    raise NotImplementedError("dispatch handles this")
