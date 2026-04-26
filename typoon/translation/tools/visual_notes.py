"""submit_visual_notes — LookAt agent output tool."""

from __future__ import annotations

from pydantic import BaseModel, Field

from typoon.llm.tool_dec import tool


class VisualNote(BaseModel):
    key: str = Field(description="Opaque bubble key")
    note: str = Field(description="Short visual observation for this bubble")


class VisualNotesArgs(BaseModel):
    notes: list[VisualNote] = Field(description="Visual notes per requested key")


@tool(strict=True)
async def submit_visual_notes(args: VisualNotesArgs) -> str:
    """Submit visual observations for the requested bubble keys.

    Rules:
    - Use exact keys from the request. Do not invent keys.
    - Each note should be a short factual observation.
    - Include: speaker identity, gender, emotion, SFX vs dialogue, reading order.
    - Say 'uncertain' if the image is unclear.
    """
    raise NotImplementedError("dispatch handles this")
