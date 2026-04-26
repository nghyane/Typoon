"""look_at — visual page inspection tool for context agent."""

from __future__ import annotations

from pydantic import BaseModel, Field

from typoon.llm.tool_dec import tool


class LookAtArgs(BaseModel):
    page: int = Field(description="Page index to inspect")
    keys: list[str] = Field(description="Bubble keys to clarify on that page")
    query: str = Field(description="What to clarify visually, e.g. speaker gender, SFX vs dialogue")


@tool(strict=True)
async def look_at(args: LookAtArgs) -> str:
    """Inspect a page image to clarify visual context for specific bubbles.

    Use when text alone is insufficient to determine:
    - speaker identity or gender (for pronouns/address)
    - whether text is dialogue, narration, SFX, or decorative
    - speaker emotion/tone visible in the panel
    - local reading order among ambiguous bubbles

    Returns keyed visual notes. Call submit_chapter_brief after
    all needed context is gathered.
    """
    raise NotImplementedError("dispatch handles this")
