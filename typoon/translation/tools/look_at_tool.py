"""look_at — visual page inspection tool for context agent."""

from __future__ import annotations

from pydantic import BaseModel, Field

from typoon.llm.tool_dec import tool


class LookAtArgs(BaseModel):
    pages: list[int] = Field(description="Page indices to inspect (1 or more)")
    keys: list[str] = Field(description="Bubble keys to clarify across those pages")
    query: str = Field(description="What to clarify visually, e.g. speaker gender, SFX vs dialogue")


@tool(strict=True)
async def look_at(args: LookAtArgs) -> str:
    """Inspect page images to clarify visual context for specific bubbles.

    May attach multiple pages when context spans across pages (e.g. a
    conversation that continues from one page to the next).

    Returns keyed visual notes. Call submit_chapter_brief after
    all needed context is gathered.
    """
    raise NotImplementedError("dispatch handles this")
