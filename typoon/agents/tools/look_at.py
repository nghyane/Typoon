"""look_at — visual page inspection tool for context agent."""

from __future__ import annotations

from pydantic import BaseModel, Field

from typoon.llm.tool_dec import tool


class LookAtArgs(BaseModel):
    pages: list[int] = Field(description="Page indices to inspect (1 or more)")
    keys: list[str] = Field(description="Bubble keys to clarify across those pages")
    query: str = Field(description="What to clarify visually")


@tool()
async def look_at(args: LookAtArgs) -> str:
    """Inspect page images to clarify visual context for specific bubbles.

    The harness attaches page images with key labels overlaid and returns
    visual observations.
    """
    raise NotImplementedError("dispatch handles this")
