"""mark_noise tool — context agent flags non-diegetic bubbles.

The OCR pipeline cannot distinguish text painted into the comic from
text overlaid by the publishing platform or scanlator. The context
agent has the full chapter view (images + every bubble's text) and is
the only stage that can rule on whether a bubble belongs to the story.

Bubbles flagged here bypass the page translator — they will not be
sent for translation and will not appear in the rendered output.
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable

from pydantic import BaseModel, Field

from typoon.llm.ir import ToolResponse
from typoon.llm.tool import Tool, tool


class MarkNoiseArgs(BaseModel):
    keys: list[str] = Field(
        description=(
            "Bubble keys to exclude from translation. A key is the hash "
            "after '#' in the chapter text (e.g. '[p3] #R7SDXKG text' → "
            "key='R7SDXKG'). Only flag NON-DIEGETIC text — text added by "
            "the host platform, viewer UI, or scanlator that is not part "
            "of the comic itself. Do not flag in-story text under any "
            "circumstance."
        )
    )
    reason: str = Field(
        default="",
        description="Short reason for the audit log",
    )


@tool
async def mark_noise(args: MarkNoiseArgs, on_mark) -> ToolResponse:
    """Flag bubble keys as non-diegetic noise; they will not be translated."""
    return await on_mark(args)


def make_mark_noise(
    on_mark: Callable[[MarkNoiseArgs], Awaitable[ToolResponse]],
) -> Tool:
    return mark_noise(on_mark=on_mark)
