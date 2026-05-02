"""look_at tool factory."""

from __future__ import annotations

from pydantic import BaseModel, Field

from typoon.llm.ir import ToolResponse
from typoon.llm.tool import Tool, tool


class LookAtArgs(BaseModel):
    pages: list[int] = Field(description="Page indices to inspect — maximum 3 pages per call. For larger ranges, make multiple targeted calls.")
    keys: list[str] = Field(description="Bubble keys to clarify on those pages")
    query: str = Field(description="What to clarify visually")


@tool
async def look_at(args: LookAtArgs, ctx, prepared, keyed) -> ToolResponse:
    """Inspect page images ONLY when text is genuinely insufficient.

    Call this ONLY when:
    - A speaker cannot be identified from names, titles, or dialogue context
    - A bubble type (dialogue vs SFX) is unclear from text alone
    - The ambiguity directly affects an address rule decision

    Do NOT call for: scene description, emotion confirmation, or any context
    already deducible from the chapter text. Max 3 pages per call.
    """
    from typoon.stages.look_at import look_at as _look_at
    notes = await _look_at(ctx, prepared, pages=args.pages, keys=args.keys, query=args.query, keyed=keyed)
    if not notes:
        return ToolResponse("No visual notes returned.")
    return ToolResponse("\n".join(f"#{k}: {v}" for k, v in notes.items()))


def make_look_at(ctx, prepared, keyed) -> Tool:
    return look_at(ctx=ctx, prepared=prepared, keyed=keyed)
