from __future__ import annotations

from pydantic import BaseModel, Field

from typoon.llm.tool_dec import tool


class UpdateSnapshotArgs(BaseModel):
    snapshot: str = Field(description="Full updated series knowledge snapshot. Must be self-contained.")


@tool(strict=True)
async def update_snapshot(args: UpdateSnapshotArgs) -> str:
    """Replace the series knowledge snapshot with an updated version.

    The snapshot is injected into future chapter prompts, so it must be self-contained.
    Include: main characters, relationships, key terms, recent events (last 3-5 chapters).
    Drop old events unless plot-critical.
    """
    raise NotImplementedError("dispatch handles this")
