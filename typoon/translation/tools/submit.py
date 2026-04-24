"""submit_translations — batch translation submission tool.

LLM calls this multiple times per turn (parallel tool calls), one per
logical batch (page, scene, difficulty group). Code merges all batches
into the same dict.
"""

from __future__ import annotations

from pydantic import BaseModel, Field

from typoon.llm.tool_dec import tool


class TranslationEdit(BaseModel):
    id: str = Field(description="Bubble ID, e.g. 'p0_b3'")
    text: str = Field(
        description=(
            "Translation in target language. "
            "Use empty string for noise/SFX that should not render."
        ),
    )
    unclear: bool = Field(
        default=False,
        description=(
            "Set true when speaker, honorific, or meaning cannot be "
            "determined from text alone. System will re-ask with image."
        ),
    )


class SubmitArgs(BaseModel):
    edits: list[TranslationEdit] = Field(
        description=(
            "One or more bubble translations. "
            "Call this tool multiple times per turn to batch logically "
            "(e.g. one call per page). Prefer several small calls over "
            "one huge list."
        ),
    )


@tool(strict=True)
async def submit_translations(args: SubmitArgs) -> str:
    """Submit translations for a batch of bubbles.

    Rules:
    - Translate each bubble ID exactly once across all calls.
    - Use empty text for noise/SFX that shouldn't render in the target.
    - Set unclear=true when speaker/honorific is ambiguous without seeing
      the panel; the system will follow up with the bubble image.
    - You may emit multiple submit_translations calls in one response.
    """
    raise NotImplementedError("dispatch handles this")
