from __future__ import annotations

from pydantic import BaseModel, Field

from typoon.llm.tool_dec import tool


class TranslationItem(BaseModel):
    id: str = Field(description="Bubble ID exactly as listed (e.g. p0_b0, p0_b1)")
    translated_text: str = Field(description="Final translated text. Empty string for noise.")


class TranslateArgs(BaseModel):
    translations: list[TranslationItem] = Field(description="Array of bubble translations")


@tool(strict=True)
async def translate(args: TranslateArgs) -> str:
    """Submit translations for one or more bubbles in a single call.

    - Submit as many bubbles as possible per call to minimize round-trips.
    - To revise: include the same id again (latest wins).
    - For non-dialogue noise (watermarks, URLs, page numbers, credits, decorative SFX):
      use empty translated_text.
    - Meaning-bearing SFX: translate/adapt naturally.
    """
    return f"ok ({len(args.translations)} bubbles)"
