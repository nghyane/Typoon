"""submit_translations — keyed batch translation submission tool."""

from __future__ import annotations

from pydantic import BaseModel, Field

from typoon.llm.tool_dec import tool


class TranslationEdit(BaseModel):
    key: str = Field(description="Opaque bubble key, e.g. 'K7Q9M2'")
    status: str = Field(
        default="ok",
        description="One of: ok, skip, need_look.",
    )
    text: str = Field(
        description=(
            "Translation in target language for status=ok. "
            "Use empty string for skip or need_look."
        ),
    )


class SubmitArgs(BaseModel):
    items: list[TranslationEdit] = Field(
        description=(
            "One or more keyed translation operations. "
            "Use status=need_look when visual context is needed."
        ),
    )
    done: bool = Field(default=False, description="Hint only; controller validates completeness.")


@tool(strict=True)
async def submit_translations(args: SubmitArgs) -> str:
    """Submit translations for a batch of bubbles.

    Rules:
    - Use each opaque key exactly as given.
    - status=ok requires final translated text.
    - status=skip uses empty text for text that should not render.
    - status=need_look asks the controller for page-level visual clarification.
    """
    raise NotImplementedError("dispatch handles this")
