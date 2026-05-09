"""mark_noise_page tool — flag entire pages as non-diegetic.

Some chapters include full-page filler that is not part of the comic
itself: scanlator credit pages, "support us" pages, host platform ads,
QR-code / "read on …" outros, end-of-chapter banners, etc.

Marking individual bubbles on these pages is wasteful — every bubble on
the page is non-story. This tool lets the context agent drop the whole
page in one call. Pages flagged here are excluded from the public
render archive: they will not be rendered, and the reader will never
see them.

Use only when EVERY visible element on the page is platform / scanlator
chrome — never to drop story pages with a few overlay bubbles.
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable

from pydantic import BaseModel, Field

from typoon.llm.ir import ToolResponse
from typoon.llm.tool import Tool, tool


class MarkNoisePageArgs(BaseModel):
    pages: list[int] = Field(
        description=(
            "0-based page indices to drop entirely. Use the `page` "
            "attribute from the chapter text. Only flag pages where "
            "EVERY visible bubble/text is non-diegetic (scanlator "
            "credits, host platform ads, QR/'read on' outro, end-of-"
            "chapter banner). Never flag a page that contains any "
            "story dialogue, narration, or in-world signage."
        )
    )
    reason: str = Field(
        default="",
        description="Short reason for the audit log",
    )


@tool
async def mark_noise_page(args: MarkNoisePageArgs, on_mark) -> ToolResponse:
    """Flag entire pages as non-diegetic; they will be dropped from render."""
    return await on_mark(args)


def make_mark_noise_page(
    on_mark: Callable[[MarkNoisePageArgs], Awaitable[ToolResponse]],
) -> Tool:
    return mark_noise_page(on_mark=on_mark)
