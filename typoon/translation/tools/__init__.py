"""Translation agent tools — Pydantic models + @tool decorator."""

from __future__ import annotations

from typoon.llm.ir import ToolDef

from .get_context import get_context
from .search_glossary import search_glossary
from .translate import translate
from .view_bubble import view_bubble
from .view_page import view_page


def build_tools(
    *,
    has_images: bool,
    has_glossary: bool,
    has_context: bool,
) -> list[ToolDef]:
    tools = [translate.definition]
    if has_images:
        tools.append(view_page.definition)
        tools.append(view_bubble.definition)
    if has_glossary:
        tools.append(search_glossary.definition)
    if has_context:
        tools.append(get_context.definition)
    return tools
