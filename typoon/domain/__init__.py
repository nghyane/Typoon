"""Domain — pure data types. No dependencies outside stdlib.

Import pattern for stage types:
    from typoon.domain import scan, translate, render
    scan.Bubble, translate.Page, render.Chapter
"""

from typoon.domain import prepared, scan, translate, render
from .prepared import Chapter as PreparedChapter, Page as PreparedPage
from .project import ChapterVariant, DiscoveredChapter, SourceInfo

__all__ = [
    # Sub-modules (preferred import style)
    "prepared", "scan", "translate", "render",
    # Convenience top-level exports
    "PreparedChapter", "PreparedPage",
    "ChapterVariant", "DiscoveredChapter", "SourceInfo",
]
