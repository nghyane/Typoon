"""Domain — pure data types. No dependencies outside stdlib.

Import pattern for stage types:
    from typoon.domain import scan, translate, render
    scan.Bubble, translate.Page, render.Chapter
"""

from typoon.domain import prepared, scan, translate, render
from .project import ChapterVariant, DiscoveredChapter, SourceInfo

__all__ = [
    "prepared", "scan", "translate", "render",
    "ChapterVariant", "DiscoveredChapter", "SourceInfo",
]
