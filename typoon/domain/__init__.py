"""Domain — pure data types. No dependencies."""

from .bubble import Bubble, Page
from .prepared import PreparedChapter, PreparedPage
from .project import ChapterVariant, DiscoveredChapter, SourceInfo

__all__ = [
    "Bubble",
    "Page",
    "PreparedChapter",
    "PreparedPage",
    "ChapterVariant",
    "DiscoveredChapter",
    "SourceInfo",
]
