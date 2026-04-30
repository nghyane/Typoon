"""Domain — pure data types. No dependencies outside stdlib."""

from .prepared import PreparedChapter, PreparedPage
from .project import ChapterVariant, DiscoveredChapter, SourceInfo
from .render import RenderedBubble, RenderedChapter, RenderedPage
from .scan import BubbleGeometry, ScannedBubble, ScannedChapter, ScannedPage
from .translate import TranslatedBubble, TranslatedChapter, TranslatedPage

__all__ = [
    "PreparedChapter", "PreparedPage",
    "ChapterVariant", "DiscoveredChapter", "SourceInfo",
    "BubbleGeometry", "ScannedBubble", "ScannedChapter", "ScannedPage",
    "TranslatedBubble", "TranslatedChapter", "TranslatedPage",
    "RenderedBubble", "RenderedChapter", "RenderedPage",
]
