"""Domain types — connector/source discovery types."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(slots=True)
class ChapterVariant:
    """One upload of a chapter (scanlation group)."""

    id: str
    url: str
    group: str | None = None
    votes: int = 0


@dataclass(slots=True)
class DiscoveredChapter:
    """A chapter discovered from a remote source."""

    number: float
    title: str | None = None
    variants: list[ChapterVariant] = field(default_factory=list)

    @property
    def best_variant(self) -> ChapterVariant:
        return max(self.variants, key=lambda v: v.votes) if self.variants else self.variants[0]


@dataclass(slots=True)
class SourceInfo:
    """Metadata discovered from a manga source URL."""

    suggested_title: str
    cover_url:       str | None = None
    description:     str | None = None
    chapters:        list[DiscoveredChapter] = field(default_factory=list)
