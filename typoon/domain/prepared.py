"""PreparedChapter — image archive is the source of truth.

Pixel data lives in a Bunle archive accessed through `PreparedReader`.
This module describes only the per-page metadata that other stages need
(width/height/index). The `Chapter` carries no filesystem path and cannot
load pixels by itself; callers pair it with a `PreparedReader` opened on
the chapter's prepared archive.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Page:
    index:  int
    width:  int
    height: int


@dataclass(frozen=True)
class Chapter:
    source:   str
    pages:    tuple[Page, ...]
    # Color signal from prepare's HSV saturation sampling. Used by the
    # context agent as a prior for tradition inference (color → manhua /
    # manhwa / colored webcomic; mono → manga / B&W manhua). The agent
    # makes the final tradition call from visual content; this flag is
    # just an explicit hint so the agent does not have to guess from the
    # storyboard alone.
    is_color: bool = False

    @property
    def page_count(self) -> int:
        return len(self.pages)
