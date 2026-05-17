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
    # How prepare produced this chapter:
    #   - "one_to_one": each prepared page is one raw entry.
    #   - "stitch":     each prepared page concatenates several raw
    #     entries (webtoon vertical-scroll re-cut).
    # Downstream stages that reason about per-page semantics (brief's
    # whole-page noise heuristic, render's stripping of cover pages)
    # must check this flag — under `stitch` a "page" is an aggregate
    # of source-page slices and per-page labels lose their meaning.
    strategy: str = "one_to_one"

    @property
    def page_count(self) -> int:
        return len(self.pages)
