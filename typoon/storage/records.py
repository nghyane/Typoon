"""Typed bridge between domain types and storage schema.

Prevents raw dict passing between domain and sqlite layer.
Key correctness is enforced at definition time, not at call site.
"""

from __future__ import annotations

import json
from dataclasses import dataclass

from typoon.domain.translate import TranslatedBubble, TranslatedChapter
from typoon.translation.brief import ChapterBrief


@dataclass(frozen=True)
class TranslationRecord:
    project_id:      int
    chapter:         float
    page_index:      int
    bubble_idx:      int
    key:             str
    source_text:     str
    translated_text: str
    status:          str
    polygon_json:    str
    font_size:       int

    @classmethod
    def from_bubble(
        cls,
        project_id: int,
        chapter: float,
        b: TranslatedBubble,
        *,
        font_size: int = 0,
    ) -> "TranslationRecord":
        return cls(
            project_id=project_id,
            chapter=chapter,
            page_index=b.page_index,
            bubble_idx=b.idx,
            key=b.translation_key,
            source_text=b.source_text,
            translated_text=b.translated_text,
            status=b.kind,
            polygon_json=json.dumps(b.source.box.polygon),
            font_size=font_size,
        )

    def as_tuple(self) -> tuple:
        return (
            self.project_id, self.chapter,
            self.page_index, self.bubble_idx,
            self.key, self.source_text, self.translated_text,
            self.status, self.polygon_json, self.font_size,
        )


@dataclass(frozen=True)
class BriefRecord:
    project_id:  int
    chapter:     float
    brief_json:  str
    summary:     str
    terms_text:  str
    facts_text:  str
    rules_text:  str

    @classmethod
    def from_brief(
        cls,
        project_id: int,
        chapter: float,
        brief: ChapterBrief,
    ) -> "BriefRecord":
        return cls(
            project_id=project_id,
            chapter=chapter,
            brief_json=json.dumps(brief.to_dict(), ensure_ascii=False),
            summary=brief.summary,
            terms_text="\n".join(f"{k} -> {v}" for k, v in brief.glossary.items()),
            facts_text="\n".join(brief.facts),
            rules_text="\n".join(brief.rules),     # correct key — enforced here
        )


def translation_records(
    project_id: int,
    chapter: float,
    translated: TranslatedChapter,
) -> list[TranslationRecord]:
    return [
        TranslationRecord.from_bubble(project_id, chapter, b)
        for b in translated.all_bubbles
        if b.kind != "skip"
    ]
