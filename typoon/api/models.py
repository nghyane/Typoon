"""API response models."""

from __future__ import annotations

from pydantic import BaseModel


class Progress(BaseModel):
    stage:      str
    page_index: int
    page_total: int


class ChapterOut(BaseModel):
    chapter_id: int
    project_id: int
    idx:        float
    title:      str | None = None
    state:      str        # idle | pending | running | error | done
    stage:      str        # scan | translate | render | ""
    page_count: int
    error:      str
    updated_at: str | None = None
    progress:   Progress | None = None


class ProjectOut(BaseModel):
    project_id:   int
    slug:         str
    title:        str
    description:  str | None = None
    cover_url:    str | None = None
    source_lang:  str
    target_lang:  str
    source_url:   str | None = None
    created_at:   str | None = None
    updated_at:   str | None = None

    model_config = {"populate_by_name": True}

    @classmethod
    def from_row(cls, row: dict) -> "ProjectOut":
        return cls(
            project_id=row["id"],
            slug=row["slug"],
            title=row["title"],
            description=row.get("description"),
            cover_url=f"/files/{row['slug']}/cover.jpg" if row.get("cover_path") else None,
            source_lang=row["source_lang"],
            target_lang=row["target_lang"],
            source_url=row.get("source_url"),
            created_at=row.get("created_at"),
            updated_at=row.get("updated_at"),
        )


# ── Discovery ─────────────────────────────────────────────────────────


class ChapterVariantOut(BaseModel):
    id:    str
    url:   str
    group: str | None = None
    votes: int = 0


class DiscoveredChapterOut(BaseModel):
    number:   float
    title:    str | None = None
    variants: list[ChapterVariantOut] = []


class SourceInfoOut(BaseModel):
    suggested_title: str
    cover_url:       str | None = None
    description:     str | None = None
    source_lang:     str  # supplied by the connector, not the discovery payload
    chapters:        list[DiscoveredChapterOut] = []


# ── Bubbles / translations ────────────────────────────────────────────


class BubbleOut(BaseModel):
    page_index:      int
    bubble_idx:      int
    source_text:     str
    translated_text: str | None = None
    kind:            str | None = None  # dialogue | sfx | skip
    confidence:      float


# ── Glossary ──────────────────────────────────────────────────────────


class GlossaryTermOut(BaseModel):
    id:          int
    source_term: str
    target_term: str
    notes:       str | None = None


# ── Workers / queue ───────────────────────────────────────────────────


class StageStatsOut(BaseModel):
    pending: int
    running: int
    stale:   int


class QueueStatsOut(BaseModel):
    stages:         dict[str, StageStatsOut]
    active_workers: list[str]


# ── Search ────────────────────────────────────────────────────────────


class SearchHit(BaseModel):
    kind: str   # bubble | translation | brief | glossary
    text: str
    chapter_idx: float | None = None
    page_index:  int | None = None


class SearchResults(BaseModel):
    hits: list[SearchHit]
