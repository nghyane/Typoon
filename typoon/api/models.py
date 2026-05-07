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
    owner_id:     int | None = None
    shared:       bool       = False
    is_owner:     bool       = False
    is_pinned:    bool       = False
    created_at:   str | None = None
    updated_at:   str | None = None

    model_config = {"populate_by_name": True}

    @classmethod
    def from_row(cls, row: dict, *, viewer_id: int | None = None) -> "ProjectOut":
        owner_id = row.get("owner_id")
        return cls(
            project_id=row["id"],
            slug=row["slug"],
            title=row["title"],
            description=row.get("description"),
            cover_url=f"/files/{row['slug']}/cover.jpg" if row.get("cover_path") else None,
            source_lang=row["source_lang"],
            target_lang=row["target_lang"],
            source_url=row.get("source_url"),
            owner_id=owner_id,
            shared=bool(row.get("shared")),
            is_owner=(viewer_id is not None and owner_id == viewer_id),
            is_pinned=bool(row.get("is_pinned")),
            created_at=row.get("created_at"),
            updated_at=row.get("updated_at"),
        )


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
