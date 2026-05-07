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
