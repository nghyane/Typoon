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
    state:      str        # idle | pending | running | error | done
    stage:      str        # scan | translate | render | ""
    page_count: int
    error:      str
    progress:   Progress | None = None


class ProjectOut(BaseModel):
    project_id:  int
    slug:        str
    title:       str
    source_lang: str
    target_lang: str

    model_config = {"populate_by_name": True}

    @classmethod
    def from_row(cls, row: dict) -> "ProjectOut":
        return cls(
            project_id=row["id"],
            slug=row["slug"],
            title=row["title"],
            source_lang=row["source_lang"],
            target_lang=row["target_lang"],
        )
