"""API response models for the Material + Translation architecture.

Mirrors the SQL schema in `typoon/storage/schema.sql`. Treat this file
as the wire contract: the web SPA + browser extension consume these
shapes verbatim (see `web/src/shared/api/api.ts`).

Conventions:
  - Snake_case fields. `id` always means "primary key of this entity".
  - Timestamps are ISO 8601 strings (Z-suffixed UTC), not datetime
    objects — Postgres adapter formats them at SQL layer.
  - Visibility / state values are typed as `str` (CHECK constraint on
    SQL side is the validator).
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel


# ── Material ──────────────────────────────────────────────────────────


MaterialOrigin = Literal["source", "extension", "upload"]


class MaterialOut(BaseModel):
    """A manga identity. Source-backed materials are cross-user (shared
    on (source, upstream_ref)); ext / upload are per-row.

    `imported_by` is audit only — NOT an ownership boundary. Every user
    has read + spawn-translation capability on any material; only ext
    / upload materials have `edit` / `delete` gated on imported_by.
    """

    id:            int
    origin:        MaterialOrigin
    work_id:       int                     # global Work identity (cross-source)
    source:        str | None = None       # NULL for ext / upload
    upstream_ref:  str | None = None       # NULL for ext / upload

    title:         str
    cover_url:     str | None = None
    description:   str | None = None
    author:        str | None = None
    status:        str | None = None
    languages:     list[str] = []

    title_native:  str | None = None
    title_alt:     list[str] = []
    cross_refs:    dict | None = None

    nsfw:          bool = False

    imported_by:   int | None = None
    created_at:    str | None = None
    updated_at:    str | None = None

    @classmethod
    def from_row(cls, row: dict) -> "MaterialOut":
        return cls(
            id=row["id"],
            origin=row["origin"],
            work_id=int(row["work_id"]),
            source=row.get("source"),
            upstream_ref=row.get("upstream_ref"),
            title=row["title"],
            cover_url=row.get("cover_url"),
            description=row.get("description"),
            author=row.get("author"),
            status=row.get("status"),
            languages=list(row.get("languages") or []),
            title_native=row.get("title_native"),
            title_alt=list(row.get("title_alt") or []),
            cross_refs=row.get("cross_refs"),
            nsfw=bool(row.get("nsfw")),
            imported_by=row.get("imported_by"),
            created_at=row.get("created_at"),
            updated_at=row.get("updated_at"),
        )


# ── Chapter + per-chapter translations overlay ────────────────────────


class ChapterTranslationOverlay(BaseModel):
    """A single translation existing for this chapter, surfaced in the
    chapter list response. Schema 19 made every non-takedown
    translation community-readable, so there's no per-viewer filtering
    here — the row is what it is."""

    id:           int
    target_lang:  str
    creator_id:   int | None
    creator_name: str | None
    state:        str                    # pending | running | done | error
    from_cache:   bool                   # True if cache hit (no quota spent)


class ChapterOut(BaseModel):
    id:            int
    material_id:   int
    position:      int
    # Canonical chapter key joined from the chapter's work_chapter.
    # Same value across sibling materials of the same Work.
    number:        str
    label:         str | None = None
    upstream_url:  str | None = None
    page_count:    int
    updated_at:    str | None = None
    translations:  list[ChapterTranslationOverlay] = []


# ── Translation Draft (Layer 2) ───────────────────────────────────────


DraftState     = Literal["pending", "running", "done", "error"]


class DraftProgress(BaseModel):
    stage: str
    index: int
    total: int


class TranslationDraftOut(BaseModel):
    id:              int
    chapter_id:      int
    source_lang:     str
    target_lang:     str
    glossary_fp:     str
    llm_model:       str
    state:           DraftState
    error_message:   str | None = None
    progress:        DraftProgress | None = None
    created_by:      int | None = None
    created_at:      str | None = None
    updated_at:      str | None = None


# ── Translation (Layer 3 — per-user wrapper) ─────────────────────────


class TranslationOut(BaseModel):
    id:             int
    chapter_id:     int
    material_id:    int
    owner_id:       int
    target_lang:    str
    draft_id:       int | None = None
    state:          str               # mirrored from draft for convenience
    archive_url:    str | None = None  # public render URL; None until done
    has_edits:      bool = False
    chapter_number: str | None = None
    chapter_label:  str | None = None
    material_title: str | None = None
    created_at:     str | None = None
    updated_at:     str | None = None


# ── Library entry ────────────────────────────────────────────────────


LinkOrigin = Literal["primary", "auto", "manual"]


class LibraryMaterialLink(BaseModel):
    material_id:  int
    link_origin:  LinkOrigin
    linked_at:    str | None = None


LibraryStatus = Literal["reading", "plan", "done", "dropped"]


class TranslationSummary(BaseModel):
    """Counts of *the viewer's* translations per draft state, keyed
    by the library_entry. Cards use this for the "Đang dịch 2 · Lỗi 1"
    chip so they never need a per-entry follow-up query."""
    pending: int = 0
    running: int = 0
    done:    int = 0
    error:   int = 0


class LibraryEntryOut(BaseModel):
    id:                   int
    title:                str
    cover_url:            str | None = None
    primary_material_id:  int | None = None

    # Reading state — drives both the filter UI and "Continue
    # reading" CTAs. Schema 19 simplified to four statuses; reading
    # history lives in its own table now.
    status:               LibraryStatus

    materials:            list[LibraryMaterialLink] = []
    translation_summary:  TranslationSummary = TranslationSummary()
    created_at:           str | None = None
    updated_at:           str | None = None


# ── Library suggestion (cross-source linking) ────────────────────────


SuggestionSignal = Literal[
    "cross_refs", "vote_high", "title_native", "vote_low", "author",
]


class LibrarySuggestionOut(BaseModel):
    """Returned by GET /api/library/suggest. The frontend renders a
    banner when one of these comes back; `signal` and `confidence`
    drive the copy ("Có vẻ là cùng manga" vs "Có thể là cùng manga")."""

    entry_id:    int
    entry_title: str
    confidence:  Literal["high", "medium", "low"]
    signal:      SuggestionSignal
    score:       int | None = None     # vote score when signal is vote_*


# ── Glossary ──────────────────────────────────────────────────────────


class GlossaryTermOut(BaseModel):
    id:          int
    source_lang: str
    target_lang: str
    source_term: str
    target_term: str
    notes:       str | None = None


# ── Feed (Hội Mê Truyện, guild-scoped) ──────────────────────────────


# ── Community feed (cross-user recent translations) ────────────────


class CommunityFeedEntryOut(BaseModel):
    """One row in /api/community/recent. A material surfaced for
    discovery, represented by its most recent translated chapter so
    repeated chapter updates don't duplicate the manga in the feed.
    `chapters_in_feed` lets the SPA show a "+N chương khác" affordance.
    """

    translation_id:   int
    chapter_id:       int
    chapter_number:   str
    chapter_label:    str | None
    material_id:      int
    material_title:   str
    material_cover:   str | None
    target_lang:      str
    creator_id:       int | None
    creator_name:     str | None
    created_at:       str | None = None
    archive_url:      str | None = None
    chapters_in_feed: int = 1


class RecentReadOut(BaseModel):
    """One row in /api/me/recent-reads — the home "Tiếp tục đọc" surface.
    Drawn from `reading_history`, deduped per Work so each manga
    surfaces once with its most recent chapter. `material_id` is the
    source the user last opened (deep-link target on click)."""

    work_id:          int
    material_id:      int
    material_title:   str
    material_cover:   str | None = None
    work_chapter_id:  int
    chapter_number:   str
    chapter_label:    str | None = None
    translation_id:   int | None
    last_read_at:     str | None = None


# ── Workers / queue ───────────────────────────────────────────────────


class StageStatsOut(BaseModel):
    pending: int
    running: int
    stale:   int


class QueueStatsOut(BaseModel):
    stages:         dict[str, StageStatsOut]
    active_workers: list[str]


# ── Reports + moderation ─────────────────────────────────────────────


ReportTargetKind = Literal["material", "chapter", "draft", "translation"]
ReportKind       = Literal["dmca", "abuse", "quality", "other"]
ReportStatus     = Literal["open", "reviewing", "resolved", "dismissed"]
ModerationAction = Literal["takedown", "restore", "delete"]


class ReportOut(BaseModel):
    id:              int
    reporter_id:     int | None
    reporter_label:  str
    target_kind:     ReportTargetKind
    target_id:       int
    kind:            ReportKind
    reason:          str
    status:          ReportStatus
    created_at:      str | None
    resolved_at:     str | None
    resolved_by:     int | None


class ModerationActionOut(BaseModel):
    id:           int
    report_id:    int | None
    target_kind:  ReportTargetKind
    target_id:    int
    action:       ModerationAction
    reason:       str
    actor_id:     int | None
    created_at:   str | None


# ── User / identity ──────────────────────────────────────────────────


class MeOut(BaseModel):
    id:            int
    display_name:  str
    avatar_url:    str | None = None
