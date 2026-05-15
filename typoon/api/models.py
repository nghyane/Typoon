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

from pydantic import BaseModel, Field


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
    title_locale:  dict[str, str] | None = None
    start_year:    int | None = None

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
            title_locale=row.get("title_locale"),
            start_year=row.get("start_year"),
            nsfw=bool(row.get("nsfw")),
            imported_by=row.get("imported_by"),
            created_at=row.get("created_at"),
            updated_at=row.get("updated_at"),
        )


class ChapterTranslationOverlay(BaseModel):
    """A single translation existing for this chapter, surfaced in the
    chapter list response. Schema 19 made every non-takedown
    translation community-readable, so there's no per-viewer filtering
    here — the row is what it is."""

    id:           int
    target_lang:  str
    creator_id:   int | None
    creator_name: str | None
    state:        str                    # pending | running | done | error | blocked
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


class WorkOut(BaseModel):
    """Minimal Work payload. Per "Cách 1 — danh bạ" decision, Works
    carry identity only (cross_refs); per-source display lives on
    each material. The SPA pulls metadata from the active material
    (selected via `?src=`) when rendering the page header.
    """
    id:         int
    cross_refs: dict | None = None
    created_at: str | None = None
    updated_at: str | None = None


class WorkChapterTranslation(BaseModel):
    """One translation surfaced on a work_chapter row.

    Cross-source by construction: a translation appears here for every
    sibling material of the same Work, regardless of which material
    the draft was spawned from. `draft_material_id` is the source
    whose pixels the reader opens; `source_lang` is the BCP-47 of the
    raw the draft was rendered from, so the UI can render
    "@userA · từ Tiếng Anh MangaDex".
    """
    id:                  int
    target_lang:         str
    source_lang:         str | None = None
    owner_id:            int
    creator_name:        str | None = None
    state:               str             # pending | running | done | error | blocked
    error_message:       str | None = None
    shared:              bool
    draft_id:            int | None = None
    draft_chapter_id:    int | None = None
    draft_material_id:   int | None = None
    uses_default_render: bool
    updated_at:          str | None = None


class WorkChapterOut(BaseModel):
    """A logical chapter inside a Work plus every (shared or
    viewer-owned) translation on it. Empty `translations` when no
    one in the community has touched this chapter yet — the SPA
    augments with the live manifest list of the active source.
    """
    id:            int
    number_norm:   str
    label:         str | None = None
    translations:  list[WorkChapterTranslation] = []


class WorkViewerEntry(BaseModel):
    """The viewer's library entry for this Work, if any. Lets the UI
    flip "Theo dõi" into the status dropdown without a second
    round-trip.
    """
    entry_id:    int
    status:      LibraryStatus
    target_lang: str


class WorkDetailOut(BaseModel):
    """Full payload for GET /api/work/{id}. One round-trip drives the
    canonical manga page: identity, sibling materials, every shared
    chapter (cross-source), and the viewer's library state.
    """
    work:         WorkOut
    materials:    list[MaterialOut]
    chapters:     list[WorkChapterOut]
    viewer_entry: WorkViewerEntry | None = None


class LinkSuggestionOut(BaseModel):
    """One row in `GET /api/work/{id}/link-suggestions` — a candidate
    material the SPA can offer for cross-source linking with this Work.

    Two sources fold into the same row shape so the UI renders one
    unified list:

      • `kind="voted"` — community has already cast at least one +1
        vote on the (own × candidate) pair. `score`/`total_votes`
        carry the aggregate; `confidence` is null.

      • `kind="ranked"` — a title-similarity ranker (server-side
        `pg_trgm` + bonuses for shared `title_native` / `title_alt`)
        surfaced the pair without any vote yet. `confidence` is the
        0..1 score; `score`/`total_votes` are 0.
        `reason` explains WHY (`title_native_exact` /
        `title_alt_overlap` / `title_trgm`).

    `own_material_id` is the sibling that triggered the suggestion;
    it scopes which (a, b) pair the next vote attaches to.
    `viewer_vote` reflects whether the viewer has already cast a vote
    so the UI renders "Đã đồng ý" / "Đã từ chối" instead of the
    buttons.
    """
    kind:                  Literal["voted", "ranked"]
    candidate_material_id: int
    candidate_title:       str
    candidate_source:      str | None = None
    candidate_cover:       str | None = None
    candidate_work_id:     int
    own_material_id:       int
    score:                 int = 0
    total_votes:           int = 0
    confidence:            float | None = None
    reason:                str | None = None
    viewer_vote:           int | None = None     # -1 | 1 | None


class LinkVoteResult(BaseModel):
    """Outcome of POST /api/work/{id}/link-vote.

    `merged` flips when the community vote crossed the inline-merge
    threshold AND the two Works were compatible (no conflicting
    cross_refs). `canonical_work_id` is the Work id the SPA should
    redirect to after a successful merge; it may equal the request's
    `work_id` (the request Work was the older sibling and stayed
    canonical) or be different (the request Work was dissolved into
    a sibling — the SPA should navigate there).

    `blocked_reason`:
      None                  — vote stored normally
      'same_work'           — already merged, idempotent
      'cross_refs_conflict' — merge refused due to hard cross_refs
                              collision; the vote still recorded so
                              moderation can review the history
    """
    vote:               int       # -1 | 1
    score:              int
    merged:             bool
    canonical_work_id:  int | None = None
    blocked_reason:     str | None = None


class SplitVoteResult(BaseModel):
    """Outcome of POST /api/work/{id}/split-vote.

    `split` flips when the community vote crossed the inline-split
    threshold. `new_work_id` is the Work the material moved to; null
    when the vote was recorded but no split fired yet.

    `blocked_reason`:
      None             — vote stored normally
      'solo_member'    — would empty the host work, refused
      'material_gone'  — race: material vanished mid-call
    """
    vote:            int       # -1 | 1
    score:           int
    split:           bool
    new_work_id:     int | None = None
    blocked_reason:  str | None = None


class WorkMemberOut(BaseModel):
    """One material attached to a Work, surfaced on the hub's
    "Nguồn đang đọc" panel. The viewer-facing equivalent of
    `MaterialOut` but trimmed to what the panel renders (cover,
    title, source, lang chip) plus the viewer's split-vote state +
    the owner undo-window hint.

    `pending_split_score` / `pending_split_threshold` drive the
    "Đang chờ tách (1/2)" affordance — same shape as the merge
    threshold counter on the suggestions panel.

    `force_link_undo_expires_at` is non-null only for the viewer
    who originally fired the force-link (and only within the undo
    window). Lets the SPA render "↩ Vừa thêm, hoàn tác (còn 8:42)"
    without a separate fetch.
    """
    material_id:                int
    title:                      str
    cover_url:                  str | None = None
    source:                     str | None = None
    languages:                  list[str] = []
    title_native:               str | None = None
    title_locale:               dict[str, str] | None = None
    viewer_split_vote:          int | None = None
    pending_split_score:        int        = 0
    pending_split_threshold:    int        = 2
    force_link_undo_expires_at: str | None = None


DraftState     = Literal["pending", "running", "done", "error", "blocked"]


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


class TranslationOut(BaseModel):
    id:               int
    work_id:          int
    work_chapter_id:  int
    chapter_id:       int               # draft's pixel chapter (= material the reader opens)
    material_id:      int
    owner_id:         int
    target_lang:      str
    draft_id:         int | None = None
    state:            str               # mirrored from draft for convenience
    archive_url:      str | None = None  # public render URL; None until done
    has_edits:        bool = False
    chapter_number:   str | None = None
    chapter_label:    str | None = None
    material_title:   str | None = None
    shared:           bool = True
    created_at:       str | None = None
    updated_at:       str | None = None


LinkOrigin = Literal["auto", "manual"]


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
    # Resolved server-side from the Work's materials against the
    # viewer's reading lang. Same canonical title the Work hub
    # renders; library_entries no longer caches it on the row.
    title:                str
    cover_url:            str | None = None
    work_id:              int
    # User's reading-language preference for this Work. BCP-47.
    target_lang:          str

    # Reading state — drives both the filter UI and "Continue
    # reading" CTAs. Schema 19 simplified to four statuses; reading
    # history lives in its own table now.
    status:               LibraryStatus

    materials:            list[LibraryMaterialLink] = []
    translation_summary:  TranslationSummary = TranslationSummary()
    created_at:           str | None = None
    updated_at:           str | None = None


class GlossaryTermOut(BaseModel):
    id:          int
    source_lang: str
    target_lang: str
    source_term: str
    target_term: str
    notes:       str | None = None


class CommunityFeedEntryOut(BaseModel):
    """One row in /api/community/recent. A Work surfaced for discovery,
    represented by its most recent translated chapter so repeated
    chapter updates don't duplicate the manga in the feed.
    `chapters_in_feed` lets the SPA show a "+N chương khác" affordance.

    `title` + `cover` resolve server-side against the viewer's
    preferred reading language across every material attached to the
    Work — matching the Work hub label rather than whichever
    per-source material surfaced the translation. `material_id`
    records WHICH source the translation came from for analytics;
    the SPA doesn't use it for navigation any more (no `?src=`).
    """

    translation_id:   int
    chapter_id:       int
    chapter_number:   str
    chapter_label:    str | None
    work_id:          int
    material_id:      int
    title:            str
    cover:            str | None
    target_lang:      str
    creator_id:       int | None
    creator_name:     str | None
    created_at:       str | None = None
    archive_url:      str | None = None
    chapters_in_feed: int = 1


class RecentReadOut(BaseModel):
    """One row in /api/me/recent-reads — the home "Tiếp tục đọc" surface.
    Drawn from `reading_history`, deduped per Work so each manga
    surfaces once with its most recent chapter. `title` + `cover`
    follow the same viewer-lang resolver the community feed uses."""

    work_id:          int
    material_id:      int
    title:            str
    cover:            str | None = None
    work_chapter_id:  int
    chapter_number:   str
    chapter_label:    str | None = None
    translation_id:   int | None
    last_read_at:     str | None = None


class StageStatsOut(BaseModel):
    pending: int
    running: int
    stale:   int
    # Tasks under a paused stage — waiting for operator action, not
    # for a worker. Surfaced separately so the header chip can show a
    # "Tạm ngưng" state instead of pretending these are normal pending.
    blocked: int = 0
    # Tasks past `MAX_TASK_ATTEMPTS` — dead-lettered, no worker will
    # touch them again until the operator requeues manually.
    failed:  int = 0


class QueueStatsOut(BaseModel):
    stages:         dict[str, StageStatsOut]
    active_workers: list[str]
    # Snapshot of `stage_pause` so the SPA can render a system-wide
    # banner without a second round-trip.
    paused_stages:  list[str] = []


# ── Admin / ops dashboard ────────────────────────────────────────────
# Wire-shape for /api/admin/ops endpoints. The store side projects
# `lifecycle_state` so the UI doesn't re-derive it from raw columns.

PipelineStageLit  = Literal["prepare", "scan", "translate", "render"]
TaskTargetKindLit = Literal["chapter", "draft", "translation"]
TaskStateLit      = Literal["pending", "running", "stale", "blocked", "failed"]
AdminActionLit    = Literal[
    "stage.pause", "stage.resume",
    "task.requeue", "task.release", "task.force_fail",
    "draft.restart", "draft.takedown",
]


class PausedStageOut(BaseModel):
    stage:     PipelineStageLit
    reason:    str
    paused_at: str
    paused_by: str | None = None


class TaskOut(BaseModel):
    """One row of the queue, projected for the ops dashboard. The
    `lifecycle_state` is computed in SQL — UI just renders it.

    The `*_lang`, `chapter_*`, `work_id`, `llm_model`, `owner_id` block
    is human-context joined from the (chapter / draft / translation)
    row that backs the task. Lets the dashboard answer "which work,
    which language pair, which model" without a second round-trip."""
    stage:             PipelineStageLit
    target_kind:       TaskTargetKindLit
    target_id:         int
    attempts:          int
    claimed_by:        str | None = None
    claimed_at:        str | None = None
    last_error:        str | None = None
    lifecycle_state:   TaskStateLit
    # NULL when the task is unclaimed; otherwise wall-clock seconds
    # since `claimed_at`. Lets the UI flag "stuck N minutes" without
    # the client doing date math.
    claim_age_seconds: int | None = None

    # ── Joined context (best-effort; NULL if the backing row is gone) ──
    work_id:           int | None = None
    work_chapter_id:   int | None = None
    chapter_id:        int | None = None
    chapter_label:     str | None = None
    # BCP-47. `target_lang` is NULL for `chapter`-kind tasks (no
    # target language yet — chapter rows are pre-translation).
    source_lang:       str | None = None
    target_lang:       str | None = None
    # Only populated for draft/translation tasks.
    llm_model:         str | None = None
    # users.id of the draft creator (draft) or translation owner
    # (translation). NULL for chapter-kind tasks.
    owner_id:          int | None = None


class TaskListOut(BaseModel):
    items:       list[TaskOut]
    next_cursor: str | None = None


class AdminActionOut(BaseModel):
    """Audit row — every ops mutation produces exactly one of these,
    inserted in the same transaction as the state change it describes.
    `prev_state` is NULL for create-only actions (stage.pause),
    populated for every mutation that overwrites state."""
    id:         int
    at:         str
    actor_id:   int | None = None
    action:     AdminActionLit
    target_ref: dict
    reason:     str
    prev_state: dict | None = None


# ── Request bodies for /api/admin/ops mutations ──────────────────────
# All mutations require `reason` (min 3 chars) — non-negotiable for
# post-mortem. Task mutations additionally carry `expected_attempts`
# and `expected_claimed_by` as optimistic-concurrency tokens taken
# from the snapshot the admin clicked on. A guard miss returns 409.

_REASON = Field(min_length=3, max_length=500)


class StagePauseIn(BaseModel):
    reason: str = _REASON


class StageResumeIn(BaseModel):
    reason: str = _REASON


class RequeueTaskIn(BaseModel):
    reason:              str         = _REASON
    expected_attempts:   int         = Field(ge=0)
    expected_claimed_by: str | None  = None


class ReleaseTaskIn(BaseModel):
    """Release a stale claim. The claim itself is the only guarded
    field — attempts are intentionally not touched, so we don't need
    `expected_attempts`. `expected_claimed_by` is required (not
    optional) because releasing an unclaimed task is a no-op the
    admin shouldn't be asking for."""
    reason:              str = _REASON
    expected_claimed_by: str


class ForceFailTaskIn(BaseModel):
    reason:              str         = _REASON
    expected_attempts:   int         = Field(ge=0)
    expected_claimed_by: str | None  = None


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


class SessionUser(BaseModel):
    """Current-user payload returned by `GET /api/auth/me` and
    `PATCH /api/me/preferences`.

    This is the single canonical "who am I + what are my prefs" shape.
    There is no `/api/me` GET — clients ask `/api/auth/me` for
    everything identity-related (the old slim `/me` was a duplicate
    subset, removed because two endpoints meant two caches, two
    types, and a real race when both lagged behind a logout).
    """
    id:                    int
    display_name:          str
    avatar_url:            str | None = None
    is_admin:              bool        = False
    preferred_target_lang: str | None  = None
