"""Store — protocol for the Material + Translation architecture.

This is the contract every storage backend (Postgres in production)
must satisfy. The route layer + worker pipeline import only this
protocol; concrete adapters are wired in `typoon/api/deps.py` and
`typoon/workers/loop.py`.

Identity:
    Users, identities, guilds, API tokens.

Reading entities:
    Materials, chapters. Source-backed materials are cross-user.

Cross-source linking:
    material_link_votes — community-voted pairs (canonical a < b).
    Suggestions surfaced by `find_library_suggestion`.

Pipeline (3 layers):
    Layer 1 — chapter scope:    bubbles, geometry, masks. Shared by
                                every translation on the chapter.
    Layer 2 — draft scope:      translation_drafts +
                                translation_draft_bubbles + briefs.
                                Keyed on (chapter, src, tgt, fp);
                                shared by visibility scope.
    Layer 3 — translation:      Per (chapter, owner, lang). Points
                                at a draft; carries sparse edits +
                                user-facing flags.

Library:
    library_entries (per user) → library_materials many-to-one
    Cross-source grouping is per-user. Linking accumulates a vote.

Queue:
    tasks (target_kind, target_id, stage). target_kind in
    {chapter, draft, translation}. LISTEN payload format:
    `<target_kind>:<target_id>`.

Glossary:
    user_glossary (per user) merged with community_glossary
    (system-curated) → glossary_fingerprint cache key.

Quota / DMCA:
    chapter_consumes (only LLM-costing events, never cache hits).
    dmca_takedowns with admin lifecycle.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal, Protocol

if TYPE_CHECKING:
    from typoon.adapters.inbox import InboxHandle


# Type aliases. Kept as Literal[...] strings (not Enum) so the
# protocol stays compatible with raw asyncpg dict rows.
MaterialOrigin   = Literal["source", "extension", "upload"]
PagesOrigin      = Literal["remote", "local"]
DraftVisibility  = Literal["private", "guild", "all_guilds"]
DraftState       = Literal["pending", "running", "done", "error"]
PipelineStage    = Literal["prepare", "scan", "translate", "render"]
TaskTargetKind   = Literal["chapter", "draft", "translation"]
LinkOrigin       = Literal["primary", "auto", "manual"]
TakedownKind     = Literal["material", "chapter", "draft", "translation"]


class Store(Protocol):
    """Storage interface — pipeline + API operations."""

    # ── Lifecycle ─────────────────────────────────────────────────
    async def close(self) -> None: ...
    async def ping(self) -> None: ...

    # ── Identity ──────────────────────────────────────────────────
    async def upsert_user_from_identity(
        self,
        *,
        provider:     str,
        external_id:  str,
        display_name: str,
        avatar_url:   str | None = None,
        email:        str | None = None,
        metadata:     dict | None = None,
    ) -> dict: ...
    async def get_user(self, user_id: int) -> dict | None: ...
    async def get_user_by_identity(
        self, provider: str, external_id: str,
    ) -> dict | None: ...
    async def get_external_id(
        self, user_id: int, provider: str = "discord",
    ) -> str | None: ...

    # ── Guild memberships (Discord) ──────────────────────────────
    async def upsert_user_guilds(
        self, user_id: int, guilds: list[dict],
    ) -> None:
        """Replace cached guilds for this user. `guilds` items:
        `{id, name?, icon_url?}`. Used during Discord OAuth exchange
        and refreshed lazily on /me."""
        ...

    async def get_user_guilds(self, user_id: int) -> list[dict]: ...

    async def user_in_guild(self, user_id: int, guild_id: str) -> bool:
        """Membership check used by visibility gates + feed access."""
        ...

    # ── API tokens (CLI / extension / worker) ────────────────────
    async def create_api_token(
        self, user_id: int, name: str, prefix: str, token_hash: str,
        *, scopes: list[str] | None = None,
    ) -> int: ...
    async def list_api_tokens(self, user_id: int) -> list[dict]: ...
    async def candidates_by_prefix(self, prefix: str) -> list[dict]: ...
    async def touch_api_token(self, token_id: int) -> None: ...
    async def revoke_api_token(self, user_id: int, token_id: int) -> bool: ...

    # ── Material ──────────────────────────────────────────────────
    async def get_or_create_source_material(
        self,
        *,
        source:       str,
        upstream_ref: str,
        title:        str,
        cover_url:    str | None = None,
        description:  str | None = None,
        author:       str | None = None,
        status:       str | None = None,
        languages:    list[str] | None = None,
        title_native: str | None = None,
        title_alt:    list[str] | None = None,
        cross_refs:   dict | None = None,
        nsfw:         bool = False,
        imported_by:  int | None = None,
    ) -> int:
        """Cross-user dedup on (source, upstream_ref). Returns existing
        row id if the pair was seen before; otherwise inserts. Display
        snapshot fields are first-write-wins."""
        ...

    async def create_local_material(
        self,
        *,
        origin:       MaterialOrigin,  # 'extension' | 'upload'
        title:        str,
        cover_url:    str | None = None,
        description:  str | None = None,
        author:       str | None = None,
        nsfw:         bool = False,
        imported_by:  int | None = None,
    ) -> int:
        """Per-row material (no cross-user dedup) for ext + upload."""
        ...

    async def get_material(self, material_id: int) -> dict | None: ...

    async def update_material_metadata(
        self, material_id: int,
        *,
        title:        str | None = None,
        cover_url:    str | None = None,
        description:  str | None = None,
        nsfw:         bool | None = None,
    ) -> None: ...

    async def delete_material(self, material_id: int) -> None:
        """Cascade: chapters, drafts, translations, bubbles, geometry,
        masks. Used by ext / upload owners only — source-backed
        materials are not user-deletable."""
        ...

    # ── Cross-source linking (community-voted) ──────────────────
    async def cast_material_link_vote(
        self, voter_id: int, material_a_id: int, material_b_id: int,
        vote: Literal[-1, 1],
    ) -> None:
        """Upsert vote with canonical ordering (a < b enforced). +1 on
        link, -1 on reject. Idempotent per (a, b, voter)."""
        ...

    async def remove_material_link_vote(
        self, voter_id: int, material_a_id: int, material_b_id: int,
    ) -> None:
        """Delete the voter's row on this pair (used when user unlinks)."""
        ...

    async def get_material_link_score(
        self, material_a_id: int, material_b_id: int,
    ) -> tuple[int, int]:
        """Returns (score, total_votes). Reads the materialized view
        if fresh; falls back to live aggregation. Canonical order
        applied internally — caller may pass any order."""
        ...

    async def refresh_material_links(self) -> None:
        """REFRESH MATERIALIZED VIEW material_links. Called by a
        periodic worker / on-demand by the suggestion endpoint when
        staleness matters."""
        ...

    # ── Chapter ──────────────────────────────────────────────────
    async def create_chapter(
        self,
        material_id:   int,
        number:        str,
        *,
        label:         str | None = None,
        upstream_url:  str | None = None,
        pages_origin:  PagesOrigin = "remote",
    ) -> int:
        """Inserts with server-managed sparse `position`."""
        ...

    async def get_chapter(self, chapter_id: int) -> dict | None: ...

    async def list_chapters(self, material_id: int) -> list[dict]:
        """Ordered by position. No translation overlay — caller joins
        separately via `list_translations_for_chapters`."""
        ...

    async def delete_chapter(self, chapter_id: int) -> bool: ...

    async def set_chapter_prepared(
        self,
        chapter_id:        int,
        *,
        prepared_hash:     str,
        prepared_backend:  str,
        prepared_locator:  str,
        page_count:        int,
    ) -> None: ...

    async def set_chapter_masks(
        self, chapter_id: int,
        *, masks_backend: str, masks_locator: str,
    ) -> None: ...

    async def find_chapter_by_prepared_hash(
        self, prepared_hash: str,
    ) -> dict | None:
        """CAS lookup. When a freshly-uploaded chapter hashes to the
        same bytes as an existing one, prepare can copy the locator
        and skip work."""
        ...

    # ── Scan output (chapter-level) ──────────────────────────────
    async def save_bubbles(
        self, chapter_id: int, bubbles: list[dict],
    ) -> None: ...
    async def get_bubbles(self, chapter_id: int) -> list[dict]: ...
    async def has_bubbles(self, chapter_id: int) -> bool: ...
    async def save_geometry(
        self, chapter_id: int, pages: list[dict],
    ) -> None: ...
    async def get_geometry(self, chapter_id: int) -> list[dict]: ...

    # ── Translation drafts (Layer 2) ──────────────────────────────
    async def find_reusable_draft(
        self,
        *,
        chapter_id:    int,
        source_lang:   str,
        target_lang:   str,
        glossary_fp:   str,
        viewer_id:     int,
        viewer_guilds: list[str],
    ) -> dict | None:
        """Look up a non-private, non-taken-down draft matching the
        cache key whose visibility lets `viewer_id` see it.

        Visibility check:
          - 'guild'       → draft.scope_guild_id in viewer_guilds
          - 'all_guilds'  → any of creator's guilds intersects viewer_guilds
          - 'private'     → excluded (cache index doesn't include them)
        """
        ...

    async def create_draft(
        self,
        *,
        chapter_id:     int,
        source_lang:    str,
        target_lang:    str,
        glossary_fp:    str,
        llm_model:      str,
        created_by:     int,
        visibility:     DraftVisibility,
        scope_guild_id: str | None,
    ) -> int:
        """Insert a new draft in state='pending'. Unique constraint
        on cache key (excl. private + taken-down) means callers MUST
        check `find_reusable_draft` first or handle UniqueViolation."""
        ...

    async def get_draft(self, draft_id: int) -> dict | None: ...

    async def update_draft_state(
        self,
        draft_id: int,
        *,
        state:    DraftState,
        error:    str | None = None,
    ) -> None: ...

    async def set_draft_progress(
        self, draft_id: int, *, stage: str, index: int, total: int,
    ) -> None: ...

    async def save_draft_bubbles(
        self, draft_id: int, bubbles: list[dict],
    ) -> None: ...
    async def get_draft_bubbles(self, draft_id: int) -> list[dict]: ...

    async def save_draft_brief(
        self, draft_id: int, brief: dict,
    ) -> None: ...
    async def get_draft_brief(self, draft_id: int) -> dict | None: ...

    async def takedown_draft(
        self, draft_id: int, reason: str,
    ) -> None: ...

    # ── Translations (Layer 3, per-user) ──────────────────────────
    async def get_or_create_translation(
        self,
        *,
        chapter_id:  int,
        owner_id:    int,
        target_lang: str,
        draft_id:    int | None,
    ) -> int:
        """Insert or fetch. UNIQUE (chapter_id, owner_id, target_lang)
        ensures one row per (chapter, owner, lang) tuple."""
        ...

    async def get_translation(self, translation_id: int) -> dict | None: ...

    async def list_translations_for_chapters(
        self,
        chapter_ids:   list[int],
        viewer_id:     int,
        viewer_guilds: list[str],
    ) -> dict[int, list[dict]]:
        """Bulk overlay: for each chapter id, return the translations
        the viewer can see. Used by GET /api/material/{id} to embed
        the per-chapter translation list."""
        ...

    async def update_translation_archive(
        self,
        translation_id: int,
        *,
        archive_backend: str,
        archive_locator: str,
    ) -> None: ...

    async def update_translation_feed(
        self,
        translation_id: int,
        *,
        in_feed:        bool,
        feed_guild_id:  str | None,
    ) -> None: ...

    async def takedown_translation(
        self, translation_id: int, reason: str,
    ) -> None: ...

    async def upsert_translation_edit(
        self, translation_id: int,
        page_index: int, bubble_idx: int, edited_text: str,
    ) -> None: ...

    async def get_translation_edits(
        self, translation_id: int,
    ) -> list[dict]: ...

    async def delete_translation_edit(
        self, translation_id: int, page_index: int, bubble_idx: int,
    ) -> bool: ...

    # ── Library entries ──────────────────────────────────────────
    async def list_library_entries(self, user_id: int) -> list[dict]:
        """Includes linked materials inline (one row per entry, with
        a list field `materials`)."""
        ...

    async def get_library_entry(
        self, entry_id: int, user_id: int,
    ) -> dict | None: ...

    async def create_library_entry(
        self,
        *,
        user_id:             int,
        title:               str,
        cover_url:           str | None,
        primary_material_id: int,
    ) -> int:
        """Create entry + link the primary material with link_origin='primary'."""
        ...

    async def update_library_entry(
        self,
        entry_id: int, user_id: int,
        *,
        title:        str | None = None,
        bookmarked:   bool | None = None,
        last_read_at: str | None = None,        # ISO; pass None to leave alone
        last_chapter_ref: dict | None = None,
    ) -> None: ...

    async def delete_library_entry(
        self, entry_id: int, user_id: int,
    ) -> bool: ...

    async def link_material_to_entry(
        self,
        *,
        entry_id:    int,
        material_id: int,
        link_origin: LinkOrigin,
        voter_id:    int,
    ) -> None:
        """Link + cast a +1 vote on the pair (existing materials in the
        entry x new material). Multi-link → multiple vote rows."""
        ...

    async def unlink_material_from_entry(
        self,
        *,
        entry_id:    int,
        material_id: int,
        voter_id:    int,
    ) -> None:
        """Unlink + remove the voter's votes on pairs involving this
        material in this entry. If the entry has no materials left,
        the caller (route) deletes the entry."""
        ...

    async def find_library_suggestion(
        self,
        *,
        user_id:     int,
        material_id: int,
    ) -> dict | None:
        """Suggestion ranking per RFC §7.4.1:

            1. cross_refs intersect with any entry's materials → high
            2. material_links.score ≥ 3 with any entry's materials → high
            3. title_native case-fold match within user's library     → medium
            4. material_links.score in [1, 2]                          → low
            5. otherwise                                                → None

        Returns dict matching `LibrarySuggestionOut`, or None."""
        ...

    async def reject_library_suggestion(
        self,
        *,
        voter_id:    int,
        material_id: int,
        candidate_material_id: int,
    ) -> None:
        """User clicks "Không phải cùng manga" — cast -1 vote on the
        pair so we don't suggest it again."""
        ...

    # ── Feed (Hội Mê Truyện, guild-scoped) ──────────────────────
    async def list_feed_entries(
        self,
        *,
        guild_id: str,
        viewer_id: int,
        limit:    int = 50,
        before:   str | None = None,   # ISO timestamp; cursor pagination
    ) -> list[dict]:
        """Translations with `in_feed=TRUE`, scoped to the guild via
        `feed_guild_id = guild_id` OR (feed_guild_id IS NULL AND the
        translation's creator belongs to this guild). Joins material
        + chapter for display. Viewer must be in the guild — caller
        enforces."""
        ...

    # ── Glossary ─────────────────────────────────────────────────
    async def list_user_glossary(
        self,
        user_id: int,
        source_lang: str | None = None,
        target_lang: str | None = None,
    ) -> list[dict]: ...
    async def upsert_user_glossary_term(
        self,
        user_id: int,
        source_lang: str, target_lang: str,
        source_term: str, target_term: str,
        notes: str | None = None,
    ) -> int: ...
    async def delete_user_glossary_term(
        self, user_id: int, term_id: int,
    ) -> bool: ...

    async def compute_glossary_fingerprint(
        self,
        *,
        user_id: int,
        source_lang: str,
        target_lang: str,
        material_id: int | None,
    ) -> str:
        """SHA256 (16-hex) of the effective glossary applied for this
        spawn. Combines user_glossary overrides + community_glossary
        (scoped + global) deterministically. Two users with identical
        effective glossary produce the same fingerprint → cache hit."""
        ...

    # ── Quota ────────────────────────────────────────────────────
    async def record_chapter_consume(
        self,
        *,
        user_id: int,
        translation_id: int,
        kind: Literal["draft_create", "render_create"],
    ) -> None: ...
    async def count_user_consumes_since(
        self, user_id: int, seconds: int,
    ) -> int: ...

    # ── Tasks queue ──────────────────────────────────────────────
    async def enqueue_task(
        self,
        *,
        target_kind: TaskTargetKind,
        target_id:   int,
        stage:       PipelineStage,
    ) -> None: ...

    async def claim_task(
        self,
        stage:     PipelineStage,
        worker_id: str,
    ) -> tuple[TaskTargetKind, int] | None:
        """Atomic FOR UPDATE SKIP LOCKED. Returns (target_kind, id)
        of the claimed task or None when queue empty."""
        ...

    async def complete_task(
        self,
        target_kind: TaskTargetKind, target_id: int,
        stage: PipelineStage,
    ) -> None: ...

    async def advance_task(
        self,
        target_kind: TaskTargetKind, target_id: int,
        completed_stage: PipelineStage, next_stage: PipelineStage,
        *,
        next_target_kind: TaskTargetKind | None = None,
        next_target_id:   int | None = None,
    ) -> None:
        """Complete current stage + enqueue next. The next stage's
        target may differ — e.g. (chapter, scan) completes and
        enqueues (draft, translate). Caller passes next_target_* when
        target shifts; defaults to same target if omitted."""
        ...

    async def fail_task(
        self,
        target_kind: TaskTargetKind, target_id: int,
        stage: PipelineStage, error: str,
    ) -> None: ...

    async def requeue_task(
        self,
        target_kind: TaskTargetKind, target_id: int,
        stage: PipelineStage, error: str,
    ) -> None: ...

    async def release_claims_by_prefix(self, prefix: str) -> int: ...

    async def queue_stats(self) -> dict: ...

    # ── Chapter inbox (multipart upload handle) ──────────────────
    async def set_inbox_handle(self, handle: "InboxHandle") -> None: ...
    async def get_inbox_handle(
        self, chapter_id: int,
    ) -> "InboxHandle | None": ...
    async def clear_inbox_handle(self, chapter_id: int) -> None: ...

    # ── DMCA ─────────────────────────────────────────────────────
    async def record_dmca_takedown(
        self,
        *,
        target_kind:    TakedownKind,
        target_id:      int,
        scope_guild_id: str | None,
        reason:         str,
        reporter:       str,
    ) -> int:
        """Inserts a takedown log + flips `takedown_at` on the target.
        Restoration reverses both (see `restore_dmca_takedown`)."""
        ...

    async def restore_dmca_takedown(self, takedown_id: int) -> bool: ...

    async def list_dmca_takedowns(
        self, *, active_only: bool = True, limit: int = 100,
    ) -> list[dict]: ...
