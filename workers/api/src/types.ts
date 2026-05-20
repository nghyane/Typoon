/**
 * Env — Cloudflare bindings for typoon-api Worker.
 *
 * Generated types (`wrangler types`) will produce worker-configuration.d.ts
 * with the same names; keep both in sync.
 */

import type { TranslationStatusDO } from "./do/translation-status";
import type { PipelineCallback }     from "./rpc/pipeline-callback";

export interface Env {
  // D1 — primary datastore
  DB: D1Database;

  // R2 — blob storage (prepared pages, scan, render archives)
  R2: R2Bucket;

  // Durable Objects
  STATUS_DO: DurableObjectNamespace<TranslationStatusDO>;

  // Service binding — invoke ChapterPipeline Workflow
  PIPELINE: Fetcher;

  // Queue — trigger material_links refresh (cron → queue → DO batch)
  LINKS_REFRESH_QUEUE: Queue;

  // Secrets
  JWT_SECRET:            string;
  DISCORD_CLIENT_ID:     string;
  DISCORD_CLIENT_SECRET: string;
  DISCORD_GUILD_ID?:     string;
  ADMIN_ROLE_ID?:        string;

  // Vars
  DISCORD_API: string;
}

// ── Domain literals ─────────────────────────────────────────────────

export type DraftState    = "pending" | "running" | "done" | "error" | "blocked";
export type PipelineStage = "prepare" | "scan" | "brief" | "translate" | "typeset" | "finalize";
export type LibraryStatus = "reading" | "plan" | "done" | "dropped";
export type MaterialOrigin = "source" | "extension" | "upload";
export type LinkOrigin    = "auto" | "manual";

// ── JWT payload ─────────────────────────────────────────────────────

export interface JwtPayload {
  sub:      string;   // user_id as string (RFC 7519)
  iss:      string;
  iat:      number;
  exp:      number;
  roles:    string[];
}

// ── Hono context variables ───────────────────────────────────────────

export interface ContextVars {
  userId:   number;
  jwtRoles: string[];
}

// ── Pipeline progress event (emitted via DO WebSocket) ──────────────

export type PipelineProgressEvent =
  | { type: "progress"; stage: PipelineStage; index?: number; total?: number }
  | { type: "done";     archive_key: string }
  | { type: "error";    stage: PipelineStage; message: string };

// ── Pipeline params (passed to Workflow) ────────────────────────────

export interface PipelineParams {
  chapter_id:   number;
  draft_id:     number;
  zip_key:      string;
  source_lang:  string;
  target_lang:  string;
  glossary_fp:  string;
  strategy?:    "auto" | "one_to_one" | "stitch";
}

// ── Pipeline finalize args (called by pipeline → api RPC) ───────────

export interface FinalizeArgs {
  chapter_id:  number;
  draft_id:    number;
  archive_key: string;
  page_count:  number;
  scan_keys:   string[];
  mask_keys:   string[];
}

// ── API Response Types ──────────────────────────────────────────────
//
// Canonical shapes returned by the API. FE client types in
// `web/src/shared/api/api.ts` should mirror these.

/** Structured error — all 4xx/5xx responses use this shape. */
export interface ApiError {
  error: {
    code:    string;   // machine-readable: "not_found", "conflict", ...
    message: string;   // human-readable (Vietnamese preferred)
    details?: Record<string, unknown>;
  };
}

// ── Material ────────────────────────────────────────────────────────

export interface ApiMaterial {
  id:            number;
  origin:        MaterialOrigin;
  work_id:       number;
  source:        string | null;
  upstream_ref:  string | null;
  title:         string;
  cover_url:     string | null;
  description:   string | null;
  author:        string | null;
  status:        string | null;
  languages:     string[];
  title_native:  string | null;
  title_alt:     string[];
  cross_refs:    Record<string, unknown> | null;
  title_locale:  Record<string, string> | null;
  start_year:    number | null;
  nsfw:          boolean;
  imported_by:   number | null;
  created_at:    string | null;
  updated_at:    string | null;
}

// ── Work ────────────────────────────────────────────────────────────

export interface ApiWork {
  id:         number;
  cross_refs: Record<string, unknown> | null;
  created_at: string | null;
  updated_at: string | null;
}

export interface ApiWorkChapterTranslation {
  id:                  number;
  target_lang:         string;
  source_lang:         string | null;
  owner_id:            number;
  creator_name:        string | null;
  state:               DraftState;
  error_message:       string | null;
  shared:              boolean;
  draft_id:            number | null;
  draft_chapter_id:    number | null;
  draft_material_id:   number | null;
  uses_default_render: boolean;
  updated_at:          string | null;
}

export interface ApiUploadingChapter {
  chapter_id:  number;
  material_id: number;
  source_lang: string | null;
  uploaded_by: number;
  created_at:  string | null;
}

export interface ApiWorkChapter {
  id:                   number;
  number_norm:          string;
  label:                string | null;
  translations:         ApiWorkChapterTranslation[];
  uploading_chapters:   ApiUploadingChapter[];
}

export interface ApiWorkViewerEntry {
  entry_id:    number;
  status:      LibraryStatus;
  target_lang: string;
}

export interface ApiWorkDetail {
  work:             ApiWork;
  materials:        ApiMaterial[];
  chapters:         ApiWorkChapter[];
  viewer_entry:     ApiWorkViewerEntry | null;
  redirected_from:  number | null;
}

// ── Translation ─────────────────────────────────────────────────────

export interface ApiBubbleEdit {
  page_index:  number;
  bubble_idx:  number;
  source_text: string;
  draft_text:  string;
  edited_text: string | null;
  kind:        "dialogue" | "sfx" | "skip";
}

export interface ApiMyTranslation {
  translation_id:       number;
  target_lang:          string;
  state:                DraftState;
  has_archive:          boolean;
  updated_at:           string | null;
  chapter_id:           number;
  chapter_number:       string;
  chapter_label:        string | null;
  chapter_position:     number;
  chapter_upstream_url: string | null;
  material_id:          number;
  material_title:       string;
  material_cover:       string | null;
  material_source:      string | null;
  material_upstream_ref:string | null;
}

export interface ApiTranslation {
  id:               number;
  work_id:          number;
  work_chapter_id:  number;
  chapter_id:       number;
  material_id:      number;
  owner_id:         number;
  target_lang:      string;
  draft_id:         number | null;
  state:            DraftState;
  archive_url:      string | null;
  has_edits:        boolean;
  chapter_number:   string | null;
  chapter_label:    string | null;
  material_title:   string | null;
  shared:           boolean;
  created_at:       string | null;
  updated_at:       string | null;
}

export interface SpawnTranslateResult {
  translation_id: number;
  draft_id:       number;
  state:          DraftState;
  cache_hit:      boolean;
  chapter_id:     number;
}

// ── Library ─────────────────────────────────────────────────────────

export interface ApiLibraryMaterialLink {
  material_id: number;
  link_origin: LinkOrigin;
  linked_at:   string | null;
}

export interface ApiTranslationSummary {
  pending: number;
  running: number;
  done:    number;
  error:   number;
}

export interface ApiLibraryEntry {
  id:                   number;
  title:                string;
  cover_url:            string | null;
  work_id:              number;
  target_lang:          string;
  status:               LibraryStatus;
  materials:            ApiLibraryMaterialLink[];
  translation_summary:  ApiTranslationSummary;
  created_at:           string | null;
  updated_at:           string | null;
}

// ── Community ───────────────────────────────────────────────────────

export interface ApiCommunityFeedEntry {
  translation_id:   number;
  chapter_id:       number;
  chapter_number:   string;
  chapter_label:    string | null;
  work_id:          number;
  material_id:      number;
  title:            string;
  cover:            string | null;
  target_lang:      string;
  creator_id:       number | null;
  creator_name:     string | null;
  created_at:       string | null;
  archive_url:      string | null;
  chapters_in_feed: number;
}

// ── Memory ──────────────────────────────────────────────────────────

export interface ApiMemoryCharacter {
  name:     string;
  aliases?: string[];
  pronouns?: { self?: string; other?: string };
  role?:    string;
  notes?:   string;
  locked?:  boolean;
  pending?: boolean;
}

export interface ApiMemoryGlossaryTerm {
  source_term: string;
  target_term: string;
  notes?:      string;
  locked?:     boolean;
  pending?:    boolean;
}

export interface ApiMemoryStyleRef {
  kind:   "translation" | "chapter";
  id:     number;
  label:  string;
  weight: number;
}

export interface ApiTranslatorMemory {
  material_id:     number;
  source_lang:     string;
  target_lang:     string;
  characters:      ApiMemoryCharacter[];
  world:           Record<string, unknown>;
  style:           Record<string, unknown>;
  glossary:        ApiMemoryGlossaryTerm[];
  style_refs:      ApiMemoryStyleRef[];
  last_chapter_id: number | null;
  updated_at:      string | null;
}

export interface ApiMemoryBrief {
  chapter_id:  number;
  position:    number;
  number:      string;
  label:       string | null;
  summary:     string | null;
  brief_json:  Record<string, unknown>;
  created_at:  string | null;
  updated_at:  string | null;
}

// ── Glossary ────────────────────────────────────────────────────────

export interface ApiGlossaryTerm {
  id:          number;
  source_lang: string;
  target_lang: string;
  source_term: string;
  target_term: string;
  notes:       string | null;
}

// ── Me ──────────────────────────────────────────────────────────────

export interface ApiTokenInfo {
  id:         number;
  name:       string;
  prefix:     string;
  last_used:  string | null;
  created_at: string | null;
}

export interface ApiTokenCreated extends ApiTokenInfo {
  token: string;
}

export interface ApiQuota {
  is_admin:       boolean;
  limit_hour:     number;
  used_hour:      number;
  remaining_hour: number;
  limit_day:      number;
  used_day:       number;
  remaining_day:  number;
}

export interface ApiRecentRead {
  work_id:         number;
  material_id:     number;
  title:           string;
  cover:           string | null;
  work_chapter_id: number;
  chapter_number:  string;
  chapter_label:   string | null;
  translation_id:  number | null;
  last_read_at:    string | null;
}

// ── Admin Ops ───────────────────────────────────────────────────────

export interface ApiPausedStage {
  stage:     PipelineStage;
  reason:    string;
  paused_at: string;
  paused_by: string | null;
}

export interface ApiTask {
  id:             number;
  stage:          PipelineStage;
  target_kind:    "chapter" | "draft" | "translation";
  target_id:      number;
  state:          "pending" | "running" | "stale" | "blocked" | "failed";
  attempts:       number;
  claimed_by:     string | null;
  claimed_at:     string | null;
  last_error:     string | null;
  work_id:        number | null;
  chapter_label:  string | null;
  source_lang:    string | null;
  target_lang:    string | null;
  owner_id:       number | null;
}

export interface ApiQueueStats {
  stages: Record<string, {
    pending: number;
    running: number;
    stale:   number;
    blocked: number;
    failed:  number;
  }>;
  active_workers: string[];
  paused_stages:  string[];
}

// ── Cursor Pagination ───────────────────────────────────────────────

export interface CursorResult<T> {
  items:       T[];
  next_cursor: string | null;
}

export interface CursorParams {
  cursor?: string | null;
  limit?:  number;
}
