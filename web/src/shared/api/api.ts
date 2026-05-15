// Web SPA API client — material/translation surface (RFC v5).
//
// One public origin in production: the DA host. The Discord URL
// Mappings front /api, /r2, /cdn, /t for everyone — DA iframe and
// plain web alike. Inside the iframe we use same-origin paths
// (`window.location.hostname` ends in `.discordsays.com`); outside we
// hit `VITE_PUBLIC_BASE_URL` cross-origin (CORS allows it).
const API_BASE = window.location.hostname.endsWith('.discordsays.com')
  ? ''
  : (import.meta.env.VITE_PUBLIC_BASE_URL ?? '')

const TOKEN_KEY = 'typoon_token'

// 401 from any request → kick the user back to /login.
function onUnauthorized() {
  localStorage.removeItem(TOKEN_KEY)
  window.dispatchEvent(new CustomEvent('typoon:unauthorized'))
}

export class BackendUnavailableError extends Error {
  constructor() {
    super('Máy chủ tạm thời không phản hồi. Có thể đang bảo trì hoặc khởi động lại.')
  }
}


/** Thrown when a `GET /work/{id}` (or any endpoint that resolves a
 *  Work through `require_work`) hits a Work id that was merged away
 *  by community vote. Carries the canonical id so the route layer
 *  can replace the URL in-place. */
export class WorkRedirectedError extends Error {
  readonly oldId: number
  readonly newId: number
  constructor(oldId: number, newId: number) {
    super(`Work ${oldId} merged into ${newId}`)
    this.name = 'WorkRedirectedError'
    this.oldId = oldId
    this.newId = newId
  }
}

async function safeFetch(input: string, init?: RequestInit): Promise<Response> {
  try {
    return await fetch(input, init)
  } catch {
    throw new BackendUnavailableError()
  }
}

function authHeaders(): Record<string, string> {
  const t = localStorage.getItem(TOKEN_KEY)
  return t ? { Authorization: `Bearer ${t}` } : {}
}

// ── Types ────────────────────────────────────────────────────────────

export type MaterialOrigin = 'source' | 'extension' | 'upload'
export type DraftState     = 'pending' | 'running' | 'done' | 'error' | 'blocked'
export type LinkOrigin     = 'auto' | 'manual'

export interface ApiMaterial {
  id:            number
  origin:        MaterialOrigin
  work_id:       number                 // global Work identity (cross-source)
  source:        string | null         // NULL for ext / upload
  upstream_ref:  string | null

  title:         string
  cover_url:     string | null
  description:   string | null
  author:        string | null
  status:        string | null
  languages:     string[]

  title_native:  string | null
  title_alt:     string[]
  cross_refs:    Record<string, unknown> | null
  /** BCP-47 → display title. Enriched from MangaDex altTitles +
   *  Anilist title.{english,romaji,native}. The title resolver
   *  picks `title_locale[targetLang]` first when present. */
  title_locale:  Record<string, string> | null
  /** First publication year. From Anilist `startDate.year` or
   *  MangaDex `year`. Null when no enrichment ever populated it. */
  start_year:    number | null

  nsfw:          boolean

  imported_by:   number | null
  created_at:    string | null
  updated_at:    string | null
}

export interface ApiChapterTranslation {
  /** translation_id */
  id:           number
  target_lang:  string
  creator_id:   number | null
  creator_name: string | null
  state:        DraftState
  /** True if the translation reuses the draft's default render (no edits). */
  from_cache:   boolean
}

export interface ApiChapter {
  id:            number
  material_id:   number
  position:      number
  /** Canonical chapter key — same value across sibling materials of
   *  the same Work. Aliased from `work_chapters.number_norm` on the
   *  server side. Use `label` for the source's free-form display
   *  string ("Chương 040", "第106话"). */
  number:        string
  label:         string | null
  upstream_url:  string | null
  page_count:    number
  updated_at:    string | null
  translations:  ApiChapterTranslation[]
}

export interface ApiWork {
  id:         number
  cross_refs: Record<string, unknown> | null
  created_at: string | null
  updated_at: string | null
}

export interface ApiWorkChapterTranslation {
  id:                  number
  target_lang:         string
  /** Source language of the draft (BCP-47), e.g. "en" / "ja" / "ko".
   *  null when the draft predates schema 24 or was created from an
   *  ext / upload material without a recorded source lang. UI uses
   *  this to render "@userA · từ Tiếng Anh MangaDex". */
  source_lang:         string | null
  owner_id:            number
  creator_name:        string | null
  state:               DraftState
  /** When `state === 'error' | 'blocked'`, the human-readable cause
   *  the worker stamped on the draft. UI surfaces this inline so the
   *  reader sees WHY a translation isn't progressing without having
   *  to ask an admin. Null while pending/running/done. */
  error_message:       string | null
  shared:              boolean
  draft_id:            number | null
  draft_chapter_id:    number | null
  /** Source material whose pixels back the draft render. The reader
   *  opens this material when the user clicks the translation. */
  draft_material_id:   number | null
  uses_default_render: boolean
  /** RFC 3339 UTC. When the translation row was last touched —
   *  drives the "3 ngày trước" hint on translation rows. */
  updated_at:          string | null
}

export interface ApiWorkChapter {
  id:           number
  number_norm:  string
  label:        string | null
  translations: ApiWorkChapterTranslation[]
}

export interface ApiWorkViewerEntry {
  entry_id:    number
  status:      LibraryStatus
  target_lang: string
}

export interface ApiWorkDetail {
  work:         ApiWork
  /** Sibling materials, oldest-first. The SPA auto-merges every
   *  installed source's manifest chapter list into a single chapter
   *  spine; no per-source URL state. */
  materials:    ApiMaterial[]
  /** Work chapters the community has touched (spawn / upload / raw
   *  history). Empty when no one has translated or read anything
   *  yet — the SPA still renders the manifest's full chapter list,
   *  just without any translation overlay. */
  chapters:     ApiWorkChapter[]
  viewer_entry: ApiWorkViewerEntry | null
  /** When non-null, the URL the client requested (`/w/<old>`) was
   *  for a Work that has since been merged into `work.id`. The
   *  payload reflects the CANONICAL Work; the route layer replaces
   *  the URL in-place so the user sees `/w/<canonical>` and any
   *  shared link keeps working. Null on the common path. */
  redirected_from: number | null
}


// ── Cross-source link voting ──────────────────────────────────────


export interface ApiLinkSuggestion {
  /** Discriminator. `voted` = community has cast at least one +1 on
   *  this pair; `ranked` = title-similarity ranker surfaced it with
   *  no vote yet. The UI shows them in one list but renders different
   *  context (vote score vs confidence + reason). */
  kind:                  'voted' | 'ranked'
  candidate_material_id: number
  candidate_title:       string
  candidate_source:      string | null
  candidate_cover:       string | null
  candidate_work_id:     number
  own_material_id:       number
  /** Voted-stream fields. 0 on ranked rows. */
  score:                 number
  total_votes:           number
  /** Ranked-stream fields. Null on voted rows. */
  confidence:            number | null
  reason:                'title_native_exact' | 'title_alt_overlap' | 'title_trgm' | null
  viewer_vote:           -1 | 1 | null
}


export interface ApiLinkVoteResult {
  vote:               -1 | 1
  score:              number
  merged:             boolean
  /** Set when `merged` is true. The SPA navigates to /w/$canonical
   *  so the user lands on the surviving Work id. May equal the
   *  request's workId (when this Work was already canonical) or
   *  differ (when this Work dissolved into a sibling). */
  canonical_work_id:  number | null
  /** 'same_work' | 'cross_refs_conflict' | null. */
  blocked_reason:     string | null
}


/** Outcome of POST /api/work/{id}/split-vote or /force-unlink.
 *  Same shape so the SPA's success path is unified. */
export interface ApiSplitVoteResult {
  vote:            -1 | 1
  score:           number
  split:           boolean
  /** Work id the material moved to when `split` is true. The SPA
   *  toasts with a "Mở →" link to navigate. */
  new_work_id:     number | null
  /** 'solo_member' | 'material_gone' | null. */
  blocked_reason:  string | null
}


/** One row in `/api/work/{id}/members` — a material attached to
 *  this Work, with the viewer's split-vote state and the
 *  owner-undo hint folded in. Drives the "Nguồn đang đọc" panel. */
export interface ApiWorkMember {
  material_id:                number
  title:                      string
  cover_url:                  string | null
  source:                     string | null
  languages:                  string[]
  title_native:               string | null
  title_locale:               Record<string, string> | null
  /** Viewer's split vote on this member. -1 / 1 / null. */
  viewer_split_vote:          number | null
  /** Aggregate split score so far (sum of ±1). */
  pending_split_score:        number
  /** Threshold the SPA uses for the "X/N" hint. Mirrors backend. */
  pending_split_threshold:    number
  /** ISO timestamp when the viewer's force-link undo window closes,
   *  or null when there's nothing to undo. The SPA renders a
   *  countdown until expiry, then hides the affordance. */
  force_link_undo_expires_at: string | null
}

export interface ApiBubbleEdit {
  page_index:  number
  bubble_idx:  number
  source_text: string
  draft_text:  string
  edited_text: string | null
  kind:        'dialogue' | 'sfx' | 'skip'
}

export interface ApiMyTranslation {
  translation_id:        number
  target_lang:           string
  state:                 DraftState
  has_archive:           boolean
  updated_at:            string | null
  chapter_id:            number
  chapter_number:        string
  chapter_label:         string | null
  chapter_position:      number
  chapter_upstream_url:  string | null
  material_id:           number
  material_title:        string
  material_cover:        string | null
  material_source:       string | null
  material_upstream_ref: string | null
}

export interface ApiTranslation {
  id:               number
  work_id:          number
  work_chapter_id:  number
  chapter_id:       number      // draft's pixel chapter (= material the reader opens)
  material_id:      number
  owner_id:         number
  target_lang:      string
  draft_id:         number | null
  state:            DraftState
  archive_url:      string | null
  has_edits:        boolean
  chapter_number:   string | null
  chapter_label:    string | null
  material_title:   string | null
  shared:           boolean
  created_at:       string | null
  updated_at:       string | null
}

export interface ApiLibraryMaterialLink {
  material_id:  number
  link_origin:  LinkOrigin
  linked_at:    string | null
}

export type LibraryStatus =
  | 'reading' | 'plan' | 'done' | 'dropped'

export interface ApiTranslationSummary {
  pending: number
  running: number
  done:    number
  error:   number
}

export interface ApiLibraryEntry {
  id:                   number
  /** Resolved server-side from the Work's materials against the
   *  viewer's reading lang. Same canonical title the Work hub
   *  renders; the entry row no longer caches this. */
  title:                string
  /** Likewise resolved. */
  cover_url:            string | null
  /** Canonical Work this entry bookmarks. SPA navigates `/w/${work_id}`
   *  to open the manga page; multiple materials of the same Work
   *  appear under `materials`. */
  work_id:              number
  /** User's reading-language preference for this Work. Drives the
   *  manifest fetch (e.g. MangaDex API needs `translatedLanguage[]=
   *  {lang}`) and the chapter list overlay badges. Editable inline
   *  via PATCH `/library/entry/{id}`. */
  target_lang:          string

  /** Reading state — what verb the user applies. `dropped` is hidden
   *  from the default list view. Schema 19 simplified to four
   *  statuses; reading history lives in its own table. */
  status:               LibraryStatus

  materials:            ApiLibraryMaterialLink[]
  /** Counts of this user's translations by draft state — drives the
   *  card's "Đang dịch / Lỗi" activity chip. */
  translation_summary:  ApiTranslationSummary

  created_at:           string | null
  updated_at:           string | null
}

export interface ApiCommunityFeedEntry {
  translation_id:   number
  chapter_id:       number
  chapter_number:   string
  chapter_label:    string | null
  work_id:          number
  material_id:      number
  /** Work-level title resolved server-side against the viewer's
   *  preferred reading lang. Same string the Work hub renders. */
  title:            string
  cover:            string | null
  target_lang:      string
  creator_id:       number | null
  creator_name:     string | null
  created_at:       string | null
  archive_url:      string | null
  /** Total translated chapters surfaced for this work in the
   *  community feed. Cards may render a "+N chương khác" hint when
   *  more than one chapter is available. */
  chapters_in_feed: number
}

export interface ApiRecentRead {
  work_id:         number
  material_id:     number
  /** Same viewer-lang-resolved title as the community feed. */
  title:           string
  cover:           string | null
  work_chapter_id: number
  chapter_number:  string
  chapter_label:   string | null
  translation_id:  number | null
  last_read_at:    string | null
}

export interface ApiGlossaryTerm {
  id:          number
  source_lang: string
  target_lang: string
  source_term: string
  target_term: string
  notes:       string | null
}

export interface ApiTokenInfo {
  id:         number
  name:       string
  prefix:     string
  last_used:  string | null
  created_at: string | null
}

export interface ApiTokenCreated extends ApiTokenInfo {
  /** Plaintext, returned only once at creation time. */
  token: string
}

export interface ApiQuota {
  is_admin:       boolean
  limit_hour:     number
  used_hour:      number
  remaining_hour: number
  limit_day:      number
  used_day:       number
  remaining_day:  number
}

export interface ApiQueueStats {
  stages: Record<string, {
    pending: number
    running: number
    stale:   number
    /** Tasks under a paused stage — waiting for admin, not workers. */
    blocked: number
    /** Dead-lettered tasks past `MAX_TASK_ATTEMPTS`. */
    failed:  number
  }>
  active_workers: string[]
  /** Stages currently in `stage_pause`. Drives the system banner. */
  paused_stages:  string[]
}

// ── Admin / ops ─────────────────────────────────────────────────────
//
// Backed by /api/admin/ops/*. RBAC = require_admin (Discord role
// snapshotted in the JWT at login). The dashboard uses these for
// pause/resume + dead-letter recovery + audit timeline.

export type PipelineStage    = 'prepare' | 'scan' | 'translate' | 'render'
export type TaskTargetKind   = 'chapter' | 'draft' | 'translation'
export type TaskState        = 'pending' | 'running' | 'stale' | 'blocked' | 'failed'
export type AdminActionKind  =
  | 'stage.pause' | 'stage.resume'
  | 'task.requeue' | 'task.release' | 'task.force_fail'
  | 'draft.restart' | 'draft.takedown'

export interface ApiPausedStage {
  stage:     PipelineStage
  reason:    string
  paused_at: string
  paused_by: string | null
}

export interface ApiTask {
  stage:             PipelineStage
  target_kind:       TaskTargetKind
  target_id:         number
  attempts:          number
  claimed_by:        string | null
  claimed_at:        string | null
  last_error:        string | null
  lifecycle_state:   TaskState
  /** Wall-clock seconds since `claimed_at`. null when unclaimed. */
  claim_age_seconds: number | null

  // Joined context from chapter / draft / translation. Lets the
  // dashboard show "which work, which language pair, which model"
  // for a failing task without a second round-trip. NULL when the
  // backing row was deleted between task enqueue and snapshot.
  work_id:           number | null
  work_chapter_id:   number | null
  chapter_id:        number | null
  chapter_label:     string | null
  source_lang:       string | null
  target_lang:       string | null
  llm_model:         string | null
  owner_id:          number | null
}

export interface ApiTaskList {
  items:       ApiTask[]
  next_cursor: string | null
}

export interface ApiAdminAction {
  id:         number
  at:         string
  actor_id:   number | null
  action:     AdminActionKind
  target_ref: {
    stage:        string
    target_kind?: string
    target_id?:   number
    source?:      string
    idem_key?:    string
  }
  reason:     string
  prev_state: Record<string, unknown> | null
}

/** Thrown when the server replies 409 to an ops mutation — the
 *  operator's snapshot of (attempts, claimed_by) no longer matches
 *  the live row, or the task vanished. The UI catches this and
 *  re-fetches the dashboard so the next click is on fresh data. */
export class OpsConflictError extends Error {
  constructor() { super('State has changed — refresh and retry') }
}

// ── Translator memory ───────────────────────────────────────────────

/** Free-form character card. The agent suggests; the user accepts +
 *  edits + locks. We keep the JSON shape loose (Record<string,unknown>)
 *  so the agent loop can extend fields without a migration — UI reads
 *  named keys (name/pronouns/role/notes) defensively. */
export interface ApiMemoryCharacter {
  name:        string
  aliases?:    string[]
  pronouns?:   { self?: string; other?: string }
  role?:       string
  notes?:      string
  locked?:     boolean
  pending?:    boolean      // agent-suggested, awaiting user accept
}

export interface ApiMemoryGlossaryTerm {
  source_term: string
  target_term: string
  notes?:      string
  locked?:     boolean
  pending?:    boolean
}

export interface ApiMemoryStyleRef {
  kind:   'translation' | 'chapter'
  id:     number
  label:  string
  weight: number
}

export interface ApiTranslatorMemory {
  material_id:     number
  source_lang:     string
  target_lang:     string
  characters:      ApiMemoryCharacter[]
  world:           Record<string, unknown>
  style:           Record<string, unknown>
  glossary:        ApiMemoryGlossaryTerm[]
  style_refs:      ApiMemoryStyleRef[]
  last_chapter_id: number | null
  updated_at:      string | null
}

export interface ApiMemoryBrief {
  chapter_id:  number
  position:    number
  number:      string
  label:       string | null
  summary:     string | null
  brief_json:  Record<string, unknown>
  created_at:  string | null
  updated_at:  string | null
}

// ── Translate spawn ──────────────────────────────────────────────────

export interface SpawnTranslateBody {
  /** Internal chapter row id. */
  chapter_id:  number
  target_lang: string
}

export interface SpawnTranslateResult {
  translation_id: number
  draft_id:       number
  state:          DraftState
  cache_hit:      boolean
  chapter_id:     number
}

// ── Transport ────────────────────────────────────────────────────────

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  // Default JSON body for plain bodies; FormData carries its own
  // boundary in `Content-Type` and MUST NOT be overridden — letting
  // fetch set it ensures the boundary stays consistent with the
  // serialised body. Anything else (string/Blob) is treated as JSON
  // per existing call sites.
  const isFormData = typeof FormData !== 'undefined'
                  && init?.body instanceof FormData
  const res = await safeFetch(`${API_BASE}/api${path}`, {
    ...init,
    headers: {
      ...(init?.body && !isFormData ? { 'Content-Type': 'application/json' } : {}),
      ...authHeaders(),
      ...(init?.headers ?? {}),
    },
  })
  if (res.status === 401) {
    onUnauthorized()
    throw new Error('401 Unauthorized')
  }
  if (res.status === 502 || res.status === 503 || res.status === 504) {
    throw new BackendUnavailableError()
  }
  if (res.status === 410) {
    // Structured 410 from `require_work`: the requested Work id has
    // been dissolved into another via community-vote merge. Surface
    // as a typed error so the route layer can redirect.
    const body = await res.json().catch(() => null) as
      | { detail?: { kind?: string; requested_id?: number; redirected_to?: number } }
      | null
    const d = body?.detail
    if (d?.kind === 'work_redirected'
        && typeof d.requested_id === 'number'
        && typeof d.redirected_to === 'number') {
      throw new WorkRedirectedError(d.requested_id, d.redirected_to)
    }
  }
  if (!res.ok) {
    const text = await res.text().catch(() => '')
    throw new Error(
      `${res.status} ${res.statusText}${text ? ` — ${text.slice(0, 200)}` : ''}`,
    )
  }
  return res.status === 204 ? (undefined as T) : res.json()
}

const json = (body: unknown) => JSON.stringify(body)

/** Mutation variant of `request` that recognises 409 from the ops
 *  endpoints and re-throws it as `OpsConflictError`. Also accepts an
 *  optional `Idempotency-Key`: the SPA generates one per click so a
 *  network retry produces one audit row, not two. Same 401/5xx
 *  semantics as `request`; everything else falls through. */
async function opsMutate(
  path: string,
  body: unknown,
  opts: { idemKey?: string } = {},
): Promise<void> {
  const res = await safeFetch(`${API_BASE}/api${path}`, {
    method:  'POST',
    headers: {
      'Content-Type':    'application/json',
      ...(opts.idemKey ? { 'Idempotency-Key': opts.idemKey } : {}),
      ...authHeaders(),
    },
    body: json(body),
  })
  if (res.status === 401) {
    onUnauthorized()
    throw new Error('401 Unauthorized')
  }
  if (res.status === 409) {
    throw new OpsConflictError()
  }
  if (res.status === 502 || res.status === 503 || res.status === 504) {
    throw new BackendUnavailableError()
  }
  if (!res.ok) {
    const text = await res.text().catch(() => '')
    throw new Error(
      `${res.status} ${res.statusText}${text ? ` — ${text.slice(0, 200)}` : ''}`,
    )
  }
}

// ── API ──────────────────────────────────────────────────────────────

export const api = {
  base: API_BASE,

  // ── Material ────────────────────────────────────────────────────
  importMaterial: (body: {
    source: string; upstream_ref: string;
    title: string; cover_url?: string | null;
    description?: string | null; author?: string | null;
    status?: string | null; languages?: string[];
    title_native?: string | null; title_alt?: string[];
    cross_refs?: Record<string, unknown> | null;
    nsfw?: boolean;
  }) =>
    request<ApiMaterial>('/material/import', {
      method: 'POST', body: json(body),
    }),

  createLocalMaterial: (body: {
    origin: 'extension' | 'upload'; title: string;
    cover_url?: string | null; description?: string | null;
    author?: string | null; nsfw?: boolean;
  }) =>
    request<ApiMaterial>('/material', { method: 'POST', body: json(body) }),

  // ── Work (canonical manga page) ───────────────────────────────
  // Drives /w/$workId — sibling materials + cross-source chapter
  // overlay + viewer library state in one round-trip. The SPA
  // auto-merges every installed source's manifest into a single
  // chapter list; no per-source URL state.
  getWork: (id: number) =>
    request<ApiWorkDetail>(`/work/${id}`),

  /** "Tạo trống" — create an empty Work the viewer can follow before
   *  any source plug-in has it indexed. No material is created here;
   *  the first chapter upload lazy-creates the upload material. */
  createBlankWork: (body: {
    title:        string
    cover_url?:   string | null
    target_lang?: string
  }) =>
    request<ApiWorkDetail>('/work', {
      method: 'POST', body: json(body),
    }),

  patchMaterial: (id: number, body: Partial<{
    title: string; cover_url: string | null;
    description: string | null; nsfw: boolean;
  }>) =>
    request<ApiMaterial>(`/material/${id}`, {
      method: 'PATCH', body: json(body),
    }),

  /** Upload a cover image for an ext / upload material. Server
   *  decodes, strips EXIF, stores via the public ArtifactStore, and
   *  writes the resulting URL into `materials.cover_url`. Returns
   *  the refreshed material so the caller can re-render without a
   *  separate refetch. Allowed MIMEs: image/jpeg, image/png,
   *  image/webp. Hard cap: 2 MiB (server enforces too). */
  uploadCover: (id: number, file: File) => {
    const fd = new FormData()
    fd.append('file', file)
    return request<ApiMaterial>(`/material/${id}/cover`, {
      method: 'POST', body: fd,
    })
  },

  /** Merge client-enriched metadata onto a material. The SPA fans
   *  search across link plugins (MangaBaka, MangaDex, …) and POSTs
   *  the discovered IDs + multilingual titles + start year + cover
   *  here. Idempotent; the server keeps existing values on conflict
   *  (additive merge — manifest data wins, enriched fields fill
   *  only the empty columns). */
  enrichMaterialMetadata: (id: number, body: {
    cross_refs?:    Record<string, string | number>
    title_native?:  string
    title_alt?:     string[]
    title_locale?:  Record<string, string>
    start_year?:    number
    description?:   string
    cover_url?:     string
    source_signals?: Array<{
      plugin:        string
      confidence:    number
      matched_title: string | null
    }>
  }) =>
    request<ApiMaterial>(`/material/${id}/enrich-metadata`, {
      method: 'POST', body: json(body),
    }),

  // ── Cross-source link voting ──────────────────────────────────
  // Community-driven `materials.work_id` merging. Each vote upserts
  // an entry in `material_link_votes`; once a (a, b) pair crosses
  // the server-side threshold (3 distinct users) AND the two Works
  // don't carry conflicting `cross_refs`, they merge inline.
  listWorkLinkSuggestions: (workId: number) =>
    request<ApiLinkSuggestion[]>(`/work/${workId}/link-suggestions`),

  castWorkLinkVote: (
    workId: number,
    body: { target_material_id: number; vote: 1 | -1; own_material_id?: number },
  ) =>
    request<ApiLinkVoteResult>(`/work/${workId}/link-vote`, {
      method: 'POST', body: json(body),
    }),

  proposeWorkLink: (
    workId: number,
    body: { target_material_id: number; own_material_id?: number },
  ) =>
    request<ApiLinkVoteResult>(`/work/${workId}/propose-link`, {
      method: 'POST', body: json(body),
    }),

  // Explicit manual-link: the viewer affirmatively picked the
  // candidate via search. Server-side merge fires immediately (no
  // community-vote threshold), but the cross_refs conflict check
  // still applies — `blocked_reason: 'cross_refs_conflict'` means
  // the two Works claim incompatible identities and a +1 vote was
  // recorded instead.
  forceWorkLink: (
    workId: number,
    body: { target_material_id: number; own_material_id?: number },
  ) =>
    request<ApiLinkVoteResult>(`/work/${workId}/force-link`, {
      method: 'POST', body: json(body),
    }),

  // ── Split / unlink (inverse of link-vote / force-link) ────────
  listWorkMembers: (workId: number) =>
    request<ApiWorkMember[]>(`/work/${workId}/members`),

  castWorkSplitVote: (
    workId: number,
    body: { material_id: number; vote: 1 | -1 },
  ) =>
    request<ApiSplitVoteResult>(`/work/${workId}/split-vote`, {
      method: 'POST', body: json(body),
    }),

  forceWorkUnlink: (
    workId: number,
    body: { material_id: number },
  ) =>
    request<ApiSplitVoteResult>(`/work/${workId}/force-unlink`, {
      method: 'POST', body: json(body),
    }),

  deleteMaterial: (id: number) =>
    request<void>(`/material/${id}`, { method: 'DELETE' }),

  // ── Chapter upload (ext / upload materials only) ────────────────
  //
  // Source-backed materials get their chapters from the manifest at
  // read time and never call these routes. The SDK
  // (`@typoon/upload-sdk`) drives the full handshake; web SPA + the
  // browser extension both implement `UploadHttpClient` against the
  // three methods below.
  uploadInit: (materialId: number, body: { byte_size: number }) =>
    request<{
      material_id: number;
      tmp_id: string; upload_id: string;
      parts: { number: number; url: string }[];
      part_size: number; expires_in: number;
    }>(
      `/material/${materialId}/chapter/upload-init`,
      { method: 'POST', body: json(body) },
    ),
  /** Per-Work convenience: server resolves (or creates) the
   *  viewer's upload-origin material lazily and returns the same
   *  init payload, including the resolved `material_id` so the SDK
   *  finalize call targets the right material. */
  workUploadInit: (workId: number, body: { byte_size: number }) =>
    request<{
      material_id: number;
      tmp_id: string; upload_id: string;
      parts: { number: number; url: string }[];
      part_size: number; expires_in: number;
    }>(
      `/work/${workId}/upload-init`,
      { method: 'POST', body: json(body) },
    ),
  uploadFinalize: (materialId: number, body: {
    tmp_id: string; upload_id: string;
    parts: { number: number; etag: string }[];
    label?: string;
    upstream_url?: string;
    number_norm?: string;
    source_lang?: string;
  }) =>
    request<ApiChapter>(
      `/material/${materialId}/chapter/upload-finalize`,
      { method: 'POST', body: json(body) },
    ),
  uploadAbort: (materialId: number, body: {
    tmp_id: string; upload_id: string;
  }) =>
    request<void>(
      `/material/${materialId}/chapter/upload-abort`,
      { method: 'POST', body: json(body) },
    ),

  // ── Translate ───────────────────────────────────────────────────
  spawnTranslate: (body: SpawnTranslateBody) =>
    request<SpawnTranslateResult>('/translate', {
      method: 'POST', body: json(body),
    }),

  listMyTranslations: () =>
    request<ApiMyTranslation[]>('/translate/mine'),

  getTranslation: (id: number) =>
    request<ApiTranslation>(`/translate/${id}`),

  listTranslationBubbles: (id: number) =>
    request<ApiBubbleEdit[]>(`/translate/${id}/bubbles`),

  patchTranslation: (id: number, body: Partial<Record<string, never>>) =>
    request<ApiTranslation>(`/translate/${id}`, {
      method: 'PATCH', body: json(body),
    }),

  redoTranslation: (id: number) =>
    request<SpawnTranslateResult>(`/translate/${id}/redo`, {
      method: 'POST',
    }),

  deleteTranslation: (id: number) =>
    request<void>(`/translate/${id}`, { method: 'DELETE' }),

  upsertEdit: (
    translationId: number,
    body: { page_index: number; bubble_idx: number; edited_text: string },
  ) =>
    request<void>(`/translate/${translationId}/edits`, {
      method: 'PUT', body: json(body),
    }),

  deleteEdit: (
    translationId: number, pageIndex: number, bubbleIdx: number,
  ) =>
    request<void>(
      `/translate/${translationId}/edits/${pageIndex}/${bubbleIdx}`,
      { method: 'DELETE' },
    ),

  // ── Library ─────────────────────────────────────────────────────
  listLibrary: (opts: { status?: LibraryStatus } = {}) => {
    const qs = new URLSearchParams()
    if (opts.status) qs.set('status', opts.status)
    const q = qs.toString()
    return request<ApiLibraryEntry[]>(`/library${q ? `?${q}` : ''}`)
  },

  getLibraryEntry: (id: number) =>
    request<ApiLibraryEntry>(`/library/entry/${id}`),

  createLibraryEntry: (body: {
    material_id:  number
    target_lang?: string
    status?:      LibraryStatus
  }) =>
    request<ApiLibraryEntry>('/library/entry', {
      method: 'POST', body: json(body),
    }),

  patchLibraryEntry: (id: number, body: Partial<{
    status:      LibraryStatus
    target_lang: string
  }>) =>
    request<ApiLibraryEntry>(`/library/entry/${id}`, {
      method: 'PATCH', body: json(body),
    }),

  deleteLibraryEntry: (id: number) =>
    request<void>(`/library/entry/${id}`, { method: 'DELETE' }),

  linkMaterial: (entryId: number, body: {
    material_id: number; link_origin?: LinkOrigin;
  }) =>
    request<void>(`/library/entry/${entryId}/link`, {
      method: 'POST', body: json(body),
    }),

  unlinkMaterial: (entryId: number, body: { material_id: number }) =>
    request<void>(`/library/entry/${entryId}/unlink`, {
      method: 'POST', body: json(body),
    }),

  // ── Community (cross-user recent translations) ─────────────────
  listCommunityRecent: (opts: { before?: string; limit?: number } = {}) => {
    const qs = new URLSearchParams()
    if (opts.before)        qs.set('before', opts.before)
    if (opts.limit != null) qs.set('limit',  String(opts.limit))
    const q = qs.toString()
    return request<ApiCommunityFeedEntry[]>(
      `/community/recent${q ? `?${q}` : ''}`,
    )
  },

  // ── Reading history ────────────────────────────────────────────
  listRecentReads: (limit?: number) => {
    const q = limit != null ? `?limit=${limit}` : ''
    return request<ApiRecentRead[]>(`/me/recent-reads${q}`)
  },

  // Translated reader: the translation already pins (Work chapter,
  // representative material) on the server side.
  recordTranslatedReading: (body: { translation_id: number }) =>
    request<void>('/me/reading/translated', { method: 'POST', body: json(body) }),

  // Raw reader: server materialises the Work chapter on demand from
  // (material, number_norm) so history dedupes per (user, Work chapter)
  // across sources. `number_norm` comes from the manifest runtime's
  // declarative normaliser.
  recordRawReading: (body: {
    material_id: number;
    number:      string;
    number_norm: string;
    label?:      string | null;
  }) =>
    request<void>('/me/reading/raw', { method: 'POST', body: json(body) }),

  // ── Glossary (per-user) ─────────────────────────────────────────
  listGlossary: (opts: {
    source_lang?: string; target_lang?: string;
  } = {}) => {
    const qs = new URLSearchParams()
    if (opts.source_lang) qs.set('source_lang', opts.source_lang)
    if (opts.target_lang) qs.set('target_lang', opts.target_lang)
    const q = qs.toString()
    return request<ApiGlossaryTerm[]>(`/glossary${q ? `?${q}` : ''}`)
  },
  upsertTerm: (body: {
    source_lang: string; target_lang: string;
    source_term: string; target_term: string;
    notes?: string | null;
  }) =>
    request<ApiGlossaryTerm>('/glossary', {
      method: 'POST', body: json(body),
    }),
  deleteTerm: (id: number) =>
    request<void>(`/glossary/${id}`, { method: 'DELETE' }),

  // ── Tokens / quota ──────────────────────────────────────────────
  //
  // Session payload (`/api/auth/me`) and the matching preferences
  // PATCH live in `@features/auth/session` — they share the
  // `['session']` React Query cache so every consumer reads from
  // one entry. Don't add a thin re-export here.
  listTokens: () => request<ApiTokenInfo[]>('/me/tokens'),
  createToken: (name: string) =>
    request<ApiTokenCreated>('/me/tokens', {
      method: 'POST', body: json({ name }),
    }),
  revokeToken: (id: number) =>
    request<void>(`/me/tokens/${id}`, { method: 'DELETE' }),
  getQuota:    () => request<ApiQuota>('/me/quota'),

  // ── Workers (admin-ish dashboard) ───────────────────────────────
  workers: () => request<ApiQueueStats>('/workers'),

  // ── Admin / ops ────────────────────────────────────────────────
  //
  // Every mutation accepts an optional `idemKey` (a UUID the UI
  // mints per click). Network retries with the same key produce
  // one audit row + one state change on the server, never two.
  // 409 from any mutation is thrown as `OpsConflictError` so the
  // UI can react with "trạng thái đã thay đổi, refresh" without
  // string-matching error messages.
  adminOps: {
    listStages: () =>
      request<ApiPausedStage[]>('/admin/ops/stages'),

    pauseStage: (
      stage:  PipelineStage,
      reason: string,
      opts?:  { idemKey?: string },
    ) =>
      opsMutate(
        `/admin/ops/stages/${stage}/pause`,
        { reason },
        opts,
      ),

    resumeStage: (
      stage:  PipelineStage,
      reason: string,
      opts?:  { idemKey?: string },
    ) =>
      opsMutate(
        `/admin/ops/stages/${stage}/resume`,
        { reason },
        opts,
      ),

    listTasks: (q: {
      stage?:       PipelineStage
      state?:       TaskState
      target_kind?: TaskTargetKind
      limit?:       number
      cursor?:      string
    } = {}) => {
      const usp = new URLSearchParams()
      if (q.stage)       usp.set('stage',       q.stage)
      if (q.state)       usp.set('state',       q.state)
      if (q.target_kind) usp.set('target_kind', q.target_kind)
      if (q.limit)       usp.set('limit',       String(q.limit))
      if (q.cursor)      usp.set('cursor',      q.cursor)
      const qs = usp.toString()
      return request<ApiTaskList>(`/admin/ops/tasks${qs ? `?${qs}` : ''}`)
    },

    requeueTask: (
      t: { stage: PipelineStage; target_kind: TaskTargetKind; target_id: number },
      body: { reason: string; expected_attempts: number; expected_claimed_by: string | null },
      opts?: { idemKey?: string },
    ) =>
      opsMutate(
        `/admin/ops/tasks/${t.stage}/${t.target_kind}/${t.target_id}/requeue`,
        body, opts,
      ),

    releaseTask: (
      t: { stage: PipelineStage; target_kind: TaskTargetKind; target_id: number },
      body: { reason: string; expected_claimed_by: string },
      opts?: { idemKey?: string },
    ) =>
      opsMutate(
        `/admin/ops/tasks/${t.stage}/${t.target_kind}/${t.target_id}/release`,
        body, opts,
      ),

    forceFailTask: (
      t: { stage: PipelineStage; target_kind: TaskTargetKind; target_id: number },
      body: { reason: string; expected_attempts: number; expected_claimed_by: string | null },
      opts?: { idemKey?: string },
    ) =>
      opsMutate(
        `/admin/ops/tasks/${t.stage}/${t.target_kind}/${t.target_id}/fail`,
        body, opts,
      ),

    /** Force-restart a draft regardless of state — admin equivalent
     *  of the user-facing redo, with the extra power of restarting
     *  `done` and `blocked` drafts (which user redo refuses). The
     *  draft's entry stage is server-derived; clients only pass
     *  reason + optional idempotency key. */
    restartDraft: (
      draftId: number,
      body:    { reason: string },
      opts?:   { idemKey?: string },
    ) =>
      opsMutate(`/admin/ops/drafts/${draftId}/restart`, body, opts),

    listActions: (q: {
      action?:      AdminActionKind
      actor_id?:    number
      stage?:       PipelineStage
      target_kind?: TaskTargetKind
      target_id?:   number
      limit?:       number
      before_id?:   number
    } = {}) => {
      const usp = new URLSearchParams()
      if (q.action)      usp.set('action',      q.action)
      if (q.actor_id)    usp.set('actor_id',    String(q.actor_id))
      if (q.stage)       usp.set('stage',       q.stage)
      if (q.target_kind) usp.set('target_kind', q.target_kind)
      if (q.target_id)   usp.set('target_id',   String(q.target_id))
      if (q.limit)       usp.set('limit',       String(q.limit))
      if (q.before_id)   usp.set('before_id',   String(q.before_id))
      const qs = usp.toString()
      return request<ApiAdminAction[]>(`/admin/ops/actions${qs ? `?${qs}` : ''}`)
    },
  },

  // ── Translator memory ───────────────────────────────────────────
  // Per (user, material, target_lang) knowledge bag. The agent loop
  // reads cards + sliding-window briefs on every spawn; the UI exposes
  // the same cards under /title/{entry_id}/memory so users can lock /
  // edit / reset without ever opening "settings".
  getMemory: (materialId: number, targetLang: string) =>
    request<ApiTranslatorMemory | null>(
      `/material/${materialId}/memory?target_lang=${encodeURIComponent(targetLang)}`,
    ),

  upsertMemory: (
    materialId: number,
    body: {
      target_lang:  string
      source_lang?: string                 // required on first write
      characters?:  unknown[] | null       // null = leave intact
      world?:       Record<string, unknown> | null
      style?:       Record<string, unknown> | null
      glossary?:    unknown[] | null
      style_refs?:  unknown[] | null
    },
  ) =>
    request<ApiTranslatorMemory>(`/material/${materialId}/memory`, {
      method: 'PUT', body: json(body),
    }),

  resetMemory: (materialId: number, targetLang: string) =>
    request<void>(
      `/material/${materialId}/memory?target_lang=${encodeURIComponent(targetLang)}`,
      { method: 'DELETE' },
    ),

  listMemoryBriefs: (
    materialId: number,
    opts: { target_lang: string; before_chapter_id?: number; limit?: number },
  ) => {
    const qs = new URLSearchParams({ target_lang: opts.target_lang })
    if (opts.before_chapter_id != null)
      qs.set('before_chapter_id', String(opts.before_chapter_id))
    if (opts.limit != null) qs.set('limit', String(opts.limit))
    return request<ApiMemoryBrief[]>(
      `/material/${materialId}/memory/briefs?${qs.toString()}`,
    )
  },

  // ── Reports ─────────────────────────────────────────────────────
  // User intake — any authenticated user can file. `kind` picks the
  // category (dmca / abuse / quality / other); admin queue handles
  // the rest. Reporter identity is taken from the auth session.
  submitReport: (body: {
    target_kind:    'material' | 'chapter' | 'draft' | 'translation'
    target_id:      number
    scope_guild_id?: string | null
    kind?:          'dmca' | 'abuse' | 'quality' | 'other'
    reason:         string
  }) =>
    request<{ report_id: number }>('/reports', {
      method: 'POST', body: json(body),
    }),
}
