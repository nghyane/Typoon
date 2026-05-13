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
  /** Sibling materials, oldest-first. The SPA picks one as the
   *  "active source" (via `?src=` URL state) and fetches its
   *  manifest chapter list live; the others render as switch
   *  targets without manifest fetches. */
  materials:    ApiMaterial[]
  /** Work chapters the community has touched (spawn / upload / raw
   *  history). Empty when no one has translated or read anything
   *  yet — the SPA still renders the manifest's full chapter list,
   *  just without any translation overlay. */
  chapters:     ApiWorkChapter[]
  viewer_entry: ApiWorkViewerEntry | null
}


// ── Cross-source link voting ──────────────────────────────────────


export interface ApiLinkSuggestion {
  candidate_material_id: number
  candidate_title:       string
  candidate_source:      string | null
  candidate_cover:       string | null
  candidate_work_id:     number
  own_material_id:       number
  score:                 number
  total_votes:           number
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
  title:                string
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
  material_title:   string
  material_cover:   string | null
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
  material_title:  string
  material_cover:  string | null
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

export interface ApiMe {
  id:           number
  display_name: string
  avatar_url:   string | null
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
  const res = await safeFetch(`${API_BASE}/api${path}`, {
    ...init,
    headers: {
      ...(init?.body ? { 'Content-Type': 'application/json' } : {}),
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
  if (!res.ok) {
    const text = await res.text().catch(() => '')
    throw new Error(
      `${res.status} ${res.statusText}${text ? ` — ${text.slice(0, 200)}` : ''}`,
    )
  }
  return res.status === 204 ? (undefined as T) : res.json()
}

const json = (body: unknown) => JSON.stringify(body)

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
  // overlay + viewer library state in one round-trip. The active
  // source (`?src=`) is a URL-only concept; the server is identity-
  // only.
  getWork: (id: number) =>
    request<ApiWorkDetail>(`/work/${id}`),

  patchMaterial: (id: number, body: Partial<{
    title: string; cover_url: string | null;
    description: string | null; nsfw: boolean;
  }>) =>
    request<ApiMaterial>(`/material/${id}`, {
      method: 'PATCH', body: json(body),
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
      tmp_id: string; upload_id: string;
      parts: { number: number; url: string }[];
      part_size: number; expires_in: number;
    }>(
      `/material/${materialId}/chapter/upload-init`,
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
    title?:       string
    cover_url?:   string | null
    status?:      LibraryStatus
  }) =>
    request<ApiLibraryEntry>('/library/entry', {
      method: 'POST', body: json(body),
    }),

  patchLibraryEntry: (id: number, body: Partial<{
    title:       string
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

  // ── Me / tokens / quota ─────────────────────────────────────────
  me:        () => request<ApiMe>('/me'),
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
