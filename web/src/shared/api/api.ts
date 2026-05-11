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
export type PagesOrigin    = 'remote' | 'local'
export type DraftState     = 'pending' | 'running' | 'done' | 'error'
export type DraftVisibility = 'private' | 'guild' | 'all_guilds'
export type LinkOrigin     = 'primary' | 'auto' | 'manual'

export interface ApiMaterial {
  id:            number
  origin:        MaterialOrigin
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
  in_feed:      boolean
  /** True if the translation reuses the draft's default render (no edits). */
  from_cache:   boolean
}

export interface ApiChapter {
  id:            number
  material_id:   number
  position:      number
  number:        string
  label:         string | null
  upstream_url:  string | null
  pages_origin:  PagesOrigin
  page_count:    number
  updated_at:    string | null
  translations:  ApiChapterTranslation[]
}

export interface ApiMaterialDetail {
  material: ApiMaterial
  chapters: ApiChapter[]
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
  id:             number
  chapter_id:     number
  owner_id:       number
  target_lang:    string
  draft_id:       number | null
  state:          DraftState
  in_feed:        boolean
  feed_guild_id:  string | null
  archive_url:    string | null
  has_edits:      boolean
  created_at:     string | null
  updated_at:     string | null
}

export interface ApiLibraryMaterialLink {
  material_id:  number
  link_origin:  LinkOrigin
  linked_at:    string | null
}

export interface ApiLibraryEntry {
  id:                   number
  title:                string
  cover_url:            string | null
  bookmarked:           boolean
  bookmarked_at:        string | null
  primary_material_id:  number | null
  last_read_at:         string | null
  last_chapter_ref:     Record<string, unknown> | null
  materials:            ApiLibraryMaterialLink[]
  created_at:           string | null
  updated_at:           string | null
}

export type SuggestionSignal =
  | 'cross_refs' | 'vote_high' | 'title_native' | 'vote_low' | 'author'

export interface ApiLibrarySuggestion {
  entry_id:    number
  entry_title: string
  confidence:  'high' | 'medium' | 'low'
  signal:      SuggestionSignal
  score:       number | null
}

export interface ApiFeedEntry {
  translation_id:  number
  chapter_id:      number
  chapter_number:  string
  chapter_label:   string | null
  material_id:     number
  material_title:  string
  material_cover:  string | null
  target_lang:     string
  creator_id:      number | null
  creator_name:    string | null
  created_at:      string | null
  archive_url:     string | null
}

export interface ApiGlossaryTerm {
  id:          number
  source_lang: string
  target_lang: string
  source_term: string
  target_term: string
  notes:       string | null
}

export interface ApiGuild {
  id:        string
  name:      string | null
  icon_url:  string | null
}

export interface ApiMe {
  id:           number
  display_name: string
  avatar_url:   string | null
  guilds:       ApiGuild[]
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
  stages: Record<string, { pending: number; running: number; stale: number }>
  active_workers: string[]
}

// ── Translate spawn ──────────────────────────────────────────────────

export interface SpawnTranslateBody {
  /** Internal chapter row id. Use when the row already exists
   *  (ext / upload after finalize, redo). */
  chapter_id?:    number
  /** Manifest coords — backend materializes the row on first use.
   *  Use when spawning from a source-backed manga detail page where
   *  no local chapter row exists yet. */
  chapter_ref?:   {
    material_id:  number
    upstream_url: string
    number:       string
    label?:       string | null
  }
  target_lang:    string
  force_private?: boolean
  /** Visibility scope when not private. Defaults server-side to 'guild'. */
  visibility?:    DraftVisibility
  /** Required when visibility='guild'. */
  scope_guild_id?: string | null
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

  getMaterial: (id: number) =>
    request<ApiMaterialDetail>(`/material/${id}`),

  patchMaterial: (id: number, body: Partial<{
    title: string; cover_url: string | null;
    description: string | null; nsfw: boolean;
  }>) =>
    request<ApiMaterial>(`/material/${id}`, {
      method: 'PATCH', body: json(body),
    }),

  translationOverlay: (
    materialId: number, upstreamUrls: string[],
  ) =>
    request<Record<string, ApiChapterTranslation[]>>(
      `/material/${materialId}/translation-overlay`,
      { method: 'POST', body: json({ upstream_urls: upstreamUrls }) },
    ),

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
    number?: string; label?: string;
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

  patchTranslation: (id: number, body: Partial<{
    in_feed: boolean; feed_guild_id: string | null;
  }>) =>
    request<ApiTranslation>(`/translate/${id}`, {
      method: 'PATCH', body: json(body),
    }),

  redoTranslation: (id: number, body: { force_private?: boolean } = {}) =>
    request<SpawnTranslateResult>(`/translate/${id}/redo`, {
      method: 'POST', body: json(body),
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
  listLibrary: () =>
    request<ApiLibraryEntry[]>('/library'),

  getLibraryEntry: (id: number) =>
    request<ApiLibraryEntry>(`/library/entry/${id}`),

  createLibraryEntry: (body: {
    material_id: number; title?: string; cover_url?: string | null;
  }) =>
    request<ApiLibraryEntry>('/library/entry', {
      method: 'POST', body: json(body),
    }),

  patchLibraryEntry: (id: number, body: Partial<{
    title: string; bookmarked: boolean;
    last_read_at: string; last_chapter_ref: Record<string, unknown>;
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

  suggestLink: (materialId: number) =>
    request<ApiLibrarySuggestion | null>(
      `/library/suggest?material_id=${materialId}`,
    ),

  rejectSuggestion: (body: {
    material_id: number; candidate_material_id: number;
  }) =>
    request<void>('/library/suggest/reject', {
      method: 'POST', body: json(body),
    }),

  // ── Feed (Hội Mê Truyện, guild-scoped) ─────────────────────────
  listFeed: (
    guildId: string,
    opts: { before?: string; limit?: number } = {},
  ) => {
    const qs = new URLSearchParams()
    if (opts.before) qs.set('before', opts.before)
    if (opts.limit  != null) qs.set('limit',  String(opts.limit))
    const q = qs.toString()
    return request<ApiFeedEntry[]>(
      `/feed/guild/${encodeURIComponent(guildId)}${q ? `?${q}` : ''}`,
    )
  },

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

  // ── DMCA ────────────────────────────────────────────────────────
  reportDmca: (body: {
    target_kind: 'material' | 'chapter' | 'draft' | 'translation';
    target_id: number; reason: string; reporter: string;
    scope_guild_id?: string | null;
  }) =>
    request<{ takedown_id: number }>('/dmca/report', {
      method: 'POST', body: json(body),
    }),
}
