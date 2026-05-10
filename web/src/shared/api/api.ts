// In Discord Activity, Discord proxy handles relative /api/* paths via URL Mappings.
// Outside DA, VITE_API_URL is set for cross-origin deploys (CF Pages → API).
const API_BASE = window.location.hostname.endsWith('.discordsays.com')
  ? ''
  : (import.meta.env.VITE_API_URL ?? '')

import type {
  UploadInitBody, UploadInitOut,
  UploadFinalizeBody, UploadAbortBody,
} from '@typoon/upload-sdk'

const TOKEN_KEY = 'typoon_token'

// 401 from any request → kick the user back to /login. We can't import
// from a TanStack Router file at module load, so use a window event the
// AppLayout listens for.
function onUnauthorized() {
  localStorage.removeItem(TOKEN_KEY)
  window.dispatchEvent(new CustomEvent('typoon:unauthorized'))
}

// `fetch` rejects with TypeError when the network layer fails — server
// down/restarting, DNS, CORS preflight, dropped connection. Surface it
// as a single recognizable message so callers (toasts, error UIs) can
// show "đang bảo trì" instead of leaking a raw browser string.
export class BackendUnavailableError extends Error {
  constructor() { super('Máy chủ tạm thời không phản hồi. Có thể đang bảo trì hoặc khởi động lại.') }
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

// ── Types ────────────────────────────────────────────────────────────────────

export interface ApiProject {
  project_id:   number
  slug:         string
  title:        string
  description:  string | null
  cover_url:    string | null
  source_lang:  string
  target_lang:  string
  source_url:   string | null
  owner_id:     number | null
  shared:       boolean
  is_owner:     boolean
  is_pinned:    boolean
  created_at:   string | null
  updated_at:   string | null
}

export type ProjectFilter = 'all' | 'mine' | 'pinned' | 'community'

export type ChapterState = 'idle' | 'pending' | 'running' | 'error' | 'done'

export interface ApiChapter {
  chapter_id: number
  project_id: number
  /** Display chapter number, free-form: "4", "4.5", "Extra". */
  number:     string
  /** Server-managed sort key. UI uses it to order rows; never to address a chapter. */
  position:   number
  title:      string | null
  state:      ChapterState
  stage:      string  // '' | 'scan' | 'translate' | 'render'
  page_count: number
  error:      string
  updated_at: string | null
  progress: {
    stage:      string
    page_index: number
    page_total: number
  } | null
  // Public URL for the rendered .bnl archive. Null until render done.
  // Browser fetches with Range requests; CDN edge handles caching.
  archive_url: string | null
}

export interface ApiBubble {
  page_index:      number
  bubble_idx:      number
  source_text:     string
  translated_text: string | null
  kind:            'dialogue' | 'sfx' | 'skip' | null
  confidence:      number
}

export interface ApiGlossaryTerm {
  id:          number
  source_term: string
  target_term: string
  notes:       string | null
}

export interface ApiQueueStats {
  stages: Record<string, { pending: number; running: number; stale: number }>
  active_workers: string[]
}

export interface ApiSearchHit {
  kind: 'bubble' | 'translation' | 'brief' | 'glossary'
  text: string
  chapter_number: string | null
  page_index:     number | null
}

export interface ApiSettings {
  project_id:  number
  target_lang: string
  title:       string
  description: string | null
  shared:      boolean
  is_owner:    boolean
  settings:    Record<string, unknown>
}

// ── Auth/me ──────────────────────────────────────────────────────────────────

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

export interface ApiMeProject {
  project_id:  number
  slug:        string
  title:       string
  cover_url:   string | null
  source_lang: string
  target_lang: string
  shared:      boolean
}

export interface ApiQuota {
  is_admin:             boolean
  limit_hour:           number
  used_hour:            number
  remaining_hour:       number
  limit_day:            number
  used_day:             number
  remaining_day:        number
  limit_concurrent:     number
  in_flight:            number
  remaining_concurrent: number
}

// ── Transport ────────────────────────────────────────────────────────────────

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
  // 502/503/504 = reverse proxy (nginx/Tailscale) is up but BE is down or
  // booting. /api/healthz returns 503 when DB is unreachable. Same UX as a
  // dropped connection: tell the user the server is temporarily down.
  if (res.status === 502 || res.status === 503 || res.status === 504) {
    throw new BackendUnavailableError()
  }
  if (!res.ok) {
    const text = await res.text().catch(() => '')
    throw new Error(`${res.status} ${res.statusText}${text ? ` — ${text.slice(0, 200)}` : ''}`)
  }
  return res.status === 204 ? (undefined as T) : res.json()
}

async function postForm<T>(path: string, fd: FormData): Promise<T> {
  // Browsers must set Content-Type with the multipart boundary themselves;
  // request() injects application/json which would corrupt the body.
  const res = await safeFetch(`${API_BASE}/api${path}`, {
    method: 'POST',
    body: fd,
    headers: authHeaders(),
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
    throw new Error(`${res.status} ${res.statusText}${text ? ` — ${text.slice(0, 200)}` : ''}`)
  }
  return res.status === 204 ? (undefined as T) : res.json()
}

const json = (body: unknown) => JSON.stringify(body)

// ── API ──────────────────────────────────────────────────────────────────────

export const api = {
  base: API_BASE,

  // Projects
  listProjects:   (filter: ProjectFilter = 'all') =>
    request<ApiProject[]>(`/projects?filter=${filter}`),
  getProject:     (id: number) => request<ApiProject>(`/projects/${id}`),
  deleteProject:  (id: number) => request<void>(`/projects/${id}`, { method: 'DELETE' }),
  createProject:  (body: { title: string; description?: string; source_lang: string; target_lang: string }) =>
    request<ApiProject>('/projects', { method: 'POST', body: json(body) }),
  uploadCover:    (id: number, file: File) => {
    const fd = new FormData()
    fd.append('file', file)
    return postForm<ApiProject>(`/projects/${id}/cover`, fd)
  },

  // Pin (bookmark)
  pinProject:   (id: number) => request<void>(`/projects/${id}/pin`, { method: 'POST' }),
  unpinProject: (id: number) => request<void>(`/projects/${id}/pin`, { method: 'DELETE' }),

  // Chapters
  listChapters:  (pid: number)              => request<ApiChapter[]>(`/projects/${pid}/chapters`),
  getChapter:    (pid: number, cid: number) => request<ApiChapter>(`/projects/${pid}/chapters/${cid}`),
  redoChapter:   (pid: number, cid: number) =>
    request<ApiChapter>(`/projects/${pid}/chapters/${cid}/redo`, { method: 'POST' }),
  deleteChapter: (pid: number, cid: number) =>
    request<void>(`/projects/${pid}/chapters/${cid}`, { method: 'DELETE' }),

  // Chapter upload — multipart-only.
  //
  // Web SPA does not call this directly: it goes through the shared
  // `@typoon/upload-sdk` driver (`uploadChapterZip`) which packs the
  // user's image set into a zip, splits into 8 MiB parts, PUTs each
  // part with a presigned URL, then calls `uploadFinalize`. The three
  // endpoints below are the engine-side handshake.
  uploadInit: (pid: number, body: UploadInitBody) =>
    request<UploadInitOut>(
      `/projects/${pid}/chapters/upload-init`,
      { method: 'POST', body: json(body) },
    ),
  uploadFinalize: (pid: number, body: UploadFinalizeBody) =>
    request<ApiChapter>(
      `/projects/${pid}/chapters/upload-finalize`,
      { method: 'POST', body: json(body) },
    ),
  uploadAbort: (pid: number, body: UploadAbortBody) =>
    request<void>(
      `/projects/${pid}/chapters/upload-abort`,
      { method: 'POST', body: json(body) },
    ),

  // Manually trigger scan on an idle chapter. Use redoChapter() instead
  // for chapters that already finished or errored.
  startChapter: (pid: number, cid: number) =>
    request<ApiChapter>(`/projects/${pid}/chapters/${cid}/start`, { method: 'POST' }),

  // Batch trigger — returns { started, total }; `started` may be less
  // than `total` because non-idle chapters in the selection are skipped.
  startChapters: (pid: number, chapterIds: number[]) =>
    request<{ started: number; total: number }>(
      `/projects/${pid}/chapters/start`,
      { method: 'POST', body: json({ chapter_ids: chapterIds }) },
    ),

  // Bubbles
  listBubbles: (pid: number, cid: number) =>
    request<ApiBubble[]>(`/projects/${pid}/chapters/${cid}/bubbles`),
  patchBubble: (
    pid: number, cid: number, page: number, bubble: number,
    body: { translated_text: string; kind?: ApiBubble['kind'] },
  ) =>
    request<ApiBubble>(
      `/projects/${pid}/chapters/${cid}/bubbles/${page}/${bubble}`,
      { method: 'PATCH', body: json(body) },
    ),

  // Brief
  getBrief: (pid: number, cid: number) =>
    request<Record<string, unknown>>(`/projects/${pid}/chapters/${cid}/brief`),

  // Glossary
  listGlossary:   (pid: number) =>
    request<ApiGlossaryTerm[]>(`/projects/${pid}/glossary`),
  createTerm:     (pid: number, body: { source_term: string; target_term: string; notes?: string | null }) =>
    request<ApiGlossaryTerm>(`/projects/${pid}/glossary`, { method: 'POST', body: json(body) }),
  updateTerm:     (pid: number, tid: number, body: { source_term: string; target_term: string; notes?: string | null }) =>
    request<ApiGlossaryTerm>(`/projects/${pid}/glossary/${tid}`, { method: 'PATCH', body: json(body) }),
  deleteTerm:     (pid: number, tid: number) =>
    request<void>(`/projects/${pid}/glossary/${tid}`, { method: 'DELETE' }),

  // Settings
  getSettings: (pid: number) => request<ApiSettings>(`/projects/${pid}/settings`),
  patchSettings: (pid: number, body: Partial<{
    target_lang: string; title: string; description: string;
    settings: Record<string, unknown>; shared: boolean;
  }>) =>
    request<ApiSettings>(`/projects/${pid}/settings`, { method: 'PATCH', body: json(body) }),

  // Workers
  workers: () => request<ApiQueueStats>('/workers'),

  // Search
  search: (params: { q: string; project_id: number; scope?: string; limit?: number }) => {
    const qs = new URLSearchParams({
      q:          params.q,
      project_id: String(params.project_id),
      ...(params.scope ? { scope: params.scope } : {}),
      ...(params.limit ? { limit: String(params.limit) } : {}),
    })
    return request<{ hits: ApiSearchHit[] }>(`/search?${qs.toString()}`)
  },

  // Asset URLs
  pageUrl: (pid: number, cid: number, page: number) =>
    `${API_BASE}/api/projects/${pid}/chapters/${cid}/pages/${page}`,

  // ── Me / API tokens ───────────────────────────────────────────────────────
  myProjects: () => request<ApiMeProject[]>('/me/projects'),

  listTokens:  () => request<ApiTokenInfo[]>('/me/tokens'),
  createToken: (name: string) =>
    request<ApiTokenCreated>('/me/tokens', { method: 'POST', body: json({ name }) }),
  revokeToken: (id: number) =>
    request<void>(`/me/tokens/${id}`, { method: 'DELETE' }),

  getQuota: () => request<ApiQuota>('/me/quota'),
}
