// VITE_API_URL is set when the web build runs cross-origin from the API
// (e.g. preview deploys). In dev/prod-same-origin it stays empty so all
// fetches go through the Vite proxy / nginx.
const API_BASE = import.meta.env.VITE_API_URL ?? ''

const TOKEN_KEY = 'typoon_token'

// 401 from any request → kick the user back to /login. We can't import
// from a TanStack Router file at module load, so use a window event the
// AppLayout listens for.
function onUnauthorized() {
  localStorage.removeItem(TOKEN_KEY)
  window.dispatchEvent(new CustomEvent('typoon:unauthorized'))
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
  idx:        number
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
  chapter_idx: number | null
  page_index:  number | null
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

// ── Transport ────────────────────────────────────────────────────────────────

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  const res = await fetch(`${API_BASE}/api${path}`, {
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
  if (!res.ok) {
    const text = await res.text().catch(() => '')
    throw new Error(`${res.status} ${res.statusText}${text ? ` — ${text.slice(0, 200)}` : ''}`)
  }
  return res.status === 204 ? (undefined as T) : res.json()
}

async function postForm<T>(path: string, fd: FormData): Promise<T> {
  // Browsers must set Content-Type with the multipart boundary themselves;
  // request() injects application/json which would corrupt the body.
  const res = await fetch(`${API_BASE}/api${path}`, {
    method: 'POST',
    body: fd,
    headers: authHeaders(),
  })
  if (res.status === 401) {
    onUnauthorized()
    throw new Error('401 Unauthorized')
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

  // Upload — single archive (PDF/CBZ/ZIP) or multiple image files
  uploadChapter: (
    pid: number,
    files: File[],
    opts: { idx?: number; title?: string } = {},
  ) => {
    const fd = new FormData()
    for (const f of files) fd.append('files', f)
    if (opts.idx   !== undefined) fd.append('idx',   String(opts.idx))
    if (opts.title)               fd.append('title', opts.title)
    return postForm<ApiChapter>(`/projects/${pid}/chapters/upload`, fd)
  },

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
  pageUrl: (pid: number, cid: number, idx: number) =>
    `${API_BASE}/api/projects/${pid}/chapters/${cid}/pages/${idx}`,

  // ── Me / API tokens (RFC-008) ─────────────────────────────────────────────
  myProjects: () => request<ApiMeProject[]>('/me/projects'),

  listTokens:  () => request<ApiTokenInfo[]>('/me/tokens'),
  createToken: (name: string) =>
    request<ApiTokenCreated>('/me/tokens', { method: 'POST', body: json({ name }) }),
  revokeToken: (id: number) =>
    request<void>(`/me/tokens/${id}`, { method: 'DELETE' }),
}
