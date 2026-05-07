// VITE_API_URL is set when the web build runs cross-origin from the API
// (e.g. preview deploys). In dev/prod-same-origin it stays empty so all
// fetches go through the Vite proxy / nginx.
const API_BASE = import.meta.env.VITE_API_URL ?? ''

export interface ApiProject {
  project_id:   number
  slug:         string
  title:        string
  description:  string | null
  cover_url:    string | null
  source_lang:  string
  target_lang:  string
  source_url:   string | null
  created_at:   string | null
  updated_at:   string | null
}

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

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  const res = await fetch(`${API_BASE}/api${path}`, init)
  if (!res.ok) {
    const text = await res.text().catch(() => '')
    throw new Error(`${res.status} ${res.statusText}${text ? ` — ${text.slice(0, 200)}` : ''}`)
  }
  return res.status === 204 ? (undefined as T) : res.json()
}

export const api = {
  base: API_BASE,

  // Queries ────────────────────────────────────────────────────────
  listProjects: () => request<ApiProject[]>('/projects'),
  getProject:   (id: number) => request<ApiProject>(`/projects/${id}`),
  listChapters: (id: number) => request<ApiChapter[]>(`/projects/${id}/chapters`),
  getChapter:   (pid: number, cid: number) =>
    request<ApiChapter>(`/projects/${pid}/chapters/${cid}`),

  // Mutations ──────────────────────────────────────────────────────
  redoChapter: (pid: number, cid: number) =>
    request<ApiChapter>(`/projects/${pid}/chapters/${cid}/redo`, { method: 'POST' }),
  deleteProject: (pid: number) =>
    request<void>(`/projects/${pid}`, { method: 'DELETE' }),

  // Asset URLs (no JSON wrapping) ──────────────────────────────────
  pageUrl: (pid: number, cid: number, idx: number) =>
    `${API_BASE}/api/projects/${pid}/chapters/${cid}/pages/${idx}`,
}
