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

export interface ApiChapter {
  chapter_id: number
  project_id: number
  idx:        number
  title:      string | null
  state:      'idle' | 'pending' | 'running' | 'error' | 'done'
  stage:      string
  page_count: number
  error:      string
  updated_at: string | null
  progress: {
    stage:      string
    page_index: number
    page_total: number
  } | null
}

async function fetchJson<T>(path: string): Promise<T> {
  const res = await fetch(`${API_BASE}/api${path}`)
  if (!res.ok) throw new Error(`${res.status} ${res.statusText}`)
  return res.json()
}

export const api = {
  base:         API_BASE,
  listProjects: () => fetchJson<ApiProject[]>('/projects'),
  listChapters: (projectId: number) =>
    fetchJson<ApiChapter[]>(`/projects/${projectId}/chapters`),
  getProject: (projectId: number) =>
    fetchJson<ApiProject>(`/projects/${projectId}`),
}
