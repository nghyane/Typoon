export interface ApiProject {
  project_id: number
  slug: string
  title: string
  source_lang: string
  target_lang: string
}

export interface ApiChapter {
  chapter_id: number
  project_id: number
  idx: number
  state: 'idle' | 'pending' | 'running' | 'error' | 'done'
  stage: string
  page_count: number
  error: string
  progress: {
    stage: string
    page_index: number
    page_total: number
  } | null
}

async function fetchJson<T>(path: string): Promise<T> {
  const res = await fetch(`/api${path}`)
  if (!res.ok) throw new Error(`${res.status} ${res.statusText}`)
  return res.json()
}

export const api = {
  listProjects: () => fetchJson<ApiProject[]>('/projects'),
  listChapters: (projectId: number) =>
    fetchJson<ApiChapter[]>(`/projects/${projectId}/chapters`),
  getProject: (projectId: number) =>
    fetchJson<ApiProject>(`/projects/${projectId}`),
}
