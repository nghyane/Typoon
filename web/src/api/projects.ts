import { client } from './client'
import type { ImportBody, ProjectOut } from './types'

export const projectKeys = {
  all:    ()                => ['projects'] as const,
  detail: (id: number)     => ['projects', id] as const,
}

export const projectsApi = {
  list:   ()               => client.get<ProjectOut[]>('/api/projects'),
  get:    (id: number)     => client.get<ProjectOut>(`/api/projects/${id}`),
  import: (body: ImportBody) => client.post<ProjectOut>('/api/projects', body),
  delete: (id: number)     => client.delete<void>(`/api/projects/${id}`),
}
