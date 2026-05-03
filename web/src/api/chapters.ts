import { client } from './client'
import type { ChapterOut, ChapterParams } from './types'

export const chapterKeys = {
  all:    (projectId: number)                  => ['chapters', projectId] as const,
  detail: (projectId: number, id: number)      => ['chapters', projectId, id] as const,
  pages:  (projectId: number, id: number)      => ['chapters', projectId, id, 'pages'] as const,
}

export const chaptersApi = {
  list: (projectId: number, _params?: ChapterParams) =>
    client.get<ChapterOut[]>(`/api/projects/${projectId}/chapters`),

  get: (projectId: number, id: number) =>
    client.get<ChapterOut>(`/api/projects/${projectId}/chapters/${id}`),

  redo: (projectId: number, id: number) =>
    client.post<ChapterOut>(`/api/projects/${projectId}/chapters/${id}/redo`),

  pageUrl: (projectId: number, id: number, index: number) =>
    `/api/projects/${projectId}/chapters/${id}/pages/${index}`,
}
