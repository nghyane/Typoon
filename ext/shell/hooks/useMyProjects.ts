import { useMemo } from 'react'
import { useQuery } from '@tanstack/react-query'
import { TypoonClient, type ApiMeProject } from '@core/typoon'
import { API_URL } from '@core/config'
import { useConfig } from './useConfig'

/** All projects the user has access to. Cached 60s; client-side filtered
 *  by `query` until the project count justifies a server search endpoint. */
export function useMyProjects(query = '') {
  const { config, authed } = useConfig()
  const q = useQuery<ApiMeProject[]>({
    queryKey: ['me', 'projects', config.token],
    queryFn: () => new TypoonClient({
      apiUrl: API_URL, token: config.token,
    }).myProjects(),
    enabled: authed,
  })

  const filtered = useMemo<ApiMeProject[]>(() => {
    const list = q.data ?? []
    const needle = query.trim().toLowerCase()
    if (!needle) return list
    return list.filter(p => p.title.toLowerCase().includes(needle))
  }, [q.data, query])

  return { ...q, projects: filtered, all: q.data ?? [] }
}
