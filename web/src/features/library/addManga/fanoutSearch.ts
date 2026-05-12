// Cross-source search fanout — runs the query against one or every
// enabled source in parallel.
//
// React Query keys per-source so adding/removing a target source
// doesn't invalidate the others. Each source query is independent;
// a slow or broken source never blocks the rest.

import { useQueries } from '@tanstack/react-query'
import {
  fetchBrowse, hasSearch,
} from '@features/browse/manifest/runtime'
import type {
  InstalledSource, MangaSummary,
} from '@features/browse/manifest/types'

export interface SearchHit {
  source:  InstalledSource
  manga:   MangaSummary
}

export interface SearchResult {
  hits:     SearchHit[]
  loading:  boolean
  failures: { sourceId: string; error: Error }[]
  /** Sources we actually queried (after the targetSourceId filter). */
  queried:  InstalledSource[]
}

const SEARCH_STALE = 5 * 60_000

export function useFanoutSearch(
  q:              string,
  sources:        InstalledSource[],
  targetSourceId: string | null = null,
): SearchResult {
  const candidates = sources.filter(
    (s) => s.enabled && hasSearch(s.manifest),
  )
  const queried = targetSourceId === null
    ? candidates
    : candidates.filter((s) => s.manifest.id === targetSourceId)

  const enabled = q.trim().length >= 2 && queried.length > 0

  const queries = useQueries({
    queries: queried.map((s) => ({
      queryKey:  ['search', s.manifest.id, q.trim()],
      queryFn:   async () =>
        await fetchBrowse(s.manifest, { search: true }, { q: q.trim() }),
      enabled,
      staleTime: SEARCH_STALE,
      retry:     false,
    })),
  })

  const hits: SearchHit[]                              = []
  const failures: { sourceId: string; error: Error }[] = []
  let loading = false

  for (let i = 0; i < queries.length; i++) {
    const result = queries[i]!
    const source = queried[i]!
    if (result.isPending && enabled) loading = true
    if (result.error) {
      failures.push({ sourceId: source.manifest.id, error: result.error as Error })
    }
    if (result.data) {
      for (const m of result.data) hits.push({ source, manga: m })
    }
  }

  return { hits, loading, failures, queried }
}
