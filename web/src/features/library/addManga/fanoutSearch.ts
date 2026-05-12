// Cross-source search fanout.
//
// Runs the same query against every enabled source that exposes
// search and merges results into a single list. Each row carries
// its source so the modal can render the `[HappyMH]` chip and pass
// the right manifest into `api.importMaterial`.
//
// Each source query is independent; one slow/broken source must not
// block the rest. We surface partial results progressively via
// React Query — the modal renders whatever has landed so far.

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
  hits:        SearchHit[]
  loading:     boolean
  /** Sources that errored out. Surfaced for hint text — never blocks
   *  the UI even when every source fails. */
  failures:    { sourceId: string; error: Error }[]
  /** All sources that participated (enabled + searchable). The modal
   *  uses this to render "Đang tìm trên N nguồn…" hints. */
  total:       number
}

const SEARCH_STALE = 5 * 60_000

export function useFanoutSearch(
  q:       string,
  sources: InstalledSource[],
): SearchResult {
  const searchable = sources.filter(
    (s) => s.enabled && hasSearch(s.manifest),
  )

  // Empty query short-circuits — every source returns []. We still
  // honour the hook contract so callers don't need to branch.
  const enabled = q.trim().length >= 2

  const queries = useQueries({
    queries: searchable.map((s) => ({
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
    const q = queries[i]!
    const s = searchable[i]!
    if (q.isPending && enabled) loading = true
    if (q.error) failures.push({ sourceId: s.manifest.id, error: q.error as Error })
    if (q.data) {
      for (const m of q.data) hits.push({ source: s, manga: m })
    }
  }

  return { hits, loading, failures, total: searchable.length }
}
