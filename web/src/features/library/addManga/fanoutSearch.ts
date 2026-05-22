// Cross-source search fanout with per-source fuzzy ranking.
//
// React Query keys per-source so adding/removing a target source
// doesn't invalidate the others. Each source query is independent;
// a slow or broken source never blocks the rest.
//
// Per-source ranking + cap:
//   • each source's raw hits get a fuzzy similarity score against the
//     query (token overlap + substring boost).
//   • we cap to `PER_SOURCE_LIMIT` after sorting — sources that return
//     50+ results don't drown other sources in the merged list.

import { useQueries, keepPreviousData } from '@tanstack/react-query'
import {
  fetchBrowse, hasSearch,
} from '@features/browse/manifest/runtime'
import { qk } from '@shared/api/keys'
import type {
  InstalledSource, MangaSummary,
} from '@features/browse/manifest/types'

export interface SearchHit {
  source:  InstalledSource
  manga:   MangaSummary
  /** Lazy match score 0..1. Used for stable per-source ordering. */
  score:   number
}

export interface SearchResult {
  hits:     SearchHit[]
  loading:  boolean
  failures: { sourceId: string; error: Error }[]
  /** Sources we actually queried (after the targetSourceId filter). */
  queried:  InstalledSource[]
}

const SEARCH_STALE     = 5 * 60_000
const PER_SOURCE_LIMIT = 8


/** Fuzzy match score in [0, 1]. Designed for short title strings,
 *  not full-text. Substring boost lets exact substring matches
 *  outrank loose token overlap. */
function fuzzyScore(query: string, title: string): number {
  const q = query.trim().toLowerCase()
  const t = title.trim().toLowerCase()
  if (!q || !t) return 0
  if (t === q)              return 1.0
  if (t.startsWith(q))      return 0.95
  if (t.includes(q))        return 0.85

  const qTokens = q.split(/\s+/).filter(Boolean)
  const tTokens = t.split(/\s+/).filter(Boolean)
  if (qTokens.length === 0) return 0

  let matched = 0
  for (const qt of qTokens) {
    if (tTokens.some(tt => tt.includes(qt))) matched++
  }
  const overlap = matched / qTokens.length

  // Character-level bigram overlap as a fallback so "narto" still
  // ranks "naruto" reasonably.
  const bigrams = (s: string) => {
    const out: string[] = []
    for (let i = 0; i < s.length - 1; i++) out.push(s.slice(i, i + 2))
    return out
  }
  const qbg = new Set(bigrams(q))
  const tbg = bigrams(t)
  if (qbg.size === 0 || tbg.length === 0) return overlap * 0.7
  let bigramHit = 0
  for (const b of tbg) if (qbg.has(b)) bigramHit++
  const bigramOverlap = bigramHit / Math.max(qbg.size, tbg.length)

  return Math.max(overlap * 0.7, bigramOverlap * 0.6)
}


function rankAndCap(query: string, source: InstalledSource, raws: MangaSummary[]): SearchHit[] {
  const scored = raws.map<SearchHit>(m => ({
    source,
    manga: m,
    score: fuzzyScore(query, m.title),
  }))
  scored.sort((a, b) => b.score - a.score)
  return scored.slice(0, PER_SOURCE_LIMIT)
}


export function useFanoutSearch(
  q:              string,
  sources:        InstalledSource[],
  targetSourceId: string | null = null,
): SearchResult {
  const candidates = sources.filter(
    s => s.enabled && hasSearch(s.manifest),
  )
  const queried = targetSourceId === null
    ? candidates
    : candidates.filter(s => s.manifest.id === targetSourceId)

  const enabled = q.trim().length >= 2 && queried.length > 0

  const queries = useQueries({
    queries: queried.map(s => ({
      queryKey:  qk.manifest.search(s.manifest.id, q.trim()),
      queryFn:   () =>
        fetchBrowse(s.manifest, { search: true as const }, { q: q.trim(), page: 1 }),
      enabled,
      staleTime:       SEARCH_STALE,
      retry:           false,
      // Keep the previous source result on screen while a new query
      // is in flight. Without this, every keystroke (post-debounce)
      // would flash the per-source list to empty and back, making
      // the modal feel jumpy on slow networks.
      placeholderData: keepPreviousData,
    })),
  })

  const hits: SearchHit[]                              = []
  const failures: { sourceId: string; error: Error }[] = []
  let loading = false

  for (let i = 0; i < queries.length; i++) {
    const result = queries[i]!
    const source = queried[i]!
    if (result.isFetching && enabled) loading = true
    if (result.error) {
      failures.push({ sourceId: source.manifest.id, error: result.error as Error })
    }
    if (result.data) {
      hits.push(...rankAndCap(q, source, result.data))
    }
  }

  return { hits, loading, failures, queried }
}
