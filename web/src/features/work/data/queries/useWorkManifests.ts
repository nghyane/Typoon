// Manifest queries — parallel fetch of source details for a Work's
// attached sources. Each query is independent so a slow source never
// blocks the others.

import { useQueries } from '@tanstack/react-query'

import { fetchMangaDetail } from '@features/browse/manifest/runtime'
import { useEnabledSources } from '@features/browse/sources'
import { qk } from '@shared/api/keys'
import type { MangaDetail, WorkSource } from '../types'


export interface ManifestResult {
  /** Same length / order as `sources`. */
  details: (MangaDetail | undefined)[]
  /** True if any manifest is still fetching. */
  loading: boolean
}


export function useWorkManifests(sources: WorkSource[]): ManifestResult {
  const installed = useEnabledSources()

  const queries = useQueries({
    queries: sources.map(m => {
      const src = installed.find(s => s.manifest.id === m.source) ?? null
      return {
        queryKey:  qk.manifest.detail(m.source, m.upstream_ref),
        queryFn:   () => fetchMangaDetail(src!.manifest, m.upstream_ref),
        enabled:   !!src,
        staleTime: 60_000,
      }
    }),
  })

  return {
    details: queries.map(q => q.data),
    loading: queries.some(q => q.isFetching && !q.data),
  }
}
