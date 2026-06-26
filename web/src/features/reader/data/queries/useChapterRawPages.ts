// useChapterRawPages — fetch raw page URLs lazily.
//
// Caller passes the version (already chosen by the resolver). The
// hook fetches via TanStack Query, returns urls + loading. When
// `enabled` is false (translated path doesn't need raw URLs), the
// hook is a no-op — no fetch fires.

import { useQuery } from '@tanstack/react-query'

import { fetchChapterPages } from '@features/browse/manifest/runtime'
import { qk } from '@shared/api/keys'
import type { SourceVersion } from '../types'


export interface RawPagesResult {
  urls:    string[]
  loading: boolean
}


export function useChapterRawPages(
  version: SourceVersion | null,
  enabled: boolean,
): RawPagesResult {
  const sourceId   = version?.source.manifest.id ?? null
  const chapterUrl = version?.ref.url ?? null

  const q = useQuery({
    queryKey: sourceId && chapterUrl
      ? qk.manifest.chapterPages(sourceId, chapterUrl)
      : ['manifest', 'chapter-pages', 'invalid'],
    queryFn:  async () => {
      if (!version) return { pages: [] } as { pages: string[] }
      return await fetchChapterPages(version.source.manifest, version.ref.url)
    },
    enabled:   enabled && !!version,
    staleTime: 5 * 60_000,
  })

  return {
    urls:    q.data?.pages ?? [],
    loading: q.isFetching,
  }
}
