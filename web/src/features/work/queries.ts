// Domain queries for the Work surface. Component-level hooks read
// THROUGH these; they never call `useQuery` against the work cache
// directly. That keeps cache keys in one place (`qk.work.*`), lets
// detail + reader hit the same cache entry, and makes invalidation
// rules easy to reason about.

import { keepPreviousData, useQuery } from '@tanstack/react-query'

import { api, type ApiWorkDetail } from '@shared/api/api'
import { qk } from '@shared/api/keys'
import { fetchMangaDetail } from '@features/browse/manifest/runtime'
import type { InstalledSource } from '@features/browse/manifest/types'


/** Fetch a Work payload — sibling materials + cross-source
 *  work_chapters + viewer entry. Cached for 30s. Polls every 5s while
 *  any translation is pending/running so the chapter list flips
 *  raw → done without a manual refresh.
 *
 *  This is THE entry point for the work cache; the detail page, the
 *  reader, the in-progress section, and any future surface all read
 *  through here and share the cache via `qk.work.byId(workId)`.
 */
export function useWork(workId: number | null) {
  return useQuery({
    queryKey: workId != null ? qk.work.byId(workId) : ['work', 'invalid'] as const,
    queryFn:  () => api.getWork(workId!),
    enabled:  workId != null && Number.isFinite(workId) && workId > 0,
    // Fresh for 2 minutes — navigating between work / reader / library
    // within this window hits the cache. Background polling below
    // overrides while there's in-flight work the user wants to see
    // tick over.
    staleTime: 2 * 60_000,
    // Keep the in-memory entry alive long enough to outlive normal
    // route switching. Pairs with IDB persistence so reload also
    // skips refetch when the persisted snapshot is still fresh.
    gcTime: 30 * 60_000,
    refetchInterval: (q) => {
      const data = q.state.data as ApiWorkDetail | undefined
      if (!data) return false
      const running = data.chapters.some((c) =>
        c.translations.some(
          (t) => t.state === 'pending' || t.state === 'running',
        ),
      )
      return running ? 5_000 : false
    },
  })
}


/** Fetch the active source's live chapter list via the manifest.
 *  Separate from `useWork` because the manifest fetch is keyed on
 *  (source, upstream_ref), not workId — sibling materials of the
 *  same Work each have their own manifest entry. Cached for 5min
 *  with `keepPreviousData` so swapping the active source doesn't
 *  blank the chapter spine while the new fetch lands.
 *
 *  Stale-while-error: persisted via `installPersistence`, so a reload
 *  rehydrates the last good snapshot. While the new fetch is in
 *  flight, RQ keeps `data` populated (the snapshot); on failure, the
 *  snapshot stays — the component reads `isError` separately if it
 *  wants to surface a "dữ liệu offline" banner. `gcTime` is long
 *  enough that the in-memory copy outlives normal navigation. */
export function useMangaDetail(
  source: InstalledSource | null,
  upstreamRef: string | null,
) {
  return useQuery({
    queryKey: qk.manifest.detail(source?.manifest.id, upstreamRef),
    queryFn:  () => fetchMangaDetail(source!.manifest, upstreamRef!),
    enabled:
      source != null
      && upstreamRef != null
      && upstreamRef.length > 0,
    staleTime: 5 * 60_000,
    // Keep the in-memory entry alive for a day past last unmount;
    // pairs with IndexedDB persistence for cross-reload survival.
    gcTime:    24 * 60 * 60_000,
    // Retry on failure — manifest endpoints flake on 3rd-party
    // outages; one retry covers transient 5xx without burning
    // through the user's patience.
    retry:     2,
    retryDelay: (attempt) => Math.min(1000 * 2 ** attempt, 8000),
    placeholderData: keepPreviousData,
  })
}
