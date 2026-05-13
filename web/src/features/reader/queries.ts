// Reader-specific queries. Built on top of the same domain hooks the
// detail page uses so cache entries are shared:
//
//   detail page  ──┐
//                  ├──> useWork(workId) ── single cache key
//   reader page  ──┘
//
// Once the user opens a chapter from detail, the reader's `useWork`
// call resolves synchronously from cache — no flash, no refetch.

import { useQuery } from '@tanstack/react-query'

import { api } from '@shared/api/api'
import { qk } from '@shared/api/keys'
import { fetchChapterPages } from '@features/browse/manifest/runtime'
import type { InstalledSource } from '@features/browse/manifest/types'


/** Translation row — gives the BNL `archive_url`, chapter context,
 *  and work_id. Cached for 60s. Once `state='done'` and the archive
 *  is materialised, this row is immutable.
 *
 *  No `placeholderData` — when the user switches chapters, the URL
 *  change re-keys this query and we WANT the data to drop so the
 *  reader can show a loading state instead of the previous chapter's
 *  archive while the new one loads. */
export function useTranslation(translationId: number | null) {
  return useQuery({
    queryKey: translationId != null
      ? qk.translation.byId(translationId)
      : ['translation', 'invalid'] as const,
    queryFn:  () => api.getTranslation(translationId!),
    enabled:  translationId != null
              && Number.isInteger(translationId)
              && translationId > 0,
    staleTime: 60_000,
    placeholderData: undefined,
  })
}


/** Manifest page URLs for a raw chapter. Cached for 15min — chapter
 *  pages are immutable once published. Persisted (see persistence.ts)
 *  so a reload + source outage still lets the user read what they've
 *  already opened.
 *
 *  Same `placeholderData: undefined` reasoning as `useTranslation`:
 *  switching chapters should clear the page list, not bleed pages
 *  from the previous chapter into the new one. */
export function useChapterPages(
  source: InstalledSource | null,
  chapterUrl: string | null,
) {
  return useQuery({
    queryKey: source && chapterUrl
      ? qk.manifest.chapterPages(source.manifest.id, chapterUrl)
      : ['manifest', 'chapter-pages', 'invalid'] as const,
    queryFn:  () => fetchChapterPages(source!.manifest, chapterUrl!),
    enabled:  source != null && chapterUrl != null && chapterUrl.length > 0,
    staleTime: 15 * 60_000,
    gcTime:    24 * 60 * 60_000,
    retry:     2,
    retryDelay: (attempt) => Math.min(1000 * 2 ** attempt, 8000),
    placeholderData: undefined,
  })
}
