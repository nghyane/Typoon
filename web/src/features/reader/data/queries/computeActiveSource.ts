// computeActiveSource — on-demand resolution for warming next chapter.
//
// Differs from `useActiveSource`: this is a one-shot async function,
// not a subscription. Reads IDB + manifest fresh when called.

import { QueryClient } from '@tanstack/react-query'

import { db, type SavedArchive } from '@shared/db'
import { fetchChapterPages } from '@features/browse/manifest/runtime'
import { qk } from '@shared/api/keys'
import type { MergedChapter } from '@features/work/data/types'
import { resolveSource } from '../selectors/resolveSource'
import { sourceCacheKey } from '../selectors/sourceCacheKey'
import type { ActiveSource, SourcePref } from '../types'


export interface ComputeArgs {
  workId:     string
  chapterRef: string
  chapter:    MergedChapter | null
  pref:       SourcePref
  targetLang: string
  qc:         QueryClient
}


export interface ComputeResult {
  active: ActiveSource
  key:    string
}


export async function computeActiveSource({
  workId, chapterRef, chapter, pref, targetLang, qc,
}: ComputeArgs): Promise<ComputeResult> {
  const id = `${workId}:${chapterRef}`

  const saved = (await db().archives.get(id)) ?? null

  const sources = {
    saved:    saved as SavedArchive | null,
    versions: chapter?.sourceVersions ?? [],
  }

  // Pass 1
  const dry = resolveSource({
    pref, chapter, sources, targetLang, rawUrls: [],
  })

  if (!dry.needsRawUrls) {
    return {
      active: dry.active,
      key:    sourceCacheKey(workId, chapterRef, dry.active),
    }
  }

  // Lazy raw fetch via the shared query cache.
  const v = dry.pickedVersion
  if (!v) {
    return {
      active: { kind: 'none' },
      key:    sourceCacheKey(workId, chapterRef, { kind: 'none' }),
    }
  }

  const pages = await qc.fetchQuery({
    queryKey: qk.manifest.chapterPages(v.source.manifest.id, v.ref.url),
    queryFn:  () => fetchChapterPages(v.source.manifest, v.ref.url),
    staleTime: 5 * 60_000,
  })

  const final = resolveSource({
    pref, chapter, sources, targetLang, rawUrls: pages.pages,
  })

  return {
    active: final.active,
    key:    sourceCacheKey(workId, chapterRef, final.active),
  }
}
