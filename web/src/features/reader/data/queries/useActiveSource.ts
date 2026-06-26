// useActiveSource — resolve the active source for a chapter.
//
// Two-pass strategy:
//   1. Run resolver with empty rawUrls. If it decides on a path that
//      doesn't need raw (saved archive) → done.
//   2. Otherwise, fetch raw URLs lazily for the version the resolver
//      picked, then re-run with the URLs.
//
// This avoids the manifest probe when the user is reading a saved
// offline archive.

import { useMemo } from 'react'

import { useWorkIdentity } from '@features/work/contexts/WorkIdentityContext'
import { useWorkChapters } from '@features/work/contexts/WorkChaptersContext'

import { useSourcePref } from '../../hooks/useSourcePref'
import { useChapterSources } from './useChapterSources'
import { useChapterRawPages } from './useChapterRawPages'
import { resolveSource } from '../selectors/resolveSource'
import { sourceCacheKey } from '../selectors/sourceCacheKey'
import type { ActiveSource } from '../types'


export interface ActiveSourceResult {
  active:  ActiveSource
  /** Stable string fingerprint for the cache pool. */
  key:     string
  loading: boolean
}


export function useActiveSource(
  workId:     string,
  chapterRef: string,
): ActiveSourceResult {
  const { work }   = useWorkIdentity()
  const { merged } = useWorkChapters()
  const pref       = useSourcePref(workId)
  const sources    = useChapterSources(workId, chapterRef)

  const chapter = useMemo(
    () => merged.find(c => c.numberNorm === chapterRef) ?? null,
    [merged, chapterRef],
  )

  // Pass 1: resolve without raw URLs.
  const dry = useMemo(
    () => resolveSource({
      pref,
      chapter,
      sources,
      targetLang: work.target_lang,
      rawUrls:    [],
    }),
    [pref, chapter, sources, work.target_lang],
  )

  // Lazy raw probe — only when the dry pass asks for URLs.
  const raw = useChapterRawPages(dry.pickedVersion, dry.needsRawUrls)

  // Pass 2: rerun resolver with URLs if needed.
  const finalResolved = useMemo(() => {
    if (!dry.needsRawUrls) return dry
    return resolveSource({
      pref,
      chapter,
      sources,
      targetLang: work.target_lang,
      rawUrls:    raw.urls,
    })
  }, [dry, pref, chapter, sources, work.target_lang, raw.urls])

  return useMemo<ActiveSourceResult>(() => ({
    active:  finalResolved.active,
    key:     sourceCacheKey(workId, chapterRef, finalResolved.active),
    loading: dry.needsRawUrls && raw.loading,
  }), [finalResolved.active, workId, chapterRef, dry.needsRawUrls, raw.loading])
}
