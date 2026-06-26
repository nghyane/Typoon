// useWarmNext — imperative warmup for a chapter ref.
//
// Reads pref + sources on-demand (not subscription) and warms the
// cache. Best-effort — failures swallowed.

import { useCallback } from 'react'
import { useQueryClient } from '@tanstack/react-query'

import { useReaderCache } from '../../cache/ReaderCacheProvider'
import { useWorkIdentity } from '@features/work/contexts/WorkIdentityContext'
import { useWorkChapters } from '@features/work/contexts/WorkChaptersContext'
import { useSourcePref } from '../../hooks/useSourcePref'
import { computeActiveSource } from './computeActiveSource'
import { openSourceFromActive } from '../../hooks/openSourceFromActive'
import { useSourceFetch } from '@features/browse/SourceFetchProvider'


export function useWarmNext() {
  const cache = useReaderCache()
  const qc    = useQueryClient()
  const sourceFetch = useSourceFetch()
  const { work }   = useWorkIdentity()
  const { merged } = useWorkChapters()
  const pref       = useSourcePref(work.id)

  return useCallback(async (chapterRef: string) => {
    const chapter = merged.find(c => c.numberNorm === chapterRef) ?? null
    const { active, key } = await computeActiveSource({
      workId:     work.id,
      chapterRef,
      chapter,
      pref,
      targetLang: work.target_lang,
      qc,
    })
    if (active.kind === 'none') return
    cache.warm(key, (signal) => openSourceFromActive(active, signal, sourceFetch))
  }, [cache, qc, work.id, work.target_lang, merged, pref, sourceFetch])
}
