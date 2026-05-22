// usePreloadNext — fires a single warm call once progress crosses
// the threshold for the current chapter.
//
// No always-call hooks for the next chapter's data — the warm path
// reads IDB / manifest on-demand inside `useWarmNext`. Cheap when
// the user isn't yet near the end of the chapter.

import { useEffect, useRef } from 'react'

import { useWarmNext } from '../data/queries/useWarmNext'
import { useWorkChapters } from '@features/work/contexts/WorkChaptersContext'


interface Args {
  workId:     string
  chapterRef: string
  /** Reading progress in [0, 1]. */
  progress:   number
  /** Trigger threshold — default 0.8. */
  threshold?: number
}


export function usePreloadNext({
  workId, chapterRef, progress, threshold = 0.8,
}: Args): void {
  const { merged } = useWorkChapters()
  const warm       = useWarmNext()
  const triggered  = useRef<string | null>(null)

  useEffect(() => {
    if (progress < threshold) return
    if (triggered.current === chapterRef) return

    const sorted = merged.slice().sort((a, b) => a.sortKey - b.sortKey)
    const idx    = sorted.findIndex(c => c.numberNorm === chapterRef)
    const next   = idx >= 0 ? sorted[idx + 1]?.numberNorm ?? null : null
    if (!next) return

    triggered.current = chapterRef
    void warm(next)
  }, [progress, threshold, chapterRef, workId, merged, warm])
}
