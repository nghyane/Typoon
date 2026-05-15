// useReadingPosition — persist + restore the user's spot in a
// chapter so re-opening lands where they left off.
//
// Scope: localStorage only for now. Server-side history already
// records "user has read this chapter" via `record*Reading` in
// `useReader`; this hook adds intra-chapter precision (which page,
// what scroll percent) without a new endpoint. Cross-device sync
// goes in a later stage when the server gains a `position` column.
//
// Key shape: `reader:pos:{workId}:{numberNorm}` → JSON
// `{ page, scrollPct, updatedAt }`.
//
// Eviction: bounded by browser localStorage (5-10MB). Reader writes
// at most one entry per chapter the user has opened; even 10k
// chapters = ~500KB. No eviction logic needed at beta scale.

import { useCallback, useEffect, useRef } from 'react'


export interface SavedPosition {
  page:      number
  scrollPct: number
  updatedAt: number
}


/** Read once on mount. Returns null when no record exists or storage
 *  is unavailable (private mode, quota error). */
export function loadPosition(
  workId:     number,
  numberNorm: string,
): SavedPosition | null {
  if (typeof window === 'undefined') return null
  try {
    const raw = window.localStorage.getItem(key(workId, numberNorm))
    if (!raw) return null
    const obj = JSON.parse(raw) as Partial<SavedPosition>
    if (typeof obj.page !== 'number') return null
    return {
      page:      obj.page,
      scrollPct: typeof obj.scrollPct === 'number' ? obj.scrollPct : 0,
      updatedAt: typeof obj.updatedAt === 'number' ? obj.updatedAt : 0,
    }
  } catch {
    return null
  }
}


/** Throttled writer. Returns an `update` function the reader body
 *  calls on scroll / page change. Writes coalesce inside a 1.5s
 *  window to avoid hammering localStorage on every scroll tick. */
export function useReadingPosition(
  workId:     number,
  numberNorm: string,
  totalPages: number,
) {
  const pendingRef = useRef<SavedPosition | null>(null)
  const timerRef   = useRef<number | null>(null)

  const flush = useCallback(() => {
    timerRef.current = null
    const p = pendingRef.current
    pendingRef.current = null
    if (!p) return
    try {
      window.localStorage.setItem(key(workId, numberNorm), JSON.stringify(p))
    } catch {
      // Storage full / blocked — silently drop. The next save attempt
      // will retry; a missed write just means resume falls back to
      // the previous saved position, not a crash.
    }
  }, [workId, numberNorm])

  const update = useCallback((p: Omit<SavedPosition, 'updatedAt'>) => {
    if (totalPages <= 0) return
    pendingRef.current = { ...p, updatedAt: Date.now() }
    if (timerRef.current !== null) return
    timerRef.current = window.setTimeout(flush, 1500)
  }, [flush, totalPages])

  // Flush on unmount so closing the tab / navigating away doesn't
  // drop the last 1500ms of progress. visibilitychange is a stronger
  // signal than unload (works on mobile background); both wired so
  // either trigger commits.
  useEffect(() => {
    const onHide = () => {
      if (timerRef.current !== null) {
        window.clearTimeout(timerRef.current)
        timerRef.current = null
      }
      flush()
    }
    window.addEventListener('visibilitychange', onHide)
    window.addEventListener('pagehide', onHide)
    return () => {
      window.removeEventListener('visibilitychange', onHide)
      window.removeEventListener('pagehide', onHide)
      onHide()
    }
  }, [flush])

  return { update }
}


function key(workId: number, numberNorm: string): string {
  return `reader:pos:${workId}:${numberNorm}`
}
