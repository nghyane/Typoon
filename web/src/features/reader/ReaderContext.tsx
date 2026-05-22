// ReaderContext — page-scoped state for the reader chrome.
//
// Holds:
//   currentPage:    visible page index (synced from view, drives chrome)
//   chromeVisible:  top/bottom bars peek state
//   navTargets:     prev/next chapter refs computed from chapter spine
//
// Page source + reader settings live in their own hooks
// (`useChapterReader`, `useReaderSettings`). This context is glue only,
// so a deep child like the bottom pill can toggle chrome without
// drilling props from the route.

import {
  createContext, useCallback, useContext, useMemo, useState,
  type ReactNode,
} from 'react'

import { useWorkChapters } from '@features/work/contexts/WorkChaptersContext'


export interface NavTarget {
  ref:    string
  label:  string
}


export interface ReaderContextValue {
  workId:        string
  chapterRef:    string
  // Page state
  page:          number
  setPage:       (next: number) => void
  /** Reading progress in [0, 1]. Pager mode = page / pageCount;
   *  strip mode = scrollTop / scrollMax. Used by the preload hook. */
  progress:      number
  setProgress:   (next: number) => void
  // Chrome (top bar + bottom pill peek/hide)
  chromeVisible: boolean
  toggleChrome:  () => void
  showChrome:    () => void
  hideChrome:    () => void
  // Nav targets across chapters
  prev:          NavTarget | null
  next:          NavTarget | null
}


const Ctx = createContext<ReaderContextValue | null>(null)


export function useReader(): ReaderContextValue {
  const v = useContext(Ctx)
  if (!v) throw new Error('useReader must be inside <ReaderProvider>')
  return v
}


interface Props {
  workId:     string
  chapterRef: string
  children:   ReactNode
}


export function ReaderProvider({ workId, chapterRef, children }: Props) {
  const { merged } = useWorkChapters()

  const [page, setPage] = useState(0)
  const [progress, setProgress] = useState(0)
  const [chromeVisible, setChromeVisible] = useState(true)

  // Prev / next computed from sorted chapter spine.
  // Order matches the work hub default (newest first), so "next"
  // chapter is at index-1 (lower in array = higher chapter number).
  const { prev, next } = useMemo<{
    prev: NavTarget | null; next: NavTarget | null
  }>(() => {
    if (!merged.length) return { prev: null, next: null }
    // Sort ascending by sortKey for predictable prev/next
    const sorted = [...merged].sort((a, b) => a.sortKey - b.sortKey)
    const idx = sorted.findIndex(c => c.numberNorm === chapterRef)
    if (idx < 0) return { prev: null, next: null }
    const before = sorted[idx - 1]
    const after  = sorted[idx + 1]
    return {
      prev: before ? { ref: before.numberNorm, label: `Ch.${before.number}` } : null,
      next: after  ? { ref: after.numberNorm,  label: `Ch.${after.number}`  } : null,
    }
  }, [merged, chapterRef])

  const toggleChrome = useCallback(() => setChromeVisible(v => !v), [])
  const showChrome   = useCallback(() => setChromeVisible(true), [])
  const hideChrome   = useCallback(() => setChromeVisible(false), [])

  const value = useMemo<ReaderContextValue>(() => ({
    workId, chapterRef,
    page, setPage,
    progress, setProgress,
    chromeVisible, toggleChrome, showChrome, hideChrome,
    prev, next,
  }), [
    workId, chapterRef, page, progress,
    chromeVisible, toggleChrome, showChrome, hideChrome,
    prev, next,
  ])

  return <Ctx.Provider value={value}>{children}</Ctx.Provider>
}
