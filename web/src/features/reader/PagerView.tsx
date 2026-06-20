// PagerView — single page at a time. Keyboard and tap navigation
// live here (no separate overlay component).
//
// Fit logic (matches legacy reader):
//   width  — page cap = pageWidth, scroll vertically when taller than viewport
//   height — page = viewport height (minus pill chrome), horizontal scroll if wider
//   free   — intrinsic size, both axes scrollable
//
// Keyboard:
//   ← / PageUp           prev page (rtl: next)
//   → / PageDown / Space next page (rtl: prev)
//   Home / End            first / last

import {
  useCallback, useEffect, useLayoutEffect, useRef,
  type CSSProperties, type MouseEvent,
} from 'react'
import { PageRenderer } from './PageRenderer'
import { useReader } from './ReaderContext'
import { useReaderSettings } from './settings'
import type { ReaderSource } from './sources'


interface Props {
  source:       ReaderSource
  pageIndex:    number
  onChangePage: (next: number) => void
  /** Read direction — 'rtl' for manga, 'ltr' for manhua/comics. */
  direction?:   'ltr' | 'rtl'
}

export function PagerView({
  source, pageIndex, onChangePage, direction = 'rtl',
}: Props) {
  const { pageWidth } = useReaderSettings()
  const { setProgress, toggleChrome } = useReader()
  const max = source.pageCount - 1
  const scrollRef = useRef<HTMLDivElement>(null)
  const sourceRef = useRef(source)
  sourceRef.current = source

  // Pager progress = current page / pageCount.
  useEffect(() => {
    if (source.pageCount > 0) {
      setProgress((pageIndex + 1) / source.pageCount)
    }
  }, [pageIndex, source.pageCount, setProgress])

  // Reset outer scroll whenever the visible page changes so a tall
  // previous page never leaks its vertical offset into the next page.
  useLayoutEffect(() => {
    if (scrollRef.current) scrollRef.current.scrollTop = 0
  }, [source, pageIndex])

  const clamp = useCallback((n: number) => Math.max(0, Math.min(max, n)), [max])

  const prev = useCallback(() => onChangePage(clamp(pageIndex - 1)), [pageIndex, clamp, onChangePage])
  const next = useCallback(() => onChangePage(clamp(pageIndex + 1)), [pageIndex, clamp, onChangePage])

  // Preload ±1 page — fire-and-forget.
  useEffect(() => {
    const src = sourceRef.current
    for (const i of [pageIndex + 1, pageIndex - 1]) {
      if (i >= 0 && i < src.pageCount) src.getUrl(i).catch(() => {})
    }
  }, [pageIndex, source])

  const handleTap = useCallback((e: MouseEvent<HTMLDivElement>) => {
    const target = e.target as HTMLElement
    if (target.closest('button,a,input,textarea,select,[role="button"]')) return

    const rect = e.currentTarget.getBoundingClientRect()
    const xPct = (e.clientX - rect.left) / rect.width

    if (xPct > 0.33 && xPct < 0.67) {
      toggleChrome()
      return
    }

    const leftTap = xPct <= 0.33
    const goNext = direction === 'rtl' ? leftTap : !leftTap
    if (goNext) next()
    else prev()
  }, [direction, next, prev, toggleChrome])

  useEffect(() => {
    const handler = (e: KeyboardEvent) => {
      if (e.target instanceof HTMLInputElement)    return
      if (e.target instanceof HTMLTextAreaElement) return
      switch (e.key) {
        case 'ArrowLeft':
        case 'PageUp':
          e.preventDefault()
          if (direction === 'rtl') next()
          else prev()
          return
        case 'ArrowRight':
        case 'PageDown':
        case ' ':
          e.preventDefault()
          if (direction === 'rtl') prev()
          else next()
          return
        case 'Home': e.preventDefault(); onChangePage(0);   return
        case 'End':  e.preventDefault(); onChangePage(max); return
      }
    }
    window.addEventListener('keydown', handler)
    return () => window.removeEventListener('keydown', handler)
  }, [direction, prev, next, onChangePage, max])

  // All pager styles use width-fit with a max cap. The image scrolls
  // vertically inside the wrapper when taller than the viewport.
  const wrapperStyle: CSSProperties = {
    maxWidth: pageWidth,
    margin:   '0 auto',
  }

  return (
    <div
      ref={scrollRef}
      className="w-full h-full overflow-y-auto bg-bg"
      onClick={handleTap}
    >
      <div style={wrapperStyle}>
        <PageRenderer
          source={source}
          index={pageIndex}
          className="w-full h-auto"
        />
      </div>
    </div>
  )
}
