// PagerView — single page at a time. Keyboard nav lives here; tap nav
// lives in `<TapZones>` which the shell mounts as an overlay.
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

import { useCallback, useEffect, useLayoutEffect, useRef } from 'react'
import { PageRenderer } from './PageRenderer'
import { useReader } from './ReaderContext'
import { useReaderSettings } from './settings'
import type { ReaderSource } from './sources'


interface Props {
  source:       ReaderSource
  sourceKey:    string
  pageIndex:    number
  onChangePage: (next: number) => void
  /** Read direction — 'rtl' for manga, 'ltr' for manhua/comics. */
  direction?:   'ltr' | 'rtl'
}

export function PagerView({
  source, sourceKey, pageIndex, onChangePage, direction = 'rtl',
}: Props) {
  const { pageWidth } = useReaderSettings()
  const { setProgress } = useReader()
  const max = source.pageCount - 1
  const scrollRef = useRef<HTMLDivElement>(null)

  // Pager progress = current page / pageCount.
  useEffect(() => {
    if (source.pageCount > 0) {
      setProgress((pageIndex + 1) / source.pageCount)
    }
  }, [pageIndex, source.pageCount, setProgress])

  // Reset outer scroll on source change so the new chapter starts at
  // the top, not inheriting the previous chapter's vertical offset.
  useLayoutEffect(() => {
    if (scrollRef.current) scrollRef.current.scrollTop = 0
  }, [sourceKey])

  // Also reset whenever the page index goes back to 0 (chapter reset
  // signal from the shell). Without this, tap-prev on page 0 within
  // the same chapter wouldn't take the reader back to the page top.
  useEffect(() => {
    if (pageIndex === 0 && scrollRef.current) {
      scrollRef.current.scrollTop = 0
    }
  }, [pageIndex])

  const clamp = useCallback((n: number) => Math.max(0, Math.min(max, n)), [max])

  const prev = useCallback(() => onChangePage(clamp(pageIndex - 1)), [pageIndex, clamp, onChangePage])
  const next = useCallback(() => onChangePage(clamp(pageIndex + 1)), [pageIndex, clamp, onChangePage])

  useEffect(() => {
    const handler = (e: KeyboardEvent) => {
      if (e.target instanceof HTMLInputElement)    return
      if (e.target instanceof HTMLTextAreaElement) return
      switch (e.key) {
        case 'ArrowLeft':
        case 'PageUp':
          e.preventDefault()
          direction === 'rtl' ? next() : prev()
          return
        case 'ArrowRight':
        case 'PageDown':
        case ' ':
          e.preventDefault()
          direction === 'rtl' ? prev() : next()
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
  const wrapperStyle: React.CSSProperties = {
    maxWidth: pageWidth,
    margin:   '0 auto',
  }

  return (
    <div ref={scrollRef} className="w-full h-full overflow-y-auto bg-bg">
      <div style={wrapperStyle}>
        <PageRenderer
          source={source}
          sourceKey={sourceKey}
          index={pageIndex}
          className="w-full h-auto"
        />
      </div>
    </div>
  )
}
