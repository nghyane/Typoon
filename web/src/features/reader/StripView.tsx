// StripView — vertical scroll, all pages stacked.
//
// Uses `@tanstack/react-virtual` so a 200-page chapter doesn't mount
// 200 <img> nodes at once. Each item reserves layout space via the BNL
// index's known dimensions (or a fallback aspect ratio for raw streams).

import { useEffect, useLayoutEffect, useRef } from 'react'
import { useVirtualizer } from '@tanstack/react-virtual'

import { PageRenderer } from './PageRenderer'
import { useReaderSettings } from './settings'
import type { ReaderSource } from './sources'

import { useReader } from './ReaderContext'

const DEFAULT_ASPECT = 1.5    // h:w fallback for unknown dimensions

interface Props {
  source:           ReaderSource
  sourceKey:        string
  /** Current page (best-effort — strip mode tracks via scroll). */
  pageIndex:        number
  onChangePage:     (next: number) => void
}

export function StripView({ source, sourceKey, pageIndex, onChangePage }: Props) {
  const { toggleChrome, setProgress } = useReader()
  const { pageWidth } = useReaderSettings()
  const pageGap = 8
  const scrollRef = useRef<HTMLDivElement | null>(null)

  const virtualizer = useVirtualizer({
    count:           source.pageCount,
    getScrollElement: () => scrollRef.current,
    estimateSize:    (index) => {
      const page = source.pages[index]
      const cap  = scrollRef.current
        ? Math.min(scrollRef.current.clientWidth, pageWidth)
        : pageWidth
      if (!page?.width || !page?.height) return cap * DEFAULT_ASPECT + pageGap
      return cap * (page.height / page.width) + pageGap
    },
    overscan:        2,
  })

  // Hard reset scroll position on every source change BEFORE paint.
  // The route remounts ReaderBody when chapter switches, but the
  // scroll container's `scrollTop` may inherit the previous chapter's
  // value because browsers restore scroll on identical layout. The
  // virtualizer's `scrollToIndex` runs AFTER paint, so a layout
  // effect that hard-zeros scrollTop wins the race.
  useLayoutEffect(() => {
    if (scrollRef.current) scrollRef.current.scrollTop = 0
  }, [sourceKey])

  // Sync intent → scroll (e.g. coming back from another route via
  // pageIndex > 0). Runs after the layout reset above.
  useEffect(() => {
    if (pageIndex > 0 && pageIndex < source.pageCount) {
      virtualizer.scrollToIndex(pageIndex, { align: 'start', behavior: 'auto' })
    }
  // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [source])  // re-run on source change only

  // Sync scroll → pageIndex
  useEffect(() => {
    const el = scrollRef.current
    if (!el) return
    let raf = 0
    const onScroll = () => {
      cancelAnimationFrame(raf)
      raf = requestAnimationFrame(() => {
        const items = virtualizer.getVirtualItems()
        const top   = el.scrollTop
        const max   = el.scrollHeight - el.clientHeight
        const pct   = max > 0 ? Math.min(1, top / max) : 0
        setProgress(pct)
        // Page whose midpoint is closest to viewport center.
        const center = top + el.clientHeight / 2
        let best = 0, bestDist = Infinity
        for (const v of items) {
          const mid = v.start + v.size / 2
          const dist = Math.abs(mid - center)
          if (dist < bestDist) { bestDist = dist; best = v.index }
        }
        if (best !== pageIndex) onChangePage(best)
      })
    }
    el.addEventListener('scroll', onScroll, { passive: true })
    return () => { el.removeEventListener('scroll', onScroll); cancelAnimationFrame(raf) }
  }, [virtualizer, pageIndex, onChangePage, setProgress])

  return (
    <div
      ref={scrollRef}
      className="w-full h-full overflow-y-auto overflow-x-hidden bg-bg"
      onClick={(e) => {
        const rect = e.currentTarget.getBoundingClientRect()
        const xPct = (e.clientX - rect.left) / rect.width
        if (xPct > 0.33 && xPct < 0.66) toggleChrome()
      }}
    >
      <div
        style={{
          height:   `${virtualizer.getTotalSize()}px`,
          position: 'relative',
          width:    '100%',
          maxWidth: `${pageWidth}px`,
          margin:   '0 auto',
        }}
      >
        {virtualizer.getVirtualItems().map(item => (
          <div
            key={item.key}
            data-index={item.index}
            ref={virtualizer.measureElement}
            style={{
              position:     'absolute',
              top:          0,
              left:         0,
              width:        '100%',
              paddingBottom: `${pageGap}px`,
              transform:    `translateY(${item.start}px)`,
            }}
          >
            <PageRenderer
              source={source}
              sourceKey={sourceKey}
              index={item.index}
              className="w-full"
            />
          </div>
        ))}
      </div>
    </div>
  )
}
