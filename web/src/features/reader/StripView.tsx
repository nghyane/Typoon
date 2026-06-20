// StripView — vertical scroll, all pages stacked.
//
// Pages use `content-visibility: auto` so the browser skips
// rendering off-screen rows without JS virtualization. The
// TranslationRuntime already fetches every page, so DOM
// existence doesn't add network cost.

import { useEffect, useLayoutEffect, useRef } from 'react'

import { PageRenderer } from './PageRenderer'
import { useReaderSettings } from './settings'
import type { ReaderSource } from './sources'

import { useReader } from './ReaderContext'

interface Props {
  source:           ReaderSource
  blobs?:           readonly (Blob | null)[]
  /** Current page (best-effort — strip mode tracks via scroll). */
  pageIndex:        number
  onChangePage:     (next: number) => void
}

export function StripView({ source, pageIndex, onChangePage }: Props) {
  const { toggleChrome, setProgress } = useReader()
  const { pageWidth } = useReaderSettings()
  const scrollRef = useRef<HTMLDivElement | null>(null)

  // Hard reset scroll position on every source change BEFORE paint.
  useLayoutEffect(() => {
    if (scrollRef.current) scrollRef.current.scrollTop = 0
  }, [source])

  // Sync intent → scroll (e.g. coming back from another route via
  // pageIndex > 0). Runs after the layout reset above.
  useEffect(() => {
    if (pageIndex > 0 && pageIndex < source.pageCount) {
      scrollRef.current
        ?.querySelector<HTMLElement>(`[data-index="${pageIndex}"]`)
        ?.scrollIntoView({ block: 'start', behavior: 'auto' })
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [source])

  // Sync scroll → pageIndex
  useEffect(() => {
    const el = scrollRef.current
    if (!el) return
    let raf = 0
    const onScroll = () => {
      cancelAnimationFrame(raf)
      raf = requestAnimationFrame(() => {
        const top   = el.scrollTop
        const max   = el.scrollHeight - el.clientHeight
        const pct   = max > 0 ? Math.min(1, top / max) : 0
        setProgress(pct)
        // Page whose midpoint is closest to viewport center.
        const viewport = el.getBoundingClientRect()
        const center = viewport.top + el.clientHeight / 2
        let best = 0, bestDist = Infinity
        for (const node of el.querySelectorAll<HTMLElement>('[data-index]')) {
          const rect = node.getBoundingClientRect()
          const mid = rect.top + rect.height / 2
          const dist = Math.abs(mid - center)
          if (dist < bestDist) { bestDist = dist; best = Number(node.dataset.index) || 0 }
        }
        if (best !== pageIndex) onChangePage(best)
      })
    }
    el.addEventListener('scroll', onScroll, { passive: true })
    return () => { el.removeEventListener('scroll', onScroll); cancelAnimationFrame(raf) }
  }, [pageIndex, onChangePage, setProgress])

  // Estimate a reasonable intrinsic height for pages whose true
  // dimensions we don't know yet. This gives the browser a usable
  // scroll-bar baseline while `content-visibility: auto` skips
  // off-screen layout.
  const estimatedHeight = 800

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
          width:    '100%',
          maxWidth: `${pageWidth}px`,
          margin:   '0 auto',
        }}
      >
        {Array.from({ length: source.pageCount }, (_, index) => (
          <div
            key={index}
            data-index={index}
            style={{
              width: '100%',
              contentVisibility: 'auto',
              containIntrinsicSize: `auto ${estimatedHeight}px`,
            }}
          >
            <PageRenderer
              source={source}
              index={index}
              className="w-full"
            />
          </div>
        ))}
      </div>
    </div>
  )
}
