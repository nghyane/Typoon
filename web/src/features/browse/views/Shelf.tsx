import { useRef, useState, useEffect, type ReactNode } from 'react'
import { ChevronRight } from 'lucide-react'
import { Link } from '@tanstack/react-router'
import { cn } from '@shared/lib/cn'

// =============================================================================
// Shelf — horizontal rail with header and "Xem tất cả" link.
//
// Layout:
//   ┌── label ──────────────────────── action ──┐
//   │                                            │
//   │  [card] [card] [card] [card] [card] →      │
//   │                                            │
//   └────────────────────────────────────────────┘
//
// Behaviour:
//   • Native overflow-x scroll, no scrollbar
//   • Edge fade mask both sides (visual hint)
//   • Optional action — usually a Link to the shelf detail page
//   • Skeleton render via `skeleton` prop while data is pending
// =============================================================================

interface Props {
  label:      string
  hint?:      string
  /** Link to full grid view (`/browse/$source/shelf/$shelfId`). When
   *  omitted, no "Xem tất cả →" link is shown. */
  more?:      { to: string; params?: Record<string, string> }
  skeleton?:  boolean
  children:   ReactNode
}

export function Shelf({ label, hint, more, skeleton = false, children }: Props) {
  const scrollerRef = useRef<HTMLDivElement>(null)
  const [edges, setEdges] = useState({ left: false, right: false })

  // Track scroll position to fade edges only when there's content
  // beyond the visible viewport. Pure cosmetic — no scroll snap, no
  // arrow buttons (covered by native flick on touch / scroll wheel).
  useEffect(() => {
    const el = scrollerRef.current
    if (!el) return
    const recompute = () => {
      const { scrollLeft, scrollWidth, clientWidth } = el
      setEdges({
        left:  scrollLeft > 2,
        right: scrollLeft + clientWidth < scrollWidth - 2,
      })
    }
    recompute()
    el.addEventListener('scroll', recompute, { passive: true })
    const ro = new ResizeObserver(recompute)
    ro.observe(el)
    return () => {
      el.removeEventListener('scroll', recompute)
      ro.disconnect()
    }
  }, [children])

  return (
    <section className="pt-8 first:pt-4">
      <header className="px-4 sm:px-6 mb-3 flex items-end justify-between gap-3">
        <div className="min-w-0">
          <h2 className="text-sm font-semibold text-text">{label}</h2>
          {hint && (
            <p className="text-[11px] text-text-subtle mt-0.5 truncate">{hint}</p>
          )}
        </div>
        {more && (
          <Link
            to={more.to}
            params={more.params as never}
            className="inline-flex items-center gap-0.5 text-xs text-text-subtle hover:text-accent-text transition-colors shrink-0"
          >
            Xem tất cả
            <ChevronRight size={12} />
          </Link>
        )}
      </header>

      <div className="relative">
        {/* Edge fade left */}
        <div
          aria-hidden
          className={cn(
            'pointer-events-none absolute left-0 top-0 bottom-0 w-8 z-10',
            'bg-gradient-to-r from-bg to-transparent transition-opacity',
            edges.left ? 'opacity-100' : 'opacity-0',
          )}
        />
        {/* Edge fade right */}
        <div
          aria-hidden
          className={cn(
            'pointer-events-none absolute right-0 top-0 bottom-0 w-8 z-10',
            'bg-gradient-to-l from-bg to-transparent transition-opacity',
            edges.right ? 'opacity-100' : 'opacity-0',
          )}
        />

        <div
          ref={scrollerRef}
          className={cn(
            'flex gap-3 overflow-x-auto px-4 sm:px-6 pb-1',
            'scrollbar-none [&::-webkit-scrollbar]:hidden',
          )}
          style={{ scrollbarWidth: 'none' }}
        >
          {skeleton ? <ShelfSkeleton /> : children}
        </div>
      </div>
    </section>
  )
}

function ShelfSkeleton() {
  return (
    <>
      {Array.from({ length: 6 }).map((_, i) => (
        <div key={i} className="flex flex-col gap-2 w-[120px] sm:w-[144px] shrink-0 animate-pulse">
          <div className="w-full aspect-[2/3] rounded-md bg-surface-2" />
          <div className="h-3 w-3/4 rounded bg-surface-2" />
        </div>
      ))}
    </>
  )
}
