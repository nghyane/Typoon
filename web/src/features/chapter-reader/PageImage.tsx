import { useEffect, useRef, useState } from 'react'
import type { Bunle, PageInfo } from '@nghyane/bunle'
import { cn } from '@shared/lib/cn'
import { Spinner } from '@shared/ui/primitives'

// =============================================================================
// PageImage — single page from a Bunle archive. The archive is opened once
// at the reader level and passed in; this component issues a per-page
// Range request via `bunle.url(index)` only when scrolled into view.
//
// Bunle owns the object-URL cache internally and revokes every URL on
// `close()`. We therefore tie this component's state to the Bunle
// instance directly — no React Query layer — so navigating between
// chapters can never hand out a `blob:` URL that belongs to a closed
// archive (Chrome reports those as `net::ERR_FILE_NOT_FOUND`).
//
// Layout: pre-allocate using known PageInfo width/height so the column
// height is correct from first paint, no jump when the image loads.
// =============================================================================

interface Props {
  bunle: Bunle | null
  info:  PageInfo
  /** intersection-observer based lazy load — defaults to true */
  lazy?:  boolean
  className?: string
}

export function PageImage({ bunle, info, lazy = true, className }: Props) {
  const ref = useRef<HTMLDivElement | null>(null)
  const [visible, setVisible] = useState(!lazy)

  useEffect(() => {
    if (visible || !ref.current) return
    const io = new IntersectionObserver(
      ([entry]) => { if (entry?.isIntersecting) setVisible(true) },
      { rootMargin: '800px 0px' },
    )
    io.observe(ref.current)
    return () => io.disconnect()
  }, [visible])

  // Resolve the page's blob URL from the live Bunle. The effect re-runs
  // whenever the Bunle instance swaps (chapter change), which discards
  // any URL belonging to the previous, now-closed archive.
  const [src, setSrc] = useState<string | null>(null)
  const [isError, setIsError] = useState(false)
  const isPending = visible && bunle !== null && src === null && !isError

  useEffect(() => {
    setSrc(null)
    setIsError(false)
    if (!visible || !bunle) return
    let cancelled = false
    bunle.url(info.index)
      .then((u) => { if (!cancelled) setSrc(u) })
      .catch(() => { if (!cancelled) setIsError(true) })
    return () => { cancelled = true }
  }, [bunle, info.index, visible])

  // Aspect ratio from PageInfo — page slot has its full final height before
  // the image loads. Eliminates layout shift across the whole reader.
  const aspect = info.width > 0 ? info.width / info.height : 2 / 3

  return (
    <div
      ref={ref}
      className={cn('relative w-full bg-surface', className)}
      style={{ aspectRatio: aspect }}
    >
      {visible && isPending && (
        <div className="absolute inset-0 flex items-center justify-center text-text-subtle">
          <Spinner size={16} />
        </div>
      )}
      {visible && isError && (
        <div className="absolute inset-0 flex items-center justify-center text-error-text text-xs">
          Không tải được trang {info.index + 1}
        </div>
      )}
      {!visible && (
        <div className="absolute inset-0 flex items-center justify-center text-text-subtle">
          <span className="text-xs tabular">Trang {info.index + 1}</span>
        </div>
      )}
      {src && (
        <img
          src={src}
          alt={`Trang ${info.index + 1}`}
          className="absolute inset-0 w-full h-full object-contain"
          loading="lazy"
        />
      )}
    </div>
  )
}
