import { useEffect, useRef, useState } from 'react'
import { useQuery } from '@tanstack/react-query'
import type { Bunle, PageInfo } from '@nghyane/bunle'
import { cn } from '@shared/lib/cn'
import { Spinner } from '@shared/ui/primitives'

// =============================================================================
// PageImage — single page from a Bunle archive. The archive is opened once
// at the reader level and passed in; this component issues a per-page
// Range request via `bunle.url(index)` only when scrolled into view.
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

  const { data: src, isPending, isError } = useQuery({
    queryKey: ['bunle-page', bunle?.pageCount, info.index, info.offset],
    queryFn:  () => bunle!.url(info.index),
    enabled:  visible && bunle !== null,
    staleTime: Infinity,  // object URL is valid for the archive's lifetime
  })

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
