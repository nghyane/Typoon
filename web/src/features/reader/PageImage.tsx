// A single page slot. Reserves its aspect-ratio box on first paint so
// the layout never jumps when blobs stream in. The <img> is only
// mounted when `src` is non-null AND `inWindow` is true — pages far
// outside the viewport keep the empty slot but skip the bitmap.

import type { ReaderPage } from './types'
import { cn } from '@shared/lib/cn'

interface Props {
  page:      ReaderPage
  /** Resolved image URL — overrides `page.url` so streamed sources
   *  (BNL) can swap blob URLs without re-rendering the whole list. */
  src:       string | null
  inWindow:  boolean
  className?: string
}

export function PageImage({ page, src, inWindow, className }: Props) {
  const aspect = page.width > 0 && page.height > 0
    ? `${page.width} / ${page.height}`
    : undefined

  return (
    <div
      className={cn('relative w-full bg-surface', className)}
      style={{ aspectRatio: aspect }}
    >
      {src && inWindow && (
        <img
          src={src}
          alt={`Trang ${page.index + 1}`}
          width={page.width || undefined}
          height={page.height || undefined}
          decoding="async"
          fetchPriority={inWindow ? 'high' : 'low'}
          draggable={false}
          className="block w-full h-full select-none"
        />
      )}
    </div>
  )
}
