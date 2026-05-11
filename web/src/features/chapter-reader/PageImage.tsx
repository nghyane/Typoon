import type { PageInfo } from '@nghyane/bunle'
import { cn } from '@shared/lib/cn'

// =============================================================================
// PageImage — a single manga page slot.
//
// Sizing rules (manga reader, no fancy fitting):
//   - width  = 100% of the column (column width itself caps at the page's
//              natural width via the parent reader).
//   - height = derived from the page's aspect ratio (width/height from the
//              Bunle index). The slot has its full final height before any
//              pixel loads — zero CLS, zero layout jump.
//   - the <img> uses `display: block` and fills the slot exactly. No
//              object-fit contain/cover (would either letterbox or crop).
//              No max-height. We trust the index dimensions.
//
// The actual blob URL is streamed in by `useChapterArchive` (single HTTP
// request for the whole chapter). When `src === null` the slot is reserved
// space waiting for the streaming bytes — no per-page request.
// =============================================================================

interface Props {
  info:       PageInfo
  src:        string | null
  className?: string
}

export function PageImage({ info, src, className }: Props) {
  const aspect = info.width > 0 && info.height > 0
    ? `${info.width} / ${info.height}`
    : undefined

  return (
    <div
      className={cn('relative w-full bg-surface', className)}
      style={{ aspectRatio: aspect }}
    >
      {src && (
        <img
          src={src}
          alt={`Trang ${info.index + 1}`}
          width={info.width}
          height={info.height}
          decoding="async"
          draggable={false}
          className="block w-full h-full select-none"
        />
      )}
    </div>
  )
}
