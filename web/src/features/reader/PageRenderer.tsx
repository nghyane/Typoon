// PageRenderer — pure image surface.
//
// No effects, no local state. Parent resolves the URL via `usePageUrl`
// and passes it down. The image element therefore doesn't re-mount on
// parent re-renders (e.g. settings slider drag) — its `src` stays the
// same blob URL across renders unless the underlying source actually
// changed.

import { cn } from '@shared/lib/cn'
import { usePageUrl } from './data/queries/usePageUrl'
import type { ReaderSource } from './sources'


interface Props {
  source:     ReaderSource
  sourceKey:  string
  index:      number
  className?: string
}


export function PageRenderer({
  source, sourceKey, index, className,
}: Props) {
  const { url, error } = usePageUrl(source, sourceKey, index)
  const page = source.pages[index]

  // Reserve layout space. When the source carries known dimensions
  // (BNL-backed) use them; otherwise fall back to a typical manga
  // aspect (1.5:1 h/w) so the placeholder has a visible height even
  // for raw-online streams.
  const aspect = page?.width && page?.height
    ? { paddingTop: `${(page.height / page.width) * 100}%` }
    : { paddingTop: '150%' }

  if (error) {
    return (
      <div
        className={cn('relative w-full bg-surface-2 flex items-center justify-center', className)}
        style={aspect}
      >
        <span className="text-xs text-text-muted">Lỗi tải trang</span>
      </div>
    )
  }

  if (!url) {
    // Source-level loading already covers chapter-switch spinner.
    // Per-page placeholder is a quiet skeleton — no double spinner.
    return (
      <div
        className={cn('relative w-full bg-surface-2', className)}
        style={aspect}
        aria-busy="true"
      />
    )
  }

  return (
    <img
      src={url}
      alt={`Trang ${index + 1}`}
      width={page?.width   ?? undefined}
      height={page?.height ?? undefined}
      className={cn('block w-full h-auto select-none', className)}
      draggable={false}
    />
  )
}
