import { cn } from '@shared/lib/cn'
import type { ReaderPage } from '../sources'


interface Props {
  page?:      ReaderPage
  aspectRatio?: number | null
  className?: string
  busy?:      boolean
}


export function PagePlaceholder({ page, aspectRatio, className, busy = false }: Props) {
  const ratio = aspectRatio ?? (page?.width && page.height ? page.height / page.width : 1.5)
  return (
    <div
      className={cn('relative w-full bg-surface-2 overflow-hidden', className)}
      style={{ paddingTop: `${ratio * 100}%` }}
      aria-busy={busy || undefined}
      aria-hidden={!busy || undefined}
    >
      {busy && (
        <>
          <div className="absolute inset-0 animate-pulse bg-surface-2" />
          <span className="absolute bottom-3 right-3 size-2 rounded-full bg-text-subtle animate-pulse" />
        </>
      )}
    </div>
  )
}
