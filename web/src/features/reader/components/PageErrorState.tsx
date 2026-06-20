import { AlertCircle } from 'lucide-react'

import { Button } from '@shared/ui/Button'
import { cn } from '@shared/lib/cn'
import type { ReaderPage } from '../sources'


interface Props {
  page?:       ReaderPage
  aspectRatio?: number | null
  error:       Error | null
  onRetry:     () => void
  className?:  string
}


export function PageErrorState({ page, aspectRatio, error, onRetry, className }: Props) {
  const ratio = aspectRatio ?? (page?.width && page.height ? page.height / page.width : 1.5)
  return (
    <div
      className={cn('relative w-full bg-surface-2', className)}
      style={{ paddingTop: `${ratio * 100}%` }}
    >
      <div className="absolute inset-0 flex flex-col items-center justify-center gap-3 px-4 text-center">
        <AlertCircle size={18} className="text-text-subtle" />
        <div className="space-y-1">
          <p className="text-sm font-medium text-text-muted">Không tải được trang</p>
          {error?.message && (
            <p className="max-w-xs truncate text-xs text-text-subtle">{error.message}</p>
          )}
        </div>
        <Button
          variant="secondary"
          size="sm"
          onClick={(event) => {
            event.stopPropagation()
            onRetry()
          }}
        >
          Thử lại
        </Button>
      </div>
    </div>
  )
}
