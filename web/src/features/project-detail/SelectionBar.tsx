import { Play, X } from 'lucide-react'
import { card } from '@shared/ui/primitives'
import { Button } from '@shared/ui/Button'
import { Spinner } from '@shared/ui/primitives'
import { cn } from '@shared/lib/cn'

interface Props {
  count:    number
  onClear:  () => void
  onStart:  () => void
  pending?: boolean
}

// Floating action bar that appears when ≥1 chapter is selected. The
// primary action used to be "Dịch lại" (redo) but redo only makes
// sense for finished/errored chapters; new uploads default to idle.
// "Bắt đầu dịch" covers both cases — the server filters non-idle ids
// out so a mixed selection still does the right thing.
export function SelectionBar({ count, onClear, onStart, pending }: Props) {
  return (
    <div className={cn(
      'fixed bottom-[calc(3.5rem+0.75rem)] sm:bottom-5 left-1/2 -translate-x-1/2 z-50 flex items-center gap-3',
      card,
      'pl-4 pr-2 py-2 shadow-[0_8px_32px_rgb(0,0,0,0.4)]',
    )}>
      <span className="text-sm text-text-muted tabular">
        <span className="text-text font-medium">{count}</span> chương đã chọn
      </span>
      <Button variant="primary" onClick={onStart} disabled={pending}>
        {pending ? <Spinner /> : <Play size={14} />}
        Bắt đầu dịch
      </Button>
      <Button variant="ghost" icon onClick={onClear} aria-label="Bỏ chọn">
        <X size={14} />
      </Button>
    </div>
  )
}
