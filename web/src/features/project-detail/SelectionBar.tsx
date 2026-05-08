import { X, RefreshCw } from 'lucide-react'
import { card } from '@shared/ui/primitives'
import { Button } from '@shared/ui/Button'
import { cn } from '@shared/lib/cn'

interface Props {
  count:   number
  onClear: () => void
  onRedo:  () => void
}

export function SelectionBar({ count, onClear, onRedo }: Props) {
  return (
    <div className={cn(
      'fixed bottom-5 left-1/2 -translate-x-1/2 z-50 flex items-center gap-3',
      card,
      'pl-4 pr-2 py-2 shadow-[0_8px_32px_rgb(0,0,0,0.4)]',
    )}>
      <span className="text-sm text-text-muted tabular">
        <span className="text-text font-medium">{count}</span> chương đã chọn
      </span>
      <Button variant="primary" onClick={onRedo}>
        <RefreshCw size={14} />
        Dịch lại
      </Button>
      <Button variant="ghost" icon onClick={onClear} aria-label="Bỏ chọn">
        <X size={14} />
      </Button>
    </div>
  )
}
