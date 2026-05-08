import { type ReactNode } from 'react'
import { type LucideIcon } from 'lucide-react'

// =============================================================================
// EmptyState — hiển thị khi list/table không có item nào.
// Dùng trong cả empty (no data ever) và no-result (filtered out).
// =============================================================================

interface Props {
  icon?:   LucideIcon
  title:   string
  hint?:   ReactNode
  action?: ReactNode
}

export function EmptyState({ icon: Icon, title, hint, action }: Props) {
  return (
    <div className="py-16 flex flex-col items-center text-center">
      {Icon && (
        <div className="size-12 rounded-md bg-surface-2 flex items-center justify-center mb-3">
          <Icon size={20} className="text-text-subtle" />
        </div>
      )}
      <p className="text-sm font-medium text-text">{title}</p>
      {hint && <p className="text-xs text-text-subtle mt-1 max-w-sm">{hint}</p>}
      {action && <div className="mt-4">{action}</div>}
    </div>
  )
}
