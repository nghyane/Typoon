import { type ReactNode } from 'react'
import { cn } from '@shared/lib/cn'

// =============================================================================
// DataTable — shared shell cho list-style tables (chương, thuật ngữ, …).
// Wraps a <table> in `bg-surface rounded-md`. Header row uses `bg-surface-2`
// with uppercase tracking-wider 11px label.
// =============================================================================

export function DataTable({
  children, className,
}: {
  children: ReactNode
  className?: string
}) {
  return (
    <div className={cn('bg-surface rounded-md overflow-hidden', className)}>
      <table className="w-full text-sm">{children}</table>
    </div>
  )
}

export function Th({ children, className }: { children: ReactNode; className?: string }) {
  return (
    <th className={cn(
      'h-9 px-3 text-left align-middle text-[11px] font-semibold uppercase tracking-wider text-text-subtle',
      className,
    )}>
      {children}
    </th>
  )
}
