import { type ReactNode } from 'react'
import { Search } from 'lucide-react'
import { cn } from '@shared/lib/cn'

// =============================================================================
// DataToolbar — toolbar đứng đầu list/table panels (Chương, Thuật ngữ, …).
// Layout: [filters/search bên trái] ······· [actions bên phải]
// =============================================================================

export function DataToolbar({
  children, right,
}: {
  /** filters, search (left side) */
  children: ReactNode
  /** primary actions (right side, end-aligned) */
  right?:   ReactNode
}) {
  return (
    <div className="flex items-center justify-between gap-3 mb-4">
      <div className="flex items-center gap-2 min-w-0 flex-1">{children}</div>
      {right && <div className="flex items-center gap-2 shrink-0">{right}</div>}
    </div>
  )
}

interface SearchProps {
  value:        string
  onChange:     (v: string) => void
  placeholder?: string
  className?:   string
}

export function SearchInput({
  value, onChange, placeholder = 'Tìm…', className,
}: SearchProps) {
  return (
    <label className={cn(
      'flex items-center gap-2 h-8 px-2.5 rounded-sm bg-surface-2',
      'hover:bg-hover focus-within:bg-surface-2 focus-within:ring-1 focus-within:ring-accent',
      'transition-colors cursor-text min-w-0',
      className ?? 'w-56',
    )}>
      <Search size={13} className="text-text-subtle shrink-0" />
      <input
        type="text"
        value={value}
        onChange={(e) => onChange(e.target.value)}
        placeholder={placeholder}
        className="flex-1 bg-transparent outline-none text-sm text-text placeholder:text-text-subtle min-w-0"
      />
    </label>
  )
}
