// LibraryStatusTabs — pill row to filter by reading status.

import { cn } from '@shared/lib/cn'
import type { LibraryStatus } from '@shared/db'

export type LibraryStatusOrAll = LibraryStatus | 'all'

const TABS: { id: LibraryStatusOrAll; label: string }[] = [
  { id: 'all',     label: 'Tất cả'    },
  { id: 'reading', label: 'Đang đọc'  },
  { id: 'plan',    label: 'Định đọc'  },
  { id: 'done',    label: 'Đã đọc'    },
  { id: 'dropped', label: 'Bỏ'        },
]

interface Props {
  value:    LibraryStatusOrAll
  onChange: (next: LibraryStatusOrAll) => void
  counts?:  Partial<Record<LibraryStatusOrAll, number>>
  className?: string
}

export function LibraryStatusTabs({ value, onChange, counts, className }: Props) {
  return (
    <div className={cn('flex flex-wrap gap-2', className)} role="tablist">
      {TABS.map(tab => {
        const count = counts?.[tab.id]
        const active = value === tab.id
        return (
          <button
            key={tab.id}
            role="tab"
            type="button"
            aria-selected={active}
            onClick={() => onChange(tab.id)}
            className={cn(
              'inline-flex items-center gap-2 h-7 px-3 rounded-full text-xs font-medium transition-colors',
              active
                ? 'bg-accent-bg text-accent-text'
                : 'bg-surface-2 text-text-muted hover:bg-hover hover:text-text',
            )}
          >
            {tab.label}
            {typeof count === 'number' && (
              <span className={cn(
                'tabular-nums',
                active ? 'text-accent-text/80' : 'text-text-subtle',
              )}>
                {count}
              </span>
            )}
          </button>
        )
      })}
    </div>
  )
}
