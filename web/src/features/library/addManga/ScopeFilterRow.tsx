// ScopeFilterRow — source tabs that appear once a query returns hits.
//
// Tab-style filter, not a dropdown. Each tab carries the per-source
// result count so the user sees the distribution at a glance and
// picks where to look. "Tất cả" (null) is the leftmost tab.

import { cn } from '@shared/lib/cn'
import type { InstalledSource } from '@features/browse/manifest/types'
import type { SearchHit } from './fanoutSearch'

interface Props {
  hits:              SearchHit[]
  searchableSources: InstalledSource[]
  scopeId:           string | null
  onChange:          (id: string | null) => void
}

export function ScopeFilterRow({
  hits, searchableSources, scopeId, onChange,
}: Props) {
  if (hits.length === 0) return null

  const counts = new Map<string, number>()
  for (const h of hits) {
    const id = h.source.manifest.id
    counts.set(id, (counts.get(id) ?? 0) + 1)
  }
  const withHits = searchableSources.filter(s => counts.has(s.manifest.id))
  if (withHits.length <= 1 && scopeId === null) {
    // Single source returned everything — tab row would be redundant.
    return null
  }

  return (
    <div
      className="flex items-center gap-1 overflow-x-auto px-0.5"
      style={{ scrollbarWidth: 'none' }}
    >
      <Tab
        active={scopeId === null}
        label="Tất cả"
        count={hits.length}
        onClick={() => onChange(null)}
      />
      {withHits.map(s => (
        <Tab
          key={s.manifest.id}
          active={scopeId === s.manifest.id}
          label={s.manifest.name}
          count={counts.get(s.manifest.id) ?? 0}
          onClick={() => onChange(s.manifest.id)}
        />
      ))}
    </div>
  )
}


function Tab({
  active, label, count, onClick,
}: {
  active:  boolean
  label:   string
  count:   number
  onClick: () => void
}) {
  return (
    <button
      type="button"
      onClick={onClick}
      className={cn(
        'inline-flex items-center gap-2 h-8 px-3 rounded-sm text-sm shrink-0',
        'transition-colors',
        active
          ? 'bg-surface-2 text-text font-medium'
          : 'text-text-muted hover:bg-hover hover:text-text',
      )}
    >
      {label}
      <span className={cn(
        'text-xs tabular-nums',
        active ? 'text-text-subtle' : 'text-text-subtle/70',
      )}>
        {count}
      </span>
    </button>
  )
}
