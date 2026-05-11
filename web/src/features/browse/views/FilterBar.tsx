import { useState, useEffect, useRef } from 'react'
import { ChevronDown, X } from 'lucide-react'
import { cn } from '@shared/lib/cn'
import type { Filter, SourceManifest } from '../manifest/types'

// =============================================================================
// FilterBar — renders manifest.filters[] as chip dropdowns.
// State is fully controlled — parent owns it and persists via URL or
// local storage as appropriate. `select` filters show the currently
// selected option's label; `multi` filters show the count of
// selections (or option labels if 1–2). Clicking opens a popover.
// =============================================================================

export type FilterState = Record<string, string | string[]>

interface Props {
  manifest: SourceManifest
  state:    FilterState
  onChange: (next: FilterState) => void
}

export function FilterBar({ manifest, state, onChange }: Props) {
  if (!manifest.filters) return null
  return (
    <div className="flex flex-wrap gap-2">
      {manifest.filters.map((f) => (
        <FilterChip
          key={f.id}
          filter={f}
          value={state[f.id]}
          onChange={(v) => onChange({ ...state, [f.id]: v })}
        />
      ))}
    </div>
  )
}

function FilterChip({
  filter, value, onChange,
}: {
  filter:   Filter
  value:    string | string[] | undefined
  onChange: (v: string | string[]) => void
}) {
  const [open, setOpen] = useState(false)
  const ref = useRef<HTMLDivElement>(null)
  useEffect(() => {
    if (!open) return
    const onDoc = (e: MouseEvent) => {
      if (ref.current && !ref.current.contains(e.target as Node)) setOpen(false)
    }
    document.addEventListener('mousedown', onDoc)
    return () => document.removeEventListener('mousedown', onDoc)
  }, [open])

  const label = displayLabel(filter, value)
  const active = filter.type === 'multi'
    ? Array.isArray(value) && value.length > 0
    : !!value && value !== filter.options[0]?.id

  return (
    <div className="relative" ref={ref}>
      <button
        onClick={() => setOpen(!open)}
        className={cn(
          'inline-flex items-center gap-1.5 h-7 px-2.5 rounded-sm text-xs font-medium cursor-pointer transition-colors',
          active
            ? 'bg-accent-bg text-accent-text'
            : 'bg-surface-2 text-text-muted hover:bg-hover hover:text-text',
        )}
      >
        <span>{filter.label}: <span className="font-semibold">{label}</span></span>
        <ChevronDown size={12} />
      </button>

      {open && (
        <div className="absolute z-30 top-full mt-1 min-w-48 max-w-72 max-h-72 overflow-y-auto rounded-md bg-surface shadow-[0_8px_24px_rgb(0,0,0,0.4)] py-1">
          {filter.options.map((opt) => (
            <FilterOptionRow
              key={opt.id}
              label={opt.label}
              selected={isSelected(filter, value, opt.id)}
              type={filter.type}
              onClick={() => {
                if (filter.type === 'select') {
                  onChange(opt.id)
                  setOpen(false)
                } else {
                  const cur = Array.isArray(value) ? value : []
                  onChange(
                    cur.includes(opt.id)
                      ? cur.filter((id) => id !== opt.id)
                      : [...cur, opt.id],
                  )
                }
              }}
            />
          ))}
          {filter.type === 'multi' && Array.isArray(value) && value.length > 0 && (
            <button
              onClick={() => { onChange([]); setOpen(false) }}
              className="w-full flex items-center gap-2 px-3 py-2 text-xs text-text-subtle hover:bg-hover cursor-pointer border-t border-border-soft mt-1"
            >
              <X size={12} />
              Xoá lựa chọn
            </button>
          )}
        </div>
      )}
    </div>
  )
}

function FilterOptionRow({
  label, selected, type, onClick,
}: {
  label:    string
  selected: boolean
  type:     'select' | 'multi'
  onClick:  () => void
}) {
  return (
    <button
      onClick={onClick}
      className={cn(
        'w-full flex items-center gap-2 px-3 py-1.5 text-xs cursor-pointer transition-colors',
        selected
          ? 'text-text bg-row-active'
          : 'text-text-muted hover:bg-hover hover:text-text',
      )}
    >
      {type === 'multi' && (
        <span
          className={cn(
            'inline-block size-3.5 rounded-xs border flex-none',
            selected
              ? 'bg-accent border-accent'
              : 'border-text-subtle/50',
          )}
        />
      )}
      <span className="flex-1 text-left truncate">{label}</span>
    </button>
  )
}

function displayLabel(filter: Filter, value: string | string[] | undefined): string {
  if (filter.type === 'select') {
    const id = typeof value === 'string' ? value : filter.options[0]?.id
    return filter.options.find((o) => o.id === id)?.label ?? '—'
  }
  const ids = Array.isArray(value) ? value : []
  if (ids.length === 0) return 'tất cả'
  if (ids.length <= 2) {
    return ids
      .map((id) => filter.options.find((o) => o.id === id)?.label ?? id)
      .join(', ')
  }
  return `${ids.length} mục`
}

function isSelected(
  filter: Filter,
  value: string | string[] | undefined,
  id: string,
): boolean {
  if (filter.type === 'select') {
    return (typeof value === 'string' ? value : filter.options[0]?.id) === id
  }
  return Array.isArray(value) && value.includes(id)
}
