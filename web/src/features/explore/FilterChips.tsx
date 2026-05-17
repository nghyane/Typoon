// FilterChips — render filters của một nguồn thành chip row.
//
// Mỗi filter group xử lý theo logic:
//   - Options có nsfw=true  → standalone toggle chip (18+)
//   - Options còn lại       → popover checklist (trigger chip + dropdown)
//
// `select` type trong popover → radio indicator.
// `multi`  type trong popover → checkbox indicator.

import { useRef, useState, useCallback } from 'react'
import { Check, ChevronDown } from 'lucide-react'
import { cn } from '@shared/lib/cn'
import { Popover } from '@shared/ui/Popover'
import type { Filter, FilterOption } from '@features/browse/manifest/types'

interface Props {
  filters:  Filter[]
  state:    Record<string, string | string[]>
  onChange: (next: Record<string, string | string[]>) => void
}

export function FilterChips({ filters, state, onChange }: Props) {
  if (filters.length === 0) return null
  return (
    <>
      {filters.map((filter) => (
        <FilterGroup key={filter.id} filter={filter} state={state} onChange={onChange} />
      ))}
    </>
  )
}

// ── helpers ────────────────────────────────────────────────────────

function getActive(state: Record<string, string | string[]>, id: string): string[] {
  const v = state[id]
  if (!v) return []
  return Array.isArray(v) ? v : [v]
}

function applyToggle(
  state:  Record<string, string | string[]>,
  filter: Filter,
  optId:  string,
): Record<string, string | string[]> {
  const next = { ...state }
  if (filter.type === 'select') {
    if (state[filter.id] === optId) delete next[filter.id]
    else next[filter.id] = optId
  } else {
    const arr     = getActive(state, filter.id)
    const updated = arr.includes(optId) ? arr.filter((x) => x !== optId) : [...arr, optId]
    if (updated.length === 0) delete next[filter.id]
    else next[filter.id] = updated
  }
  return next
}

// ── FilterGroup ────────────────────────────────────────────────────
// Tách nsfw options ra, render riêng.

function FilterGroup({ filter, state, onChange }: {
  filter:   Filter
  state:    Record<string, string | string[]>
  onChange: (next: Record<string, string | string[]>) => void
}) {
  const nsfwOpts   = filter.options.filter((o) => o.nsfw)
  const normalOpts = filter.options.filter((o) => !o.nsfw)
  const active     = getActive(state, filter.id)

  return (
    <>
      {/* normal options → popover (chỉ hiện khi có ≥ 2 options thường) */}
      {normalOpts.length > 1 && (
        <PopoverGroup
          filter={filter}
          options={normalOpts}
          active={active}
          onChange={onChange}
          state={state}
        />
      )}

      {/* nsfw options → toggle chips riêng */}
      {nsfwOpts.map((opt) => (
        <NsfwChip
          key={opt.id}
          opt={opt}
          on={active.includes(opt.id)}
          onClick={() => onChange(applyToggle(state, filter, opt.id))}
        />
      ))}
    </>
  )
}

// ── NsfwChip — standalone 18+ toggle ──────────────────────────────

function NsfwChip({ opt, on, onClick }: {
  opt:     FilterOption
  on:      boolean
  onClick: () => void
}) {
  return (
    <button
      type="button"
      onClick={onClick}
      className={cn(
        'inline-flex items-center h-8 px-3 rounded-full text-sm font-medium',
        'whitespace-nowrap shrink-0 cursor-pointer transition-colors duration-150',
        on
          ? 'bg-error/15 text-error'
          : 'text-text-muted hover:text-text',
      )}
    >
      {opt.label}
    </button>
  )
}

// ── PopoverGroup — trigger chip + checklist ────────────────────────

function PopoverGroup({ filter, options, active, onChange, state }: {
  filter:   Filter
  options:  FilterOption[]
  active:   string[]
  onChange: (next: Record<string, string | string[]>) => void
  state:    Record<string, string | string[]>
}) {
  const [open, setOpen] = useState(false)
  const ref             = useRef<HTMLButtonElement>(null)
  const close           = useCallback(() => setOpen(false), [])

  // Chỉ đếm active trong nhóm options này (không tính nsfw)
  const activeNormal = active.filter((id) => options.some((o) => o.id === id))
  const hasActive    = activeNormal.length > 0

  let label = filter.label
  if (activeNormal.length === 1) {
    label = options.find((o) => o.id === activeNormal[0])?.label ?? filter.label
  } else if (activeNormal.length > 1) {
    label = `${activeNormal.length} thể loại`
  }

  return (
    <>
      <button
        ref={ref}
        type="button"
        onClick={() => setOpen((v) => !v)}
        className={cn(
          'inline-flex items-center gap-1 h-8 px-3 rounded-full text-sm font-medium',
          'whitespace-nowrap shrink-0 cursor-pointer transition-colors duration-150',
          hasActive
            ? 'text-text'
            : 'text-text-muted hover:text-text',
          open && 'text-text',
        )}
      >
        {label}
        <ChevronDown size={10} className={cn(
          'transition-transform duration-150', open && 'rotate-180',
        )} />
      </button>

      <Popover
        open={open}
        onClose={close}
        anchorRef={ref}
        align="start"
        minWidth={180}
        maxWidth={260}
      >
        <div className="py-1 max-h-72 overflow-y-auto">
          {options.map((opt) => {
            const checked = active.includes(opt.id)
            return (
              <button
                key={opt.id}
                type="button"
                onClick={() => onChange(applyToggle(state, filter, opt.id))}
                className={cn(
                  'w-full flex items-center gap-2.5 px-3 py-1.5 text-sm text-left',
                  'transition-colors duration-100 cursor-pointer',
                  checked ? 'text-text' : 'text-text-muted hover:text-text hover:bg-hover',
                )}
              >
                {filter.type === 'multi' ? (
                  <span className={cn(
                    'size-3.5 rounded-sm border flex items-center justify-center shrink-0 transition-colors',
                    checked ? 'bg-accent border-accent' : 'border-border-strong',
                  )}>
                    {checked && <Check size={9} strokeWidth={3} className="text-accent-fg" />}
                  </span>
                ) : (
                  <span className={cn(
                    'size-3.5 rounded-full border flex items-center justify-center shrink-0 transition-colors',
                    checked ? 'border-accent' : 'border-border-strong',
                  )}>
                    {checked && <span className="size-2 rounded-full bg-accent" />}
                  </span>
                )}
                {opt.label}
              </button>
            )
          })}
        </div>
      </Popover>
    </>
  )
}
