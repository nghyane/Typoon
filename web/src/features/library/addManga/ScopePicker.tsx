import { ChevronDown, Globe } from 'lucide-react'
import { useState } from 'react'
import { cn } from '@shared/lib/cn'
import type { InstalledSource } from '@features/browse/manifest/types'

// =============================================================================
// ScopePicker — inline source scope filter for SearchPane (Option B).
//
// Compact chip-style trigger that opens a popover menu. Sits right
// next to the input so the user can change scope without leaving the
// search row. Sources that don't support search appear disabled —
// hiding them would contradict the installed-source count in Settings.
//
// `lockedTo` puts the trigger into read-only mode while a URL paste
// is being resolved (the source is implied by the URL).
// =============================================================================

interface Props {
  sources:        InstalledSource[]
  searchableIds:  Set<string>
  value:          string | null      // null = all
  onChange:       (id: string | null) => void
  lockedTo:       string | null
}

export function ScopePicker({
  sources, searchableIds, value, onChange, lockedTo,
}: Props) {
  const [open, setOpen] = useState(false)

  if (lockedTo) {
    const src = sources.find((s) => s.manifest.id === lockedTo)
    return (
      <span className="inline-flex items-center gap-1.5 h-10 px-3 rounded-sm bg-surface-2 text-sm text-text-muted shrink-0">
        <Globe size={13} />
        <span className="max-w-[120px] truncate">
          {src?.manifest.name ?? lockedTo}
        </span>
      </span>
    )
  }

  const active = value === null
    ? null
    : sources.find((s) => s.manifest.id === value)

  return (
    <div className="relative shrink-0">
      <button
        type="button"
        onClick={() => setOpen(!open)}
        className={cn(
          'inline-flex items-center gap-1.5 h-10 px-3 rounded-sm',
          'bg-surface-2 text-sm text-text hover:bg-hover transition-colors cursor-pointer',
        )}
        title="Chọn nguồn để tìm"
      >
        <Globe size={13} className="text-text-subtle" />
        <span className="max-w-[120px] truncate">
          {active?.manifest.name ?? 'Tất cả'}
        </span>
        <ChevronDown
          size={12}
          className={cn(
            'text-text-subtle transition-transform',
            open && 'rotate-180',
          )}
        />
      </button>

      {open && (
        <>
          <div
            className="fixed inset-0 z-10"
            onMouseDown={() => setOpen(false)}
          />
          <Menu
            sources={sources}
            searchableIds={searchableIds}
            value={value}
            onPick={(id) => { onChange(id); setOpen(false) }}
          />
        </>
      )}
    </div>
  )
}


function Menu({
  sources, searchableIds, value, onPick,
}: {
  sources:       InstalledSource[]
  searchableIds: Set<string>
  value:         string | null
  onPick:        (id: string | null) => void
}) {
  return (
    <div
      role="listbox"
      className="absolute top-full mt-1 left-0 z-20 w-64 max-h-80 overflow-auto rounded-md bg-surface border border-border-soft shadow-lg py-1"
    >
      <MenuItem
        active={value === null}
        disabled={false}
        title="Tất cả"
        subtitle={`${searchableIds.size} nguồn tìm được`}
        onClick={() => onPick(null)}
      />
      <div className="my-1 mx-2 h-px bg-border-soft" />
      {sources.map((s) => {
        const searchable = searchableIds.has(s.manifest.id)
        return (
          <MenuItem
            key={s.manifest.id}
            active={value === s.manifest.id}
            disabled={!searchable}
            title={s.manifest.name}
            subtitle={searchable
              ? s.manifest.host
              : `${s.manifest.host} · chỉ dán link`
            }
            onClick={() => onPick(s.manifest.id)}
          />
        )
      })}
    </div>
  )
}


function MenuItem({
  active, disabled, title, subtitle, onClick,
}: {
  active:    boolean
  disabled:  boolean
  title:     string
  subtitle:  string
  onClick:   () => void
}) {
  return (
    <button
      type="button"
      onClick={onClick}
      disabled={disabled}
      className={cn(
        'w-full block px-3 py-1.5 text-left transition-colors',
        active
          ? 'bg-surface-2'
          : disabled
          ? 'cursor-not-allowed opacity-50'
          : 'hover:bg-hover cursor-pointer',
      )}
    >
      <div className={cn(
        'text-[13px] truncate',
        active ? 'text-text font-medium' : 'text-text-muted',
      )}>
        {title}
      </div>
      <div className="text-[11px] text-text-subtle truncate">
        {subtitle}
      </div>
    </button>
  )
}
