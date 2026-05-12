import { useState } from 'react'
import { ChevronDown, Globe, Check, Link as LinkIcon } from 'lucide-react'
import { cn } from '@shared/lib/cn'
import type { InstalledSource } from '@features/browse/manifest/types'

// =============================================================================
// SourcePicker — compact dropdown.
//
// Pro pattern (Linear / Raycast / GitHub command palette): a single
// trigger chip that opens a menu. Even with 2–3 sources, an inline
// chip row eats horizontal space that the search input needs.
//
// `searchableIds` flags which manifests expose a search endpoint.
// Sources outside this set still appear in the menu but render in a
// muted, non-selectable state with a hint pointing the user at URL
// paste instead.
//
// `lockedTo` puts the picker in read-only mode (used while a URL
// paste is being resolved — the source is implied by the URL).
// =============================================================================

interface Props {
  sources:        InstalledSource[]
  searchableIds:  Set<string>
  value:          string | null   // null = "all"
  onChange:       (id: string | null) => void
  lockedTo?:      string | null
}

export function SourcePicker({
  sources, searchableIds, value, onChange, lockedTo,
}: Props) {
  const [open, setOpen] = useState(false)

  if (lockedTo) {
    const src = sources.find((s) => s.manifest.id === lockedTo)
    return (
      <span className="inline-flex items-center gap-1.5 h-8 px-2.5 rounded-sm bg-surface-2 text-sm text-text-muted shrink-0">
        <Globe size={12} />
        <span className="max-w-[120px] truncate">
          {src?.manifest.name ?? lockedTo}
        </span>
      </span>
    )
  }

  const activeLabel = value === null
    ? 'Tất cả'
    : sources.find((s) => s.manifest.id === value)?.manifest.name ?? '?'

  return (
    <div className="relative shrink-0">
      <button
        type="button"
        onClick={() => setOpen(!open)}
        className={cn(
          'inline-flex items-center gap-1.5 h-8 px-2.5 rounded-sm',
          'bg-surface-2 text-sm text-text hover:bg-hover transition-colors cursor-pointer',
        )}
        title="Chọn nguồn để tìm"
      >
        <Globe size={12} className="text-text-subtle" />
        <span className="max-w-[120px] truncate">{activeLabel}</span>
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
          <div className="absolute top-full mt-1 left-0 z-20 min-w-[220px] rounded-sm bg-surface border border-border-soft shadow-lg overflow-hidden">
            <Option
              active={value === null}
              disabled={false}
              onClick={() => { onChange(null); setOpen(false) }}
            >
              <span className="flex-1 truncate">Tất cả nguồn</span>
            </Option>
            <div className="border-t border-border-soft" />
            {sources.map((s) => {
              const searchable = searchableIds.has(s.manifest.id)
              return (
                <Option
                  key={s.manifest.id}
                  active={value === s.manifest.id}
                  disabled={!searchable}
                  title={searchable
                    ? undefined
                    : `${s.manifest.name} chưa hỗ trợ tìm — dán link manga để thêm trực tiếp`}
                  onClick={() => {
                    if (!searchable) return
                    onChange(s.manifest.id); setOpen(false)
                  }}
                >
                  <span className="flex-1 truncate inline-flex items-center gap-1.5">
                    {!searchable && <LinkIcon size={10} className="text-text-subtle" />}
                    {s.manifest.name}
                  </span>
                  <span className="text-[11px] text-text-subtle ml-2 uppercase shrink-0">
                    {s.manifest.languages.slice(0, 3).join('/')}
                  </span>
                </Option>
              )
            })}
          </div>
        </>
      )}
    </div>
  )
}


function Option({
  active, disabled, title, onClick, children,
}: {
  active:   boolean
  disabled: boolean
  title?:   string
  onClick:  () => void
  children: React.ReactNode
}) {
  return (
    <button
      type="button"
      onClick={onClick}
      disabled={disabled}
      title={title}
      className={cn(
        'w-full flex items-center justify-between gap-2 h-8 px-3 text-sm text-left transition-colors',
        disabled
          ? 'text-text-subtle/60 cursor-not-allowed'
          : 'hover:bg-hover cursor-pointer',
        active && !disabled && 'bg-surface-2 text-text font-medium',
      )}
    >
      {children}
      {active && !disabled && (
        <Check size={13} className="text-success-text shrink-0" />
      )}
    </button>
  )
}
