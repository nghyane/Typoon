import { useState } from 'react'
import { ChevronDown, Globe, Check, Link as LinkIcon } from 'lucide-react'
import { cn } from '@shared/lib/cn'
import type { InstalledSource } from '@features/browse/manifest/types'

// =============================================================================
// SourcePicker — chip row when ≤4 enabled sources, dropdown when more.
//
// `searchableIds` flags which manifests expose a search endpoint.
// Sources outside this set still appear in the picker but render in
// a muted, non-selectable state with a hint that they require a URL
// paste instead. We don't hide them — that contradicts the "X nguồn
// đã cài" badge in settings and made users wonder why some sources
// vanished.
//
// `lockedTo` puts the picker in disabled mode (used while a URL paste
// is being resolved — the source is implied by the URL, switching
// would lose context).
// =============================================================================

const CHIP_THRESHOLD = 4

interface Props {
  sources:        InstalledSource[]
  searchableIds:  Set<string>
  value:          string | null   // null = "all"
  onChange:       (id: string | null) => void
  lockedTo?:      string | null
}

export function SourcePicker(props: Props) {
  if (props.lockedTo) {
    const src = props.sources.find((s) => s.manifest.id === props.lockedTo)
    return (
      <span className="inline-flex items-center gap-1.5 h-8 px-2.5 rounded-sm bg-surface-2 text-sm text-text-muted shrink-0">
        <Globe size={12} />
        {src?.manifest.name ?? props.lockedTo}
      </span>
    )
  }
  return props.sources.length > CHIP_THRESHOLD
    ? <DropdownPicker {...props} />
    : <ChipRowPicker  {...props} />
}


function ChipRowPicker({
  sources, searchableIds, value, onChange,
}: Props) {
  return (
    <div className="flex items-center gap-1 shrink-0 overflow-x-auto">
      <Chip active={value === null} disabled={false} onClick={() => onChange(null)}>
        Tất cả
      </Chip>
      {sources.map((s) => {
        const searchable = searchableIds.has(s.manifest.id)
        return (
          <Chip
            key={s.manifest.id}
            active={value === s.manifest.id}
            disabled={!searchable}
            title={searchable
              ? undefined
              : `${s.manifest.name} chưa hỗ trợ tìm kiếm — dán đường dẫn manga để thêm trực tiếp`}
            onClick={() => onChange(s.manifest.id)}
          >
            {!searchable && <LinkIcon size={10} />}
            {s.manifest.name}
          </Chip>
        )
      })}
    </div>
  )
}


function Chip({
  active, disabled, title, onClick, children,
}: {
  active:    boolean
  disabled:  boolean
  title?:    string
  onClick:   () => void
  children:  React.ReactNode
}) {
  return (
    <button
      type="button"
      onClick={onClick}
      disabled={disabled}
      title={title}
      className={cn(
        'inline-flex items-center gap-1 h-8 px-2.5 rounded-sm text-[13px] font-medium transition-colors shrink-0',
        disabled
          ? 'bg-surface-2 text-text-subtle/60 cursor-not-allowed'
          : active
          ? 'bg-text text-bg cursor-pointer'
          : 'bg-surface-2 text-text-muted hover:bg-hover hover:text-text cursor-pointer',
      )}
    >
      {children}
    </button>
  )
}


function DropdownPicker({
  sources, searchableIds, value, onChange,
}: Props) {
  const [open, setOpen] = useState(false)
  const label = value === null
    ? 'Tất cả nguồn'
    : sources.find((s) => s.manifest.id === value)?.manifest.name ?? '?'
  return (
    <div className="relative shrink-0">
      <button
        type="button"
        onClick={() => setOpen(!open)}
        className="inline-flex items-center gap-1.5 h-8 px-2.5 rounded-sm bg-surface-2 text-sm text-text hover:bg-hover transition-colors cursor-pointer"
      >
        <Globe size={12} className="text-text-subtle" />
        <span className="max-w-[140px] truncate">{label}</span>
        <ChevronDown size={12} className="text-text-subtle" />
      </button>
      {open && (
        <>
          <div
            className="fixed inset-0 z-10"
            onMouseDown={() => setOpen(false)}
          />
          <div className="absolute top-full mt-1 right-0 z-20 min-w-[220px] rounded-sm bg-surface-2 border border-border-soft shadow-lg overflow-hidden">
            <Option
              active={value === null}
              disabled={false}
              onClick={() => { onChange(null); setOpen(false) }}
            >
              Tất cả nguồn
            </Option>
            <div className="border-t border-border-soft" />
            {sources.map((s) => {
              const searchable = searchableIds.has(s.manifest.id)
              return (
                <Option
                  key={s.manifest.id}
                  active={value === s.manifest.id}
                  disabled={!searchable}
                  onClick={() => { onChange(s.manifest.id); setOpen(false) }}
                  title={searchable
                    ? undefined
                    : 'Chưa hỗ trợ tìm — dán đường dẫn manga để thêm'}
                >
                  <span className="flex-1 truncate inline-flex items-center gap-1.5">
                    {!searchable && <LinkIcon size={10} />}
                    {s.manifest.name}
                  </span>
                  <span className="text-[11px] text-text-subtle ml-2 uppercase">
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
        active && !disabled && 'text-text font-medium',
      )}
    >
      {children}
      {active && !disabled && (
        <Check size={13} className="text-success-text shrink-0" />
      )}
    </button>
  )
}
