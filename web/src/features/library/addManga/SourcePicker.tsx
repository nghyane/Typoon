import { useState } from 'react'
import { ChevronDown, Globe, Check } from 'lucide-react'
import { cn } from '@shared/lib/cn'
import type { InstalledSource } from '@features/browse/manifest/types'

// =============================================================================
// SourcePicker — chip row when ≤4 enabled sources, dropdown when more.
//
// Pro pattern (Linear / Raycast): a "Tất cả" option is always present
// first, then each source. Active state inverts the chip so the
// current scope is unmistakable at a glance.
//
// `lockedTo` puts the picker in disabled mode (used while a URL paste
// is being resolved — the source is implied by the URL, switching
// would lose context).
// =============================================================================

const CHIP_THRESHOLD = 4

interface Props {
  sources:    InstalledSource[]
  value:      string | null   // null = "all"
  onChange:   (id: string | null) => void
  lockedTo?:  string | null
}

export function SourcePicker({ sources, value, onChange, lockedTo }: Props) {
  if (lockedTo) {
    const src = sources.find((s) => s.manifest.id === lockedTo)
    return (
      <span className="inline-flex items-center gap-1.5 h-8 px-2.5 rounded-sm bg-surface-2 text-sm text-text-muted shrink-0">
        <Globe size={12} />
        {src?.manifest.name ?? lockedTo}
      </span>
    )
  }

  return sources.length > CHIP_THRESHOLD
    ? <DropdownPicker  sources={sources} value={value} onChange={onChange} />
    : <ChipRowPicker   sources={sources} value={value} onChange={onChange} />
}


function ChipRowPicker({
  sources, value, onChange,
}: { sources: InstalledSource[]; value: string | null; onChange: (id: string | null) => void }) {
  return (
    <div className="flex items-center gap-1 shrink-0 overflow-x-auto">
      <Chip active={value === null} onClick={() => onChange(null)}>
        Tất cả
      </Chip>
      {sources.map((s) => (
        <Chip
          key={s.manifest.id}
          active={value === s.manifest.id}
          onClick={() => onChange(s.manifest.id)}
        >
          {s.manifest.name}
        </Chip>
      ))}
    </div>
  )
}


function Chip({
  active, onClick, children,
}: { active: boolean; onClick: () => void; children: React.ReactNode }) {
  return (
    <button
      type="button"
      onClick={onClick}
      className={cn(
        'h-8 px-2.5 rounded-sm text-[13px] font-medium transition-colors cursor-pointer shrink-0',
        active
          ? 'bg-text text-bg'
          : 'bg-surface-2 text-text-muted hover:bg-hover hover:text-text',
      )}
    >
      {children}
    </button>
  )
}


function DropdownPicker({
  sources, value, onChange,
}: { sources: InstalledSource[]; value: string | null; onChange: (id: string | null) => void }) {
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
          <div className="absolute top-full mt-1 right-0 z-20 min-w-[200px] rounded-sm bg-surface-2 border border-border-soft shadow-lg overflow-hidden">
            <Option
              active={value === null}
              onClick={() => { onChange(null); setOpen(false) }}
            >
              Tất cả nguồn
            </Option>
            <div className="border-t border-border-soft" />
            {sources.map((s) => (
              <Option
                key={s.manifest.id}
                active={value === s.manifest.id}
                onClick={() => { onChange(s.manifest.id); setOpen(false) }}
              >
                {s.manifest.name}
                <span className="text-[11px] text-text-subtle ml-2 uppercase">
                  {s.manifest.languages.slice(0, 3).join('/')}
                </span>
              </Option>
            ))}
          </div>
        </>
      )}
    </div>
  )
}


function Option({
  active, onClick, children,
}: { active: boolean; onClick: () => void; children: React.ReactNode }) {
  return (
    <button
      type="button"
      onClick={onClick}
      className={cn(
        'w-full flex items-center justify-between gap-2 h-8 px-3 text-sm text-left',
        'hover:bg-hover transition-colors cursor-pointer',
        active && 'text-text font-medium',
      )}
    >
      <span className="flex-1 truncate">{children}</span>
      {active && <Check size={13} className="text-success-text shrink-0" />}
    </button>
  )
}
