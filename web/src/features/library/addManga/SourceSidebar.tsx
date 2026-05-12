import { Globe } from 'lucide-react'
import { cn } from '@shared/lib/cn'
import type { InstalledSource } from '@features/browse/manifest/types'
import { Favicon, CapabilityPills } from './SourceBits'

// =============================================================================
// SourceSidebar — vertical source navigator inside the modal.
//
// One row per source plus a top "Tất cả" row that selects cross-source
// fanout. Non-searchable sources stay visible but render disabled —
// hiding them would contradict the source count in Settings and
// confuse users who installed a source for paste-link only.
//
// When a URL paste matches a source, `lockedSourceId` puts that row
// in an emphasised state (filled background) so the user sees where
// the modal will route them. Other rows are still readable but the
// picker effectively ignores clicks.
// =============================================================================

interface Props {
  sources:        InstalledSource[]
  searchableIds:  Set<string>
  value:          string | null      // null = all
  onChange:       (id: string | null) => void
  lockedSourceId: string | null
}

export function SourceSidebar({
  sources, searchableIds, value, onChange, lockedSourceId,
}: Props) {
  const locked = lockedSourceId !== null
  const effective = lockedSourceId ?? value
  return (
    <aside
      className="w-44 shrink-0 border-r border-border-soft py-2 overflow-y-auto"
      aria-label="Chọn nguồn"
    >
      <Row
        active={effective === null}
        disabled={locked}
        onClick={() => onChange(null)}
        icon={<AllTile />}
        label="Tất cả"
        hint={`${searchableIds.size} tìm`}
      />
      <div className="my-1 mx-3 h-px bg-border-soft" />
      {sources.map((s) => {
        const searchable = searchableIds.has(s.manifest.id)
        const isLocked   = lockedSourceId === s.manifest.id
        const active     = effective === s.manifest.id
        return (
          <Row
            key={s.manifest.id}
            active={active}
            disabled={!searchable && !isLocked}
            onClick={() => {
              if (locked) return
              if (!searchable) return
              onChange(s.manifest.id)
            }}
            icon={<Favicon host={s.manifest.host} size={16} />}
            label={s.manifest.name}
            hint={<CapabilityPills searchable={searchable} />}
            titleOverride={searchable
              ? undefined
              : `${s.manifest.name} chưa hỗ trợ tìm theo tên — dán đường dẫn để thêm`
            }
          />
        )
      })}
    </aside>
  )
}


function Row({
  active, disabled, onClick, icon, label, hint, titleOverride,
}: {
  active:         boolean
  disabled:       boolean
  onClick:        () => void
  icon:           React.ReactNode
  label:          string
  hint:           React.ReactNode
  titleOverride?: string
}) {
  return (
    <button
      type="button"
      onClick={onClick}
      disabled={disabled}
      title={titleOverride}
      className={cn(
        'w-full flex items-center gap-2 h-9 px-3 text-left text-[13px] transition-colors',
        active
          ? 'bg-surface-2 text-text font-medium'
          : disabled
          ? 'text-text-subtle/60 cursor-not-allowed'
          : 'text-text-muted hover:bg-hover hover:text-text cursor-pointer',
      )}
    >
      <span className="shrink-0">{icon}</span>
      <span className="flex-1 truncate">{label}</span>
      <span className="shrink-0 text-[11px] text-text-subtle">
        {hint}
      </span>
    </button>
  )
}


function AllTile() {
  return (
    <span className="size-4 rounded-xs bg-surface-2 flex items-center justify-center">
      <Globe size={10} className="text-text-muted" />
    </span>
  )
}
