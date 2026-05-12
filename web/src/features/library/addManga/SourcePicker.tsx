import { useState } from 'react'
import { ChevronDown, Globe, Check, Link as LinkIcon, Search } from 'lucide-react'
import { cn } from '@shared/lib/cn'
import type { InstalledSource } from '@features/browse/manifest/types'

// =============================================================================
// SourcePicker — compact dropdown with rich option cards.
//
// Each option is 2 lines: name + capability pills on the top row,
// host + language tag on the bottom row. A 24px favicon on the left
// gives the source identity at a glance. We pull favicons from
// Google's S2 endpoint — same as Chrome/Firefox URL bar — so no
// server-side asset pipeline is required.
//
// `searchableIds` flags manifests with a search endpoint. URL paste
// works on every enabled source, so it's always-on in the legend.
// `lockedTo` puts the picker in read-only mode (URL paste implies
// the source).
// =============================================================================

interface Props {
  sources:        InstalledSource[]
  searchableIds:  Set<string>
  value:          string | null   // null = "all"
  onChange:       (id: string | null) => void
  lockedTo?:      string | null
}

const FAVICON = (host: string) =>
  `https://www.google.com/s2/favicons?domain=${encodeURIComponent(host)}&sz=64`


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

  const active = value === null
    ? null
    : sources.find((s) => s.manifest.id === value)

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
        {active
          ? <Favicon host={active.manifest.host} size={14} />
          : <Globe size={12} className="text-text-subtle" />
        }
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
          <div
            role="listbox"
            className="absolute top-full mt-1 left-0 z-20 w-[320px] max-h-[360px] overflow-auto rounded-md bg-surface border border-border-soft shadow-lg"
          >
            <AllOption
              active={value === null}
              total={sources.length}
              searchable={searchableIds.size}
              onClick={() => { onChange(null); setOpen(false) }}
            />
            <div className="h-px bg-border-soft" />
            {sources.map((s) => (
              <SourceOption
                key={s.manifest.id}
                source={s}
                searchable={searchableIds.has(s.manifest.id)}
                active={value === s.manifest.id}
                onClick={() => {
                  if (!searchableIds.has(s.manifest.id)) return
                  onChange(s.manifest.id); setOpen(false)
                }}
              />
            ))}
          </div>
        </>
      )}
    </div>
  )
}


// ── All sources row ─────────────────────────────────────────────────

function AllOption({
  active, total, searchable, onClick,
}: {
  active:     boolean
  total:      number
  searchable: number
  onClick:    () => void
}) {
  return (
    <button
      type="button"
      onClick={onClick}
      className={cn(
        'w-full flex items-center gap-3 px-3 py-2.5 text-left transition-colors cursor-pointer',
        'hover:bg-hover',
        active && 'bg-surface-2',
      )}
    >
      <span className="size-8 rounded-sm bg-surface-2 flex items-center justify-center shrink-0">
        <Globe size={14} className="text-text-muted" />
      </span>
      <span className="flex-1 min-w-0">
        <span className="block text-sm text-text font-medium">Tất cả nguồn</span>
        <span className="block text-[11px] text-text-subtle mt-0.5">
          {searchable} tìm được · {total} dán link
        </span>
      </span>
      {active && <Check size={13} className="text-success-text shrink-0" />}
    </button>
  )
}


// ── Source row ──────────────────────────────────────────────────────

function SourceOption({
  source, searchable, active, onClick,
}: {
  source:     InstalledSource
  searchable: boolean
  active:     boolean
  onClick:    () => void
}) {
  const { manifest } = source
  return (
    <button
      type="button"
      onClick={onClick}
      disabled={!searchable}
      title={searchable
        ? undefined
        : `${manifest.name} chưa hỗ trợ tìm — dán đường dẫn để thêm`}
      className={cn(
        'w-full flex items-center gap-3 px-3 py-2.5 text-left transition-colors',
        searchable
          ? 'hover:bg-hover cursor-pointer'
          : 'cursor-not-allowed opacity-60',
        active && searchable && 'bg-surface-2',
      )}
    >
      <Favicon host={manifest.host} size={28} />
      <span className="flex-1 min-w-0">
        <span className="flex items-center gap-1.5">
          <span className="text-sm text-text font-medium truncate">
            {manifest.name}
          </span>
          <CapPill kind="search" active={searchable} />
          <CapPill kind="link"   active={true} />
        </span>
        <span className="block text-[11px] text-text-subtle mt-0.5 truncate">
          {manifest.host}
          {manifest.languages.length > 0 && (
            <span className="uppercase ml-1.5">
              · {manifest.languages.slice(0, 3).join('/')}
            </span>
          )}
        </span>
      </span>
      {active && searchable && (
        <Check size={13} className="text-success-text shrink-0" />
      )}
    </button>
  )
}


// ── Bits ────────────────────────────────────────────────────────────

function Favicon({ host, size }: { host: string; size: number }) {
  return (
    <span
      className="rounded-sm bg-surface-2 overflow-hidden flex items-center justify-center shrink-0"
      style={{ width: size, height: size }}
    >
      <img
        src={FAVICON(host)}
        alt=""
        width={size}
        height={size}
        loading="lazy"
        onError={(e) => {
          // Fallback to the host's first letter when Google's S2 has
          // no record. Avoid stretched broken-image icons in the menu.
          const el = e.currentTarget
          el.style.display = 'none'
          el.parentElement?.classList.add('font-bold', 'text-text-muted', 'text-[11px]')
          if (el.parentElement) el.parentElement.textContent = host[0]?.toUpperCase() ?? '?'
        }}
        className="w-full h-full object-contain"
      />
    </span>
  )
}


function CapPill({
  kind, active,
}: {
  kind: 'search' | 'link'; active: boolean
}) {
  const Icon = kind === 'search' ? Search : LinkIcon
  return (
    <span
      title={kind === 'search'
        ? (active ? 'Hỗ trợ tìm theo tên' : 'Chưa hỗ trợ tìm')
        : 'Hỗ trợ dán đường dẫn'
      }
      className={cn(
        'inline-flex items-center justify-center size-4 rounded-xs shrink-0',
        active ? 'bg-surface-2 text-text-muted' : 'text-text-subtle/40',
      )}
    >
      <Icon size={9} />
    </span>
  )
}
