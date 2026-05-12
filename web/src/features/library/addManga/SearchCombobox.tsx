import { forwardRef, useEffect, useRef, useState } from 'react'
import {
  Search, Link as LinkIcon, AlertTriangle, CheckCircle2, Globe,
  ChevronDown,
} from 'lucide-react'
import { input as inputCls } from '@shared/ui/primitives'
import { cn } from '@shared/lib/cn'
import type { InstalledSource } from '@features/browse/manifest/types'
import type { matchSource } from './parseUrl'

// =============================================================================
// SearchCombobox — input + scope chip + autocomplete suggestions.
//
// Pro pattern (Linear, Stripe Checkout, cmd-K palettes): a single
// input that lets the user scope the search via inline suggestions.
// While typing, an overlay menu appears below the input with two
// kinds of entries:
//
//   ① "Tìm 'naruto' trên Tất cả"             (default scope row)
//   ② "Tìm 'naruto' trên HappyMH"            (one per searchable source)
//
// Click a row → scope sets + menu closes + parent's `query` commits.
// No separate dropdown trigger — the input IS the dropdown trigger.
//
// The menu uses `position: absolute` inside a `position: relative`
// wrapper, so it never pushes the modal body. Wrapper z-index is
// elevated so the menu floats above subsequent content.
// =============================================================================

interface Props {
  /** The committed query — what fanoutSearch is currently running. */
  query:        string
  setQuery:     (q: string) => void

  /** Active scope. null = all searchable sources. */
  scopeId:      string | null
  setScopeId:   (id: string | null) => void

  /** All enabled sources. The combobox itself filters down to
   *  searchable ones for the suggestion list. */
  sources:      InstalledSource[]
  searchableSources: InstalledSource[]

  /** When the input matches a URL, scope picker is locked and the
   *  match badge replaces the suggestions menu. */
  isUrl:        boolean
  urlMatch:     ReturnType<typeof matchSource>
}

export function SearchCombobox({
  query, setQuery, scopeId, setScopeId,
  sources, searchableSources, isUrl, urlMatch,
}: Props) {
  const [draft, setDraft] = useState(query)
  const [open, setOpen]   = useState(false)
  const wrapperRef        = useRef<HTMLDivElement | null>(null)
  const inputRef          = useRef<HTMLInputElement | null>(null)

  // Keep draft in sync if parent resets it (e.g. modal close).
  useEffect(() => {
    if (query !== draft) setDraft(query)
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [query])

  // Close menu when clicking outside the combobox.
  useEffect(() => {
    if (!open) return
    const onDown = (e: MouseEvent) => {
      if (!wrapperRef.current) return
      if (!wrapperRef.current.contains(e.target as Node)) setOpen(false)
    }
    window.addEventListener('mousedown', onDown)
    return () => window.removeEventListener('mousedown', onDown)
  }, [open])

  const commit = (q: string, sid: string | null) => {
    setDraft(q)
    setQuery(q)
    setScopeId(sid)
    setOpen(false)
    inputRef.current?.focus()
  }

  const activeScope: InstalledSource | null = scopeId === null
    ? null
    : sources.find((s) => s.manifest.id === scopeId) ?? null

  return (
    <div ref={wrapperRef} className="relative">
      {/* Input row */}
      <div className="flex items-stretch gap-0">
        <ScopeChip
          locked={!!urlMatch}
          lockedSource={urlMatch ? urlMatch.source : null}
          activeScope={activeScope}
          totalSearchable={searchableSources.length}
          onClick={() => {
            if (urlMatch) return
            setOpen(!open)
            inputRef.current?.focus()
          }}
        />
        <InputCell
          ref={inputRef}
          draft={draft}
          isUrl={isUrl}
          urlMatch={urlMatch}
          onChange={(v) => {
            setDraft(v)
            setQuery(v)
            if (!isUrlLikeQuick(v)) setOpen(true)
          }}
          onFocus={() => {
            if (!isUrl) setOpen(true)
          }}
        />
      </div>

      {/* Suggestion overlay — only when typing text (not URL paste) */}
      {open && !isUrl && (
        <SuggestionsMenu
          query={draft}
          scopeId={scopeId}
          searchableSources={searchableSources}
          onPick={commit}
        />
      )}
    </div>
  )
}


// ── Scope chip on the left of the input ──────────────────────────────

function ScopeChip({
  locked, lockedSource, activeScope, totalSearchable, onClick,
}: {
  locked:          boolean
  lockedSource:    InstalledSource | null
  activeScope:     InstalledSource | null
  totalSearchable: number
  onClick:         () => void
}) {
  if (locked && lockedSource) {
    return (
      <span className="inline-flex items-center gap-1.5 h-10 px-3 rounded-l-sm bg-surface-2 border border-r-0 border-transparent text-sm text-text-muted shrink-0">
        <Globe size={13} />
        <span className="max-w-[120px] truncate">{lockedSource.manifest.name}</span>
      </span>
    )
  }
  return (
    <button
      type="button"
      onClick={onClick}
      className={cn(
        'inline-flex items-center gap-1.5 h-10 px-3 rounded-l-sm shrink-0',
        'bg-surface-2 text-sm text-text hover:bg-hover transition-colors cursor-pointer',
      )}
      title="Đổi nguồn tìm"
    >
      <Globe size={13} className="text-text-subtle" />
      <span className="max-w-[120px] truncate">
        {activeScope?.manifest.name ?? `Tất cả ${totalSearchable}`}
      </span>
      <ChevronDown size={12} className="text-text-subtle" />
    </button>
  )
}


// ── Input cell ───────────────────────────────────────────────────────

const InputCell = forwardRef<HTMLInputElement, {
  draft:    string
  isUrl:    boolean
  urlMatch: ReturnType<typeof matchSource>
  onChange: (v: string) => void
  onFocus:  () => void
}>(function InputCell({ draft, isUrl, urlMatch, onChange, onFocus }, ref) {
  const Icon = isUrl ? LinkIcon : Search
  return (
    <div className="relative flex-1 min-w-0">
      <Icon
        size={14}
        className="absolute left-3 top-1/2 -translate-y-1/2 text-text-subtle pointer-events-none"
      />
      <input
        ref={ref}
        autoFocus
        type="text"
        value={draft}
        onChange={(e) => onChange(e.target.value)}
        onFocus={onFocus}
        placeholder="Tìm tên truyện hoặc dán đường dẫn manga"
        className={cn(
          inputCls,
          'rounded-l-none h-10 pl-9',
          isUrl && (urlMatch ? 'pr-36' : 'pr-32'),
        )}
      />
      {isUrl && <UrlBadge urlMatch={urlMatch} />}
    </div>
  )
})


function UrlBadge({
  urlMatch,
}: {
  urlMatch: ReturnType<typeof matchSource>
}) {
  const base =
    'absolute right-2 top-1/2 -translate-y-1/2 inline-flex items-center gap-1 ' +
    'h-6 px-2 rounded-xs text-[11px] font-medium pointer-events-none'
  if (urlMatch) {
    return (
      <span className={cn(base, 'bg-success/15 text-success-text')}>
        <CheckCircle2 size={10} />
        {urlMatch.source.manifest.name}
      </span>
    )
  }
  return (
    <span className={cn(base, 'bg-warning/15 text-warning-text')}>
      <AlertTriangle size={10} />
      Chưa hỗ trợ
    </span>
  )
}


// ── Suggestions menu ─────────────────────────────────────────────────

function SuggestionsMenu({
  query, scopeId, searchableSources, onPick,
}: {
  query:             string
  scopeId:           string | null
  searchableSources: InstalledSource[]
  onPick:            (q: string, sid: string | null) => void
}) {
  const q = query.trim()
  if (q.length === 0 && scopeId === null) return null

  return (
    <div
      role="listbox"
      className={cn(
        'absolute top-full left-0 right-0 mt-1 z-50',
        'rounded-md bg-surface border border-border-soft shadow-lg overflow-hidden',
        'max-h-72 overflow-y-auto py-1',
      )}
    >
      <SuggestionRow
        active={scopeId === null}
        primary={q ? `Tìm "${q}"` : 'Tìm trên tất cả nguồn'}
        secondary={`Tất cả · ${searchableSources.length} nguồn fanout`}
        onClick={() => onPick(q, null)}
      />
      {searchableSources.length > 0 && (
        <div className="my-1 mx-2 h-px bg-border-soft" />
      )}
      {searchableSources.map((s) => {
        const active = scopeId === s.manifest.id
        return (
          <SuggestionRow
            key={s.manifest.id}
            active={active}
            primary={q ? `Tìm "${q}" trên ${s.manifest.name}` : `Chỉ tìm trên ${s.manifest.name}`}
            secondary={s.manifest.host}
            onClick={() => onPick(q, s.manifest.id)}
          />
        )
      })}
    </div>
  )
}


function SuggestionRow({
  active, primary, secondary, onClick,
}: {
  active:    boolean
  primary:   string
  secondary: string
  onClick:   () => void
}) {
  return (
    <button
      type="button"
      onClick={onClick}
      className={cn(
        'w-full block px-3 py-1.5 text-left transition-colors cursor-pointer',
        active ? 'bg-surface-2' : 'hover:bg-hover',
      )}
    >
      <div className={cn(
        'text-[13px] truncate',
        active ? 'text-text font-medium' : 'text-text-muted',
      )}>
        {primary}
      </div>
      <div className="text-[11px] text-text-subtle truncate">
        {secondary}
      </div>
    </button>
  )
}


// Quick URL detection without re-importing parseUrl to avoid a cycle.
function isUrlLikeQuick(s: string): boolean {
  return /^https?:\/\//i.test(s.trim())
}
