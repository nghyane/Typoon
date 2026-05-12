import { useMemo, useState } from 'react'
import { Search, Link as LinkIcon, AlertTriangle, CheckCircle2 } from 'lucide-react'
import { input as inputCls } from '@shared/ui/primitives'
import { cn } from '@shared/lib/cn'
import { useDebouncedValue } from '@shared/lib/useDebouncedValue'
import { hasSearch } from '@features/browse/manifest/runtime'
import { useAllSources, useSources } from '@features/browse/sources'
import type { InstalledSource } from '@features/browse/manifest/types'
import { isUrlLike, matchSource } from './parseUrl'
import { useFanoutSearch } from './fanoutSearch'
import { UrlPasteCard } from './UrlPasteCard'
import { ResultsList } from './ResultsList'
import { ManualCreateRow } from './ManualCreateRow'
import { ScopeFilterRow } from './ScopeFilterRow'
import type { Picked } from './types'

// =============================================================================
// SearchPane — single column, three vertical zones:
//
//   ① Input          full width. Detects URL paste; renders the
//                    'HappyMH ✓' badge inline when matched.
//   ② Scope tabs     'Tất cả N · HappyMH 5 · MangaDex 7'. Only
//                    appears after a query returns results — no
//                    upfront scope decision required.
//   ③ Body           URL card / empty hint / scoped results list.
//
// No overlays, no dropdowns, no chip-next-to-input. The three zones
// flow top to bottom inside the modal body; modal height never
// shifts because of menu state.
// =============================================================================

interface Props {
  query:          string
  setQuery:       (s: string) => void
  sources:        InstalledSource[]
  onPick:         (p: Picked) => void
  onManualCreate: (seed: string) => void
}

export function SearchPane({
  query, setQuery, sources, onPick, onManualCreate,
}: Props) {
  const searchable = useMemo(
    () => sources.filter((s) => hasSearch(s.manifest)),
    [sources],
  )
  const urlMatch = useMemo(
    () => isUrlLike(query) ? matchSource(query, sources) : null,
    [query, sources],
  )
  const isUrl = isUrlLike(query)

  // Local scope state — resets implicitly whenever the user changes
  // the query (cleared via setQuery).
  const [scopeId, setScopeId] = useState<string | null>(null)

  // Fanout always queries every searchable source; scope filter is
  // applied client-side over the merged hit list. That way the user
  // can switch source without re-running the request.
  // Debounce the query before passing to the network. Each keystroke
  // updates the input synchronously (no laggy typing) but fanout only
  // fires after 250ms of stability — fewer requests, no flicker.
  const debouncedQuery = useDebouncedValue(query, 250)
  const { hits, loading, failures } = useFanoutSearch(debouncedQuery, searchable)

  const scopedHits = useMemo(() => {
    if (scopeId === null) return hits
    return hits.filter((h) => h.source.manifest.id === scopeId)
  }, [hits, scopeId])

  const visibleSources = useMemo(() => {
    if (scopeId === null) return searchable
    return searchable.filter((s) => s.manifest.id === scopeId)
  }, [searchable, scopeId])

  return (
    <div className="space-y-3 min-h-[420px]">
      <InputRow
        query={query}
        setQuery={(v) => { setQuery(v); setScopeId(null) }}
        isUrl={isUrl}
        urlMatch={urlMatch}
      />

      {isUrl ? (
        <UrlPasteCard
          url={query}
          match={urlMatch}
          onPick={onPick}
          onManualCreate={onManualCreate}
        />
      ) : debouncedQuery.trim().length < 2 ? (
        <SourceListHint />
      ) : (
        <>
          <ScopeFilterRow
            hits={hits}
            searchableSources={searchable}
            scopeId={scopeId}
            onChange={setScopeId}
          />
          <ResultsList
            query={query}
            hits={scopedHits}
            loading={loading}
            failures={failures}
            searchableSources={visibleSources}
            onPick={onPick}
          />
          <ManualCreateRow
            query={query}
            hits={scopedHits.length}
            onManualCreate={onManualCreate}
          />
        </>
      )}
    </div>
  )
}


function InputRow({
  query, setQuery, isUrl, urlMatch,
}: {
  query:    string
  setQuery: (v: string) => void
  isUrl:    boolean
  urlMatch: ReturnType<typeof matchSource>
}) {
  const Icon = isUrl ? LinkIcon : Search
  return (
    <div className="relative">
      <Icon
        size={14}
        className="absolute left-3 top-1/2 -translate-y-1/2 text-text-subtle pointer-events-none"
      />
      <input
        autoFocus
        type="text"
        value={query}
        onChange={(e) => setQuery(e.target.value)}
        placeholder="Tìm tên truyện hoặc dán đường dẫn manga"
        className={cn(
          inputCls, 'pl-9 h-10',
          isUrl && (urlMatch ? 'pr-36' : 'pr-32'),
        )}
      />
      {isUrl && <UrlBadge urlMatch={urlMatch} />}
    </div>
  )
}


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


// Empty-state hint. Shows the source roster as chips so the user
// knows what the app can reach without reading instructions. Each
// chip carries a Link icon when the source supports search ('Tìm
// được') vs a muted state for paste-only sources.

// Domain chip list — empty-state replacement for hint text. Each
// installed source becomes a toggle chip. Active = included in
// fanout search. Disabled chips dim to text-subtle. Chip itself is
// the affordance: name + host text reads as a domain, click toggles
// inclusion. The user can also paste a URL from any of those hosts
// regardless of the toggle — paste mode always works.

function SourceListHint() {
  const sources    = useAllSources()
  const setEnabled = useSources((s) => s.setEnabled)

  if (sources.length === 0) {
    return (
      <div className="rounded-md bg-surface-2 border border-dashed border-border-soft px-4 py-6 text-center">
        <p className="text-sm text-text-muted">Chưa cài nguồn nào</p>
        <p className="text-[11px] text-text-subtle mt-1">
          Mở Cài đặt để cài nguồn đầu tiên.
        </p>
      </div>
    )
  }
  return (
    <div className="space-y-2">
      <p className="text-[11px] text-text-subtle px-0.5">
        Bấm để bật/tắt nguồn cho fanout search
      </p>
      <ul className="flex flex-wrap gap-1.5">
        {sources.map((s) => {
          const searchable = hasSearch(s.manifest)
          const on = s.enabled && searchable
          return (
            <li key={s.manifest.id}>
              <button
                type="button"
                onClick={() => searchable && setEnabled(s.manifest.id, !s.enabled)}
                disabled={!searchable}
                title={searchable
                  ? (on ? `Tắt ${s.manifest.name}` : `Bật ${s.manifest.name}`)
                  : `${s.manifest.name} chưa hỗ trợ tìm — dán link để thêm`
                }
                className={cn(
                  'inline-flex items-center gap-2 h-8 pl-2 pr-3 rounded-sm text-[12px] transition-colors',
                  !searchable
                    ? 'bg-surface-2 text-text-subtle cursor-not-allowed border border-border-soft opacity-50'
                    : on
                    ? 'bg-accent/15 text-text border border-accent/30 hover:bg-accent/20 cursor-pointer'
                    : 'bg-surface-2 text-text-muted border border-border-soft hover:bg-hover hover:text-text cursor-pointer',
                )}
              >
                <span
                  className={cn(
                    'size-1.5 rounded-full shrink-0',
                    on
                      ? 'bg-accent'
                      : searchable
                      ? 'bg-text-subtle/40'
                      : 'bg-text-subtle/20',
                  )}
                />
                <span className="font-medium truncate max-w-[140px]">
                  {s.manifest.name}
                </span>
                <span className="text-[11px] text-text-subtle truncate">
                  {s.manifest.host}
                </span>
              </button>
            </li>
          )
        })}
      </ul>
    </div>
  )
}
