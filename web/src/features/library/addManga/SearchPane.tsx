import { useMemo, useState } from 'react'
import { Search, Link as LinkIcon, AlertTriangle, CheckCircle2 } from 'lucide-react'
import { input as inputCls } from '@shared/ui/primitives'
import { cn } from '@shared/lib/cn'
import { hasSearch } from '@features/browse/manifest/runtime'
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
  const { hits, loading, failures } = useFanoutSearch(query, searchable)

  const scopedHits = useMemo(() => {
    if (scopeId === null) return hits
    return hits.filter((h) => h.source.manifest.id === scopeId)
  }, [hits, scopeId])

  const visibleSources = useMemo(() => {
    if (scopeId === null) return searchable
    return searchable.filter((s) => s.manifest.id === scopeId)
  }, [searchable, scopeId])

  return (
    <div className="space-y-3">
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
      ) : query.trim().length < 2 ? (
        <EmptyState
          total={sources.length}
          searchable={searchable.length}
        />
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


function EmptyState({
  total, searchable,
}: {
  total: number; searchable: number
}) {
  return (
    <div className="rounded-md bg-surface-2 border border-dashed border-border-soft px-4 py-5">
      <div className="flex items-start gap-3">
        <span className="size-9 rounded-sm bg-bg/40 flex items-center justify-center shrink-0">
          <Search size={16} className="text-text-subtle" />
        </span>
        <div className="flex-1 min-w-0">
          <p className="text-sm text-text">
            Gõ tên truyện hoặc dán đường dẫn manga
          </p>
          <div className="mt-2 flex items-center gap-3 text-[11px] text-text-subtle">
            <span className="inline-flex items-center gap-1">
              <Search size={10} />
              Tìm: <span className="text-text-muted">{searchable} nguồn</span>
            </span>
            <span className="inline-flex items-center gap-1">
              <LinkIcon size={10} />
              Dán link: <span className="text-text-muted">{total} nguồn</span>
            </span>
          </div>
        </div>
      </div>
    </div>
  )
}
