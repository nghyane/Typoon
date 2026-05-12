import { useMemo } from 'react'
import { Search, Link as LinkIcon, AlertTriangle, CheckCircle2 } from 'lucide-react'
import { input as inputCls } from '@shared/ui/primitives'
import { cn } from '@shared/lib/cn'
import { hasSearch } from '@features/browse/manifest/runtime'
import type { InstalledSource } from '@features/browse/manifest/types'
import { isUrlLike, matchSource } from './parseUrl'
import { useFanoutSearch } from './fanoutSearch'
import { ScopePicker } from './ScopePicker'
import { UrlPasteCard } from './UrlPasteCard'
import { ResultsList } from './ResultsList'
import { ManualCreateRow } from './ManualCreateRow'
import type { Picked } from './types'

// =============================================================================
// SearchPane — Option B single-column layout.
//
// Top: input with inline scope picker (ScopePicker) + URL detection
// badge (right side of input). One row, no sidebar.
// Body: scrollable result section. Search hits group per source;
// URL paste shows a loading / error / matched card; empty state
// shows a scope-aware hint.
//
// The pane never lays out wider than the modal it lives in; mode
// changes (search → picked → manual) swap content but keep the same
// column width — no modal resize.
// =============================================================================

interface Props {
  query:            string
  setQuery:         (s: string) => void
  sources:          InstalledSource[]
  selectedSourceId: string | null
  setSelectedSourceId: (id: string | null) => void
  onPick:           (p: Picked) => void
  onManualCreate:   (seed: string) => void
}

export function SearchPane({
  query, setQuery, sources, selectedSourceId, setSelectedSourceId,
  onPick, onManualCreate,
}: Props) {
  const searchableIds = useMemo(
    () => new Set(sources.filter((s) => hasSearch(s.manifest))
                         .map((s) => s.manifest.id)),
    [sources],
  )

  const urlMatch = useMemo(
    () => isUrlLike(query) ? matchSource(query, sources) : null,
    [query, sources],
  )
  const lockedSourceId = urlMatch?.source.manifest.id ?? null
  const isUrl          = isUrlLike(query)
  const effective      = lockedSourceId ?? selectedSourceId

  const scopedSources = useMemo(() => {
    const searchable = sources.filter((s) => searchableIds.has(s.manifest.id))
    if (effective === null) return searchable
    return searchable.filter((s) => s.manifest.id === effective)
  }, [sources, searchableIds, effective])

  const { hits, loading, failures } = useFanoutSearch(query, scopedSources)

  return (
    <div className="space-y-4">
      {/* Input row */}
      <div className="flex items-center gap-2">
        <ScopePicker
          sources={sources}
          searchableIds={searchableIds}
          value={selectedSourceId}
          onChange={setSelectedSourceId}
          lockedTo={lockedSourceId}
        />
        <InputWithBadge
          query={query}
          setQuery={setQuery}
          isUrl={isUrl}
          urlMatch={urlMatch}
        />
      </div>

      {/* Body */}
      {isUrl ? (
        <UrlPasteCard
          url={query}
          match={urlMatch}
          onPick={onPick}
          onManualCreate={onManualCreate}
        />
      ) : query.trim().length < 2 ? (
        <ScopeHint
          selected={effective ? sources.find((s) => s.manifest.id === effective) : null}
          searchableCount={searchableIds.size}
        />
      ) : (
        <>
          <ResultsList
            query={query}
            hits={hits}
            loading={loading}
            failures={failures}
            searchableSources={scopedSources}
            onPick={onPick}
          />
          <ManualCreateRow
            query={query}
            hits={hits.length}
            onManualCreate={onManualCreate}
          />
        </>
      )}
    </div>
  )
}


function InputWithBadge({
  query, setQuery, isUrl, urlMatch,
}: {
  query:    string
  setQuery: (s: string) => void
  isUrl:    boolean
  urlMatch: ReturnType<typeof matchSource>
}) {
  const Icon = isUrl ? LinkIcon : Search
  return (
    <div className="relative flex-1 min-w-0">
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
        className={cn(inputCls, 'pl-9 h-10', isUrl && (urlMatch ? 'pr-36' : 'pr-32'))}
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


function ScopeHint({
  selected, searchableCount,
}: {
  selected:        InstalledSource | null | undefined
  searchableCount: number
}) {
  return (
    <div className="rounded-md bg-surface-2 border border-dashed border-border-soft px-5 py-8">
      <div className="flex items-center gap-3">
        <span className="size-10 rounded-sm bg-bg/40 flex items-center justify-center shrink-0">
          <Search size={18} className="text-text-subtle" />
        </span>
        <div className="flex-1 min-w-0">
          <p className="text-sm text-text">
            {selected
              ? `Tìm trên ${selected.manifest.name}`
              : `Tìm trên ${searchableCount} nguồn cùng lúc`
            }
          </p>
          {selected && (
            <p className="text-[11px] text-text-subtle truncate">
              {selected.manifest.host}
            </p>
          )}
          <p className="text-[11px] text-text-subtle mt-1">
            Hoặc dán đường dẫn manga vào ô trên để thêm trực tiếp.
          </p>
        </div>
      </div>
    </div>
  )
}
