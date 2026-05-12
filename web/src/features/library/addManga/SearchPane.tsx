import { useMemo } from 'react'
import { Search } from 'lucide-react'
import { hasSearch } from '@features/browse/manifest/runtime'
import type { InstalledSource } from '@features/browse/manifest/types'
import { isUrlLike, matchSource } from './parseUrl'
import { useFanoutSearch } from './fanoutSearch'
import { SearchCombobox } from './SearchCombobox'
import { UrlPasteCard } from './UrlPasteCard'
import { ResultsList } from './ResultsList'
import { ManualCreateRow } from './ManualCreateRow'
import type { Picked } from './types'

// =============================================================================
// SearchPane — owns the input + scope state, dispatches to results.
//
// Layout: single column. Combobox row up top, results body below.
// Combobox menu floats absolute so it never pushes the modal body
// or affects modal height.
// =============================================================================

interface Props {
  query:            string
  setQuery:         (s: string) => void
  scopeId:          string | null
  setScopeId:       (id: string | null) => void
  sources:          InstalledSource[]
  onPick:           (p: Picked) => void
  onManualCreate:   (seed: string) => void
}

export function SearchPane({
  query, setQuery, scopeId, setScopeId,
  sources, onPick, onManualCreate,
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
  const effectiveScopeId = urlMatch?.source.manifest.id ?? scopeId

  const scopedSources = useMemo(() => {
    if (effectiveScopeId === null) return searchable
    return searchable.filter((s) => s.manifest.id === effectiveScopeId)
  }, [searchable, effectiveScopeId])

  const { hits, loading, failures } = useFanoutSearch(query, scopedSources)

  return (
    <div className="space-y-4">
      <SearchCombobox
        query={query}
        setQuery={setQuery}
        scopeId={scopeId}
        setScopeId={setScopeId}
        sources={sources}
        searchableSources={searchable}
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
        <EmptyHint
          scopeId={effectiveScopeId}
          sources={sources}
          searchableCount={searchable.length}
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


function EmptyHint({
  scopeId, sources, searchableCount,
}: {
  scopeId:         string | null
  sources:         InstalledSource[]
  searchableCount: number
}) {
  const sel = scopeId
    ? sources.find((s) => s.manifest.id === scopeId)
    : null
  return (
    <div className="rounded-md bg-surface-2 border border-dashed border-border-soft px-4 py-5">
      <div className="flex items-start gap-3">
        <span className="size-9 rounded-sm bg-bg/40 flex items-center justify-center shrink-0">
          <Search size={16} className="text-text-subtle" />
        </span>
        <div className="flex-1 min-w-0">
          <p className="text-sm text-text">
            {sel
              ? `Tìm trên ${sel.manifest.name}`
              : `Tìm trên ${searchableCount} nguồn cùng lúc`
            }
          </p>
          {sel && (
            <p className="text-[11px] text-text-subtle truncate">
              {sel.manifest.host}
            </p>
          )}
          <p className="text-[11px] text-text-subtle mt-1">
            Hoặc dán đường dẫn manga trực tiếp.
          </p>
        </div>
      </div>
    </div>
  )
}
