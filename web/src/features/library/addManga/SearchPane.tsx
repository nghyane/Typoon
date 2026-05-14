import { useMemo, useState } from 'react'
import { Search, Link as LinkIcon, AlertTriangle, CheckCircle2 } from 'lucide-react'

import { input as inputCls } from '@shared/ui/primitives'
import { cn } from '@shared/lib/cn'
import { useDebouncedValue } from '@shared/lib/useDebouncedValue'
import { fetchMangaDetail, hasSearch } from '@features/browse/manifest/runtime'
import { useAllSources, useSources } from '@features/browse/sources'
import type { InstalledSource } from '@features/browse/manifest/types'

import { isUrlLike, matchSource } from './parseUrl'
import { useFanoutSearch, type SearchHit } from './fanoutSearch'
import { UrlPasteCard } from './UrlPasteCard'
import { ResultsList } from './ResultsList'
import { hitKey } from './hitKey'
import { BlankCreateRow } from './BlankCreateRow'
import { ScopeFilterRow } from './ScopeFilterRow'
import type { ImportToLibrary } from './useImportToLibrary'

// Three vertical zones: input → scope tabs → body (URL card / source
// roster / scoped results + blank-create fallback). Fanout runs across
// every searchable source; scope filter is applied client-side so the
// user can switch sources without re-issuing the request.

export function SearchPane({
  query, setQuery, sources, importer,
}: {
  query:    string
  setQuery: (s: string) => void
  sources:  InstalledSource[]
  importer: ImportToLibrary
}) {
  const searchable = useMemo(
    () => sources.filter((s) => hasSearch(s.manifest)),
    [sources],
  )
  const urlMatch = useMemo(
    () => isUrlLike(query) ? matchSource(query, sources) : null,
    [query, sources],
  )
  const isUrl = isUrlLike(query)

  const [scopeId, setScopeId] = useState<string | null>(null)
  const [pendingKey, setPendingKey] = useState<string | null>(null)

  const debouncedQuery = useDebouncedValue(query, 250)
  const { hits, loading, failures } = useFanoutSearch(debouncedQuery, searchable)

  const scopedHits = useMemo(
    () => scopeId === null
      ? hits
      : hits.filter((h) => h.source.manifest.id === scopeId),
    [hits, scopeId],
  )
  const visibleSources = useMemo(
    () => scopeId === null
      ? searchable
      : searchable.filter((s) => s.manifest.id === scopeId),
    [searchable, scopeId],
  )

  // Pick handler: resolve canonical detail (description / author /
  // languages) before importing. Falls back to the search snapshot
  // on fetch failure so a flaky upstream doesn't block the save.
  // The wire payload is built inside `importHit` — this callsite
  // does not touch `ImportBody` shape.
  const handlePick = async (hit: SearchHit) => {
    if (importer.isPending || pendingKey) return
    const key = hitKey(hit)
    setPendingKey(key)
    try {
      const detail = await fetchMangaDetail(hit.source.manifest, hit.manga.url)
        .catch(() => null)
      importer.importHit({ hit, detail })
    } finally {
      setPendingKey(null)
    }
  }

  return (
    <div className="space-y-3 min-h-[420px]">
      <InputRow
        query={query}
        setQuery={(v) => { setQuery(v); setScopeId(null) }}
        isUrl={isUrl}
        urlMatch={urlMatch}
        disabled={importer.isPending}
      />

      {isUrl ? (
        <UrlPasteCard url={query} match={urlMatch} importer={importer} />
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
            hits={scopedHits}
            loading={loading}
            failures={failures}
            searchableSources={visibleSources}
            pendingKey={pendingKey}
            onPick={handlePick}
          />
          <BlankCreateRow
            query={query}
            hits={scopedHits.length}
            importer={importer}
          />
        </>
      )}
    </div>
  )
}


function InputRow({
  query, setQuery, isUrl, urlMatch, disabled,
}: {
  query:    string
  setQuery: (v: string) => void
  isUrl:    boolean
  urlMatch: ReturnType<typeof matchSource>
  disabled: boolean
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
        disabled={disabled}
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


function UrlBadge({ urlMatch }: { urlMatch: ReturnType<typeof matchSource> }) {
  const base =
    'absolute right-2 top-1/2 -translate-y-1/2 inline-flex items-center gap-1 ' +
    'h-6 px-2 rounded-xs text-xs font-medium pointer-events-none'
  if (urlMatch) {
    return (
      <span className={cn(base, 'bg-success/15 text-success-text')}>
        <CheckCircle2 size={12} />
        {urlMatch.source.manifest.name}
      </span>
    )
  }
  return (
    <span className={cn(base, 'bg-warning/15 text-warning-text')}>
      <AlertTriangle size={12} />
      Chưa hỗ trợ
    </span>
  )
}


// Empty-state hint when the query is too short. Renders the installed
// source roster as toggle chips so the user can enable / disable
// individual sources for fanout search without leaving the modal.
function SourceListHint() {
  const sources    = useAllSources()
  const setEnabled = useSources((s) => s.setEnabled)

  if (sources.length === 0) {
    return (
      <div className="rounded-md bg-surface-2 border border-dashed border-border-soft px-4 py-6 text-center">
        <p className="text-sm text-text-muted">Chưa cài nguồn nào</p>
        <p className="text-xs text-text-subtle mt-1">
          Mở Cài đặt để cài nguồn đầu tiên.
        </p>
      </div>
    )
  }
  return (
    <div className="space-y-2">
      <p className="text-xs text-text-subtle px-0.5">
        Bấm để bật/tắt nguồn cho fanout search
      </p>
      <ul className="flex flex-wrap gap-2">
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
                  'inline-flex items-center gap-2 h-8 pl-2 pr-3 rounded-sm text-xs transition-colors',
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
                <span className="text-xs text-text-subtle truncate">
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
