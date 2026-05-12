import { useMemo, useState } from 'react'
import { AlertTriangle, Loader2 } from 'lucide-react'
import { Cover } from '@shared/ui/Cover'
import { cn } from '@shared/lib/cn'
import { fetchMangaDetail } from '@features/browse/manifest/runtime'
import { proxify } from '@features/browse/proxy'
import type {
  InstalledSource, MangaSummary,
} from '@features/browse/manifest/types'
import type { SearchHit } from './fanoutSearch'
import type { Picked } from './types'

// =============================================================================
// ResultsList — grouped, fuzzy-ranked search results.
//
// Hits arrive per-source already capped at PER_SOURCE_LIMIT (8) and
// sorted by fuzzy score; this component groups them by source and
// stamps a header (source name + host + count). When the search is
// scoped to a single source the header is suppressed — the scope
// picker already shows where the user is.
// =============================================================================

export function ResultsList({
  query, hits, loading, failures, searchableSources, onPick,
}: {
  query:             string
  hits:              SearchHit[]
  loading:           boolean
  failures:          { sourceId: string; error: Error }[]
  searchableSources: InstalledSource[]
  onPick:            (p: Picked) => void
}) {
  const groups = useMemo(() => {
    const by: Record<string, { source: InstalledSource; hits: SearchHit[] }> = {}
    for (const h of hits) {
      const id = h.source.manifest.id
      if (!by[id]) by[id] = { source: h.source, hits: [] }
      by[id]!.hits.push(h)
    }
    return searchableSources
      .map((s) => by[s.manifest.id])
      .filter((g): g is { source: InstalledSource; hits: SearchHit[] } => !!g)
  }, [hits, searchableSources])

  const singleSource = searchableSources.length === 1

  if (loading && hits.length === 0) {
    return (
      <div className="flex items-center gap-2.5 px-4 py-3 rounded-md bg-surface-2">
        <Loader2 size={14} className="text-info-text animate-spin shrink-0" />
        <p className="text-sm text-text-muted">
          Đang tìm trên {searchableSources.length} nguồn…
        </p>
      </div>
    )
  }

  return (
    <div className="space-y-4">
      {(query || groups.length > 0) && (
        <div className="flex items-center justify-between gap-2 px-0.5">
          <p className="text-[11px] uppercase tracking-wider text-text-subtle">
            {hits.length} kết quả
            {loading && <span className="ml-1.5 normal-case">· đang tìm thêm…</span>}
          </p>
          {failures.length > 0 && (
            <span className="text-[11px] text-warning-text inline-flex items-center gap-1">
              <AlertTriangle size={10} />
              {failures.length} nguồn lỗi
            </span>
          )}
        </div>
      )}

      {groups.map(({ source, hits: g }) => (
        <SourceGroup
          key={source.manifest.id}
          source={source}
          hits={g}
          onPick={onPick}
          hideHeader={singleSource}
        />
      ))}
    </div>
  )
}


function SourceGroup({
  source, hits, onPick, hideHeader,
}: {
  source:     InstalledSource
  hits:       SearchHit[]
  onPick:     (p: Picked) => void
  hideHeader: boolean
}) {
  const manifest = source.manifest
  return (
    <section>
      {!hideHeader && (
        <header className="flex items-baseline gap-2 px-1 mb-1.5">
          <span className="text-[12px] font-medium text-text">
            {manifest.name}
          </span>
          <span className="text-[11px] text-text-subtle">
            {manifest.host} · {hits.length}
          </span>
        </header>
      )}
      <ul className="rounded-md bg-surface-2 divide-y divide-border-soft overflow-hidden">
        {hits.map(({ manga }) => (
          <ResultRow
            key={`${manifest.id}::${manga.id}`}
            source={source}
            manga={manga}
            onPick={onPick}
          />
        ))}
      </ul>
    </section>
  )
}


function ResultRow({
  source, manga, onPick,
}: {
  source: InstalledSource; manga: MangaSummary; onPick: (p: Picked) => void
}) {
  const manifest = source.manifest
  const [resolving, setResolving] = useState(false)

  const pick = async () => {
    setResolving(true)
    try {
      const d = await fetchMangaDetail(manifest, manga.url)
      onPick({
        source, upstreamRef: manga.url,
        title:       d.title || manga.title,
        cover:       d.cover ?? manga.cover,
        description: d.description,
        author:      d.author,
        status:      d.status,
        languages:   d.availableLanguages ?? manifest.languages,
        nsfw:        !!manifest.nsfw,
      })
    } catch {
      onPick({
        source, upstreamRef: manga.url,
        title:       manga.title,
        cover:       manga.cover,
        description: null, author: null, status: null,
        languages:   manifest.languages,
        nsfw:        !!manifest.nsfw,
      })
    } finally {
      setResolving(false)
    }
  }

  return (
    <li>
      <button
        type="button"
        onClick={pick}
        disabled={resolving}
        className={cn(
          'w-full flex items-center gap-3 px-3 py-2 text-left',
          'hover:bg-hover transition-colors cursor-pointer',
          resolving && 'opacity-60 cursor-wait',
        )}
      >
        <Cover
          src={manga.cover ? proxify(manga.cover) : null}
          title={manga.title}
          className="w-10 aspect-[2/3] rounded-xs shrink-0"
          fontSize="text-[10px]"
        />
        <div className="flex-1 min-w-0">
          <p className="text-sm text-text truncate">{manga.title}</p>
          {manifest.languages.length > 0 && (
            <p className="text-[11px] text-text-subtle mt-0.5 uppercase">
              {manifest.languages.slice(0, 3).join('/')}
            </p>
          )}
        </div>
        {resolving && (
          <Loader2 size={14} className="text-text-subtle animate-spin shrink-0" />
        )}
      </button>
    </li>
  )
}
