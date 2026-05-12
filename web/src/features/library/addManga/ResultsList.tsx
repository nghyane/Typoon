import { useMemo, useState } from 'react'
import { AlertTriangle, Loader2, ChevronDown } from 'lucide-react'
import { Cover } from '@shared/ui/Cover'
import { cn } from '@shared/lib/cn'
import { fetchMangaDetail } from '@features/browse/manifest/runtime'
import { proxify } from '@features/browse/proxy'
import type { InstalledSource } from '@features/browse/manifest/types'
import type { SearchHit } from './fanoutSearch'
import type { Picked } from './types'

// =============================================================================
// ResultsList — grouped per source with score-ranked preview + expand.
//
// Each source group renders its top INITIAL_PREVIEW hits (sorted by
// fuzzy score). When the group has more, a 'Xem thêm N' row reveals
// the rest in place — modal stays compact by default; deeper digs
// available on demand.
//
// Source scope filter (set externally via ScopeFilterRow) narrows
// `searchableSources`; when only one source remains the per-group
// header collapses to a single inline row.
// =============================================================================

const INITIAL_PREVIEW = 3
const PER_GROUP_MAX   = 8

export function ResultsList({
  hits, loading, failures, searchableSources, onPick,
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
    for (const id in by) {
      by[id]!.hits.sort((a, b) => b.score - a.score)
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
    <div className="space-y-3">
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
  const [expanded, setExpanded] = useState(false)

  // When the user has scoped to a single source via ScopeFilterRow,
  // `hideHeader` is true — that's also our signal to skip the
  // 'Xem thêm' collapse: the user already committed to this source,
  // no need to hide hits behind another click.
  const capped = hits.slice(0, PER_GROUP_MAX)
  const visible = hideHeader || expanded
    ? capped
    : capped.slice(0, INITIAL_PREVIEW)
  const more = capped.length - visible.length

  return (
    <section>
      {!hideHeader && (
        <header className="flex items-baseline justify-between gap-2 px-1 mb-1.5">
          <div className="flex items-baseline gap-2 min-w-0">
            <span className="text-[12px] font-medium text-text truncate">
              {manifest.name}
            </span>
            <span className="text-[11px] text-text-subtle truncate">
              {manifest.host}
            </span>
          </div>
          <span className="text-[11px] text-text-subtle shrink-0">
            {hits.length}
          </span>
        </header>
      )}
      <ul className="rounded-md bg-surface-2 divide-y divide-border-soft overflow-hidden">
        {visible.map((hit) => (
          <ResultRow
            key={`${manifest.id}::${hit.manga.id}`}
            hit={hit}
            onPick={onPick}
          />
        ))}
        {more > 0 && (
          <li>
            <button
              type="button"
              onClick={() => setExpanded(true)}
              className="w-full inline-flex items-center justify-center gap-1.5 h-8 text-[12px] text-text-muted hover:bg-hover hover:text-text transition-colors cursor-pointer"
            >
              <ChevronDown size={12} />
              Xem thêm {more}
            </button>
          </li>
        )}
      </ul>
    </section>
  )
}


function ResultRow({
  hit, onPick,
}: {
  hit:    SearchHit
  onPick: (p: Picked) => void
}) {
  const { source, manga } = hit
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
          'w-full flex items-center gap-2.5 px-2.5 py-1.5 text-left',
          'hover:bg-hover transition-colors cursor-pointer',
          resolving && 'opacity-60 cursor-wait',
        )}
      >
        <Cover
          src={manga.cover ? proxify(manga.cover) : null}
          title={manga.title}
          className="w-8 aspect-[2/3] rounded-xs shrink-0"
          fontSize="text-[9px]"
        />
        <div className="flex-1 min-w-0">
          <p className="text-[13px] text-text truncate leading-tight">
            {manga.title}
          </p>
          {manifest.languages.length > 0 && (
            <p className="text-[11px] text-text-subtle uppercase mt-0.5">
              {manifest.languages.slice(0, 3).join('/')}
            </p>
          )}
        </div>
        {resolving && (
          <Loader2 size={13} className="text-text-subtle animate-spin shrink-0" />
        )}
      </button>
    </li>
  )
}
