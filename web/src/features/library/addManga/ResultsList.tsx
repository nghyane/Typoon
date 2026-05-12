import { useMemo, useState } from 'react'
import { AlertTriangle, Loader2 } from 'lucide-react'
import { Cover } from '@shared/ui/Cover'
import { cn } from '@shared/lib/cn'
import { fetchMangaDetail } from '@features/browse/manifest/runtime'
import { proxify } from '@features/browse/proxy'
import type { InstalledSource } from '@features/browse/manifest/types'
import type { SearchHit } from './fanoutSearch'
import type { Picked } from './types'

// =============================================================================
// ResultsList — compact, score-sorted flat list.
//
// Hits arrive per-source already capped at PER_SOURCE_LIMIT (8) and
// scored against the query inside `fanoutSearch`. This component:
//   • flattens every source into one list,
//   • sorts globally by fuzzy score (highest first),
//   • renders each row with a small source label so the user knows
//     where the match came from without breaking flow.
//
// Caps the displayed list at MAX_VISIBLE to keep the modal compact;
// 'Còn N kết quả' note at the bottom hints at the rest.
// =============================================================================

const MAX_VISIBLE = 20

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
  const sorted = useMemo(
    () => [...hits].sort((a, b) => b.score - a.score),
    [hits],
  )
  const visible = sorted.slice(0, MAX_VISIBLE)
  const hidden  = Math.max(0, sorted.length - MAX_VISIBLE)
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
    <div className="space-y-2">
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

      <ul className="rounded-md bg-surface-2 divide-y divide-border-soft overflow-hidden">
        {visible.map((hit) => (
          <ResultRow
            key={`${hit.source.manifest.id}::${hit.manga.id}`}
            hit={hit}
            showSource={!singleSource}
            onPick={onPick}
          />
        ))}
      </ul>

      {hidden > 0 && (
        <p className="text-[11px] text-text-subtle text-center">
          Còn {hidden} kết quả khớp ít hơn · gõ chính xác hơn để thu hẹp
        </p>
      )}
    </div>
  )
}


function ResultRow({
  hit, showSource, onPick,
}: {
  hit:        SearchHit
  showSource: boolean
  onPick:     (p: Picked) => void
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
          <p className="text-[11px] text-text-subtle truncate mt-0.5">
            {showSource && (
              <>
                <span>{manifest.name}</span>
                <span className="mx-1 opacity-40">·</span>
              </>
            )}
            <span className="uppercase">
              {manifest.languages.slice(0, 3).join('/')}
            </span>
          </p>
        </div>
        {resolving && (
          <Loader2 size={13} className="text-text-subtle animate-spin shrink-0" />
        )}
      </button>
    </li>
  )
}
