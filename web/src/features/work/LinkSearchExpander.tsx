// LinkSearchExpander — user-driven cross-source link search.
//
// Sits inline in `LinkSuggestionPanel`. When the auto-ranker misses
// (no title overlap between, say, a Chinese raw and the Vietnamese
// scanlation already in the library), the user can type their target
// and pick from the fanout result list. One UI, no source-vs-DB
// distinction the user has to learn — every candidate looks the
// same, only the trailing badge whispers whether it's already in the
// library or about to be imported.
//
// Three states:
//
//   collapsed      Just the affordance button. Most users 1-click the
//                  auto-ranker and never open this.
//
//   open / empty   Input + "Bắt đầu gõ" hint.
//
//   open / typing  Debounced fanout via `useFanoutSearch` (same hook
//                  the AddMangaModal uses). Results filter out
//                  candidates whose Work already equals OUR work
//                  (self-link blocked) and badge anything we recognize
//                  in `existingByRef` as "Đã có ở thư viện".

import { useMemo, useState } from 'react'
import { Plus, Search, X } from 'lucide-react'

import { Cover } from '@shared/ui/Cover'
import { Spinner } from '@shared/ui/primitives'
import { cn } from '@shared/lib/cn'
import { useAllSources } from '@features/browse/sources'
import { useFanoutSearch, type SearchHit } from '@features/library/addManga/fanoutSearch'
import { hasSearch } from '@features/browse/manifest/runtime'
import type { ApiMaterial } from '@shared/api/api'


interface Props {
  /** Materials already on this Work — used to filter self-link
   *  candidates and to render "Đã có ở thư viện" badges. */
  ownMaterials:  ApiMaterial[]
  /** Triggered when the user picks a candidate. Parent decides
   *  whether it's a vote-only flow (candidate already in DB) or
   *  an import-and-vote flow (candidate is fresh from a source
   *  fanout). */
  onPick: (hit: SearchHit) => void
}


export function LinkSearchExpander({ ownMaterials, onPick }: Props) {
  const [open, setOpen] = useState(false)

  if (!open) {
    return (
      <button
        type="button"
        onClick={() => setOpen(true)}
        className={cn(
          'inline-flex items-center gap-1.5 h-7 px-2.5 rounded-sm text-xs',
          'text-text-muted hover:text-text bg-surface-2/50 hover:bg-surface-2',
          'cursor-pointer transition-colors',
        )}
      >
        <Plus size={12} />
        Liên kết thủ công
      </button>
    )
  }

  return <SearchBody ownMaterials={ownMaterials} onPick={onPick} onClose={() => setOpen(false)} />
}


// ── Body ──────────────────────────────────────────────────────


/** Set of "{source}::{upstream_ref}" strings — the canonical key
 *  the manifest layer uses for the same upstream chapter / manga.
 *  Lets us answer "is this candidate already in our library?" in
 *  O(1) without an extra round-trip. */
function ownUpstreamSet(materials: ApiMaterial[]): Set<string> {
  const out = new Set<string>()
  for (const m of materials) {
    if (m.source && m.upstream_ref) out.add(`${m.source}::${m.upstream_ref}`)
  }
  return out
}


function SearchBody({
  ownMaterials, onPick, onClose,
}: {
  ownMaterials: ApiMaterial[]
  onPick:       (hit: SearchHit) => void
  onClose:      () => void
}) {
  const [q, setQ] = useState('')

  // Fanout reuses the AddMangaModal hook: same plugins, same scoring,
  // same cache key — switching between AddManga and this surface
  // doesn't refetch when the user typed the same query recently.
  const allSources = useAllSources()
  const searchable = useMemo(
    () => allSources.filter((s) => s.enabled && hasSearch(s.manifest)),
    [allSources],
  )
  const { hits, loading } = useFanoutSearch(q, searchable)

  const ownSet = useMemo(() => ownUpstreamSet(ownMaterials), [ownMaterials])

  // Drop self-links: any candidate whose URL already lives on this
  // Work is just our own row coming back from a manifest search,
  // never a useful merge target.
  const filtered = hits.filter((h) => {
    return !ownSet.has(`${h.source.manifest.id}::${h.manga.url}`)
  })

  return (
    <div className="bg-surface-2/30 rounded-md p-2.5 space-y-2">
      <div className="flex items-center gap-2">
        <div className="relative flex-1 min-w-0">
          <Search
            size={13}
            className="absolute left-2 top-1/2 -translate-y-1/2 text-text-subtle"
          />
          <input
            value={q}
            onChange={(e) => setQ(e.target.value)}
            placeholder="Gõ tên hoặc dán link manga"
            autoFocus
            className={cn(
              'w-full h-8 pl-7 pr-2 rounded-sm text-sm',
              'bg-bg text-text placeholder:text-text-subtle',
              'focus:outline-hidden focus:ring-1 focus:ring-accent/40',
            )}
          />
        </div>
        <button
          type="button"
          onClick={onClose}
          className="h-8 w-8 inline-flex items-center justify-center rounded-sm text-text-subtle hover:text-text hover:bg-hover cursor-pointer"
          title="Đóng"
        >
          <X size={14} />
        </button>
      </div>

      <SearchResults
        q={q}
        hits={filtered}
        loading={loading}
        ownMaterials={ownMaterials}
        onPick={onPick}
      />
    </div>
  )
}


function SearchResults({
  q, hits, loading, ownMaterials, onPick,
}: {
  q:            string
  hits:         SearchHit[]
  loading:      boolean
  ownMaterials: ApiMaterial[]
  onPick:       (hit: SearchHit) => void
}) {
  if (q.trim().length < 2) {
    return (
      <p className="text-xs text-text-subtle px-1 py-2">
        Gõ ít nhất 2 ký tự để bắt đầu tìm.
      </p>
    )
  }
  if (loading && hits.length === 0) {
    return (
      <div className="py-4 flex justify-center">
        <Spinner size={14} />
      </div>
    )
  }
  if (hits.length === 0) {
    return (
      <p className="text-xs text-text-subtle px-1 py-2">
        Không tìm thấy. Thử từ khoá khác hoặc dán đường dẫn manga.
      </p>
    )
  }

  return (
    <ul className="space-y-1 max-h-72 overflow-y-auto">
      {hits.slice(0, 12).map((h) => (
        <SearchRow
          key={`${h.source.manifest.id}::${h.manga.url}`}
          hit={h}
          ownMaterials={ownMaterials}
          onPick={() => onPick(h)}
        />
      ))}
    </ul>
  )
}


function SearchRow({
  hit, ownMaterials, onPick,
}: {
  hit:          SearchHit
  ownMaterials: ApiMaterial[]
  onPick:       () => void
}) {
  // "Already in this Work" — the candidate's (source, url) maps to a
  // material we already have on the current Work. Block the action;
  // it's a no-op merge.
  const inOwnWork = ownMaterials.some(
    (m) => m.source === hit.source.manifest.id && m.upstream_ref === hit.manga.url,
  )
  const badge = inOwnWork ? 'Đã trong work này' : 'Liên kết'

  return (
    <li>
      <button
        type="button"
        onClick={onPick}
        disabled={inOwnWork}
        className={cn(
          'w-full flex items-center gap-3 p-2 rounded-sm text-left',
          'transition-colors',
          inOwnWork
            ? 'opacity-60 cursor-not-allowed'
            : 'bg-bg/40 hover:bg-hover cursor-pointer',
        )}
      >
        <div className="w-10 h-14 shrink-0 rounded-sm overflow-hidden">
          <Cover
            src={hit.manga.cover ?? null}
            title={hit.manga.title}
            className="w-full h-full"
            fontSize="text-xs"
          />
        </div>
        <div className="flex-1 min-w-0">
          <div className="text-sm text-text truncate">{hit.manga.title}</div>
          <div className="text-xs text-text-subtle truncate">
            {hit.source.manifest.name}
          </div>
        </div>
        <span
          className={cn(
            'text-xs px-2 h-6 inline-flex items-center rounded-sm shrink-0',
            inOwnWork
              ? 'bg-surface-2 text-text-subtle'
              : 'bg-accent/15 text-accent',
          )}
        >
          {badge}
        </span>
      </button>
    </li>
  )
}
