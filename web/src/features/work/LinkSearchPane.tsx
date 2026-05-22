// LinkSearchPane — search + attach UI inside `LinkSearchModal`.
//
// Mirrors `SearchPane`'s shell (debounced fanout, scope filter,
// grouped results via shared `ResultsList`) but the pick handler
// runs `useAttachSource` on the host Work instead of importing into
// the library. The modal stays open so the user can attach multiple
// sources in one session; rows the viewer has picked show a check
// + go disabled.

import { useMemo, useState } from 'react'
import { Search } from 'lucide-react'

import { input as inputCls } from '@shared/ui/primitives'
import { cn } from '@shared/lib/cn'
import { useDebouncedValue } from '@shared/lib/useDebouncedValue'
import { hasSearch } from '@features/browse/manifest/runtime'
import { useAllSources } from '@features/browse/sources'
import {
  useFanoutSearch, type SearchHit,
} from '@features/library/addManga/fanoutSearch'
import { ResultsList } from '@features/library/addManga/ResultsList'
import { hitKey } from '@features/library/addManga/hitKey'
import { ScopeFilterRow } from '@features/library/addManga/ScopeFilterRow'
import type { WorkSource } from '@features/works/queries'


interface Props {
  /** Sources already attached to this Work — drops self-link
   *  candidates before they reach the result list. */
  ownSources: WorkSource[]
  /** Initial search query. Parent seeds with the work title so the
   *  pane lands on relevant siblings out of the box. */
  initialQuery?: string
  /** Fires for each candidate the viewer picks. Parent runs the
   *  attach round-trip. */
  onPick: (hit: SearchHit) => void
  /** Keys of candidates this session already picked. */
  pickedKeys: Set<string>
  busy:       boolean
  pendingKey: string | null
}


export function LinkSearchPane({
  ownSources, initialQuery = '', onPick, pickedKeys, busy, pendingKey,
}: Props) {
  const [q, setQ]  = useState(initialQuery)
  const debouncedQ = useDebouncedValue(q, 250)

  const allSources = useAllSources()
  const searchable = useMemo(
    () => allSources.filter(s => s.enabled && hasSearch(s.manifest)),
    [allSources],
  )

  const [scopeId, setScopeId] = useState<string | null>(null)
  const { hits, loading, failures } = useFanoutSearch(debouncedQ, searchable)

  const ownSet = useMemo(() => ownUpstreamSet(ownSources), [ownSources])
  const visibleHits = useMemo(
    () => hits.filter(h => !ownSet.has(`${h.source.manifest.id}::${h.manga.url}`)),
    [hits, ownSet],
  )

  const scopedHits = useMemo(
    () => scopeId === null
      ? visibleHits
      : visibleHits.filter(h => h.source.manifest.id === scopeId),
    [visibleHits, scopeId],
  )
  const visibleSources = useMemo(
    () => scopeId === null
      ? searchable
      : searchable.filter(s => s.manifest.id === scopeId),
    [searchable, scopeId],
  )

  return (
    <div className="space-y-3 min-h-[420px]">
      <div className="relative">
        <Search
          size={14}
          className="absolute left-3 top-1/2 -translate-y-1/2 text-text-subtle pointer-events-none"
        />
        <input
          autoFocus
          type="text"
          value={q}
          onChange={(e) => { setQ(e.target.value); setScopeId(null) }}
          disabled={busy}
          placeholder="Tìm tên truyện ở nguồn khác"
          className={cn(inputCls, 'pl-9 h-10')}
        />
      </div>

      {debouncedQ.trim().length < 2 ? (
        <p className="text-sm text-text-subtle px-0.5">
          Gõ ít nhất 2 ký tự để bắt đầu tìm trên {searchable.length} nguồn.
        </p>
      ) : (
        <>
          <ScopeFilterRow
            hits={visibleHits}
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
            pickedKeys={pickedKeys}
            onPick={(h) => { if (!pickedKeys.has(hitKey(h))) onPick(h) }}
          />
        </>
      )}
    </div>
  )
}


/** Canonical "{source}::{upstream_ref}" key for an own source. */
function ownUpstreamSet(sources: WorkSource[]): Set<string> {
  const out = new Set<string>()
  for (const s of sources) {
    out.add(`${s.source}::${s.upstream_ref}`)
  }
  return out
}
