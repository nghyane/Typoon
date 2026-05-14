import { useMemo, useState } from 'react'
import { AlertTriangle, Check, ChevronDown, Loader2 } from 'lucide-react'

import { Cover } from '@shared/ui/Cover'
import { cn } from '@shared/lib/cn'
import type { InstalledSource } from '@features/browse/manifest/types'

import type { SearchHit } from './fanoutSearch'
import { hitKey } from './hitKey'

// Per-source groups with score-ranked preview (top 3) and an expand
// row. Pure presentation: emits `onPick(hit)` and lets the parent
// decide what "pick" means (import to library, link to work, …).
// The parent passes `pendingKey` to flag the row currently resolving
// and `pickedKeys` to flag rows the user has already acted on so
// they show a "đã chọn" check instead of being clickable again.

const INITIAL_PREVIEW = 3
const PER_GROUP_MAX   = 8


export function ResultsList({
  hits, loading, failures, searchableSources,
  pendingKey = null, pickedKeys,
  onPick,
}: {
  hits:              SearchHit[]
  loading:           boolean
  failures:          { sourceId: string; error: Error }[]
  searchableSources: InstalledSource[]
  /** Key of the row whose detail-fetch / mutation is in flight. */
  pendingKey?:       string | null
  /** Keys of rows the user has already picked (link flow uses this
   *  to keep the modal open while marking completed picks). */
  pickedKeys?:       Set<string>
  onPick:            (hit: SearchHit) => void
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
        <p className="text-xs uppercase tracking-wider text-text-subtle">
          {hits.length} kết quả
          {loading && <span className="ml-1.5 normal-case">· đang tìm thêm…</span>}
        </p>
        {failures.length > 0 && (
          <span className="text-xs text-warning-text inline-flex items-center gap-1">
            <AlertTriangle size={12} />
            {failures.length} nguồn lỗi
          </span>
        )}
      </div>

      {groups.map(({ source, hits: g }) => (
        <SourceGroup
          key={source.manifest.id}
          source={source}
          hits={g}
          pendingKey={pendingKey}
          pickedKeys={pickedKeys}
          onPick={onPick}
          hideHeader={singleSource}
        />
      ))}
    </div>
  )
}


function SourceGroup({
  source, hits, pendingKey, pickedKeys, onPick, hideHeader,
}: {
  source:     InstalledSource
  hits:       SearchHit[]
  pendingKey: string | null
  pickedKeys: Set<string> | undefined
  onPick:     (hit: SearchHit) => void
  hideHeader: boolean
}) {
  const manifest = source.manifest
  const [expanded, setExpanded] = useState(false)

  // hideHeader == single-source scope: also skip the "Xem thêm"
  // collapse since the user already committed to this source.
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
            <span className="text-xs font-medium text-text truncate">
              {manifest.name}
            </span>
            <span className="text-xs text-text-subtle truncate">
              {manifest.host}
            </span>
          </div>
          <span className="text-xs text-text-subtle shrink-0">
            {hits.length}
          </span>
        </header>
      )}
      <ul className="rounded-md bg-surface-2 divide-y divide-border-soft overflow-hidden">
        {visible.map((hit) => {
          const key = hitKey(hit)
          return (
            <ResultRow
              key={key}
              hit={hit}
              pending={pendingKey === key}
              picked={pickedKeys?.has(key) ?? false}
              busy={pendingKey !== null}
              onPick={() => onPick(hit)}
            />
          )
        })}
        {more > 0 && (
          <li>
            <button
              type="button"
              onClick={() => setExpanded(true)}
              className="w-full inline-flex items-center justify-center gap-2 h-8 text-xs text-text-muted hover:bg-hover hover:text-text transition-colors cursor-pointer"
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
  hit, pending, picked, busy, onPick,
}: {
  hit:     SearchHit
  pending: boolean
  picked:  boolean
  /** Some other row on the page is mid-flight. Disables click. */
  busy:    boolean
  onPick:  () => void
}) {
  const { source, manga } = hit
  const manifest = source.manifest
  const disabled = busy || picked

  return (
    <li>
      <button
        type="button"
        onClick={onPick}
        disabled={disabled}
        className={cn(
          'w-full flex items-center gap-2.5 px-2.5 py-1.5 text-left',
          'hover:bg-hover transition-colors cursor-pointer',
          picked  && 'opacity-70 cursor-default',
          pending && 'opacity-60 cursor-wait',
          disabled && !picked && !pending && 'opacity-60 cursor-not-allowed',
        )}
      >
        <Cover
          src={manga.cover}
          title={manga.title}
          className="w-8 aspect-[2/3] rounded-xs shrink-0"
          fontSize="text-[9px]"
        />
        <div className="flex-1 min-w-0">
          <p className="text-sm text-text truncate leading-tight">
            {manga.title}
          </p>
          {manifest.languages.length > 0 && (
            <p className="text-xs text-text-subtle uppercase mt-0.5">
              {manifest.languages.slice(0, 3).join('/')}
            </p>
          )}
        </div>
        {pending && (
          <Loader2 size={14} className="text-text-subtle animate-spin shrink-0" />
        )}
        {picked && (
          <Check size={14} className="text-success-text shrink-0" />
        )}
      </button>
    </li>
  )
}
