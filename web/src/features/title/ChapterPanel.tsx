import { useMemo, useState } from 'react'
import {
  Sparkles, Loader2, AlertCircle, BookOpen,
  ArrowDown, ArrowUp, Clock, X,
} from 'lucide-react'
import { getRouteApi } from '@tanstack/react-router'
import { Button } from '@shared/ui/Button'
import { EmptyState } from '@shared/ui/EmptyState'
import { SearchInput } from '@shared/ui/DataToolbar'
import { card } from '@shared/ui/primitives'
import { cn } from '@shared/lib/cn'
import { timeAgo } from '@shared/lib/time'
import {
  preferredReadable, inFlight, lastError,
  stripChapterPrefix,
  type HubChapter, type HubVersion,
} from './mergeChapters'

// =============================================================================
// ChapterPanel
//
// Two user intents:
//   READ      — find chapters that already have a VI translation → Đọc
//   TRANSLATE — find chapters without VI → Dịch (slice 15)
//
// Toolbar: status pills (Tất cả / Đã dịch / Chưa có / Đang dịch / Lỗi)
//          + search + sort. Nothing else.
//
// Row: number | label | time | action
//   action = "Đọc VI" link when raw upstreamUrl available (slice 13)
//          = null otherwise (slice 14/15 not wired yet)
//   sub-line = creator info only for running/error states
// =============================================================================

type StatusFilter = 'all' | 'translated' | 'running' | 'error' | 'raw'
type Sort         = 'chapter_desc' | 'chapter_asc' | 'updated_desc'

const titleRoute = getRouteApi('/title/$entryId')

interface Props {
  chapters:   HubChapter[]
  targetLang: string | null
  loading:    boolean
}

export function ChapterPanel({ chapters, targetLang, loading }: Props) {
  const search = titleRoute.useSearch()
  const nav    = titleRoute.useNavigate()
  const filter = search.filter ?? 'all'
  const q      = search.q      ?? ''
  const sort   = search.sort   ?? 'chapter_desc'

  const setFilter = (v: StatusFilter) =>
    nav({ search: (s) => ({ ...s, filter: v === 'all' ? undefined : v }) })
  const setQ = (v: string) =>
    nav({ search: (s) => ({ ...s, q: v || undefined }) })
  const setSort = (v: Sort) =>
    nav({ search: (s) => ({ ...s, sort: v === 'chapter_desc' ? undefined : v }) })

  const [sel, setSel] = useState<Set<string>>(new Set())

  const counts = useMemo(
    () => countByStatus(chapters, targetLang),
    [chapters, targetLang],
  )

  const visible = useMemo(() => {
    const term = q.trim().toLowerCase()
    const list = chapters.filter((c) => {
      if (term && !`${c.number} ${c.label ?? ''}`.toLowerCase().includes(term)) return false
      if (filter === 'all') return true
      return chapterStatus(c, targetLang) === filter
    })
    return list.slice().sort(sortFn(sort))
  }, [chapters, q, filter, sort, targetLang])

  const eligibleSpawn = useMemo(
    () => [...sel].filter((n) => {
      const ch = chapters.find((c) => c.number === n)
      return ch ? chapterStatus(ch, targetLang) === 'raw' : false
    }),
    [sel, chapters, targetLang],
  )

  const toggle = (n: string) =>
    setSel((prev) => { const s = new Set(prev); s.has(n) ? s.delete(n) : s.add(n); return s })

  const toggleAll = () => {
    const allOn = visible.every((c) => sel.has(c.number))
    setSel(allOn ? new Set() : new Set(visible.map((c) => c.number)))
  }

  return (
    <section className="px-4 sm:px-6">
      <Toolbar
        filter={filter} setFilter={setFilter} counts={counts}
        q={q} setQ={setQ}
        sort={sort} setSort={setSort}
      />

      {loading && chapters.length === 0 ? (
        <div className="space-y-px rounded-md overflow-hidden border border-border-soft">
          {Array.from({ length: 8 }).map((_, i) => (
            <div key={i} className="h-12 bg-surface-2 animate-pulse" />
          ))}
        </div>
      ) : visible.length === 0 ? (
        <EmptyState
          icon={Sparkles}
          title={
            chapters.length === 0
              ? 'Chưa có chương nào'
              : filter === 'translated'
              ? 'Chưa có chương nào được dịch'
              : filter === 'raw'
              ? 'Tất cả chương đã có bản dịch'
              : 'Không tìm thấy chương phù hợp'
          }
          hint={
            chapters.length === 0
              ? 'Nguồn chưa trả về chương nào.'
              : q
              ? 'Thử từ khoá khác.'
              : undefined
          }
        />
      ) : (
        <ChapterTable
          chapters={visible}
          targetLang={targetLang}
          sel={sel}
          onToggle={toggle}
          onToggleAll={toggleAll}
          allChecked={visible.length > 0 && visible.every((c) => sel.has(c.number))}
        />
      )}

      {sel.size > 0 && (
        <SelectionBar
          selected={sel.size}
          eligibleSpawn={eligibleSpawn.length}
          onClear={() => setSel(new Set())}
        />
      )}
    </section>
  )
}


// ── Toolbar ─────────────────────────────────────────────────────────

const STATUS_FILTERS: { key: StatusFilter; label: string }[] = [
  { key: 'all',        label: 'Tất cả' },
  { key: 'translated', label: 'Đã dịch' },
  { key: 'raw',        label: 'Chưa có' },
  { key: 'running',    label: 'Đang dịch' },
  { key: 'error',      label: 'Lỗi' },
]

const SORT_META: Record<Sort, { label: string; icon: typeof ArrowDown }> = {
  chapter_desc: { label: 'Mới nhất', icon: ArrowDown },
  chapter_asc:  { label: 'Cũ nhất',  icon: ArrowUp },
  updated_desc: { label: 'Cập nhật', icon: Clock },
}

function Toolbar({
  filter, setFilter, counts, q, setQ, sort, setSort,
}: {
  filter:    StatusFilter
  setFilter: (v: StatusFilter) => void
  counts:    Record<StatusFilter, number>
  q:         string
  setQ:      (v: string) => void
  sort:      Sort
  setSort:   (v: Sort) => void
}) {
  const sortOrder: Sort[] = ['chapter_desc', 'chapter_asc', 'updated_desc']
  const { label: sortLabel, icon: SortIcon } = SORT_META[sort]

  return (
    <div className="flex items-center gap-2 mb-4 flex-wrap">
      {/* Status pills — scroll-x on mobile */}
      <div
        className="overflow-x-auto -mx-1 px-1 flex-1 min-w-0"
        style={{ scrollbarWidth: 'none' }}
      >
        <div className="inline-flex items-center gap-0.5">
          {STATUS_FILTERS.map(({ key, label }) => {
            const n = counts[key]
            const active = filter === key
            if (!active && key !== 'all' && n === 0) return null
            return (
              <button
                key={key}
                type="button"
                onClick={() => setFilter(key)}
                className={cn(
                  'h-8 px-3 rounded-sm text-[13px] cursor-pointer transition-colors whitespace-nowrap',
                  'inline-flex items-center gap-1.5',
                  active
                    ? 'bg-surface-2 text-text font-medium'
                    : 'text-text-muted hover:bg-hover hover:text-text',
                )}
              >
                {label}
                {n > 0 && (
                  <span className={cn(
                    'tabular text-[11px]',
                    active ? 'text-text-muted' : 'text-text-subtle',
                  )}>
                    {n}
                  </span>
                )}
              </button>
            )
          })}
        </div>
      </div>

      {/* Search + sort — right cluster, same surface */}
      <div className="flex items-center gap-1 shrink-0">
        <SearchInput value={q} onChange={setQ} placeholder="Tìm chương…" className="w-44" />
        <button
          type="button"
          onClick={() => setSort(sortOrder[(sortOrder.indexOf(sort) + 1) % sortOrder.length]!)}
          title="Đổi cách sắp xếp"
          className="inline-flex items-center gap-1.5 h-8 px-2.5 rounded-sm text-[13px] text-text-muted bg-surface-2 hover:bg-hover hover:text-text transition-colors cursor-pointer shrink-0"
        >
          <SortIcon size={12} className="text-text-subtle" />
          <span className="hidden sm:inline">{sortLabel}</span>
        </button>
      </div>
    </div>
  )
}


// ── Selection bar ────────────────────────────────────────────────────

function SelectionBar({
  selected, eligibleSpawn, onClear,
}: {
  selected:      number
  eligibleSpawn: number
  onClear:       () => void
}) {
  return (
    <div className={cn(
      'fixed left-1/2 -translate-x-1/2 z-50 flex items-center gap-3',
      'bottom-[calc(3.5rem+0.75rem+var(--saib))] sm:bottom-[calc(1.25rem+var(--saib))]',
      card,
      'pl-4 pr-2 py-2 shadow-[0_8px_32px_rgb(0,0,0,0.4)]',
    )}>
      <span className="text-sm text-text-muted tabular">
        <span className="text-text font-medium">{selected}</span> chương đã chọn
        {eligibleSpawn < selected && (
          <span className="text-text-subtle ml-1.5">· {eligibleSpawn} chưa có bản dịch</span>
        )}
      </span>
      <Button
        variant="primary" size="sm" disabled
        title="Bulk spawn sẽ wire ở slice 15"
      >
        <Sparkles size={12} />
        Dịch ({eligibleSpawn})
      </Button>
      <Button variant="ghost" size="sm" icon onClick={onClear} aria-label="Bỏ chọn">
        <X size={14} />
      </Button>
    </div>
  )
}


// ── Table ────────────────────────────────────────────────────────────

function ChapterTable({
  chapters, targetLang, sel, onToggle, onToggleAll, allChecked,
}: {
  chapters:    HubChapter[]
  targetLang:  string | null
  sel:         Set<string>
  onToggle:    (n: string) => void
  onToggleAll: () => void
  allChecked:  boolean
}) {
  const anySelected = sel.size > 0
  return (
    <div className="rounded-md overflow-hidden border border-border-soft">
      <table className="w-full">
        <thead>
          <tr className="text-[11px] font-medium tracking-wide text-text-subtle uppercase bg-surface-2/40 border-b border-border-soft">
            <th className="pl-3 pr-2 py-2 w-8">
              <Checkbox checked={allChecked} onClick={onToggleAll} ariaLabel="Chọn tất cả" />
            </th>
            <th className="pr-3 py-2 text-right">#</th>
            <th className="px-3 py-2 text-left">
              {anySelected ? `${sel.size} đã chọn` : 'Chương'}
            </th>
            <th className="px-3 py-2 hidden sm:table-cell text-right">Cập nhật</th>
            <th className="pl-2 pr-3 py-2 text-right">Thao tác</th>
          </tr>
        </thead>
        <tbody>
          {chapters.map((c) => (
            <ChapterRow
              key={c.number}
              chapter={c}
              targetLang={targetLang}
              checked={sel.has(c.number)}
              anySelected={anySelected}
              onToggle={() => onToggle(c.number)}
            />
          ))}
        </tbody>
      </table>
    </div>
  )
}


// ── Row ──────────────────────────────────────────────────────────────

function ChapterRow({
  chapter, targetLang, checked, anySelected, onToggle,
}: {
  chapter:     HubChapter
  targetLang:  string | null
  checked:     boolean
  anySelected: boolean
  onToggle:    () => void
}) {
  const status   = chapterStatus(chapter, targetLang)
  const readable = preferredReadable(chapter, targetLang)
  const running  = inFlight(chapter, targetLang)
  const errored  = lastError(chapter, targetLang)
  const label    = stripChapterPrefix(chapter.label, chapter.number)

  const stripeColor =
    status === 'running' ? 'var(--color-info)'
  : status === 'error'   ? 'var(--color-error)'
                         : 'transparent'

  return (
    <tr className={cn(
      'group transition-colors border-b border-border-soft last:border-0',
      checked ? 'bg-row-active' : 'hover:bg-hover',
    )}>
      <td
        className="pl-3 pr-1 py-3 w-8"
        style={{ boxShadow: `inset 2px 0 0 0 ${stripeColor}` }}
      >
        <div className={cn(
          'transition-opacity',
          checked || anySelected
            ? 'opacity-100'
            : 'opacity-0 group-hover:opacity-100',
        )}>
          <Checkbox checked={checked} onClick={onToggle} ariaLabel="Chọn chương" />
        </div>
      </td>

      <td className="pr-3 py-3 whitespace-nowrap tabular text-right font-semibold text-text">
        {chapter.number}
      </td>

      <td className="px-3 py-3 min-w-0 w-full">
        {label
          ? <div className="text-sm text-text-muted truncate">{label}</div>
          : <div className="text-sm text-text-subtle">—</div>
        }
        {/* Sub-line: only for in-progress states where the action
            cell can't carry the full context. */}
        {status === 'running' && (
          <div className="mt-0.5 inline-flex items-center gap-1 text-xs text-info-text">
            <Loader2 size={10} className="animate-spin shrink-0" />
            {running?.creatorName ? `@${running.creatorName}` : 'Đang dịch'}
          </div>
        )}
        {status === 'error' && (
          <div className="mt-0.5 inline-flex items-center gap-1 text-xs text-error-text">
            <AlertCircle size={10} className="shrink-0" />
            {errored?.creatorName ? `@${errored.creatorName}` : 'Lỗi dịch'}
          </div>
        )}
      </td>

      <td className="px-3 py-3 hidden sm:table-cell whitespace-nowrap text-right w-px">
        <TimeCell iso={chapter.updatedAt} />
      </td>

      <td className="pl-2 pr-3 py-3 whitespace-nowrap text-right">
        <RowAction status={status} readable={readable} />
      </td>
    </tr>
  )
}


// ── Action cell ──────────────────────────────────────────────────────
//
// Only renders when there's something the user can actually DO now.
// Disabled placeholders for unwired slices are removed — they signal
// "broken" not "coming soon".
//
// Wired (slice 13):
//   raw + upstreamUrl → external link to source page
//
// Not yet wired:
//   translation done  → reader (slice 14)
//   raw no url        → spawn dialog (slice 15)
//   error             → retry (slice 15)
//   running           → nothing (wait)

function RowAction({
  status, readable,
}: {
  status:   StatusFilter
  readable: HubVersion | null
}) {
  if (readable?.kind === 'raw' && readable.upstreamUrl) {
    return (
      <a
        href={readable.upstreamUrl}
        target="_blank"
        rel="noopener noreferrer"
        title={`Mở trên ${readable.sourceName ?? 'nguồn'}`}
        className="inline-flex"
      >
        <Button size="sm" variant="secondary" tabIndex={-1}>
          <BookOpen size={12} />
          Đọc {readable.lang.toUpperCase()}
        </Button>
      </a>
    )
  }
  // All other states: nothing actionable yet.
  return null
}


// ── Time cell ────────────────────────────────────────────────────────
// Color shifts by freshness so the eye naturally finds new chapters.

function TimeCell({ iso }: { iso: string | null }) {
  if (!iso) return <span className="text-sm text-text-subtle">—</span>
  const ageH = (Date.now() - new Date(iso).getTime()) / 36e5
  const cls  = ageH < 24 ? 'text-text font-medium' : ageH < 168 ? 'text-text-muted' : 'text-text-subtle'
  return <span className={cn('text-sm', cls)} title={iso}>{timeAgo(iso)}</span>
}


// ── Checkbox ─────────────────────────────────────────────────────────

function Checkbox({ checked, onClick, ariaLabel }: {
  checked: boolean; onClick: () => void; ariaLabel: string
}) {
  return (
    <button
      type="button"
      onClick={onClick}
      aria-label={ariaLabel}
      className={cn(
        'size-4 rounded-xs border flex items-center justify-center cursor-pointer transition-colors',
        checked ? 'bg-accent border-accent text-accent-fg' : 'border-text-subtle hover:border-text-muted',
      )}
    >
      {checked && (
        <svg width="9" height="9" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="3">
          <path d="M5 12l5 5 9-11" />
        </svg>
      )}
    </button>
  )
}


// ── Helpers ──────────────────────────────────────────────────────────

function chapterStatus(c: HubChapter, targetLang: string | null): StatusFilter {
  const tgt = targetLang?.toLowerCase() ?? null
  if (!tgt) return 'raw'
  if (preferredReadable(c, tgt)) return 'translated'
  if (inFlight(c, tgt))          return 'running'
  if (lastError(c, tgt))         return 'error'
  return 'raw'
}

function countByStatus(
  chapters: HubChapter[],
  targetLang: string | null,
): Record<StatusFilter, number> {
  const out: Record<StatusFilter, number> = {
    all: chapters.length, translated: 0, running: 0, error: 0, raw: 0,
  }
  for (const c of chapters) out[chapterStatus(c, targetLang)]++
  return out
}

function sortFn(s: Sort): (a: HubChapter, b: HubChapter) => number {
  switch (s) {
    case 'chapter_desc':  return (a, b) => b.sortKey - a.sortKey
    case 'chapter_asc':   return (a, b) => a.sortKey - b.sortKey
    case 'updated_desc':  return (a, b) => (b.updatedAt ?? '').localeCompare(a.updatedAt ?? '')
  }
}
