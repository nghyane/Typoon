import { useEffect, useMemo, useState } from 'react'
import {
  Sparkles, Loader2, AlertCircle,
  BookOpen, ArrowDown, ArrowUp, Clock,
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
// ChapterPanel — hub's chapter list with bulk-select support.
//
// State model:
//   • filter / q / sort   live on the URL (TanStack Router search
//                         params). Refresh preserves them, links
//                         share them.
//   • sel                 transient client state — doesn't belong
//                         in the URL.
//
// UX agreed in the design pass:
//   • Toolbar: segmented filter (counts inline) + search + sort
//     cycle button. Mirrors the ProjectDetail DataToolbar pattern.
//   • SelectionBar floats at bottom-center while ANY chapter is
//     checked, primary action 'Dịch hàng loạt' eligibility-filters
//     to chapters that don't yet have a readable target_lang version.
//   • Row: checkbox + 'Ch.N + label' + lang chips + smart action.
//     Smart action picks 'Đọc {lang}' when a readable version
//     exists, 'Xem tiến độ' when running/pending, 'Thử lại' on
//     error, 'Dịch' otherwise.
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
  const search   = titleRoute.useSearch()
  const nav      = titleRoute.useNavigate()
  const filter = search.filter ?? 'all'
  const q      = search.q      ?? ''
  const sort   = search.sort   ?? 'chapter_desc'

  const setFilter = (next: StatusFilter) =>
    nav({ search: (s) => ({ ...s, filter: next === 'all' ? undefined : next }) })
  const setQ = (next: string) =>
    nav({ search: (s) => ({ ...s, q: next || undefined }) })
  const setSort = (next: Sort) =>
    nav({ search: (s) => ({ ...s, sort: next === 'chapter_desc' ? undefined : next }) })

  const [sel, setSel] = useState<Set<string>>(new Set())

  const counts = useMemo(() => countByStatus(chapters, targetLang), [chapters, targetLang])

  const visible = useMemo(() => {
    const term = q.trim().toLowerCase()
    let list = chapters.filter((c) => {
      if (term && !`${c.number} ${c.label ?? ''}`.toLowerCase().includes(term)) return false
      if (filter === 'all') return true
      return chapterStatus(c, targetLang) === filter
    })
    return list.slice().sort(sortFn(sort))
  }, [chapters, q, filter, sort, targetLang])

  const eligibleForSpawn = useMemo(
    () => [...sel].filter((n) => {
      const ch = chapters.find((c) => c.number === n)
      if (!ch) return false
      const status = chapterStatus(ch, targetLang)
      return status === 'raw'
    }),
    [sel, chapters, targetLang],
  )

  const toggle = (n: string) => {
    setSel((prev) => {
      const next = new Set(prev)
      if (next.has(n)) next.delete(n)
      else next.add(n)
      return next
    })
  }
  const toggleAllVisible = () => {
    const allOn = visible.every((c) => sel.has(c.number))
    setSel(() => allOn ? new Set() : new Set(visible.map((c) => c.number)))
  }

  return (
    <section className="px-4 sm:px-6">
      <Toolbar
        filter={filter} setFilter={setFilter} counts={counts}
        q={q} setQ={setQ}
        sort={sort} setSort={setSort}
      />

      {loading && chapters.length === 0 ? (
        <div className="space-y-1.5">
          {Array.from({ length: 6 }).map((_, i) => (
            <div key={i} className="h-12 rounded-md bg-surface-2 animate-pulse" />
          ))}
        </div>
      ) : visible.length === 0 ? (
        <EmptyState
          icon={Sparkles}
          title={chapters.length === 0 ? 'Không có chương đọc được' : 'Không tìm thấy chương phù hợp'}
          hint={chapters.length === 0
            ? 'Nguồn không trả về chương nào ở ngôn ngữ này.'
            : 'Thử từ khoá khác hoặc bỏ bộ lọc.'}
        />
      ) : (
        <ChapterTable
          chapters={visible}
          targetLang={targetLang}
          sel={sel}
          onToggle={toggle}
          onToggleAll={toggleAllVisible}
          allChecked={visible.length > 0 && visible.every((c) => sel.has(c.number))}
        />
      )}

      {sel.size > 0 && (
        <SelectionBar
          selected={sel.size}
          eligibleSpawn={eligibleForSpawn.length}
          onClear={() => setSel(new Set())}
        />
      )}
    </section>
  )
}


// ── Toolbar ─────────────────────────────────────────────────────────
//
// Two intents: READ (find chapters with VI translation) and TRANSLATE
// (find chapters without VI to spawn). Toolbar reflects exactly that:
//
//   [Tất cả] [Đã dịch] [Chưa có] [Đang dịch] [Lỗi]   [🔍 Tìm…]  [↓ Sort]
//
// "Chưa có" = raw chapters with no target-lang translation yet.
// Pills hide when count = 0 (except "Tất cả" and the active one).
// Search filters by chapter number or label.
// Sort cycles: mới nhất → cũ nhất → cập nhật gần đây.

function Toolbar({
  filter, setFilter, counts, q, setQ, sort, setSort,
}: {
  filter:    StatusFilter
  setFilter: (v: StatusFilter) => void
  counts:    Record<StatusFilter, number>
  q:         string
  setQ:      (v: string) => void
  sort:      Sort
  setSort:   (s: Sort) => void
}) {
  return (
    <div className="flex items-center gap-2 mb-4 flex-wrap">
      <div className="overflow-x-auto -mx-1 px-1 flex-1 min-w-0" style={{ scrollbarWidth: 'none' }}>
        <StatusPills value={filter} onChange={setFilter} counts={counts} />
      </div>
      <div className="flex items-center gap-1 shrink-0">
        <SearchInput value={q} onChange={setQ} placeholder="Tìm chương…" className="w-44" />
        <SortCycle value={sort} onChange={setSort} />
      </div>
    </div>
  )
}


const STATUS_FILTERS: { key: StatusFilter; label: string }[] = [
  { key: 'all',        label: 'Tất cả' },
  { key: 'translated', label: 'Đã dịch' },
  { key: 'raw',        label: 'Chưa có' },
  { key: 'running',    label: 'Đang dịch' },
  { key: 'error',      label: 'Lỗi' },
]

function StatusPills({
  value, onChange, counts,
}: {
  value:    StatusFilter
  onChange: (v: StatusFilter) => void
  counts:   Record<StatusFilter, number>
}) {
  return (
    <div className="inline-flex items-center gap-0.5">
      {STATUS_FILTERS.map(({ key, label }) => {
        const n = counts[key]
        const active = value === key
        if (!active && key !== 'all' && n === 0) return null
        return (
          <button
            key={key}
            type="button"
            onClick={() => onChange(key)}
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
              <span className={cn('tabular text-[11px]', active ? 'text-text-muted' : 'text-text-subtle')}>
                {n}
              </span>
            )}
          </button>
        )
      })}
    </div>
  )
}


const SORT_LABELS: Record<Sort, string> = {
  chapter_desc: 'Mới nhất',
  chapter_asc:  'Cũ nhất',
  updated_desc: 'Cập nhật',
}
const SORT_ICONS: Record<Sort, typeof ArrowDown> = {
  chapter_desc: ArrowDown,
  chapter_asc:  ArrowUp,
  updated_desc: Clock,
}

function SortCycle({ value, onChange }: { value: Sort; onChange: (s: Sort) => void }) {
  const order: Sort[] = ['chapter_desc', 'chapter_asc', 'updated_desc']
  const Icon = SORT_ICONS[value]
  return (
    <button
      type="button"
      onClick={() => onChange(order[(order.indexOf(value) + 1) % order.length]!)}
      title="Đổi cách sắp xếp"
      className="inline-flex items-center gap-1.5 h-8 px-2.5 rounded-sm text-[13px] text-text-muted bg-surface-2 hover:bg-hover hover:text-text transition-colors cursor-pointer shrink-0"
    >
      <Icon size={12} className="text-text-subtle" />
      <span className="hidden sm:inline">{SORT_LABELS[value]}</span>
    </button>
  )
}


// ── Selection bar ───────────────────────────────────────────────────
//
// Floating bottom-center bar, same pattern as the old ProjectDetail
// SelectionBar — stays visible while the user scrolls a long chapter
// list, instead of trapping itself at the top of the panel. The bulk
// spawn primary action is disabled with an explanatory tooltip until
// slice 15 wires the spawn dialog.

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
          <span className="text-text-subtle ml-1.5">
            · {eligibleSpawn} dịch được
          </span>
        )}
      </span>
      <Button
        variant="primary"
        size="sm"
        disabled
        title="Bulk spawn sẽ wire ở slice 15"
      >
        <Sparkles size={12} />
        Dịch hàng loạt ({eligibleSpawn})
      </Button>
      <Button variant="ghost" size="sm" icon onClick={onClear} aria-label="Bỏ chọn">
        <X size={14} />
      </Button>
    </div>
  )
}


// ── Table ───────────────────────────────────────────────────────────
//
// Real <table> — the chapter list IS tabular data (number, label,
// time, actions). Table primitive gives us automatic content-driven
// column widths (so chapter numbers '1' / '999' / '1.5' align across
// rows without min-width tricks), ARIA semantics, and predictable
// row geometry the browser optimizes for.
//
// Earlier mistake: hard-coding w-N on every <td>, keeping <thead>
// (4-line noise above 40 chapters of data the user already
// understands), and nesting <a><Button/></a> inside an
// <td text-right><div></div></td> action cell — that's the layout
// bug from the screenshot, not the <table> itself.
//
// This pass:
//   • No <thead>. Column labels are obvious from the data.
//   • <td>s let content drive width (no w-N except where a min is
//     genuinely required to avoid jitter on short labels).
//   • Action cell flattens: one <td> containing the action group,
//     no nested <a> wrapping the <Button/>. The action <a> sits as
//     a sibling of any secondary spawn so they share the cell.
//   • State stripe via box-shadow on the first <td> — paints over
//     row background reliably, no ::before pseudo gymnastics.

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
            {/* Checkbox lane — no label, just visual lane alignment.
                When selection mode is active, the master 'select all'
                checkbox lives here. */}
            <th className="pl-3 pr-2 py-2 w-8 text-left">
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


function ChapterRow({
  chapter, targetLang, checked, anySelected, onToggle,
}: {
  chapter:     HubChapter
  targetLang:  string | null
  checked:     boolean
  anySelected: boolean
  onToggle:    () => void
}) {
  const status     = chapterStatus(chapter, targetLang)
  const readable   = preferredReadable(chapter, targetLang)
  const running    = inFlight(chapter, targetLang)
  const errored    = lastError(chapter, targetLang)
  const label      = stripChapterPrefix(chapter.label, chapter.number)
  const hasRaw     = chapter.versions.some((v) => v.kind === 'raw')

  // State stripe color — first <td> renders this via an inset box
  // shadow on the LEFT edge. Box-shadow is the cleanest way to paint
  // a 2px stripe inside a cell without affecting layout, and unlike
  // a pseudo-element on <tr> it actually renders reliably across
  // engines.
  const stripeColor =
    status === 'running' ? 'var(--color-info)'
  : status === 'error'   ? 'var(--color-error)'
                         : 'transparent'

  return (
    <tr
      className={cn(
        'group transition-colors border-b border-border-soft last:border-0',
        checked ? 'bg-row-active' : 'hover:bg-hover',
      )}
    >
      {/* Checkbox cell — at-rest invisible, reveals on row hover or
          when any row is already selected. State stripe via
          box-shadow on this cell's left edge. */}
      <td
        className="pl-3 pr-1 py-3 w-8"
        style={{ boxShadow: `inset 2px 0 0 0 ${stripeColor}` }}
      >
        <div className={cn(
          'transition-opacity',
          checked ? 'opacity-100' : anySelected ? 'opacity-60 group-hover:opacity-100' : 'opacity-0 group-hover:opacity-100',
        )}>
          <Checkbox checked={checked} onClick={onToggle} ariaLabel="Chọn chương" />
        </div>
      </td>

      {/* Number — tabular, right-aligned, content-driven width. */}
      <td className="pr-3 py-3 whitespace-nowrap tabular text-right font-semibold text-text">
        {chapter.number}
      </td>

      {/* Label + sub-line — flex container fills available width. */}
      <td className="px-3 py-3 min-w-0 w-full">
        {label ? (
          <div className="text-sm text-text-muted truncate">{label}</div>
        ) : (
          <div className="text-sm text-text-subtle italic">Chương {chapter.number}</div>
        )}
        <SubLine
          status={status}
          running={running}
          errored={errored}
        />
      </td>

      <td className="px-3 py-3 hidden sm:table-cell whitespace-nowrap text-right w-px">
        <TimeCell iso={chapter.updatedAt} />
      </td>

      <td className="pl-2 pr-3 py-3 whitespace-nowrap text-right">
        <div className="inline-flex items-center gap-1 justify-end">
          <Action
            chapter={chapter}
            status={status}
            readable={readable}
            running={running}
            errored={errored}
          />
        </div>
      </td>
    </tr>
  )
}


// ── Time cell — color shifts by freshness ───────────────────────────
//
// 'mắt tự kéo về chương mới' — recent updates render in text-text,
// old ones fade to text-subtle. No NEW badge needed.

function TimeCell({ iso }: { iso: string | null }) {
  if (!iso) return <span className="text-sm text-text-subtle">—</span>
  const ageHours = (Date.now() - new Date(iso).getTime()) / 36e5
  const tone =
    ageHours < 24       ? 'text-text font-medium'
  : ageHours < 24 * 7   ? 'text-text-muted'
                        : 'text-text-subtle'
  return (
    <span className={cn('text-sm', tone)} title={iso}>
      {timeAgo(iso)}
    </span>
  )
}


// ── Secondary spawn button — '+Dịch' on rows that already have a
// readable target lang. Power-user affordance to commission a new
// translation when an existing one exists but a different source/
// quality is wanted. Disabled until the spawn dialog (slice 15) ships.


// ── Smart action button ─────────────────────────────────────────────
//
// Action mapping (slice 13 scope):
//   • readable.raw  → external link to upstreamUrl (opens source in
//                     new tab). Reader for raws ships with slice 14;
//                     until then the source page is the next best
//                     thing for a translated-read flow.
//   • readable.translation → disabled with explicit 'slice 14' hint.
//     Translation reader needs the page-by-page render fetch which
//     isn't wired yet.
//   • running       → disabled progress indicator; spawn-driven
//                     refresh will flip the row to readable once the
//                     translation lands.
//   • errored       → disabled retry; spawn dialog (slice 15) is
//                     where retry will live.
//   • raw (no target match yet) → disabled 'Dịch'; same dialog.
//
// Each disabled branch carries a `title` so the user understands
// WHY it's inert, not just THAT it is.

function Action({
  status, readable, running, errored,
}: {
  chapter:   HubChapter
  status:    StatusFilter
  readable:  HubVersion | null
  running:   HubVersion | null
  errored:   HubVersion | null
}) {
  if (status === 'running' && running) {
    return (
      <span className="inline-flex items-center gap-1 text-sm text-info-text">
        <Loader2 size={12} className="animate-spin" />
        Đang dịch
      </span>
    )
  }
  if (status === 'error' && errored) {
    // Retry not wired yet — show error state only, no action.
    return (
      <span className="inline-flex items-center gap-1 text-sm text-error-text">
        <AlertCircle size={12} />
        Lỗi
      </span>
    )
  }
  if (readable?.kind === 'raw' && readable.upstreamUrl) {
    // Raw with upstream URL — only actionable case right now.
    return (
      <a
        href={readable.upstreamUrl}
        target="_blank"
        rel="noopener noreferrer"
        title={`Mở chương trên ${readable.sourceName ?? 'nguồn'}`}
        className="inline-flex"
      >
        <Button size="sm" variant="secondary" tabIndex={-1}>
          <BookOpen size={12} />
          Đọc {readable.lang.toUpperCase()}
        </Button>
      </a>
    )
  }
  // Translation reader (slice 14) and spawn dialog (slice 15) not
  // wired yet — render nothing rather than a disabled placeholder.
  return null
}


// ── Row sub-line ────────────────────────────────────────────────────
//
// Only carries text when it ADDS information beyond the action button.
// The action button already encodes the readable state ('Đọc VI' /
// 'Đang dịch' / 'Thử lại' / 'Dịch'), so we skip a sub-line entirely
// on the happy 'translated' path and let the action button speak for
// the row. Sub-line shows up for:
//
//   • running   → '@creator' so the user knows whose draft is running.
//   • error     → 'Lỗi · @creator' so the user knows what failed.
//   • raw       → '+EN +KO' chip row of OTHER langs available, so the
//                 user can decide whether to spawn a translation or
//                 read an existing scanlation in a lang they know.
//
// On a 40/40 vi→vi title the sub-line is invisible across the board.

function SubLine({
  status, running, errored,
}: {
  status:   StatusFilter
  running:  HubVersion | null
  errored:  HubVersion | null
}) {
  if (status === 'running') {
    return (
      <div className="mt-1 inline-flex items-center gap-1.5 text-xs text-info-text min-w-0">
        <Loader2 size={11} className="animate-spin shrink-0" />
        <span className="truncate">
          {running?.creatorName ? `@${running.creatorName}` : 'Đang dịch'}
        </span>
      </div>
    )
  }
  if (status === 'error') {
    return (
      <div
        className="mt-1 inline-flex items-center gap-1.5 text-xs text-error-text min-w-0"
        title={errored?.creatorName
          ? `Lần dịch gần nhất của @${errored.creatorName} thất bại`
          : 'Lần dịch gần nhất thất bại'}
      >
        <AlertCircle size={11} className="shrink-0" />
        <span className="truncate">
          {errored?.creatorName ? `@${errored.creatorName}` : 'Lỗi'}
        </span>
      </div>
    )
  }
  return null
}


// ── Checkbox primitive ──────────────────────────────────────────────

function Checkbox({
  checked, onClick, ariaLabel,
}: {
  checked:   boolean
  onClick:   () => void
  ariaLabel: string
}) {
  return (
    <button
      type="button"
      onClick={onClick}
      aria-label={ariaLabel}
      className={cn(
        'size-4 rounded-xs border flex items-center justify-center cursor-pointer transition-colors',
        checked
          ? 'bg-accent border-accent text-accent-fg'
          : 'border-text-subtle hover:border-text-muted',
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


// ── Status derivation ──────────────────────────────────────────────

function chapterStatus(c: HubChapter, targetLang: string | null): StatusFilter {
  const tgt = targetLang?.toLowerCase() ?? null
  if (!tgt) return 'raw'
  if (preferredReadable(c, tgt)) return 'translated'
  if (inFlight(c, tgt))           return 'running'
  if (lastError(c, tgt))          return 'error'
  return 'raw'
}


function countByStatus(
  chapters: HubChapter[],
  targetLang: string | null,
): Record<StatusFilter, number> {
  const out: Record<StatusFilter, number> = {
    all: chapters.length,
    translated: 0, running: 0, error: 0, raw: 0,
  }
  for (const c of chapters) out[chapterStatus(c, targetLang)]++
  return out
}


// ── Facet derivation ───────────────────────────────────────────────
//
// Distinct langs + uploaders across all loaded chapters, with chapter
// counts. Counts measure "chapters that have ≥1 matching version",
// not version totals — what the user actually filters on.


function sortFn(s: Sort): (a: HubChapter, b: HubChapter) => number {
  switch (s) {
    case 'chapter_desc':   return (a, b) => b.sortKey - a.sortKey
    case 'chapter_asc':    return (a, b) => a.sortKey - b.sortKey
    case 'updated_desc':   return (a, b) => (b.updatedAt ?? '').localeCompare(a.updatedAt ?? '')
  }
}
