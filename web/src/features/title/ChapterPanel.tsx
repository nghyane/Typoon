import { useEffect, useMemo, useRef, useState } from 'react'
import {
  Sparkles, Loader2, AlertCircle,
  BookOpen, ArrowDown, ArrowUp, Clock, RefreshCw, X, Check,
  Globe, UserCircle2,
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
  const filter   = search.filter   ?? 'all'
  const q        = search.q        ?? ''
  const sort     = search.sort     ?? 'chapter_desc'
  const lang     = search.lang     ?? null
  const uploader = search.uploader ?? null

  const setFilter = (next: StatusFilter) =>
    nav({ search: (s) => ({ ...s, filter: next === 'all' ? undefined : next }) })
  const setQ = (next: string) =>
    nav({ search: (s) => ({ ...s, q: next || undefined }) })
  const setSort = (next: Sort) =>
    nav({ search: (s) => ({ ...s, sort: next === 'chapter_desc' ? undefined : next }) })
  const setLang = (next: string | null) =>
    nav({ search: (s) => ({ ...s, lang: next ?? undefined }) })
  const setUploader = (next: string | null) =>
    nav({ search: (s) => ({ ...s, uploader: next ?? undefined }) })

  const [sel, setSel] = useState<Set<string>>(new Set())

  const counts = useMemo(() => countByStatus(chapters, targetLang), [chapters, targetLang])
  // Distinct dimensions across all loaded chapters. Drives the lang
  // and uploader filter menus — only render the menu when there's
  // an actual choice to make (≥ 2 options).
  const facets = useMemo(() => collectFacets(chapters), [chapters])

  const visible = useMemo(() => {
    const term = q.trim().toLowerCase()
    let list = chapters.filter((c) => {
      if (term && !`${c.number} ${c.label ?? ''}`.toLowerCase().includes(term)) {
        return false
      }
      if (lang && !c.versions.some((v) => v.lang === lang)) return false
      if (uploader && !c.versions.some(
        (v) => v.kind === 'translation' && v.creatorName === uploader,
      )) return false
      if (filter === 'all') return true
      const status = chapterStatus(c, targetLang)
      return status === filter
    })
    list = list.slice().sort(sortFn(sort))
    return list
  }, [chapters, q, filter, sort, lang, uploader, targetLang])

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
        q={q} setQ={setQ}
        filter={filter} setFilter={setFilter} counts={counts}
        sort={sort} setSort={setSort}
        lang={lang} setLang={setLang}
        uploader={uploader} setUploader={setUploader}
        facets={facets}
        visibleCount={visible.length}
        totalCount={chapters.length}
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
// One row, two clusters, divider in between:
//
//   [Tất cả 40] [Đã dịch 38] [Lỗi 2]  │  [🌐 VI] [👤 @nick] [🔍 Tìm…] [↓] [40/40]
//   └── status filter (ghost pills) ─┘   └────────── view tools (surface-2) ──────────┘
//
// Status pills stay ghost so the active one reads as a tab. The
// right cluster shares bg-surface-2 across every control — Lang
// menu, Uploader menu, SearchInput, Sort, and the X/Y count chip.
// Lang and Uploader collapse to nothing when there's only one
// option (don't ask the user to pick from a list of 1).
//
// Mobile: row wraps, pills scroll-x, tools row falls below.

interface ToolbarProps {
  q:            string
  setQ:         (v: string) => void
  filter:       StatusFilter
  setFilter:    (v: StatusFilter) => void
  counts:       Record<StatusFilter, number>
  sort:         Sort
  setSort:      (s: Sort) => void
  lang:         string | null
  setLang:      (v: string | null) => void
  uploader:     string | null
  setUploader:  (v: string | null) => void
  facets:       Facets
  visibleCount: number
  totalCount:   number
}

function Toolbar({
  q, setQ, filter, setFilter, counts, sort, setSort,
  lang, setLang, uploader, setUploader, facets,
  visibleCount, totalCount,
}: ToolbarProps) {
  const showLang     = facets.langs.length >= 2
  const showUploader = facets.uploaders.length >= 2
  return (
    <div className="flex flex-wrap items-center gap-2 mb-4">
      <div
        className="overflow-x-auto -mx-1 px-1 min-w-0"
        style={{ scrollbarWidth: 'none' }}
      >
        <Segmented value={filter} onChange={setFilter} counts={counts} />
      </div>

      <div className="hidden sm:block h-5 w-px bg-border" aria-hidden />

      <div className="flex items-center gap-1 flex-1 sm:flex-initial min-w-0">
        {showLang && (
          <FilterMenu
            icon={Globe}
            label={lang ? lang.toUpperCase() : 'Ngôn ngữ'}
            active={lang !== null}
            options={facets.langs.map((l) => ({
              value: l,
              label: l.toUpperCase(),
              count: facets.langCount[l] ?? 0,
            }))}
            value={lang}
            onPick={setLang}
          />
        )}
        {showUploader && (
          <FilterMenu
            icon={UserCircle2}
            label={uploader ? `@${uploader}` : 'Người dịch'}
            active={uploader !== null}
            options={facets.uploaders.map((u) => ({
              value: u,
              label: `@${u}`,
              count: facets.uploaderCount[u] ?? 0,
            }))}
            value={uploader}
            onPick={setUploader}
          />
        )}
        <SearchInput
          value={q}
          onChange={setQ}
          placeholder="Tìm chương…"
          className="flex-1 sm:w-56 sm:flex-initial min-w-0"
        />
        <SortCycle value={sort} onChange={setSort} />
        <CountChip visible={visibleCount} total={totalCount} />
      </div>
    </div>
  )
}


// ── Count chip — anchored to the right of the tools cluster ─────────
//
// Only renders when a filter is actually narrowing the list. When
// visible == total, the hero's aggregate progress bar already
// communicates the count; doubling it here is noise.

function CountChip({ visible, total }: { visible: number; total: number }) {
  if (visible === total) return null
  return (
    <span
      className={cn(
        'inline-flex items-center h-8 px-2.5 rounded-sm shrink-0',
        'text-[12px] tabular bg-surface-2 text-text-muted',
      )}
      title={`Đang lọc — ${visible} trong ${total} chương`}
    >
      <span className="text-text font-medium">{visible}</span>
      <span className="opacity-50 mx-0.5">/</span>
      {total}
    </span>
  )
}


// ── Lang / Uploader popover menu ────────────────────────────────────
//
// Click-outside / Escape to dismiss. Native HTML <details> would be
// shorter but the disclosure arrow is visually noisy and styling is
// browser-specific. We keep it manual for full control over the
// surface and dock direction.

interface MenuOption {
  value: string
  label: string
  count: number
}

function FilterMenu({
  icon: Icon, label, active, options, value, onPick,
}: {
  icon:    typeof Globe
  label:   string
  active:  boolean
  options: MenuOption[]
  value:   string | null
  onPick:  (v: string | null) => void
}) {
  const [open, setOpen] = useState(false)
  const ref = useRef<HTMLDivElement>(null)

  useEffect(() => {
    if (!open) return
    const onDocPointer = (e: PointerEvent) => {
      if (!ref.current?.contains(e.target as Node)) setOpen(false)
    }
    const onKey = (e: KeyboardEvent) => {
      if (e.key === 'Escape') setOpen(false)
    }
    document.addEventListener('pointerdown', onDocPointer)
    document.addEventListener('keydown', onKey)
    return () => {
      document.removeEventListener('pointerdown', onDocPointer)
      document.removeEventListener('keydown', onKey)
    }
  }, [open])

  return (
    <div ref={ref} className="relative shrink-0">
      <button
        type="button"
        onClick={() => setOpen((v) => !v)}
        className={cn(
          'inline-flex items-center gap-1.5 h-8 px-2.5 rounded-sm',
          'text-[13px] bg-surface-2 transition-colors cursor-pointer',
          'hover:bg-hover',
          active ? 'text-text' : 'text-text-muted hover:text-text',
        )}
        aria-haspopup="listbox"
        aria-expanded={open}
      >
        <Icon size={12} className={active ? 'text-text' : 'text-text-subtle'} />
        <span className="max-w-[10ch] truncate">{label}</span>
        {active && (
          <span
            role="button"
            tabIndex={0}
            onPointerDown={(e) => {
              e.stopPropagation()
              onPick(null)
            }}
            className="ml-0.5 -mr-1 size-4 inline-flex items-center justify-center rounded-xs hover:bg-hover text-text-subtle hover:text-text"
            aria-label="Bỏ lọc"
          >
            <X size={11} />
          </span>
        )}
      </button>

      {open && (
        <div
          className={cn(
            card,
            'absolute top-full left-0 mt-1 z-30 min-w-[200px] py-1',
            'shadow-[0_8px_24px_rgb(0,0,0,0.35)] border border-border-soft',
          )}
          role="listbox"
        >
          <button
            type="button"
            onClick={() => { onPick(null); setOpen(false) }}
            className={cn(
              'w-full flex items-center justify-between gap-3 px-3 h-8',
              'text-[13px] text-left cursor-pointer transition-colors hover:bg-hover',
              value === null ? 'text-text font-medium' : 'text-text-muted',
            )}
          >
            <span>Tất cả</span>
            {value === null && <Check size={12} className="text-accent" />}
          </button>
          <div className="h-px bg-border-soft my-1" />
          {options.map((opt) => {
            const selected = value === opt.value
            return (
              <button
                key={opt.value}
                type="button"
                onClick={() => { onPick(opt.value); setOpen(false) }}
                className={cn(
                  'w-full flex items-center justify-between gap-3 px-3 h-8',
                  'text-[13px] text-left cursor-pointer transition-colors hover:bg-hover',
                  selected ? 'text-text font-medium' : 'text-text-muted',
                )}
              >
                <span className="truncate">{opt.label}</span>
                <span className="flex items-center gap-2 shrink-0">
                  <span className="text-[11px] tabular text-text-subtle">{opt.count}</span>
                  {selected && <Check size={12} className="text-accent" />}
                </span>
              </button>
            )
          })}
        </div>
      )}
    </div>
  )
}


const FILTERS: { key: StatusFilter; label: string }[] = [
  { key: 'all',        label: 'Tất cả' },
  { key: 'translated', label: 'Đã dịch' },
  { key: 'running',    label: 'Đang dịch' },
  { key: 'error',      label: 'Lỗi' },
  { key: 'raw',        label: 'Raw' },
]

function Segmented({
  value, onChange, counts,
}: {
  value:    StatusFilter
  onChange: (v: StatusFilter) => void
  counts:   Record<StatusFilter, number>
}) {
  return (
    <div className="inline-flex items-center gap-0.5">
      {FILTERS.map(({ key, label }) => {
        const n = counts[key]
        const active = value === key
        // Hide empty buckets EXCEPT 'all' and the currently selected
        // one — keeps the toolbar from showing 'Lỗi 0' all the time
        // while still letting the user stay on a filter that just
        // emptied (so they can switch off it).
        if (!active && key !== 'all' && n === 0) return null
        return (
          <button
            key={key}
            type="button"
            onClick={() => onChange(key)}
            className={cn(
              'h-8 px-3 rounded-sm text-[13px] cursor-pointer transition-colors',
              'inline-flex items-center gap-1.5 whitespace-nowrap',
              active
                ? 'bg-surface-2 text-text font-medium'
                : 'text-text-muted hover:bg-hover hover:text-text',
            )}
          >
            {label}
            {n > 0 && (
              <span className={cn(
                'tabular text-[11px] tracking-tight',
                active ? 'text-text-muted' : 'text-text-subtle',
              )}>
                {n}
              </span>
            )}
          </button>
        )
      })}
    </div>
  )
}


const SORT_LABEL: Record<Sort, string> = {
  chapter_desc: 'Mới nhất',
  chapter_asc:  'Cũ nhất',
  updated_desc: 'Cập nhật',
}

const SORT_ICON: Record<Sort, typeof ArrowDown> = {
  chapter_desc: ArrowDown,
  chapter_asc:  ArrowUp,
  updated_desc: Clock,
}

// Same h-8 surface-2 affordance as SearchInput so the two read as
// one "view tools" cluster. Click cycles through the 3 modes — a
// dropdown would be heavier UI for a 3-option pick.
function SortCycle({
  value, onChange,
}: {
  value: Sort; onChange: (s: Sort) => void
}) {
  const order: Sort[] = ['chapter_desc', 'chapter_asc', 'updated_desc']
  const next = () => {
    const i = order.indexOf(value)
    onChange(order[(i + 1) % order.length]!)
  }
  const Icon = SORT_ICON[value]
  return (
    <button
      type="button"
      onClick={next}
      title="Đổi cách sắp xếp"
      className={cn(
        'inline-flex items-center gap-1.5 h-8 px-2.5 rounded-sm shrink-0',
        'text-[13px] text-text-muted bg-surface-2',
        'hover:bg-hover hover:text-text transition-colors cursor-pointer',
      )}
    >
      <Icon size={12} className="text-text-subtle" />
      <span className="hidden sm:inline">{SORT_LABEL[value]}</span>
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
              {anySelected && (
                <Checkbox checked={allChecked} onClick={onToggleAll} ariaLabel="Chọn tất cả" />
              )}
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
  const canSpawn   = hasRaw && status !== 'running'

  // State stripe color — first <td> renders this via an inset box
  // shadow on the LEFT edge. Box-shadow is the cleanest way to paint
  // a 2px stripe inside a cell without affecting layout, and unlike
  // a pseudo-element on <tr> it actually renders reliably across
  // engines.
  const stripeColor =
    status === 'running' ? 'var(--color-info)'
  : status === 'error'   ? 'var(--color-error)'
  : status === 'raw'     ? 'var(--color-warning)'
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

      <td className="px-3 py-3 hidden sm:table-cell whitespace-nowrap text-right">
        <TimeCell iso={chapter.updatedAt} />
      </td>

      <td className="pl-2 pr-3 py-3 whitespace-nowrap text-right">
        <div className="inline-flex items-center gap-1 justify-end">
          {canSpawn && status !== 'raw' && status !== 'error' && (
            <SecondarySpawn />
          )}
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
  if (!iso) return <span className="text-xs text-text-subtle">—</span>
  const ageHours = (Date.now() - new Date(iso).getTime()) / 36e5
  const tone =
    ageHours < 24       ? 'text-text font-medium'
  : ageHours < 24 * 7   ? 'text-text-muted'
                        : 'text-text-subtle'
  return (
    <span className={cn('text-xs', tone)} title={iso}>
      {timeAgo(iso)}
    </span>
  )
}


// ── Secondary spawn button — '+Dịch' on rows that already have a
// readable target lang. Power-user affordance to commission a new
// translation when an existing one exists but a different source/
// quality is wanted. Disabled until the spawn dialog (slice 15) ships.

function SecondarySpawn() {
  return (
    <button
      type="button"
      disabled
      title="Dịch lại từ nguồn khác — slice 15 sẽ wire popover chọn source/lang"
      className={cn(
        'inline-flex items-center gap-0.5 h-7 px-2 rounded-sm shrink-0',
        'text-[12px] text-text-subtle cursor-not-allowed',
        'opacity-0 group-hover:opacity-100 transition-opacity',
        'hover:bg-hover hover:text-text-muted',
      )}
    >
      <Sparkles size={11} />
      <span className="hidden sm:inline">Dịch</span>
    </button>
  )
}


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
      <Button
        size="sm"
        variant="secondary"
        disabled
        title="Pipeline đang dịch. Trạng thái sẽ tự cập nhật khi xong."
      >
        <Loader2 size={12} className="animate-spin" />
        Đang dịch
      </Button>
    )
  }
  if (status === 'error' && errored) {
    return (
      <Button
        size="sm"
        variant="secondary"
        disabled
        title="Dialog dịch lại sẽ wire ở slice 15."
      >
        <RefreshCw size={12} />
        Thử lại
      </Button>
    )
  }
  if (readable) {
    if (readable.kind === 'raw' && readable.upstreamUrl) {
      // Raw read = open source in a new tab. Same as the row's
      // primary click target — having the button as a discrete
      // anchor lets users click either spot with the same effect.
      return (
        <a
          href={readable.upstreamUrl}
          target="_blank"
          rel="noopener noreferrer"
          // Don't double-trigger when the row link is also clicked.
          onClick={(e) => e.stopPropagation()}
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
    return (
      <Button
        size="sm"
        variant="secondary"
        disabled
        title="Reader sẽ wire ở slice 14."
      >
        <BookOpen size={12} />
        Đọc {readable.lang.toUpperCase()}
      </Button>
    )
  }
  return (
    <Button
      size="sm"
      variant="primary"
      disabled
      title="Dialog dịch sẽ wire ở slice 15."
    >
      <Sparkles size={12} />
      Dịch
    </Button>
  )
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

interface Facets {
  langs:          string[]
  langCount:      Record<string, number>
  uploaders:      string[]
  uploaderCount:  Record<string, number>
}

function collectFacets(chapters: HubChapter[]): Facets {
  const langSet     = new Set<string>()
  const uploaderSet = new Set<string>()
  const langCount: Record<string, number>     = {}
  const uploaderCount: Record<string, number> = {}

  for (const c of chapters) {
    const seenLangs:     Set<string> = new Set()
    const seenUploaders: Set<string> = new Set()
    for (const v of c.versions) {
      if (v.lang && v.lang !== '?') {
        seenLangs.add(v.lang)
        langSet.add(v.lang)
      }
      if (v.kind === 'translation' && v.creatorName) {
        seenUploaders.add(v.creatorName)
        uploaderSet.add(v.creatorName)
      }
    }
    for (const l of seenLangs)     langCount[l]     = (langCount[l]     ?? 0) + 1
    for (const u of seenUploaders) uploaderCount[u] = (uploaderCount[u] ?? 0) + 1
  }

  // Langs: most chapters first; ties → alpha.
  const langs = [...langSet].sort((a, b) => {
    const da = langCount[b]! - langCount[a]!
    return da !== 0 ? da : a.localeCompare(b)
  })
  const uploaders = [...uploaderSet].sort((a, b) => {
    const da = uploaderCount[b]! - uploaderCount[a]!
    return da !== 0 ? da : a.localeCompare(b)
  })

  return { langs, langCount, uploaders, uploaderCount }
}


function sortFn(s: Sort): (a: HubChapter, b: HubChapter) => number {
  switch (s) {
    case 'chapter_desc':   return (a, b) => b.sortKey - a.sortKey
    case 'chapter_asc':    return (a, b) => a.sortKey - b.sortKey
    case 'updated_desc':   return (a, b) => (b.updatedAt ?? '').localeCompare(a.updatedAt ?? '')
  }
}
