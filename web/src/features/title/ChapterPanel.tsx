import { useMemo, useState } from 'react'
import {
  BookOpen, Sparkles, Loader2, AlertCircle, CheckCircle2,
  ArrowDown, RefreshCw, Check, X,
} from 'lucide-react'
import { Button } from '@shared/ui/Button'
import { EmptyState } from '@shared/ui/EmptyState'
import { DataTable, Th } from '@shared/ui/DataTable'
import { SearchInput, DataToolbar } from '@shared/ui/DataToolbar'
import { card } from '@shared/ui/primitives'
import { cn } from '@shared/lib/cn'
import { timeAgo } from '@shared/lib/time'
import {
  preferredReadable, inFlight, lastError, chapterLangs,
  type HubChapter, type HubVersion,
} from './mergeChapters'

// =============================================================================
// ChapterPanel — hub's chapter list with bulk-select support.
//
// UX agreed in the design pass (see chat history):
//   • Toolbar: search left, status filter + sort right.
//   • SelectionBar appears once at least one chapter is checked;
//     primary action 'Dịch hàng loạt' eligibility-filters to chapters
//     that don't yet have a readable target_lang version.
//   • Row: checkbox + 'Ch.N + label' + lang chips + smart action.
//     Smart action picks 'Đọc {lang}' when a readable version exists,
//     'Xem tiến độ' when running/pending, 'Thử lại' on error, 'Dịch'
//     otherwise. Caret ▾ on multi-version rows opens an Action menu
//     with every version + 'Dịch riêng' force-spawn.
// =============================================================================

type StatusFilter = 'all' | 'translated' | 'running' | 'error' | 'raw'
type Sort         = 'chapter_desc' | 'chapter_asc' | 'updated_desc'

interface Props {
  chapters:   HubChapter[]
  targetLang: string | null
  loading:    boolean
}

export function ChapterPanel({ chapters, targetLang, loading }: Props) {
  const [q,      setQ]      = useState('')
  const [filter, setFilter] = useState<StatusFilter>('all')
  const [sort,   setSort]   = useState<Sort>('chapter_desc')
  const [sel,    setSel]    = useState<Set<string>>(new Set())

  const counts = useMemo(() => countByStatus(chapters, targetLang), [chapters, targetLang])

  const visible = useMemo(() => {
    const term = q.trim().toLowerCase()
    let list = chapters.filter((c) => {
      if (term && !`${c.number} ${c.label ?? ''}`.toLowerCase().includes(term)) {
        return false
      }
      if (filter === 'all') return true
      const status = chapterStatus(c, targetLang)
      return status === filter
    })
    list = list.slice().sort(sortFn(sort))
    return list
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
        q={q} setQ={setQ}
        filter={filter} setFilter={setFilter} counts={counts}
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

      {!loading && visible.length > 0 && (
        <p className="text-xs text-text-subtle mt-3 tabular">
          Hiển thị <span className="text-text-muted">{visible.length}</span> trong{' '}
          <span className="text-text-muted">{chapters.length}</span> chương
        </p>
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

// Segmented filter + search, mirroring ProjectDetail's DataToolbar
// pattern. Counts surface on the pills directly so the user sees
// state distribution at a glance — no dropdown to open.
function Toolbar({
  q, setQ, filter, setFilter, counts, sort, setSort,
}: {
  q:         string
  setQ:      (v: string) => void
  filter:    StatusFilter
  setFilter: (v: StatusFilter) => void
  counts:    Record<StatusFilter, number>
  sort:      Sort
  setSort:   (s: Sort) => void
}) {
  return (
    <DataToolbar right={<SortCycle value={sort} onChange={setSort} />}>
      <div className="overflow-x-auto">
        <Segmented value={filter} onChange={setFilter} counts={counts} />
      </div>
      <SearchInput
        value={q}
        onChange={setQ}
        placeholder="Tìm chương…"
        className="flex-1 min-w-32"
      />
    </DataToolbar>
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
        return (
          <button
            key={key}
            type="button"
            onClick={() => onChange(key)}
            className={cn(
              'h-8 px-3 rounded-sm text-[13px] cursor-pointer transition-colors',
              'inline-flex items-center gap-2 whitespace-nowrap',
              active
                ? 'bg-surface-2 text-text font-medium'
                : 'text-text-muted hover:bg-hover hover:text-text',
            )}
          >
            {label}
            {n > 0 && (
              <span className={cn(
                'tabular text-[11px]',
                active ? 'text-text-subtle' : 'text-text-subtle/80',
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
  chapter_desc: 'Mới → cũ',
  chapter_asc:  'Cũ → mới',
  updated_desc: 'Cập nhật',
}

// Cycle through the 3 sort modes. A 3-state toggle keeps the toolbar
// flat — no popover, no native <select> chrome leaking into the
// design system.
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
  return (
    <button
      type="button"
      onClick={next}
      title="Đổi cách sắp xếp"
      className="inline-flex items-center gap-1.5 h-8 px-2.5 rounded-sm text-[13px] text-text-muted hover:bg-hover hover:text-text transition-colors cursor-pointer"
    >
      <ArrowDown size={12} className="text-text-subtle" />
      {SORT_LABEL[value]}
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
  return (
    <DataTable className="overflow-x-auto">
      <thead>
        <tr className="bg-surface-2">
          <Th className="w-10">
            <Checkbox checked={allChecked} onClick={onToggleAll} ariaLabel="Chọn tất cả" />
          </Th>
          <Th>Chương</Th>
          <Th className="w-72 hidden sm:table-cell">Trạng thái</Th>
          <Th className="w-24 hidden sm:table-cell">Cập nhật</Th>
          <Th className="w-36 text-right pr-3">Thao tác</Th>
        </tr>
      </thead>
      <tbody>
        {chapters.map((c) => (
          <ChapterRow
            key={c.number}
            chapter={c}
            targetLang={targetLang}
            checked={sel.has(c.number)}
            onToggle={() => onToggle(c.number)}
          />
        ))}
      </tbody>
    </DataTable>
  )
}


function ChapterRow({
  chapter, targetLang, checked, onToggle,
}: {
  chapter:    HubChapter
  targetLang: string | null
  checked:    boolean
  onToggle:   () => void
}) {
  const status   = chapterStatus(chapter, targetLang)
  const readable = preferredReadable(chapter, targetLang)
  const running  = inFlight(chapter, targetLang)
  const errored  = lastError(chapter, targetLang)
  const langs    = chapterLangs(chapter)

  return (
    <tr
      className={cn(
        'transition-colors border-b border-border-soft last:border-0',
        checked ? 'bg-row-active' : 'hover:bg-hover',
      )}
      style={checked ? { boxShadow: 'inset 2px 0 0 0 var(--color-accent)' } : undefined}
    >
      <td className="px-3 py-3 w-10">
        <Checkbox checked={checked} onClick={onToggle} ariaLabel="Chọn chương" />
      </td>

      <td className="px-3 py-3 min-w-0">
        <div className="flex items-baseline gap-2 min-w-0">
          <span className="font-semibold text-text tabular shrink-0">
            Ch.{chapter.number}
          </span>
          {chapter.label && (
            <span className="text-sm text-text-muted truncate">
              {chapter.label}
            </span>
          )}
        </div>
        <LangChips
          langs={langs}
          targetLang={targetLang}
          className="mt-1.5 sm:hidden"
        />
        <div className="sm:hidden mt-1">
          <StatusInline status={status} readable={readable} />
        </div>
      </td>

      <td className="px-3 py-3 w-72 hidden sm:table-cell">
        <StatusInline status={status} readable={readable} />
        <LangChips langs={langs} targetLang={targetLang} className="mt-1.5" />
      </td>

      <td className="px-3 py-3 text-xs text-text-subtle whitespace-nowrap w-24 tabular hidden sm:table-cell">
        {chapter.updatedAt ? timeAgo(chapter.updatedAt) : '—'}
      </td>

      <td className="px-3 py-3 w-36">
        <div className="flex items-center gap-1 justify-end">
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


// ── Smart action button ─────────────────────────────────────────────

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
      <Button size="sm" variant="secondary" disabled>
        <Loader2 size={12} className="animate-spin" />
        Đang dịch
      </Button>
    )
  }
  if (status === 'error' && errored) {
    return (
      <Button size="sm" variant="secondary" disabled>
        <RefreshCw size={12} />
        Thử lại
      </Button>
    )
  }
  if (readable) {
    // TODO(slice 14): wire reader route /title/$entryId/ch/$cid?tx=
    return (
      <Button size="sm" variant="secondary" disabled>
        <BookOpen size={12} />
        Đọc {readable.lang.toUpperCase()}
      </Button>
    )
  }
  return (
    <Button size="sm" variant="primary" disabled>
      <Sparkles size={12} />
      Dịch
    </Button>
  )
}


// ── Inline status text (column + mobile) ────────────────────────────

function StatusInline({
  status, readable,
}: {
  status:   StatusFilter
  readable: HubVersion | null
}) {
  if (status === 'translated' && readable) {
    return (
      <span className="inline-flex items-center gap-1.5 text-xs text-success-text">
        <CheckCircle2 size={11} />
        Đã dịch
        {readable.creatorName && (
          <span className="text-text-subtle ml-1 truncate">
            · @{readable.creatorName}
          </span>
        )}
      </span>
    )
  }
  if (status === 'running') {
    return (
      <span className="inline-flex items-center gap-1.5 text-xs text-info-text">
        <Loader2 size={11} className="animate-spin" />
        Đang dịch
      </span>
    )
  }
  if (status === 'error') {
    return (
      <span className="inline-flex items-center gap-1.5 text-xs text-error-text">
        <AlertCircle size={11} />
        Lỗi
      </span>
    )
  }
  return (
    <span className="inline-flex items-center gap-1.5 text-xs text-text-subtle">
      <span className="size-1.5 rounded-full bg-text-subtle" />
      Raw
    </span>
  )
}


function LangChips({
  langs, targetLang, className,
}: {
  langs:      string[]
  targetLang: string | null
  className?: string
}) {
  if (langs.length === 0) return null
  return (
    <div className={cn('flex flex-wrap items-center gap-1', className)}>
      {langs.slice(0, 5).map((l) => {
        const active = targetLang && l === targetLang.toLowerCase()
        return (
          <span
            key={l}
            className={cn(
              'inline-flex items-center h-4 px-1 rounded-xs text-[10px] font-semibold uppercase',
              active
                ? 'bg-success/15 text-success-text'
                : 'bg-bg/30 text-text-subtle',
            )}
          >
            {l}
          </span>
        )
      })}
      {langs.length > 5 && (
        <span className="text-[10px] text-text-subtle">+{langs.length - 5}</span>
      )}
    </div>
  )
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


function sortFn(s: Sort): (a: HubChapter, b: HubChapter) => number {
  switch (s) {
    case 'chapter_desc':   return (a, b) => b.sortKey - a.sortKey
    case 'chapter_asc':    return (a, b) => a.sortKey - b.sortKey
    case 'updated_desc':   return (a, b) => (b.updatedAt ?? '').localeCompare(a.updatedAt ?? '')
  }
}
