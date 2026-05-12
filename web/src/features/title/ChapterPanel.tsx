import { useMemo, useState } from 'react'
import {
  BookOpen, Sparkles, Loader2, AlertCircle, CheckCircle2,
  ChevronDown, ArrowDown, RefreshCw,
} from 'lucide-react'
import { Button } from '@shared/ui/Button'
import { EmptyState } from '@shared/ui/EmptyState'
import { DataTable, Th } from '@shared/ui/DataTable'
import { SearchInput } from '@shared/ui/DataToolbar'
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

      {sel.size > 0 && (
        <SelectionBar
          selected={sel.size}
          eligibleSpawn={eligibleForSpawn.length}
          onClear={() => setSel(new Set())}
          onBulkSpawn={() => {
            // TODO(slice 15): wire bulk spawn dialog.
            // eslint-disable-next-line no-alert
            alert(`Sẽ dịch ${eligibleForSpawn.length} chương — slice 15 sẽ wire`)
          }}
        />
      )}

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
    </section>
  )
}


// ── Toolbar ─────────────────────────────────────────────────────────

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
    <div className="flex flex-wrap items-center gap-2 mb-3">
      <SearchInput
        value={q}
        onChange={setQ}
        placeholder="Tìm chương…"
        className="flex-1 min-w-40 max-w-sm"
      />
      <div className="flex items-center gap-1 ml-auto">
        <StatusSelect value={filter} onChange={setFilter} counts={counts} />
        <SortSelect value={sort} onChange={setSort} />
      </div>
    </div>
  )
}


function StatusSelect({
  value, onChange, counts,
}: {
  value:    StatusFilter
  onChange: (v: StatusFilter) => void
  counts:   Record<StatusFilter, number>
}) {
  return (
    <label className="inline-flex items-center gap-1.5 h-8 px-2.5 rounded-sm bg-surface-2 hover:bg-hover transition-colors cursor-pointer">
      <select
        value={value}
        onChange={(e) => onChange(e.target.value as StatusFilter)}
        className="bg-transparent text-[13px] text-text outline-none cursor-pointer pr-1"
      >
        <option value="all">Tất cả ({counts.all})</option>
        <option value="translated">Đã dịch ({counts.translated})</option>
        <option value="running">Đang dịch ({counts.running})</option>
        <option value="error">Lỗi ({counts.error})</option>
        <option value="raw">Raw ({counts.raw})</option>
      </select>
      <ChevronDown size={11} className="text-text-subtle" aria-hidden />
    </label>
  )
}


function SortSelect({
  value, onChange,
}: {
  value: Sort; onChange: (s: Sort) => void
}) {
  return (
    <label className="inline-flex items-center gap-1.5 h-8 px-2.5 rounded-sm bg-surface-2 hover:bg-hover transition-colors cursor-pointer">
      <ArrowDown size={12} className="text-text-subtle" />
      <select
        value={value}
        onChange={(e) => onChange(e.target.value as Sort)}
        className="bg-transparent text-[13px] text-text outline-none cursor-pointer"
      >
        <option value="chapter_desc">Chương mới → cũ</option>
        <option value="chapter_asc">Chương cũ → mới</option>
        <option value="updated_desc">Cập nhật gần đây</option>
      </select>
    </label>
  )
}


// ── Selection bar ───────────────────────────────────────────────────

function SelectionBar({
  selected, eligibleSpawn, onClear, onBulkSpawn,
}: {
  selected:      number
  eligibleSpawn: number
  onClear:       () => void
  onBulkSpawn:   () => void
}) {
  return (
    <div className="flex items-center gap-2 mb-3 px-3 h-10 rounded-md bg-accent/10 border border-accent/20">
      <span className="text-[13px] text-text">
        <span className="font-medium">{selected}</span> đã chọn
        {eligibleSpawn < selected && (
          <span className="text-text-subtle ml-1.5">
            · {eligibleSpawn} dịch được
          </span>
        )}
      </span>
      <button
        type="button"
        onClick={onClear}
        className="text-xs text-text-subtle hover:text-text underline-offset-2 hover:underline cursor-pointer"
      >
        Bỏ chọn
      </button>
      <div className="flex-1" />
      <Button
        variant="primary"
        size="sm"
        onClick={onBulkSpawn}
        disabled={eligibleSpawn === 0}
        title={eligibleSpawn === 0
          ? 'Tất cả chương đã chọn đều đã có bản dịch — không cần dịch lại'
          : `Dịch ${eligibleSpawn} chương chưa có bản dịch`
        }
      >
        <Sparkles size={12} />
        Dịch hàng loạt ({eligibleSpawn})
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
