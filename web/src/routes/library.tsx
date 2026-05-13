import { createFileRoute, useNavigate } from '@tanstack/react-router'
import { useEffect, useState } from 'react'
import {
  Plus, Library, BookOpen, BookmarkCheck, CheckCircle2, Layers,
  Loader2, AlertCircle,
} from 'lucide-react'
import { cn } from '@shared/lib/cn'
import { EmptyState } from '@shared/ui/EmptyState'
import { Spinner } from '@shared/ui/primitives'
import { Button } from '@shared/ui/Button'
import { useSources } from '@features/browse/sources'
import { useUnifiedLibrary, type LibraryFilter } from '@features/library/unified'
import { LibraryItemCard } from '@features/library/views/LibraryCard'
import { AddMangaModal } from '@features/library/addManga/AddMangaModal'

// =============================================================================
// /library — unified grid.
//
// One list, one filter row. Chips combine reading status (5 enum
// values: reading / plan / on_hold / done / dropped) with two
// activity buckets sourced from translation_summary:
//
//   • Đang dịch   running > 0 OR pending > 0
//   • Lỗi          error > 0
//
// Activity chips are not a separate "view" — they're filters that
// cut across status. A manga can be `status=reading` AND `Đang dịch`
// at the same time; the chip is just one slice.
// =============================================================================

interface SearchParams { filter?: LibraryFilter }

const STATUS_CHIPS: Array<{
  id:    LibraryFilter
  label: string
  icon:  typeof Layers
}> = [
  { id: 'all',     label: 'Tất cả',    icon: Layers        },
  { id: 'reading', label: 'Đang đọc',  icon: BookOpen      },
  { id: 'plan',    label: 'Để dành',   icon: BookmarkCheck },
  { id: 'done',    label: 'Đã xong',   icon: CheckCircle2  },
]

const ACTIVITY_CHIPS: Array<{
  id:    LibraryFilter
  label: string
  icon:  typeof Layers
  tone:  'info' | 'error'
}> = [
  { id: 'translating', label: 'Đang dịch', icon: Loader2,     tone: 'info'  },
  { id: 'errored',     label: 'Lỗi',       icon: AlertCircle, tone: 'error' },
]

const EMPTY_HINT: Record<LibraryFilter, { title: string; sub: string }> = {
  all: {
    title: 'Thư viện đang trống',
    sub:   'Dán đường dẫn manga hoặc gõ tên để thêm vào theo dõi',
  },
  reading: {
    title: 'Chưa có truyện đang đọc',
    sub:   'Truyện vừa thêm sẽ tự vào đây',
  },
  plan: {
    title: 'Chưa có truyện để dành',
    sub:   'Đánh dấu "Để dành" để lưu lại đọc sau',
  },
  done: {
    title: 'Chưa hoàn thành truyện nào',
    sub:   'Đánh dấu "Đã xong" khi đọc xong một bộ',
  },
  dropped: {
    title: 'Không có truyện đã bỏ',
    sub:   'Truyện bị bỏ theo dõi sẽ hiện ở đây',
  },
  translating: {
    title: 'Không có bản dịch đang xử lý',
    sub:   'Mở một chương rồi bấm "Dịch" để chạy bản dịch đầu tiên',
  },
  errored: {
    title: 'Không có bản dịch lỗi',
    sub:   'Khi pipeline dịch gặp lỗi, manga liên quan sẽ hiện ở đây',
  },
}

function LibraryPage() {
  const { filter = 'all' } = Route.useSearch()
  const nav = useNavigate()

  const { items, loading, counts } = useUnifiedLibrary(filter)

  const ensureBundled = useSources((s) => s.ensureBundled)
  useEffect(() => { ensureBundled() }, [ensureBundled])

  const [addOpen, setAddOpen] = useState(false)

  const isEmpty = !loading && items.length === 0
  const hint    = EMPTY_HINT[filter]

  return (
    <div className="px-4 sm:px-6 pt-4 sm:pt-6">
      <Toolbar onAdd={() => setAddOpen(true)} />

      <FilterChips
        filter={filter}
        counts={counts}
        onPick={(id) => nav({ to: '/library', search: { filter: id } })}
      />

      {loading && items.length === 0 ? (
        <div className="flex items-center justify-center py-24">
          <Spinner size={20} />
        </div>
      ) : isEmpty ? (
        <div className="py-12">
          <EmptyState
            icon={Library}
            title={hint.title}
            hint={hint.sub}
            action={filter === 'all' ? (
              <Button variant="primary" onClick={() => setAddOpen(true)}>
                <Plus size={14} />
                Thêm manga đầu tiên
              </Button>
            ) : undefined}
          />
        </div>
      ) : (
        <div
          className={cn(
            'grid gap-x-3 gap-y-5 pb-16',
            'grid-cols-[repeat(auto-fill,minmax(120px,1fr))]',
            'sm:grid-cols-[repeat(auto-fill,minmax(140px,1fr))]',
          )}
        >
          {items.map((it) => (
            <LibraryItemCard key={it.key} item={it} />
          ))}
        </div>
      )}

      <AddMangaModal open={addOpen} onClose={() => setAddOpen(false)} />
    </div>
  )
}


// ── Toolbar — Add button ─────────────────────────────────────────────

function Toolbar({ onAdd }: { onAdd: () => void }) {
  return (
    <div className="flex items-center justify-between mb-4 gap-3">
      <h1 className="text-lg font-semibold text-text tracking-tight">Thư viện</h1>
      <Button variant="primary" onClick={onAdd}>
        <Plus size={14} />
        Thêm manga
      </Button>
    </div>
  )
}


// ── Filter chip row ──────────────────────────────────────────────────
//
// Layout: status chips on the left, a thin divider, then activity
// chips. Activity chips hide entirely when their count is 0 — keeps
// the row dense for the casual reader who never spawns translations.

function FilterChips({
  filter, counts, onPick,
}: {
  filter: LibraryFilter
  counts: Record<LibraryFilter, number>
  onPick: (id: LibraryFilter) => void
}) {
  const showActivity = counts.translating > 0 || counts.errored > 0
  return (
    <div
      className="flex items-center gap-2 mb-5 -mx-4 px-4 sm:mx-0 sm:px-0 overflow-x-auto"
      style={{ scrollbarWidth: 'none' }}
    >
      {STATUS_CHIPS.map(({ id, label, icon: Icon }) => {
        const active = filter === id
        const count  = counts[id]
        if (id !== 'all' && id !== 'reading' && count === 0) return null
        return (
          <Chip
            key={id}
            active={active}
            count={count}
            onClick={() => onPick(id)}
          >
            <Icon size={12} />
            {label}
          </Chip>
        )
      })}

      {showActivity && (
        <span
          aria-hidden
          className="shrink-0 w-px h-5 bg-border-soft mx-1"
        />
      )}

      {ACTIVITY_CHIPS.map(({ id, label, icon: Icon, tone }) => {
        const count = counts[id]
        if (count === 0) return null
        const active = filter === id
        return (
          <Chip
            key={id}
            active={active}
            count={count}
            tone={tone}
            onClick={() => onPick(id)}
          >
            <Icon
              size={12}
              className={tone === 'info' && count > 0 ? 'animate-spin' : ''}
            />
            {label}
          </Chip>
        )
      })}
    </div>
  )
}


function Chip({
  active, count, tone, onClick, children,
}: {
  active:   boolean
  count:    number
  tone?:    'info' | 'error'
  onClick:  () => void
  children: React.ReactNode
}) {
  const accent = tone === 'info'
    ? 'bg-info/15 text-info-text hover:bg-info/25'
    : tone === 'error'
    ? 'bg-error/15 text-error-text hover:bg-error/25'
    : 'bg-surface text-text-muted hover:bg-surface-2 hover:text-text'

  return (
    <button
      type="button"
      onClick={onClick}
      className={cn(
        'inline-flex items-center gap-2 h-8 px-3 rounded-full text-sm shrink-0',
        'transition-colors cursor-pointer',
        active
          ? 'bg-text text-bg font-medium'
          : accent,
      )}
    >
      {children}
      {count > 0 && (
        <span className={cn(
          'text-xs tabular',
          active ? 'text-bg/70' : 'opacity-70',
        )}>
          {count}
        </span>
      )}
    </button>
  )
}


// ── Route ────────────────────────────────────────────────────────────

const VALID_FILTERS: ReadonlyArray<LibraryFilter> = [
  'all', 'reading', 'plan', 'done', 'dropped',
  'translating', 'errored',
]

export const Route = createFileRoute('/library')({
  validateSearch: (search: Record<string, unknown>): SearchParams => {
    const f = search.filter
    return {
      filter: VALID_FILTERS.includes(f as LibraryFilter)
        ? (f as LibraryFilter)
        : 'all',
    }
  },
  component: LibraryPage,
})
