import { createFileRoute, useNavigate } from '@tanstack/react-router'
import { useEffect, useState } from 'react'
import {
  Plus, Library, BookOpen, BookmarkCheck, Pause, CheckCircle2, Layers,
  Sparkles,
} from 'lucide-react'
import { cn } from '@shared/lib/cn'
import { EmptyState } from '@shared/ui/EmptyState'
import { Spinner } from '@shared/ui/primitives'
import { Button } from '@shared/ui/Button'
import { useHeaderStore } from '../store/header'
import { useSources } from '@features/browse/sources'
import { useUnifiedLibrary, type LibraryFilter } from '@features/library/unified'
import { LibraryItemCard } from '@features/library/views/LibraryCard'
import { AddMangaModal } from '@features/library/addManga/AddMangaModal'

// =============================================================================
// /library — unified hub. Backend-backed.
//
// Two view modes share the same data + grid (M5 will split translation
// into a per-row list; slice 12 keeps the manga grid plus an inline
// "đang dịch" filter chip that pivots through the same component):
//
//   view=manga          (default) status chips, grid cards
//   view=translation    chips narrow to entries with at least one
//                       running/error/pending translation
//
// Both views write through the URL so back/forward navigation
// preserves state.
// =============================================================================

type LibraryView = 'manga' | 'translation'

interface SearchParams {
  filter?: LibraryFilter
  view?:   LibraryView
}

const CHIPS: Array<{
  id:    LibraryFilter
  label: string
  icon:  typeof Layers
}> = [
  { id: 'all',     label: 'Tất cả',    icon: Layers        },
  { id: 'reading', label: 'Đang đọc',  icon: BookOpen      },
  { id: 'plan',    label: 'Kế hoạch',  icon: BookmarkCheck },
  { id: 'on_hold', label: 'Tạm dừng',  icon: Pause         },
  { id: 'done',    label: 'Đã xong',   icon: CheckCircle2  },
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
    title: 'Chưa có truyện trong kế hoạch',
    sub:   'Đánh dấu "Kế hoạch" để lưu lại đọc sau',
  },
  on_hold: {
    title: 'Không có truyện tạm dừng',
    sub:   'Đánh dấu "Tạm dừng" khi muốn quay lại sau',
  },
  done: {
    title: 'Chưa hoàn thành truyện nào',
    sub:   'Đánh dấu "Đã xong" khi đọc xong một bộ',
  },
  dropped: {
    title: 'Không có truyện đã bỏ',
    sub:   'Truyện bị bỏ theo dõi sẽ hiện ở đây',
  },
}

function LibraryPage() {
  const { filter = 'all', view = 'manga' } = Route.useSearch()
  const nav = useNavigate()

  const { items, loading, counts } = useUnifiedLibrary(filter)

  const setHeader   = useHeaderStore((s) => s.set)
  const clearHeader = useHeaderStore((s) => s.clear)
  useEffect(() => {
    setHeader('Thư viện', [])
    return () => clearHeader()
  }, [setHeader, clearHeader])

  const ensureBundled = useSources((s) => s.ensureBundled)
  useEffect(() => { ensureBundled() }, [ensureBundled])

  const [addOpen, setAddOpen] = useState(false)

  // Translation view narrows further: entries that have at least one
  // running/error/pending translation owned by this user.
  const visible = view === 'translation'
    ? items.filter((it) =>
        it.summary.running > 0 || it.summary.error > 0 || it.summary.pending > 0,
      )
    : items

  const isEmpty = !loading && visible.length === 0

  return (
    <div className="px-4 sm:px-6 pt-4 sm:pt-6">
      <Toolbar
        view={view}
        onSetView={(v) => nav({ to: '/library', search: { filter, view: v } })}
        onAdd={() => setAddOpen(true)}
      />

      {view === 'manga' && (
        <FilterChips
          filter={filter}
          counts={counts}
          onPick={(id) => nav({ to: '/library', search: { filter: id, view } })}
        />
      )}

      {loading && visible.length === 0 ? (
        <div className="flex items-center justify-center py-24">
          <Spinner size={20} />
        </div>
      ) : isEmpty ? (
        <div className="py-12">
          <EmptyState
            icon={view === 'translation' ? Sparkles : Library}
            title={view === 'translation'
              ? 'Không có bản dịch đang xử lý'
              : EMPTY_HINT[filter].title
            }
            hint={view === 'translation'
              ? 'Bản dịch đang chạy / lỗi / chờ sẽ hiện ở đây.'
              : EMPTY_HINT[filter].sub
            }
            action={view === 'manga' && filter === 'all' ? (
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
          {visible.map((it) => (
            <LibraryItemCard key={it.key} item={it} />
          ))}
        </div>
      )}

      <AddMangaModal open={addOpen} onClose={() => setAddOpen(false)} />
    </div>
  )
}


// ── Toolbar — view toggle + Add button ───────────────────────────────

function Toolbar({
  view, onSetView, onAdd,
}: {
  view:       LibraryView
  onSetView:  (v: LibraryView) => void
  onAdd:      () => void
}) {
  return (
    <div className="flex items-center gap-2 mb-4">
      <div className="inline-flex items-center gap-0.5 p-0.5 rounded-sm bg-surface-2">
        <ViewBtn label="Manga"    active={view === 'manga'}       onClick={() => onSetView('manga')} />
        <ViewBtn label="Bản dịch" active={view === 'translation'} onClick={() => onSetView('translation')} />
      </div>
      <div className="flex-1" />
      <Button variant="primary" onClick={onAdd}>
        <Plus size={14} />
        Thêm
      </Button>
    </div>
  )
}

function ViewBtn({
  label, active, onClick,
}: {
  label: string; active: boolean; onClick: () => void
}) {
  return (
    <button
      type="button"
      onClick={onClick}
      className={cn(
        'h-7 px-3 rounded-xs text-xs font-medium transition-colors cursor-pointer',
        active
          ? 'bg-bg text-text'
          : 'text-text-muted hover:text-text',
      )}
    >
      {label}
    </button>
  )
}


// ── Filter chip row ──────────────────────────────────────────────────

function FilterChips({
  filter, counts, onPick,
}: {
  filter: LibraryFilter
  counts: Record<LibraryFilter, number>
  onPick: (id: LibraryFilter) => void
}) {
  return (
    <div
      className="flex items-center gap-1.5 mb-5 -mx-4 px-4 sm:mx-0 sm:px-0 overflow-x-auto"
      style={{ scrollbarWidth: 'none' }}
    >
      {CHIPS.map(({ id, label, icon: Icon }) => {
        const active = filter === id
        const count  = counts[id]
        if (id !== 'all' && id !== 'reading' && count === 0) return null
        return (
          <button
            key={id}
            onClick={() => onPick(id)}
            className={cn(
              'inline-flex items-center gap-1.5 h-8 px-3 rounded-full text-[13px] shrink-0',
              'transition-colors cursor-pointer',
              active
                ? 'bg-text text-bg font-medium'
                : 'bg-surface text-text-muted hover:bg-surface-2 hover:text-text',
            )}
          >
            <Icon size={12} />
            {label}
            {count > 0 && (
              <span className={cn(
                'text-[11px] tabular',
                active ? 'text-bg/70' : 'text-text-subtle',
              )}>
                {count}
              </span>
            )}
          </button>
        )
      })}
    </div>
  )
}


// ── Route ────────────────────────────────────────────────────────────

const VALID_FILTERS: ReadonlyArray<LibraryFilter> = [
  'all', 'reading', 'plan', 'on_hold', 'done', 'dropped',
]

export const Route = createFileRoute('/library')({
  validateSearch: (search: Record<string, unknown>): SearchParams => {
    const f = search.filter
    const v = search.view
    return {
      filter: VALID_FILTERS.includes(f as LibraryFilter)
        ? (f as LibraryFilter)
        : 'all',
      view: v === 'translation' ? 'translation' : 'manga',
    }
  },
  component: LibraryPage,
})
