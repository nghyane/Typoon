import { createFileRoute, Link, useNavigate } from '@tanstack/react-router'
import { useEffect } from 'react'
import { BookOpen, BookmarkCheck, Pause, CheckCircle2, Layers, FolderOpen } from 'lucide-react'
import { cn } from '@shared/lib/cn'
import { EmptyState } from '@shared/ui/EmptyState'
import { Spinner } from '@shared/ui/primitives'
import { useHeaderStore } from '../store/header'
import { useUnifiedLibrary, type LibraryFilter } from '@features/library/unified'
import { LibraryItemCard } from '@features/library/views/LibraryCard'

// =============================================================================
// /library — unified backend-backed surface.
//
// One grid, all entries. Filter chips operate on the server `status`
// enum so the URL stays canonical (filter=reading bookmarks fine via
// browser back-button). `dropped` is excluded from the default list;
// users wanting to see them can pass ?filter=dropped explicitly.
// =============================================================================

interface SearchParams { filter?: LibraryFilter }

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
    sub:   'Mở một truyện ở Duyệt nguồn rồi bấm "Theo dõi" để bắt đầu',
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
  const { filter = 'all' } = Route.useSearch()
  const nav = useNavigate()
  const { items, loading, counts } = useUnifiedLibrary(filter)

  const setHeader   = useHeaderStore((s) => s.set)
  const clearHeader = useHeaderStore((s) => s.clear)
  useEffect(() => {
    setHeader('Thư viện', [])
    return () => clearHeader()
  }, [setHeader, clearHeader])

  return (
    <div className="px-4 sm:px-6 pt-4 sm:pt-6">
      {/* Filter chips. Horizontal-scroll on narrow viewports; bleeds
          to the edge so the row feels native on mobile. */}
      <div
        className="flex items-center gap-1.5 mb-5 -mx-4 px-4 sm:mx-0 sm:px-0 overflow-x-auto"
        style={{ scrollbarWidth: 'none' }}
      >
        {CHIPS.map(({ id, label, icon: Icon }) => {
          const active = filter === id
          const count  = counts[id]
          // Hide secondary chips when they would render zero — keeps
          // the chrome dense for the typical reader.
          if (id !== 'all' && id !== 'reading' && count === 0) return null
          return (
            <button
              key={id}
              onClick={() => nav({ to: '/library', search: { filter: id } })}
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

      {loading && items.length === 0 ? (
        <div className="flex items-center justify-center py-24">
          <Spinner size={20} />
        </div>
      ) : items.length === 0 ? (
        <div className="py-12">
          <EmptyState
            icon={FolderOpen}
            title={EMPTY_HINT[filter].title}
            hint={EMPTY_HINT[filter].sub}
            action={
              <Link
                to="/browse"
                className="inline-flex items-center gap-1.5 h-8 px-3 rounded-sm bg-accent text-accent-fg text-[13px] font-medium hover:bg-accent-strong transition-colors cursor-pointer"
              >
                Duyệt nguồn
              </Link>
            }
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
    </div>
  )
}

const VALID_FILTERS: ReadonlyArray<LibraryFilter> = [
  'all', 'reading', 'plan', 'on_hold', 'done', 'dropped',
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
