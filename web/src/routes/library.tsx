import { createFileRoute, Link, useNavigate } from '@tanstack/react-router'
import { useEffect } from 'react'
import { Bookmark, BookOpen, Sparkles, FolderOpen, Layers } from 'lucide-react'
import { cn } from '@shared/lib/cn'
import { EmptyState } from '@shared/ui/EmptyState'
import { Spinner } from '@shared/ui/primitives'
import { useHeaderStore } from '../store/header'
import type { LibraryFilter } from '@features/library/hooks'
import { useUnifiedLibrary } from '@features/library/unified'
import { LibraryItemCard } from '@features/library/views/LibraryCard'

// =============================================================================
// /library — single unified surface.
//
// Pattern lifted from Letterboxd / AniList / Apple Music Library:
//   • One grid, all entries (external + internal).
//   • Filter chips, NOT tabs. Chips are a filter on the same list —
//     route doesn't change, scroll position survives.
//   • "Có chương mới" is a chip too, BUT entries with hasNew also
//     float to the top of every other filter via default sort, so
//     the chip is just a focus mode, not the only way to see them.
//
// State sources:
//   • External entries → local zustand store (typoon.library.v1)
//   • Internal projects → /api/projects (React Query, 60s stale)
//
// Adapter merges both into a uniform LibraryItem (see unified.ts).
// =============================================================================

interface SearchParams { filter?: LibraryFilter }

const CHIPS: Array<{
  id:    LibraryFilter
  label: string
  icon:  typeof Bookmark
}> = [
  { id: 'all',       label: 'Tất cả',         icon: Layers   },
  { id: 'reading',   label: 'Đang đọc',       icon: BookOpen },
  { id: 'bookmarks', label: 'Đã lưu',         icon: Bookmark },
  { id: 'updates',   label: 'Có chương mới',  icon: Sparkles },
]

const EMPTY_HINT: Record<LibraryFilter, { title: string; sub: string }> = {
  all: {
    title: 'Thư viện đang trống',
    sub:   'Mở một truyện ở Duyệt nguồn, hoặc tạo dự án để bắt đầu',
  },
  reading: {
    title: 'Chưa có truyện đang đọc',
    sub:   'Truyện sẽ hiện ở đây sau khi bạn mở chương đầu tiên',
  },
  bookmarks: {
    title: 'Chưa lưu truyện nào',
    sub:   'Bấm "Lưu" ở trang truyện để thêm vào đây',
  },
  updates: {
    title: 'Chưa có chương mới',
    sub:   'Khi truyện đã lưu có chương mới, sẽ hiện ở đây',
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
      {/* Filter chips. Horizontal-scroll on narrow viewports so the
          full set is always reachable; bleeds to the screen edge on
          mobile to feel native. */}
      <div
        className="flex items-center gap-1.5 mb-5 -mx-4 px-4 sm:mx-0 sm:px-0 overflow-x-auto"
        style={{ scrollbarWidth: 'none' }}
      >
        {CHIPS.map(({ id, label, icon: Icon }) => {
          const active = filter === id
          const count  = counts[id]
          // Hide "Có chương mới" chip entirely when count is 0 —
          // empty filter is just dead space at the top of the page.
          if (id === 'updates' && count === 0) return null
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
                  'text-[10px] tabular',
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
                className="inline-flex items-center gap-1.5 h-8 px-3 rounded-sm bg-accent text-accent-fg text-[13px] font-medium hover:bg-accent-hover transition-colors cursor-pointer"
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

export const Route = createFileRoute('/library')({
  validateSearch: (search: Record<string, unknown>): SearchParams => {
    const f = search.filter
    return {
      filter: (f === 'all' || f === 'reading' || f === 'bookmarks' || f === 'updates')
        ? f
        : 'all',
    }
  },
  component: LibraryPage,
})
