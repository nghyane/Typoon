import { Link } from '@tanstack/react-router'
import { Bookmark, Sparkles } from 'lucide-react'
import { Cover } from '@shared/ui/Cover'
import { Tag } from '@shared/ui/primitives'
import { cn } from '@shared/lib/cn'
import { proxify } from '@features/browse/proxy'
import { getSource } from '@features/browse/sources'
import { useLibrary, type LibraryEntry } from '../store'
import type { LibraryItem } from '../unified'

// =============================================================================
// LibraryItemCard — single card shape for any source.
//
// Click target picks the best resumption point:
//   • External with last-read chapter → straight into the reader
//   • External without            → manga detail page
//   • Internal project            → project detail
//
// Visual state cluster on cover top-right:
//   • "Mới" tag when hasNew
//   • Bookmark fill icon when bookmarked
//   • Ownership chip on bottom-left for internal (Mine / Shared)
// =============================================================================

interface Props {
  item: LibraryItem
}

const OWNERSHIP_LABEL = {
  mine:   'Của tôi',
  pinned: 'Đã lưu',
  shared: 'Hội Mê Truyện',
} as const

export function LibraryItemCard({ item }: Props) {
  // External: try to deep-link straight into the chapter the user
  // was last reading. Internal: open the project route.
  const entry = useLibrary((s) =>
    item.kind === 'external'
      ? s.items[`${item.source}::${item.ref}`]
      : undefined,
  )

  const installed = getSource(item.source)
  const sourceName = installed?.manifest.name ?? item.source

  const body = (
    <>
      <div className="relative w-full aspect-[2/3] rounded-md overflow-hidden">
        <Cover
          src={item.cover ? (item.kind === 'external' ? proxify(item.cover) : item.cover) : null}
          title={item.title}
          className="absolute inset-0 group-hover:brightness-110 transition-[filter]"
        />

        {/* Top-right state cluster */}
        <div className="absolute top-1.5 right-1.5 flex flex-col items-end gap-1">
          {item.hasNew && (
            <Tag tone="success" size="sm" uppercase className="inline-flex items-center gap-0.5">
              <Sparkles size={9} />
              Mới
            </Tag>
          )}
          {item.bookmarked && (
            <span
              className="size-6 rounded-sm bg-bg/80 backdrop-blur flex items-center justify-center"
              title="Đã lưu"
            >
              <Bookmark size={11} className="fill-warning text-warning" />
            </span>
          )}
        </div>

        {/* Bottom-left ownership chip (internal only) */}
        {item.kind === 'internal' && item.ownership && (
          <div className="absolute bottom-1.5 left-1.5">
            <span className={cn(
              'text-[10px] font-medium uppercase tracking-wide px-1.5 py-0.5 rounded-xs',
              'bg-bg/80 backdrop-blur text-text',
            )}>
              {OWNERSHIP_LABEL[item.ownership]}
            </span>
          </div>
        )}

        {/* Bottom gradient + chapter label (external with history only) */}
        {item.chapterLabel && (
          <>
            <div
              aria-hidden
              className="absolute inset-x-0 bottom-0 h-2/5 bg-gradient-to-t from-black/85 via-black/40 to-transparent"
            />
            <div className="absolute inset-x-0 bottom-0 px-2 py-1.5">
              <p className="text-[10px] font-semibold text-white/95 truncate">
                {item.chapterLabel}
              </p>
            </div>
          </>
        )}
      </div>

      <div className="min-w-0">
        <p className="text-[13px] font-medium text-text leading-snug line-clamp-2 group-hover:text-accent-text transition-colors">
          {item.title}
        </p>
        <p className="text-[10px] text-text-subtle truncate mt-0.5">
          {item.kind === 'internal' ? 'Hội Mê Truyện' : sourceName}
        </p>
      </div>
    </>
  )

  const cls = 'group flex flex-col gap-2'

  // Branch the Link wrapper so each `to` keeps its type-safe params
  // shape. Trying to spread a unioned linkProps object loses the
  // discriminant and trips TS2698.
  if (item.kind === 'internal') {
    return (
      <Link
        to="/projects/$projectId"
        params={{ projectId: item.ref }}
        className={cls}
      >
        {body}
      </Link>
    )
  }
  if (entry?.lastChapterRead) {
    return (
      <Link
        to="/browse/$source/manga/$mangaId/chapter/$chapterId"
        params={{
          source:    item.source,
          mangaId:   encodeURIComponent(item.ref),
          chapterId: encodeURIComponent(entry.lastChapterRead.url),
        }}
        className={cls}
      >
        {body}
      </Link>
    )
  }
  return (
    <Link
      to="/browse/$source/manga/$mangaId"
      params={{
        source:  item.source,
        mangaId: encodeURIComponent(item.ref),
      }}
      className={cls}
    >
      {body}
    </Link>
  )
}

// =============================================================================
// BookmarkButton — toggle for MangaPage hero (external sources only).
// Reads + writes through the library store directly.
// =============================================================================

interface BookmarkButtonProps {
  source:   string
  mangaUrl: string
  title:    string
  cover:    string | null
  size?:    'sm' | 'md'
}

export function BookmarkButton({
  source, mangaUrl, title, cover, size = 'md',
}: BookmarkButtonProps) {
  const entry  = useLibrary((s) => s.items[`${source}::${mangaUrl}`])
  const toggle = useLibrary((s) => s.toggleBookmark)
  const on = !!entry?.bookmarked

  const dims = size === 'sm'
    ? 'h-7 px-2.5 text-xs'
    : 'h-8 px-3 text-[13px]'

  return (
    <button
      type="button"
      onClick={() => toggle({ source, mangaUrl, title, cover })}
      className={cn(
        'inline-flex items-center gap-1.5 rounded-sm cursor-pointer transition-colors',
        dims,
        on
          ? 'bg-warning/15 text-warning-text hover:bg-warning/25'
          : 'bg-surface-2 text-text-muted hover:bg-hover hover:text-text',
      )}
      title={on ? 'Bỏ lưu khỏi thư viện' : 'Lưu vào thư viện'}
    >
      <Bookmark
        size={size === 'sm' ? 11 : 13}
        className={on ? 'fill-warning text-warning' : ''}
      />
      {on ? 'Đã lưu' : 'Lưu'}
    </button>
  )
}

// =============================================================================
// LibraryRailCard — slim per-source variant for browse hub
// "Tiếp tục đọc" rail. Different from grid card: fixed width,
// external-only context (per-source). Keeps the cover overlay
// chapter label like the legacy ContinueCard.
// =============================================================================

interface RailProps {
  entry: LibraryEntry
}

export function LibraryRailCard({ entry }: RailProps) {
  const to = entry.lastChapterRead
    ? '/browse/$source/manga/$mangaId/chapter/$chapterId' as const
    : '/browse/$source/manga/$mangaId' as const
  const params = entry.lastChapterRead
    ? {
        source:    entry.source,
        mangaId:   encodeURIComponent(entry.mangaUrl),
        chapterId: encodeURIComponent(entry.lastChapterRead.url),
      }
    : {
        source:  entry.source,
        mangaId: encodeURIComponent(entry.mangaUrl),
      }

  return (
    <Link
      to={to}
      params={params as never}
      className="group flex flex-col gap-2 w-[140px] sm:w-[168px] shrink-0"
    >
      <div className="relative w-full aspect-[2/3] rounded-md overflow-hidden">
        <Cover
          src={entry.cover ? proxify(entry.cover) : null}
          title={entry.title}
          className="absolute inset-0 group-hover:brightness-110 transition-[filter]"
        />
        {entry.lastChapterRead && (
          <>
            <div
              aria-hidden
              className="absolute inset-x-0 bottom-0 h-2/5 bg-gradient-to-t from-black/85 via-black/40 to-transparent"
            />
            <div className="absolute inset-x-0 bottom-0 px-2 py-1.5">
              <p className="text-[10px] font-semibold text-white/95 truncate">
                {entry.lastChapterRead.label}
              </p>
            </div>
          </>
        )}
      </div>
      <p className="text-[13px] font-medium text-text leading-snug line-clamp-2 group-hover:text-accent-text transition-colors">
        {entry.title}
      </p>
    </Link>
  )
}
