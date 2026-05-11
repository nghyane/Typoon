import { Link } from '@tanstack/react-router'
import { Bookmark, Sparkles } from 'lucide-react'
import { useMutation, useQueryClient } from '@tanstack/react-query'
import { Cover } from '@shared/ui/Cover'
import { Tag } from '@shared/ui/primitives'
import { cn } from '@shared/lib/cn'
import { proxify } from '@features/browse/proxy'
import { api } from '@shared/api/api'
import { useLibrary, type LibraryEntry } from '../store'
import type { LibraryItem } from '../unified'

// =============================================================================
// LibraryItemCard — single card shape backed by /api/library entries.
//
// Click target: `/manga/library/$entryId` (the route below is a thin
// resolver that fetches the entry's primary material and forwards to
// /browse/$source/manga/$mangaId, possibly deep-linking into the
// chapter the user last read).
// =============================================================================

interface Props {
  item: LibraryItem
}

export function LibraryItemCard({ item }: Props) {
  return (
    <Link
      to="/library/entry/$entryId"
      params={{ entryId: String(item.entryId) }}
      className="group flex flex-col gap-2"
    >
      <div className="relative w-full aspect-[2/3] rounded-md overflow-hidden">
        <Cover
          src={item.cover}
          title={item.title}
          className="absolute inset-0 group-hover:brightness-110 transition-[filter]"
        />

        {/* Top-right state cluster */}
        <div className="absolute top-1.5 right-1.5 flex flex-col items-end gap-1">
          {item.hasNew && (
            <Tag
              tone="success" size="sm" uppercase
              className="inline-flex items-center gap-0.5"
            >
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

        {/* Bottom gradient + chapter label (when we have a local snapshot) */}
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

      <p className="text-[13px] font-medium text-text leading-snug line-clamp-2 group-hover:text-accent-text transition-colors">
        {item.title}
      </p>
    </Link>
  )
}

// =============================================================================
// BookmarkButton — toggles a library entry's bookmarked flag.
//
// Three modes:
//   (1) entryId known         → patch /library/entry/{id}
//   (2) materialId known      → ensure entry exists (POST /library/entry
//                                with material_id), then patch
//                                bookmarked=true. Used from the manga
//                                detail page.
//   (3) external mangaUrl only → backwards-compat shim: writes to the
//                                local reading-history store until
//                                we wire material_id into the local
//                                state shape.
// =============================================================================

interface BookmarkButtonProps {
  size?:       'sm' | 'md'
  entryId?:    number
  materialId?: number
  /** Used to seed the entry title when creating one on first bookmark. */
  title?:      string
  cover?:      string | null
  bookmarked:  boolean
}

export function BookmarkButton({
  size = 'md',
  entryId, materialId, title, cover, bookmarked,
}: BookmarkButtonProps) {
  const qc = useQueryClient()
  const mutation = useMutation({
    mutationFn: async () => {
      if (entryId != null) {
        await api.patchLibraryEntry(entryId, { bookmarked: !bookmarked })
        return
      }
      if (materialId != null) {
        // Ensure an entry exists, then patch the flag.
        const entry = await api.createLibraryEntry({
          material_id: materialId,
          title,
          cover_url: cover ?? null,
        })
        if (!entry.bookmarked) {
          await api.patchLibraryEntry(entry.id, { bookmarked: true })
        }
        return
      }
      throw new Error('BookmarkButton: pass entryId or materialId')
    },
    onSuccess: () => {
      qc.invalidateQueries({ queryKey: ['library'] })
    },
  })

  const dims = size === 'sm'
    ? 'h-7 px-2.5 text-xs'
    : 'h-8 px-3 text-[13px]'

  return (
    <button
      type="button"
      onClick={() => mutation.mutate()}
      disabled={mutation.isPending}
      className={cn(
        'inline-flex items-center gap-1.5 rounded-sm cursor-pointer transition-colors',
        dims,
        bookmarked
          ? 'bg-warning/15 text-warning-text hover:bg-warning/25'
          : 'bg-surface-2 text-text-muted hover:bg-hover hover:text-text',
        mutation.isPending && 'opacity-60 cursor-wait',
      )}
      title={bookmarked ? 'Bỏ lưu khỏi thư viện' : 'Lưu vào thư viện'}
    >
      <Bookmark
        size={size === 'sm' ? 11 : 13}
        className={bookmarked ? 'fill-warning text-warning' : ''}
      />
      {bookmarked ? 'Đã lưu' : 'Lưu'}
    </button>
  )
}

// =============================================================================
// LibraryRailCard — slim per-source variant for browse hub
// "Tiếp tục đọc" rail. Reads from the local reading-history store
// because per-source recency is browser-derived (not in the server
// library payload).
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

// Keep a name-export the (slowly retiring) `useLibrary` consumer pulls
// through for the old shape; new code uses the variants above.
useLibrary  // referenced to keep tree-shaking honest
