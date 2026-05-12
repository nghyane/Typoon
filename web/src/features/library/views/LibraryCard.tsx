import { Link } from '@tanstack/react-router'
import { BookmarkPlus, Check, Sparkles } from 'lucide-react'
import { useMutation, useQueryClient } from '@tanstack/react-query'
import { Cover } from '@shared/ui/Cover'
import { Tag } from '@shared/ui/primitives'
import { cn } from '@shared/lib/cn'
import { proxify } from '@features/browse/proxy'
import { api, type LibraryStatus } from '@shared/api/api'
import { useLibrary, type LibraryEntry } from '../store'
import type { LibraryItem } from '../unified'

// =============================================================================
// LibraryItemCard — single card shape backed by /api/library entries.
//
// Click target: `/library/entry/$entryId` resolves to the entry's hub
// page (the post-M4 /title/{id} route). Status badge floats top-right
// when the entry is anything other than 'reading' — the default state
// doesn't deserve visual noise.
// =============================================================================

const STATUS_LABEL: Record<LibraryStatus, string> = {
  reading: 'Đang đọc',
  plan:    'Kế hoạch',
  on_hold: 'Tạm dừng',
  done:    'Đã xong',
  dropped: 'Đã bỏ',
}

const STATUS_TONE: Record<LibraryStatus, 'success' | 'info' | 'warning' | 'neutral' | 'error'> = {
  reading: 'success',
  plan:    'info',
  on_hold: 'warning',
  done:    'neutral',
  dropped: 'error',
}

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

        {/* Top-right state cluster — "Mới" badge floats above status
            so even at glance the user sees what's actionable. */}
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
          {item.status !== 'reading' && (
            <Tag tone={STATUS_TONE[item.status]} size="sm" uppercase>
              {STATUS_LABEL[item.status]}
            </Tag>
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
// FollowButton — single CTA for "follow this manga".
//
// Two entry conditions:
//   (1) Already in library  → click cycles status (reading → dropped → ...);
//                              UI uses STATUS_LABEL.
//   (2) Not in library yet   → click POSTs /library/entry with default
//                              `status='reading'` and `target_lang=null`
//                              (the hub asks the user to pick at first
//                              open). Caller may supply target_lang
//                              from MaterialPage context to skip the
//                              prompt.
// =============================================================================

interface FollowButtonProps {
  size?:        'sm' | 'md'
  entryId?:     number
  materialId?:  number
  /** Seed title/cover when creating an entry on first follow. */
  title?:       string
  cover?:       string | null
  /** When known, baked into the new entry so the hub doesn't prompt. */
  targetLang?:  string | null
  /** Current status; null when not in library yet. */
  status:       LibraryStatus | null
}

export function FollowButton({
  size = 'md',
  entryId, materialId, title, cover, targetLang, status,
}: FollowButtonProps) {
  const qc = useQueryClient()
  const inLibrary = status !== null && status !== 'dropped'

  const mutation = useMutation({
    mutationFn: async () => {
      // Toggle off → drop. Toggle on → reading.
      const next: LibraryStatus = inLibrary ? 'dropped' : 'reading'
      if (entryId != null) {
        await api.patchLibraryEntry(entryId, { status: next })
        return
      }
      if (materialId != null) {
        await api.createLibraryEntry({
          material_id:  materialId,
          title,
          cover_url:    cover ?? null,
          target_lang:  targetLang ?? null,
          status:       'reading',
        })
        return
      }
      throw new Error('FollowButton: pass entryId or materialId')
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
        inLibrary
          ? 'bg-success/15 text-success-text hover:bg-success/25'
          : 'bg-accent text-accent-fg hover:bg-accent-strong',
        mutation.isPending && 'opacity-60 cursor-wait',
      )}
      title={inLibrary ? 'Bỏ theo dõi' : 'Theo dõi truyện này'}
    >
      {inLibrary ? (
        <Check size={size === 'sm' ? 11 : 13} />
      ) : (
        <BookmarkPlus size={size === 'sm' ? 11 : 13} />
      )}
      {inLibrary ? (status && STATUS_LABEL[status]) : 'Theo dõi'}
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
