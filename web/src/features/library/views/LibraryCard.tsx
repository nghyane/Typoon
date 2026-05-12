import { Link } from '@tanstack/react-router'
import {
  BookmarkPlus, Check, Sparkles, Loader2, AlertCircle,
} from 'lucide-react'
import { useMutation, useQueryClient } from '@tanstack/react-query'
import { Cover } from '@shared/ui/Cover'
import { Tag, type TagTone } from '@shared/ui/primitives'
import { cn } from '@shared/lib/cn'
import { proxify } from '@features/browse/proxy'
import { api, type LibraryStatus } from '@shared/api/api'
import { useLibrary, type LibraryEntry } from '../store'
import type { LibraryItem } from '../unified'

// =============================================================================
// LibraryItemCard — grid card backed by /api/library entries.
//
// Visual contract:
//   • Cover-first (2:3 aspect). Hover lifts brightness.
//   • Top-right cluster (stacked, top to bottom):
//       Mới       → at least one upstream chapter past lastReadAt.
//       Hoạt động → translation_summary chips (running/error/pending).
//       Trạng thái → status enum tag, hidden when status='reading'
//                    (the default — would just be noise).
//   • Bottom gradient + chapter label, when local snapshot has one.
//
// Click target: /library/entry/$entryId — the post-M4 hub route.
// =============================================================================

const STATUS_LABEL: Record<LibraryStatus, string> = {
  reading: 'Đang đọc',
  plan:    'Kế hoạch',
  on_hold: 'Tạm dừng',
  done:    'Đã xong',
  dropped: 'Đã bỏ',
}

const STATUS_TONE: Record<LibraryStatus, TagTone> = {
  reading: 'success',
  plan:    'info',
  on_hold: 'warning',
  done:    'neutral',
  dropped: 'error',
}

interface Props { item: LibraryItem }

export function LibraryItemCard({ item }: Props) {
  return (
    <Link
      to="/title/$entryId"
      params={{ entryId: String(item.entryId) }}
      className="group flex flex-col gap-2"
    >
      <div className="relative w-full aspect-[2/3] rounded-md overflow-hidden">
        <Cover
          src={item.cover}
          title={item.title}
          className="absolute inset-0 group-hover:brightness-110 transition-[filter]"
        />

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
          <ActivityChips summary={item.summary} />
          {item.status !== 'reading' && (
            <Tag tone={STATUS_TONE[item.status]} size="sm" uppercase>
              {STATUS_LABEL[item.status]}
            </Tag>
          )}
        </div>

        {item.chapterLabel && (
          <>
            <div
              aria-hidden
              className="absolute inset-x-0 bottom-0 h-2/5 bg-gradient-to-t from-black/85 via-black/40 to-transparent"
            />
            <div className="absolute inset-x-0 bottom-0 px-2 py-1.5">
              <p className="text-[11px] font-semibold text-white/95 truncate">
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


// ── Activity chips ───────────────────────────────────────────────────
//
// Translation activity is what the user actively cares about — far
// more actionable than reading status. Order is `running → error →
// pending`. `done` count never surfaces; it's the resting state.

function ActivityChips({
  summary,
}: {
  summary: LibraryItem['summary']
}) {
  if (summary.running === 0 && summary.error === 0 && summary.pending === 0) {
    return null
  }
  return (
    <div className="flex flex-col items-end gap-1">
      {summary.running > 0 && (
        <Tag tone="info" size="sm" className="inline-flex items-center gap-0.5">
          <Loader2 size={9} className="animate-spin" />
          {summary.running} đang dịch
        </Tag>
      )}
      {summary.error > 0 && (
        <Tag tone="error" size="sm" className="inline-flex items-center gap-0.5">
          <AlertCircle size={9} />
          {summary.error} lỗi
        </Tag>
      )}
      {summary.pending > 0 && (
        <Tag tone="warning" size="sm">{summary.pending} chờ</Tag>
      )}
    </div>
  )
}


// =============================================================================
// FollowButton — used by MaterialPage / hub hero (slice 13+).
//
// Toggles between `reading` and `dropped`. Other statuses (`plan`,
// `on_hold`, `done`) are reachable through the status menu — kept in
// a separate component once the hub ships.
// =============================================================================

interface FollowButtonProps {
  size?:        'sm' | 'md'
  entryId?:     number
  materialId?:  number
  title?:       string
  cover?:       string | null
  targetLang?:  string | null
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
          : 'bg-accent text-accent-fg hover:brightness-110',
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
// LibraryRailCard — slim per-source variant for browse-mode rails.
// Reads from the local reading-history store; not the server library.
// =============================================================================

interface RailProps { entry: LibraryEntry }

export function LibraryRailCard({ entry }: RailProps) {
  // Source-routed link is dead in v2; rail navigates to the entry
  // resolver instead so the user reaches the hub regardless of which
  // source first introduced the manga. The store key is enough to
  // identify the manga locally; the resolver picks up the entry id.
  return (
    <Link
      to="/library"
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
              <p className="text-[11px] font-semibold text-white/95 truncate">
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

// Keep useLibrary referenced so this module's store import survives
// dead-code elimination during the M5 hub rebuild.
useLibrary
