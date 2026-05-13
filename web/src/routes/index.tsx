// Home (/) — landing surface for logged-in users.
//
// Two stacked sections, both data-driven:
//
//   ① Tiếp tục đọc       /api/me/recent-reads — manga the viewer
//                         opened recently, deduped per material.
//                         Empty for fresh accounts.
//
//   ② Mới trong cộng đồng /api/community/recent — newest community
//                         translations, deduped per material.
//
// Both sections click through to `/m/$materialId` so the manga page
// is the single canonical surface.

import { useEffect } from 'react'
import { createFileRoute, useNavigate, Link } from '@tanstack/react-router'
import { useQuery } from '@tanstack/react-query'
import { BookOpen, Sparkles } from 'lucide-react'

import {
  api,
  type ApiCommunityFeedEntry, type ApiRecentRead,
} from '@shared/api/api'
import { qk } from '@shared/api/keys'
import { Cover } from '@shared/ui/Cover'
import { EmptyState } from '@shared/ui/EmptyState'
import { Spinner } from '@shared/ui/primitives'
import { cn } from '@shared/lib/cn'
import { timeAgo } from '@shared/lib/time'
import { useSources } from '@features/browse/sources'


function HomePage() {
  const ensureBundled = useSources((s) => s.ensureBundled)
  useEffect(() => { ensureBundled() }, [ensureBundled])

  const recent = useQuery({
    queryKey: qk.me.recentReads(),
    queryFn:  () => api.listRecentReads(20),
    staleTime: 30_000,
  })

  const community = useQuery({
    queryKey: qk.community.recent(),
    queryFn:  () => api.listCommunityRecent({ limit: 60 }),
    staleTime: 30_000,
  })

  return (
    <div className="px-4 sm:px-6 py-4 sm:py-6 space-y-8">
      <RecentSection
        items={recent.data ?? []}
        loading={recent.isPending}
      />
      <CommunitySection
        items={community.data ?? []}
        loading={community.isPending}
      />
    </div>
  )
}


// ── Tiếp tục đọc ────────────────────────────────────────────────────


function RecentSection({
  items, loading,
}: {
  items:   ApiRecentRead[]
  loading: boolean
}) {
  const nav = useNavigate()

  if (loading) {
    return (
      <section>
        <SectionHeading title="Tiếp tục đọc" />
        <div className="flex items-center justify-center py-12">
          <Spinner size={18} />
        </div>
      </section>
    )
  }
  if (items.length === 0) {
    return (
      <section>
        <SectionHeading title="Tiếp tục đọc" />
        <EmptyState
          icon={BookOpen}
          title="Chưa có lịch sử đọc"
          hint="Mở một chương từ Thư viện hoặc Khám phá để bắt đầu."
        />
      </section>
    )
  }

  return (
    <section>
      <SectionHeading
        title="Tiếp tục đọc"
        action={
          <Link to="/library" className="text-xs text-text-subtle hover:text-text-muted">
            Thư viện →
          </Link>
        }
      />
      <ul className="grid grid-cols-2 sm:grid-cols-3 md:grid-cols-4 lg:grid-cols-6 gap-3 sm:gap-4">
        {items.map((it) => (
          <RecentCard
            key={`r:${it.work_id}`}
            item={it}
            onOpenManga={() => nav({
              to:     '/w/$workId',
              params: { workId: String(it.work_id) },
              search: { src: it.material_id },
            })}
            onContinue={() => {
              // Unified reader resolves translation vs raw from the
              // cached Work payload; URL stays stable across spawns.
              nav({
                to:     '/r/$workId/$numberNorm',
                params: {
                  workId:     String(it.work_id),
                  numberNorm: it.chapter_number,
                },
                search: { src: it.material_id ?? undefined },
              })
            }}
          />
        ))}
      </ul>
    </section>
  )
}


function RecentCard({
  item, onOpenManga, onContinue,
}: {
  item:        ApiRecentRead
  onOpenManga: () => void
  onContinue:  () => void
}) {
  return (
    <li className="group flex flex-col gap-2">
      <button
        type="button"
        onClick={onOpenManga}
        className="relative aspect-[2/3] rounded-md overflow-hidden bg-surface-2 cursor-pointer"
      >
        <Cover
          src={item.material_cover}
          title={item.material_title}
          className="absolute inset-0 group-hover:brightness-110 transition-[filter]"
        />
        <div className="absolute bottom-0 left-0 right-0 bg-gradient-to-t from-black/85 via-black/40 to-transparent px-2 py-1.5">
          <span className="text-xs font-medium text-white/95 tabular">
            Ch.{item.chapter_number}
          </span>
        </div>
      </button>
      <div className="min-w-0">
        <button
          type="button"
          onClick={onOpenManga}
          className="block w-full text-left"
        >
          <p className="text-sm font-medium text-text truncate group-hover:text-accent-text transition-colors">
            {item.material_title}
          </p>
        </button>
        <button
          type="button"
          onClick={onContinue}
          className="text-xs text-text-subtle hover:text-text-muted truncate w-full text-left"
        >
          Tiếp tục Ch.{item.chapter_number}
          {item.last_read_at && <> · {timeAgo(item.last_read_at)}</>}
        </button>
      </div>
    </li>
  )
}


// ── Mới trong cộng đồng ────────────────────────────────────────────


function CommunitySection({
  items, loading,
}: {
  items:   ApiCommunityFeedEntry[]
  loading: boolean
}) {
  const nav = useNavigate()

  if (loading) {
    return (
      <section>
        <SectionHeading title="Mới trong cộng đồng" />
        <div className="flex items-center justify-center py-12">
          <Spinner size={18} />
        </div>
      </section>
    )
  }
  if (items.length === 0) {
    return (
      <section>
        <SectionHeading title="Mới trong cộng đồng" />
        <EmptyState
          icon={Sparkles}
          title="Cộng đồng chưa có bản dịch"
          hint="Khi có thành viên dịch chương, bản dịch sẽ xuất hiện ở đây."
        />
      </section>
    )
  }

  return (
    <section>
      <SectionHeading
        title="Mới trong cộng đồng"
        action={
          <Link to="/explore" className="text-xs text-text-subtle hover:text-text-muted">
            Khám phá →
          </Link>
        }
      />
      <ul className="grid grid-cols-2 sm:grid-cols-3 md:grid-cols-4 lg:grid-cols-6 gap-3 sm:gap-4">
        {items.map((e) => (
          <CommunityCard
            key={`c:${e.work_id}`}
            entry={e}
            onClick={() => nav({
              to:     '/w/$workId',
              params: { workId: String(e.work_id) },
              search: { src: e.material_id },
            })}
          />
        ))}
      </ul>
    </section>
  )
}


function CommunityCard({
  entry, onClick,
}: {
  entry:   ApiCommunityFeedEntry
  onClick: () => void
}) {
  const extra = Math.max(0, entry.chapters_in_feed - 1)
  return (
    <li>
      <button
        type="button"
        onClick={onClick}
        className={cn(
          'group w-full text-left flex flex-col gap-2 cursor-pointer',
        )}
      >
        <div className="relative aspect-[2/3] rounded-md overflow-hidden bg-surface-2">
          <Cover
            src={entry.material_cover}
            title={entry.material_title}
            className="absolute inset-0 group-hover:brightness-110 transition-[filter]"
          />
          {extra > 0 && (
            <span
              className="absolute top-1.5 right-1.5 rounded-full bg-black/70 text-white text-[10px] font-medium px-1.5 py-0.5 tabular"
              title={`${entry.chapters_in_feed} chương đã dịch`}
            >
              +{extra}
            </span>
          )}
          <div className="absolute bottom-0 left-0 right-0 bg-gradient-to-t from-black/85 via-black/40 to-transparent px-2 py-1.5">
            <span className="text-xs font-medium text-white/95 tabular">
              Ch.{entry.chapter_number}
            </span>
          </div>
        </div>
        <div className="min-w-0">
          <p className="text-sm font-medium text-text truncate group-hover:text-accent-text transition-colors">
            {entry.material_title}
          </p>
          <p className="text-xs text-text-subtle truncate">
            {entry.creator_name ? `@${entry.creator_name}` : 'Ẩn danh'}
            {entry.created_at && <> · {timeAgo(entry.created_at)}</>}
          </p>
        </div>
      </button>
    </li>
  )
}


// ── Section heading ────────────────────────────────────────────────


function SectionHeading({
  title, action,
}: {
  title:   string
  action?: React.ReactNode
}) {
  return (
    <div className="flex items-baseline justify-between mb-3">
      <h2 className="text-base sm:text-lg font-semibold tracking-tight text-text">
        {title}
      </h2>
      {action}
    </div>
  )
}


export const Route = createFileRoute('/')({
  component: HomePage,
})
