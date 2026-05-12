import { useEffect } from 'react'
import { Link } from '@tanstack/react-router'
import {
  ArrowLeft, AlertTriangle, Clock,
} from 'lucide-react'
import { Cover, coverUrl } from '@shared/ui/Cover'
import { EmptyState } from '@shared/ui/EmptyState'
import { Spinner, Tag } from '@shared/ui/primitives'
import { timeAgo } from '@shared/lib/time'
import { useHeaderStore } from '../../store/header'
import { useSources } from '@features/browse/sources'
import { FollowButton } from '@features/library/views/LibraryCard'
import type {
  ApiLibraryEntry, ApiMaterial, LibraryStatus,
} from '@shared/api/api'
import { useHubData, type HubChapter } from './useHubData'

// =============================================================================
// TitleHub — `/title/$entryId` detail page, pro-design.
//
// Reference: pre-refactor ProjectDetail (commit f8977dd). Same density
// ladder, same DataTable shell, same DataToolbar filter pattern. The
// material-architecture rebuild keeps the visual contract.
//
// Layout:
//   ① Hero          80px cover, title, lang pair + status + age,
//                   description inline (not collapsible), action
//                   cluster right.
//   ② Toolbar       filter segmented + search input.
//   ③ DataTable     Chương / Trạng thái / Cập nhật / Thao tác cols.
//
// Action buttons are disabled placeholders for slice 14 (reader) and
// slice 15 (inline spawn). Everything else is wired.
// =============================================================================

interface Props { entryId: number }

export function TitleHub({ entryId }: Props) {
  // Source registry hydration on direct refresh — same pattern as
  // /library and /settings.
  const ensureBundled = useSources((s) => s.ensureBundled)
  useEffect(() => { ensureBundled() }, [ensureBundled])

  const {
    entry, primaryMaterial, chapters,
    loading, chaptersLoading, error,
  } = useHubData(entryId)

  const setHeader   = useHeaderStore((s) => s.set)
  const clearHeader = useHeaderStore((s) => s.clear)
  useEffect(() => {
    if (primaryMaterial) {
      setHeader(primaryMaterial.material.title, [{ label: 'Thư viện', to: '/library' }])
    } else {
      setHeader('', [{ label: 'Thư viện', to: '/library' }])
    }
    return () => clearHeader()
  }, [primaryMaterial, setHeader, clearHeader])

  if (loading) {
    return (
      <div className="flex items-center justify-center py-24">
        <Spinner size={20} />
      </div>
    )
  }
  if (error || !entry) {
    return (
      <div className="px-4 sm:px-6 pt-12">
        <EmptyState
          icon={AlertTriangle}
          title="Không tải được"
          hint={(error as Error)?.message ?? 'Entry không tồn tại hoặc đã bị xoá.'}
        />
      </div>
    )
  }
  if (!primaryMaterial) {
    return (
      <div className="px-4 sm:px-6 pt-12">
        <EmptyState
          icon={AlertTriangle}
          title="Entry chưa có material chính"
          hint="Mở Cài đặt để link một material làm chính cho entry này."
        />
      </div>
    )
  }

  return (
    <div className="pb-16">
      <MobileBack />
      <Hero entry={entry} material={primaryMaterial.material} />
      <ChapterPanel
        chapters={chapters}
        targetLang={entry.target_lang}
        loading={chaptersLoading}
      />
    </div>
  )
}


// ── Mobile back link ────────────────────────────────────────────────

function MobileBack() {
  return (
    <div className="sm:hidden px-4 pt-4">
      <Link
        to="/library"
        className="inline-flex items-center gap-1.5 text-sm text-text-subtle hover:text-text"
      >
        <ArrowLeft size={14} />
        Thư viện
      </Link>
    </div>
  )
}


// ── Hero ────────────────────────────────────────────────────────────

const STATUS_LABEL: Record<LibraryStatus, string> = {
  reading: 'Đang đọc',
  plan:    'Kế hoạch',
  on_hold: 'Tạm dừng',
  done:    'Đã xong',
  dropped: 'Đã bỏ',
}

function Hero({
  entry, material,
}: {
  entry: ApiLibraryEntry; material: ApiMaterial
}) {
  return (
    <div className="px-4 sm:px-6 pt-4 sm:pt-6 pb-4 sm:pb-5 flex items-start gap-3 sm:gap-4">
      <Cover
        src={coverUrl(material.cover_url, material.updated_at)}
        title={material.title}
        fontSize="text-xl"
        className="w-20 aspect-[2/3] rounded-md shrink-0"
      />
      <div className="flex-1 min-w-0">
        <div className="flex flex-col sm:flex-row sm:items-start sm:justify-between gap-2 sm:gap-3">
          <div className="min-w-0">
            <h1 className="text-lg sm:text-2xl font-semibold tracking-tight text-text line-clamp-2">
              {material.title}
            </h1>

            <div className="flex items-center gap-2 mt-2 flex-wrap text-xs text-text-subtle">
              <span className="inline-flex items-center gap-1 h-[22px] px-2 rounded-xs bg-surface-2 text-[11px] font-semibold uppercase tracking-wider text-text-muted">
                {material.languages[0]?.toUpperCase() ?? '?'}
                <span className="text-text-subtle">→</span>
                {(entry.target_lang ?? '?').toUpperCase()}
              </span>
              <Tag tone="neutral" size="sm">{STATUS_LABEL[entry.status]}</Tag>
              {material.author && <span>{material.author}</span>}
              {material.status && <span>· {material.status}</span>}
              {material.nsfw && (
                <span className="text-[11px] uppercase font-semibold px-1.5 py-0.5 rounded-xs bg-error/15 text-error-text">
                  NSFW
                </span>
              )}
              {material.updated_at && (
                <>
                  <span>·</span>
                  <span className="inline-flex items-center gap-1">
                    <Clock size={11} />
                    Cập nhật {timeAgo(material.updated_at)}
                  </span>
                </>
              )}
            </div>

            {material.description && (
              <p className="mt-3 text-sm text-text-muted leading-relaxed line-clamp-2 max-w-2xl">
                {material.description}
              </p>
            )}

            <ActivityRow summary={entry.translation_summary} />
          </div>

          <div className="flex items-center gap-2 shrink-0 self-start">
            <FollowButton
              entryId={entry.id}
              materialId={material.id}
              title={material.title}
              cover={material.cover_url}
              targetLang={entry.target_lang}
              status={entry.status}
            />
          </div>
        </div>
      </div>
    </div>
  )
}


function ActivityRow({
  summary,
}: {
  summary: ApiLibraryEntry['translation_summary']
}) {
  if (!summary) return null
  if (summary.running === 0 && summary.error === 0 && summary.pending === 0) {
    return null
  }
  return (
    <div className="mt-2 flex items-center gap-1.5">
      {summary.running > 0 && (
        <Tag tone="info" size="sm">{summary.running} đang dịch</Tag>
      )}
      {summary.error > 0 && (
        <Tag tone="error" size="sm">{summary.error} lỗi</Tag>
      )}
      {summary.pending > 0 && (
        <Tag tone="warning" size="sm">{summary.pending} chờ</Tag>
      )}
    </div>
  )
}

// ── Chapter panel ───────────────────────────────────────────────────
// Stub. Slice in progress — DataTable + bulk select + version-aware
// action lands in the next commit.

function ChapterPanel({
  chapters, targetLang, loading,
}: {
  chapters:   HubChapter[]
  targetLang: string | null
  loading:    boolean
}) {
  return (
    <section className="px-4 sm:px-6">
      <p className="text-sm text-text-muted">
        {loading ? 'Đang tải…' : `${chapters.length} chương`}
        {targetLang && ` · đọc bằng ${targetLang.toUpperCase()}`}
      </p>
    </section>
  )
}
