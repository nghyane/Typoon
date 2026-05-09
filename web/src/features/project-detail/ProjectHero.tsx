import { useMutation, useQueryClient } from '@tanstack/react-query'
import { Plus, Share2, Bookmark, Clock } from 'lucide-react'
import type { ApiProject } from '@shared/api/api'
import { api } from '@shared/api/api'
import { Cover } from '@shared/ui/Cover'
import { Button } from '@shared/ui/Button'
import { Badge } from '@shared/ui/primitives'
import { toast } from '@shared/ui/Toaster'
import { timeAgo } from '@shared/lib/time'
import type { ChapterStats } from './chapter'

interface Props {
  project: ApiProject
  stats:   ChapterStats
  isOwner: boolean
  onAddChapters: () => void
}

export function ProjectHero({ project, stats, isOwner, onAddChapters }: Props) {
  const qc = useQueryClient()

  const togglePin = useMutation({
    mutationFn: () =>
      project.is_pinned ? api.unpinProject(project.project_id) : api.pinProject(project.project_id),
    onSuccess: () => qc.invalidateQueries({ queryKey: ['projects'] }),
  })

  const toggleShare = useMutation({
    mutationFn: () => api.patchSettings(project.project_id, { shared: !project.shared }),
    onSuccess: () => {
      qc.invalidateQueries({ queryKey: ['projects'] })
      toast.success(project.shared ? 'Đã tắt chia sẻ' : 'Đã bật chia sẻ')
    },
    onError: (e: Error) => toast.error(e.message),
  })

  // Inline progress — replaces the redundant 4-card KPI strip. Filter pills
  // and per-row badges already cover state breakdown; here we just need a
  // single "X/Y" with a thin bar so the user sees overall completion.
  const pct = stats.total > 0 ? Math.round((stats.done / stats.total) * 100) : 0

  return (
    <div className="px-4 sm:px-6 pt-4 sm:pt-6 pb-4 sm:pb-5 flex items-start gap-3 sm:gap-4">
      <Cover
        src={project.cover_url}
        title={project.title}
        fontSize="text-xl"
        version={project.updated_at}
        className="w-20 aspect-[2/3] rounded-md shrink-0"
      />

      <div className="flex-1 min-w-0">
        <div className="flex flex-col sm:flex-row sm:items-start sm:justify-between gap-2 sm:gap-3">
          <div className="min-w-0">
            <h1 className="text-lg sm:text-2xl font-semibold tracking-tight text-text line-clamp-2">
              {project.title}
            </h1>

            <div className="flex items-center gap-2 mt-2 flex-wrap">
              <span className="inline-flex items-center gap-1 h-[22px] px-2 rounded-xs bg-surface-2 text-[11px] font-semibold text-text-muted uppercase tracking-wider">
                {project.source_lang}
                <span className="text-text-subtle">→</span>
                {project.target_lang}
              </span>
              {project.shared && (
                <Badge tone="success" dot={false}>
                  <Share2 size={10} /> Đã chia sẻ
                </Badge>
              )}
              {!isOwner && <Badge tone="neutral" dot={false}>Chỉ xem</Badge>}
              {project.updated_at && (
                <>
                  <span className="text-xs text-text-subtle">·</span>
                  <span className="inline-flex items-center gap-1.5 text-xs text-text-subtle">
                    <Clock size={11} />Cập nhật {timeAgo(project.updated_at)}
                  </span>
                </>
              )}
            </div>

            {project.description && (
              <p className="mt-3 text-sm text-text-muted leading-relaxed line-clamp-2 max-w-2xl">
                {project.description}
              </p>
            )}

            {/* Inline progress — only when there's something to track */}
            {stats.total > 0 && (
              <div className="mt-3 flex items-center gap-3 max-w-md">
                <div className="flex-1 h-1 rounded-full bg-surface-2 overflow-hidden">
                  <div
                    className="h-full rounded-full bg-success transition-[width] duration-300"
                    style={{ width: `${pct}%` }}
                  />
                </div>
                <span className="text-xs text-text-subtle tabular shrink-0">
                  <span className="text-text-muted font-medium">{stats.done}</span>
                  <span className="opacity-50">/</span>
                  {stats.total}
                </span>
              </div>
            )}
          </div>

          <div className="flex items-center gap-2 shrink-0 self-start">
            {!isOwner && (
              <Button
                onClick={() => togglePin.mutate()}
                title={project.is_pinned ? 'Bỏ lưu' : 'Lưu'}
              >
                <Bookmark size={14} fill={project.is_pinned ? 'currentColor' : 'none'} />
                {project.is_pinned ? 'Đã lưu' : 'Lưu'}
              </Button>
            )}
            {isOwner && (
              <Button
                onClick={() => toggleShare.mutate()}
                disabled={toggleShare.isPending}
                title={project.shared ? 'Tắt chia sẻ' : 'Chia sẻ với cộng đồng'}
              >
                <Share2 size={14} />
                {project.shared ? 'Đang chia sẻ' : 'Chia sẻ'}
              </Button>
            )}
            {isOwner && (
              <Button variant="primary" onClick={onAddChapters}>
                <Plus size={14} />Tải chương
              </Button>
            )}
          </div>
        </div>
      </div>
    </div>
  )
}
