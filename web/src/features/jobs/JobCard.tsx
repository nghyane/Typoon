// JobCard — one row in the activity / recent-jobs list.
//
// Variants:
//   compact   single line, no progress bar
//   full      with progress bar + chapter ref + relative time

import { Download, Trash2, FileArchive } from 'lucide-react'
import { Link } from '@tanstack/react-router'

import { cn } from '@shared/lib/cn'
import { Tag } from '@shared/ui/primitives'
import type { ApiJob } from '@shared/api/api'
import { JobStateBadge } from './JobStateBadge'

interface Props {
  job:        ApiJob
  /** Resolved title from works lookup (work_id → Work.title). */
  workTitle?: string | null
  variant?:   'compact' | 'full'
  onDelete?:  (id: number) => void
  className?: string
}

const formatRelative = (iso: string): string => {
  const diff = Date.now() - new Date(iso).getTime()
  const sec  = Math.floor(diff / 1000)
  if (sec < 60)         return `${sec}s trước`
  if (sec < 3600)       return `${Math.floor(sec / 60)} phút trước`
  if (sec < 86_400)     return `${Math.floor(sec / 3600)} giờ trước`
  if (sec < 7 * 86_400) return `${Math.floor(sec / 86_400)} ngày trước`
  try { return new Date(iso).toLocaleDateString() } catch { return iso }
}

export function JobCard({ job, workTitle, variant = 'compact', onDelete, className }: Props) {
  const pct = job.progress_total && job.progress_index !== null
    ? Math.round((job.progress_index / Math.max(job.progress_total, 1)) * 100)
    : null
  const title = workTitle ?? job.work_id ?? `Job #${job.id}`

  return (
    <div className={cn(
      'group flex items-center gap-3 px-3 py-2.5 rounded-sm bg-surface hover:bg-hover transition-colors',
      className,
    )}>
      <FileArchive size={16} className="text-text-subtle flex-none" />

      <Link
        to="/jobs/$id"
        params={{ id: String(job.id) }}
        className="min-w-0 flex-1"
      >
        <div className="flex items-center gap-2">
          <span className="text-sm font-medium text-text truncate">{title}</span>
          {job.kind === 'analyze' && (
            <Tag tone="info" size="sm">phân tích</Tag>
          )}
        </div>
        <div className="flex items-center gap-2 text-xs text-text-muted mt-0.5">
          <span>{job.page_count ?? job.estimated_pages ?? '?'} trang</span>
          <span>·</span>
          <span>{job.source_lang} → {job.target_lang}</span>
          <span>·</span>
          <span>{formatRelative(job.created_at)}</span>
        </div>

        {variant === 'full' && pct !== null && (
          <div className="h-1 bg-surface-2 rounded-full overflow-hidden mt-2">
            <div
              className="h-full bg-accent transition-all"
              style={{ width: `${pct}%` }}
            />
          </div>
        )}
      </Link>

      <div className="flex items-center gap-1 flex-none">
        <JobStateBadge state={job.state} />

        {job.state === 'done' && job.archive_url && (
          <a
            href={job.archive_url}
            download
            className="p-1.5 rounded-sm text-text-subtle hover:text-text hover:bg-surface-2 transition-colors"
            title="Tải về .bnl"
          >
            <Download size={14} />
          </a>
        )}

        {onDelete && (
          <button
            type="button"
            onClick={(e) => { e.preventDefault(); e.stopPropagation(); onDelete(job.id) }}
            className="p-1.5 rounded-sm text-text-subtle hover:text-error-text hover:bg-surface-2 transition-colors opacity-0 group-hover:opacity-100"
            title="Xóa job"
          >
            <Trash2 size={14} />
          </button>
        )}
      </div>
    </div>
  )
}
