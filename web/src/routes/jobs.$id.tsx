// /jobs/$id — detail view with live progress.

import { createFileRoute, useNavigate } from '@tanstack/react-router'
import { Download, FileArchive, ArrowLeft, Trash2, BookOpen, Save } from 'lucide-react'

import {
  useJob, useDeleteJob,
} from '@features/jobs/queries'
import { useWork } from '@features/works/queries'
import {
  useDownloadTranslatedArchive, useSavedArchive,
} from '@features/reader/archives'
import { JobStateBadge } from '@features/jobs/JobStateBadge'
import { StageChecklist } from '@features/jobs/StageChecklist'
import { Spinner, Tag } from '@shared/ui/primitives'
import { Button } from '@shared/ui/Button'
import { EmptyState } from '@shared/ui/EmptyState'
import { toast } from '@shared/ui/Toaster'


function JobDetailPage() {
  const { id }   = Route.useParams()
  const jobId    = Number(id)
  const nav      = useNavigate()
  const job      = useJob(jobId)
  const del      = useDeleteJob()
  const download = useDownloadTranslatedArchive()
  const workQ    = useWork(job.data?.work_id ?? null)

  // Look up if this job's archive is saved offline already.
  const chapter_ref = `job:${jobId}`        // jobs without explicit chapter_ref still get a placeholder so archives keyed by composite still work
  const work_id     = job.data?.work_id ?? null
  const saved       = useSavedArchive(work_id, chapter_ref)

  if (job.isPending) {
    return (
      <div className="flex items-center justify-center py-16">
        <Spinner size={20} />
      </div>
    )
  }

  if (!job.data) {
    return (
      <div className="max-w-2xl mx-auto px-4 sm:px-6 py-10">
        <EmptyState
          title="Không tìm thấy job"
          hint="Job có thể đã hết hạn hoặc bị xóa."
        />
      </div>
    )
  }

  const j = job.data
  const workTitle = j.work_id
    ? (workQ.data?.title ?? j.work_id)
    : `Job #${j.id}`

  async function handleDownload() {
    if (!j.archive_url || !j.work_id) return
    try {
      await download.mutateAsync({
        work_id:     j.work_id,
        chapter_ref,
        job_id:      j.id,
        archive_url: j.archive_url,
      })
      toast.success('Đã lưu offline. Có thể đọc khi mất mạng.')
    } catch (e) {
      toast.error((e as Error).message)
    }
  }

  async function handleDelete() {
    if (!confirm('Xóa job này? Archive và context cache sẽ mất.')) return
    await del.mutateAsync(j.id)
    nav({ to: '/jobs' })
  }

  return (
    <div className="max-w-2xl mx-auto px-4 sm:px-6 py-6 space-y-6">
      <header className="flex items-center justify-between">
        <button
          onClick={() => nav({ to: '/jobs' })}
          className="inline-flex items-center gap-2 text-sm text-text-subtle hover:text-text transition-colors"
        >
          <ArrowLeft size={14} /> Quay lại
        </button>
        <JobStateBadge state={j.state} />
      </header>

      <section className="space-y-1">
        <h1 className="text-lg font-semibold text-text">{workTitle}</h1>
        <div className="flex items-center gap-2 text-xs text-text-muted">
          <FileArchive size={12} />
          <span>{j.page_count ?? j.estimated_pages ?? '?'} trang</span>
          <span>·</span>
          <span>{j.source_lang} → {j.target_lang}</span>
          {j.kind === 'analyze' && <Tag tone="info" size="sm">phân tích</Tag>}
        </div>
      </section>

      {/* Progress */}
      <section className="rounded-md bg-surface p-4 space-y-3">
        <StageChecklist job={j} />
        {j.state === 'error' && j.error_message && (
          <div className="text-xs text-error-text bg-error-bg px-3 py-2 rounded-sm">
            {j.error_message}
          </div>
        )}
      </section>

      {/* Actions */}
      {j.state === 'done' && j.kind === 'translate' && (
        <section className="space-y-2">
          <h2 className="text-xs font-semibold text-text-muted uppercase tracking-wide">
            Khi đã xong
          </h2>
          <div className="flex flex-wrap gap-2">
            {j.archive_url && (
              <a href={j.archive_url} download className="inline-flex">
                <Button variant="primary" size="md">
                  <Download size={14} /> Tải file .bnl
                </Button>
              </a>
            )}
            <Button
              variant="secondary"
              size="md"
              onClick={handleDownload}
              disabled={download.isPending || !!saved}
            >
              <Save size={14} />
              {saved ? 'Đã lưu offline' : 'Lưu offline'}
            </Button>
            {j.work_id && (
              <Button
                variant="ghost"
                size="md"
                onClick={() => nav({
                  to: '/r/$workId/$numberNorm',
                  params: { workId: j.work_id!, numberNorm: chapter_ref },
                })}
              >
                <BookOpen size={14} /> Đọc ngay
              </Button>
            )}
          </div>
        </section>
      )}

      {/* Meta */}
      <section className="text-xs text-text-muted space-y-1">
        <div>Tạo: {new Date(j.created_at).toLocaleString()}</div>
        <div>Hết hạn server: {new Date(j.expires_at).toLocaleString()}</div>
        {j.context_version !== null && j.work_id && (
          <div>Context: cập nhật lên v{j.context_version}</div>
        )}
      </section>

      <footer className="pt-4 border-t border-border-soft">
        <Button variant="danger" size="sm" onClick={handleDelete}>
          <Trash2 size={14} /> Xóa job
        </Button>
      </footer>
    </div>
  )
}

export const Route = createFileRoute('/jobs/$id')({
  component: JobDetailPage,
  staticData: { auth: 'required' },
})
