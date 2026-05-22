// Home — drag-drop translation entry point + recent activity.

import { useState } from 'react'
import { useNavigate, createFileRoute } from '@tanstack/react-router'

import { useSession } from '@features/auth/session'
import { useQuota } from '@features/jobs/useQuota'
import { useSubmitJob } from '@features/jobs/useSubmitJob'
import { useMyJobs } from '@features/jobs/queries'
import { useLibraryWorks } from '@features/library/queries'
import { HeroDropZone, detectSourceLangFromName } from '@features/jobs/HeroDropZone'
import { buildZipFromDrop } from '@features/jobs/buildZipFromDrop'
import { QuotaMeter } from '@features/jobs/QuotaMeter'
import { JobCard } from '@features/jobs/JobCard'
import { Field, input as inputCls } from '@shared/ui/primitives'
import { Button } from '@shared/ui/Button'
import { toast } from '@shared/ui/Toaster'


function HomePage() {
  const nav     = useNavigate()
  const session = useSession()
  const quota   = useQuota()
  const jobs    = useMyJobs()
  const library = useLibraryWorks()
  const { submit, progress } = useSubmitJob()

  const [sourceLang, setSourceLang] = useState('ja')
  const [targetLang, setTargetLang] = useState<string | null>(null)
  const [workId,     setWorkId]     = useState<string>('')   // empty = blank work
  const [kind,       setKind]       = useState<'translate' | 'analyze'>('translate')
  const [submitting, setSubmitting] = useState(false)

  if (session.status === 'loading') return null

  const lookupTitle = (id: string | null): string | null =>
    !id ? null : (library.data?.find(it => it.id === id)?.title ?? null)

  async function handleDrop(dt: DataTransfer | null, files: FileList | null) {
    if (submitting) return
    setSubmitting(true)
    try {
      const fallback = files?.[0] ?? null
      const built = await buildZipFromDrop(dt, fallback)
      const sl = detectSourceLangFromName(built.filename) ?? sourceLang
      const { job_id } = await submit({
        zip:         built.blob,
        source_lang: sl,
        target_lang: targetLang ?? undefined,
        work_id:     workId || undefined,
        kind,
        onProgress:  () => {},
      })
      toast.success(built.passthrough
        ? 'Đã tạo job, đang xử lý…'
        : `Đã pack ${built.page_count} trang, đang xử lý…`,
      )
      nav({ to: '/jobs/$id', params: { id: String(job_id) } })
    } catch (e) {
      toast.error((e as Error).message)
    } finally {
      setSubmitting(false)
    }
  }

  const uploadPct = progress.total > 0
    ? Math.round((progress.loaded / progress.total) * 100)
    : null

  return (
    <div className="max-w-3xl mx-auto px-4 sm:px-6 py-8 space-y-8">
      {/* Hero */}
      <HeroDropZone
        onDrop={handleDrop}
        disabled={submitting}
        hint={uploadPct !== null ? `Đang tải lên: ${uploadPct}%` : undefined}
      />

      {/* Form */}
      <div className="grid grid-cols-1 sm:grid-cols-2 gap-4">
        <Field label="Ngôn ngữ nguồn">
          <select className={inputCls} value={sourceLang} onChange={e => setSourceLang(e.target.value)}>
            <option value="ja">Tiếng Nhật</option>
            <option value="ko">Tiếng Hàn</option>
            <option value="zh">Tiếng Trung</option>
            <option value="en">Tiếng Anh</option>
          </select>
        </Field>
        <Field label="Ngôn ngữ đích">
          <select
            className={inputCls}
            value={targetLang ?? ''}
            onChange={e => setTargetLang(e.target.value || null)}
          >
            <option value="">{session.user?.preferred_target_lang ? `Mặc định (${session.user.preferred_target_lang})` : 'Mặc định'}</option>
            <option value="vi">Tiếng Việt</option>
            <option value="en">Tiếng Anh</option>
          </select>
        </Field>
        <Field label="Gắn vào truyện trong thư viện">
          <select
            className={inputCls}
            value={workId}
            onChange={e => setWorkId(e.target.value)}
          >
            <option value="">— Không gắn (Work mới) —</option>
            {(library.data ?? []).map(it => (
              <option key={it.id} value={it.id}>{it.title}</option>
            ))}
          </select>
        </Field>
        <Field label="Chế độ">
          <div className="flex gap-2">
            <Button
              variant={kind === 'translate' ? 'primary' : 'ghost'}
              size="sm"
              onClick={() => setKind('translate')}
            >Dịch đầy đủ</Button>
            <Button
              variant={kind === 'analyze' ? 'primary' : 'ghost'}
              size="sm"
              onClick={() => setKind('analyze')}
            >Chỉ phân tích context</Button>
          </div>
        </Field>
      </div>

      {/* Quota */}
      {quota.data && <QuotaMeter quota={quota.data} />}

      {/* Recent jobs */}
      {(jobs.data?.length ?? 0) > 0 && (
        <section>
          <h2 className="text-xs font-semibold text-text-muted uppercase tracking-wide mb-3">
            Hoạt động gần đây
          </h2>
          <div className="space-y-1.5">
            {jobs.data!.slice(0, 5).map(job => (
              <JobCard key={job.id} job={job} workTitle={lookupTitle(job.work_id)} />
            ))}
          </div>
        </section>
      )}
    </div>
  )
}

export const Route = createFileRoute('/')({
  component: HomePage,
  staticData: { auth: 'required' },
})
