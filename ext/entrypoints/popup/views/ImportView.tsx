// Popup-driven import flow with a background upload queue.
//
// Top of the popup: form (project + chapter number + title) +
// thumb strip of the picked images. User clicks [Tải lên] →
// the SW queues the job and immediately we clear the form so the
// user can pick the next chapter (auto-extract if domain is trained).
//
// Below the form: queue list with progress, retry, dismiss.

import { useEffect, useMemo, useState } from 'react'
import { Button } from '@shared/ui/Button'
import { Field, input, Spinner } from '@shared/ui/Field'
import { ProjectPicker } from './ProjectPicker'
import { CreateProjectModal } from './CreateProjectModal'
import { useConfig } from '@shell/hooks/useConfig'
import { useMyProjects } from '@shell/hooks/useMyProjects'
import { useProfile } from '@shell/hooks/useProfile'
import { useQueue } from '@shell/hooks/useQueue'
import {
  cancelJob, clearAllJobs, clearFinishedJobs, consumePendingPick,
  dismissJob, enqueueUpload, retryJob, startPickerOnActiveTab,
  suggestNextNumber, tryAutoPick,
  type PickedSelection,
} from '@shell/pipeline'
import type { ApiMeProject } from '@core/typoon'
import type { JobPhase, QueuedJob } from '@core/upload/state'

type LocalPhase =
  | { kind: 'loading' }
  | { kind: 'idle' }
  | { kind: 'ready'; selection: PickedSelection }

const PHASE_LABEL: Record<JobPhase, string> = {
  queued:     'Trong hàng đợi',
  fetching:   'Đang tải ảnh',
  packing:    'Đang đóng gói',
  uploading:  'Đang upload',
  finalizing: 'Engine đang xử lý',
  done:       'Đã upload',
  error:      'Lỗi',
}

const PHASE_TONE: Record<JobPhase, string> = {
  queued:     'text-text-subtle',
  fetching:   'text-info-text',
  packing:    'text-info-text',
  uploading:  'text-info-text',
  finalizing: 'text-info-text',
  done:       'text-success-text',
  error:      'text-error-text',
}

export function ImportView() {
  const { config, save } = useConfig()
  const { all: projects } = useMyProjects()
  const profile = useProfile()
  const queue   = useQueue()

  const [local,    setLocal]    = useState<LocalPhase>({ kind: 'loading' })
  const [project,  setProject]  = useState<ApiMeProject | null>(null)
  const [title,    setTitle]    = useState('')
  const [number,   setNumber]   = useState('')
  const [creating, setCreating] = useState<string | null>(null)

  // On mount: prefer a pending pick written by the content script
  // (after the user picked something on the page and the popup
  // auto-closed). Otherwise try a silent auto-extract using the
  // saved selector. Otherwise show the empty state.
  useEffect(() => {
    let alive = true
    void (async () => {
      const pending = await consumePendingPick()
      if (!alive) return
      if (pending) { setLocal({ kind: 'ready', selection: pending }); return }

      const auto = await tryAutoPick()
      if (!alive) return
      setLocal(auto ? { kind: 'ready', selection: auto } : { kind: 'idle' })
    })()
    return () => { alive = false }
  }, [])

  // Restore last-used project once the projects list arrives.
  useEffect(() => {
    if (project || !projects.length) return
    const last = projects.find(p => p.project_id === config.lastProjectId)
    setProject(last ?? null)
  }, [projects, config.lastProjectId, project])

  // Persist the project pick to lastProjectId immediately, not just
  // after upload — popup may close before upload completes.
  useEffect(() => {
    if (project && project.project_id !== config.lastProjectId) {
      void save({ lastProjectId: project.project_id })
    }
  }, [project, config.lastProjectId, save])

  // Auto-suggest the next chapter number when the user picks a
  // project. Engine round-trip; we need the latest chapter list +
  // the latest queued/in-flight numbers so we don't suggest the
  // same number for two consecutive enqueues.
  useEffect(() => {
    if (!project || number) return
    let alive = true
    void suggestNextNumber(config.token, project.project_id).then(n => {
      if (!alive || !n) return
      // Bump past any queued/active jobs targeting this project so
      // the user doesn't accidentally enqueue duplicate numbers.
      const inFlight = queue.jobs
        .filter(j => j.job.projectId === project.project_id)
        .map(j => j.job.number)
        .filter((s): s is string => Boolean(s))
        .map(parseFloat)
        .filter(n => !isNaN(n))
      const max = inFlight.length ? Math.max(...inFlight, parseFloat(n) - 1) : parseFloat(n) - 1
      setNumber(String(Math.floor(max) + 1))
    })
    return () => { alive = false }
  }, [project, config.token, number, queue.jobs])

  async function startPicker(autoScroll = false) {
    try {
      await startPickerOnActiveTab({ autoScroll })
    } catch (e) {
      console.error('[typoon] picker start failed', e)
    }
  }

  async function submit(selection: PickedSelection) {
    if (!project) return
    try {
      await enqueueUpload({
        selection,
        projectId:    project.project_id,
        projectTitle: project.title,
        number:       number.trim() || undefined,
        title:        title.trim()  || undefined,
      })
      // Reset form so the user can immediately pick the next chapter.
      setTitle('')
      setNumber('')
      setLocal({ kind: 'loading' })
      const auto = await tryAutoPick()
      setLocal(auto ? { kind: 'ready', selection: auto } : { kind: 'idle' })
    } catch (e) {
      console.error('[typoon] enqueue failed', e)
    }
  }

  const canSubmit = useMemo(
    () => local.kind === 'ready' && Boolean(project),
    [local, project],
  )

  return (
    <div className="w-[380px] p-4 space-y-3">
      <header className="flex items-center justify-between gap-2">
        <div className="flex items-center gap-2 min-w-0">
          {profile?.avatar_url ? (
            <img
              src={profile.avatar_url}
              alt=""
              className="size-6 rounded-full flex-none"
              referrerPolicy="no-referrer"
            />
          ) : (
            <div className="size-6 rounded-full bg-surface-2 flex-none" />
          )}
          <span className="text-sm font-medium text-text truncate">
            {profile?.display_name ?? 'Đang tải…'}
          </span>
        </div>
        <Button variant="ghost" size="sm" onClick={() => save({ token: '' })}>
          Đăng xuất
        </Button>
      </header>

      {local.kind === 'loading' && (
        <div className="bg-surface rounded-md p-3 text-xs text-text-muted flex items-center gap-2">
          <Spinner size={14} /> Đang đọc trang…
        </div>
      )}

      {local.kind === 'idle' && <IdlePanel onPick={() => startPicker(false)} />}

      {local.kind === 'ready' && (
        <Form
          selection={local.selection}
          project={project}
          onProject={setProject}
          onCreate={setCreating}
          number={number}
          onNumber={setNumber}
          title={title}
          onTitle={setTitle}
          canSubmit={canSubmit}
          onSubmit={() => submit(local.selection)}
          onCancel={() => setLocal({ kind: 'idle' })}
          onRepick={() => startPicker(false)}
          onRepickFull={() => startPicker(true)}
        />
      )}

      {queue.jobs.length > 0 && (
        <QueuePanel
          jobs={queue.jobs}
          onDismiss={dismissJob}
          onCancel={cancelJob}
          onRetry={retryJob}
          onClearFinished={clearFinishedJobs}
          onClearAll={clearAllJobs}
        />
      )}

      {creating !== null && (
        <CreateProjectModal
          initialTitle={creating}
          onCreated={(p) => { setProject(p); setCreating(null) }}
          onCancel={() => setCreating(null)}
        />
      )}
    </div>
  )
}

// ── sub-views ──────────────────────────────────────────────────────

function IdlePanel({ onPick }: { onPick: () => void }) {
  return (
    <div className="bg-surface rounded-md p-3 text-xs text-text-muted space-y-2">
      <p>Mở chương truyện cần upload trên trình duyệt, sau đó:</p>
      <Button variant="primary" size="md" onClick={onPick}>
        Chọn ảnh trên trang này
      </Button>
    </div>
  )
}

function Form({
  selection, project, onProject, onCreate,
  number, onNumber, title, onTitle,
  canSubmit, onSubmit, onCancel, onRepick, onRepickFull,
}: {
  selection:    PickedSelection
  project:      ApiMeProject | null
  onProject:    (p: ApiMeProject) => void
  onCreate:     (suggested: string) => void
  number:       string
  onNumber:     (s: string) => void
  title:        string
  onTitle:      (s: string) => void
  canSubmit:    boolean
  onSubmit:     () => void
  onCancel:     () => void
  onRepick:     () => void
  onRepickFull: () => void
}) {
  return (
    <>
      <ThumbStrip selection={selection} onRepick={onRepick} />

      <Field label="Project">
        <ProjectPicker
          value={project}
          onChange={onProject}
          onCreate={onCreate}
        />
      </Field>

      <div className="grid grid-cols-[80px_1fr] gap-2">
        <Field label="Số chương">
          <input
            className={input}
            type="text"
            inputMode="decimal"
            placeholder="—"
            value={number}
            onChange={e => onNumber(e.target.value)}
          />
        </Field>
        <Field label="Tên chương">
          <input
            className={input}
            type="text"
            placeholder="Tuỳ chọn"
            value={title}
            onChange={e => onTitle(e.target.value)}
          />
        </Field>
      </div>

      <div className="flex items-center justify-between pt-1">
        <button
          type="button"
          onClick={onRepickFull}
          className="text-[11px] text-text-subtle hover:text-text-muted underline-offset-2 hover:underline"
          title="Mở picker và bật auto-scroll để load các ảnh lazy ở viewer dài"
        >
          Thiếu ảnh? Chọn lại với auto-scroll
        </button>
        <div className="flex justify-end gap-2 ml-auto">
          <Button variant="ghost" size="sm" onClick={onCancel}>
            Hủy
          </Button>
          <Button
            variant="primary"
            size="md"
            disabled={!canSubmit}
            onClick={onSubmit}
          >
            Tải lên
          </Button>
        </div>
      </div>
    </>
  )
}

function ThumbStrip({
  selection, onRepick,
}: {
  selection: PickedSelection
  onRepick:  () => void
}) {
  const sample = selection.images.slice(0, 6)
  return (
    <div className="bg-surface rounded-md p-2 space-y-2">
      <div className="flex items-center justify-between text-xs">
        <span className="text-text-muted truncate">
          <span className="text-accent-text font-medium">{selection.images.length} ảnh</span>
          <span className="text-text-subtle"> · {selection.domain}</span>
        </span>
        <button
          type="button"
          onClick={onRepick}
          className="text-text-subtle hover:text-text underline-offset-2 hover:underline"
        >
          Chọn lại
        </button>
      </div>
      <div className="flex gap-1 overflow-x-auto">
        {sample.map((img, i) => (
          <img
            key={i}
            src={img.url}
            alt=""
            loading="lazy"
            className="size-12 object-cover rounded-sm bg-surface-2 flex-none"
            referrerPolicy="no-referrer"
            onError={(e) => { (e.target as HTMLImageElement).style.opacity = '0.2' }}
          />
        ))}
        {selection.images.length > sample.length && (
          <div className="size-12 grid place-items-center bg-surface-2 rounded-sm flex-none text-[10px] text-text-subtle">
            +{selection.images.length - sample.length}
          </div>
        )}
      </div>
    </div>
  )
}

function QueuePanel({
  jobs, onDismiss, onCancel, onRetry, onClearFinished, onClearAll,
}: {
  jobs: QueuedJob[]
  onDismiss: (id: string) => void
  onCancel:  (id: string) => void
  onRetry:   (id: string) => void
  onClearFinished: () => void
  onClearAll:      () => void
}) {
  const finishedCount = jobs.filter(j => j.phase === 'done' || j.phase === 'error').length
  const idleCount     = jobs.filter(j => j.phase === 'queued' || j.phase === 'done' || j.phase === 'error').length

  return (
    <div className="space-y-1.5">
      <div className="flex items-center justify-between">
        <h2 className="text-[11px] uppercase tracking-wider text-text-subtle font-medium">
          Hàng đợi · {jobs.length}
        </h2>
        <div className="flex items-center gap-3">
          {finishedCount > 0 && (
            <button
              type="button"
              onClick={onClearFinished}
              className="text-[11px] text-text-subtle hover:text-text-muted underline-offset-2 hover:underline"
            >
              Xóa đã xong
            </button>
          )}
          {idleCount > 0 && (
            <button
              type="button"
              onClick={() => {
                if (confirm('Xóa toàn bộ hàng đợi (trừ job đang chạy)?')) onClearAll()
              }}
              className="text-[11px] text-error-text hover:brightness-110 underline-offset-2 hover:underline"
            >
              Xóa tất cả
            </button>
          )}
        </div>
      </div>
      <ul className="space-y-1.5">
        {jobs.map(j => (
          <JobRow
            key={j.id}
            job={j}
            onDismiss={() => onDismiss(j.id)}
            onCancel={() => onCancel(j.id)}
            onRetry={() => onRetry(j.id)}
          />
        ))}
      </ul>
    </div>
  )
}

function JobRow({
  job, onDismiss, onCancel, onRetry,
}: {
  job: QueuedJob
  onDismiss: () => void
  onCancel:  () => void
  onRetry:   () => void
}) {
  const { phase, fetched, total, bytesSent, bytesTotal, speedBps, etaSeconds } = job
  const progressing = phase === 'fetching' || phase === 'packing'
                   || phase === 'uploading' || phase === 'finalizing'

  // Two phases drive the bar:
  //   fetching → fraction of pages pulled from the CDN
  //   uploading → fraction of bytes pushed to the inbox
  // Packing + finalizing are short and indeterminate; we render the
  // bar at 100% (packing) or 0% restart (finalizing reuses uploading
  // numbers so it stays at 100%).
  const pct = phase === 'fetching' && total > 0
    ? Math.round((fetched / total) * 100)
    : phase === 'uploading' && bytesTotal && bytesTotal > 0
      ? Math.round(((bytesSent ?? 0) / bytesTotal) * 100)
      : phase === 'packing' || phase === 'finalizing'
        ? 100
        : 0

  const label = job.job.title?.trim()
    ? job.job.title
    : job.job.number
      ? `Chương ${job.job.number}`
      : 'Chương'

  const detail = phase === 'fetching'
    ? `${fetched}/${total}`
    : phase === 'uploading' && bytesTotal
      ? fmtUploadDetail(bytesSent ?? 0, bytesTotal, speedBps, etaSeconds)
      : null

  return (
    <li className="bg-surface rounded-md p-2 space-y-1">
      <div className="flex items-center justify-between gap-2 text-xs">
        <span className="truncate">
          <span className="text-text">{label}</span>
          {job.job.projectTitle && (
            <span className="text-text-subtle"> · {job.job.projectTitle}</span>
          )}
        </span>
        <span className={`text-[11px] font-medium tabular ${PHASE_TONE[phase]} flex-none`}>
          {phase === 'done' && job.chapterNumber
            ? `✓ Ch.${job.chapterNumber}`
            : detail
              ? detail
              : PHASE_LABEL[phase]}
        </span>
      </div>

      {progressing && (
        <div className="h-1 bg-surface-2 rounded-full overflow-hidden">
          <div
            className="h-full bg-accent transition-[width]"
            style={{ width: `${pct}%` }}
          />
        </div>
      )}

      {phase === 'error' && job.error && (
        <p className="text-[11px] text-error-text">{job.error}</p>
      )}

      <div className="flex justify-end gap-2">
        {phase === 'queued' && (
          <button
            type="button"
            onClick={onCancel}
            className="text-[11px] text-text-subtle hover:text-text-muted underline-offset-2 hover:underline"
          >
            Hủy
          </button>
        )}
        {phase === 'error' && (
          <>
            <button
              type="button"
              onClick={onDismiss}
              className="text-[11px] text-text-subtle hover:text-text-muted underline-offset-2 hover:underline"
            >
              Bỏ
            </button>
            <button
              type="button"
              onClick={onRetry}
              className="text-[11px] text-accent-text hover:brightness-110 underline-offset-2 hover:underline"
            >
              Thử lại
            </button>
          </>
        )}
        {phase === 'done' && (
          <button
            type="button"
            onClick={onDismiss}
            className="text-[11px] text-text-subtle hover:text-text-muted underline-offset-2 hover:underline"
          >
            Xóa
          </button>
        )}
      </div>
    </li>
  )
}

function fmtUploadDetail(
  sent: number, total: number,
  speedBps: number | undefined, etaSeconds: number | undefined,
): string {
  const head = `${fmtSize(sent)}/${fmtSize(total)}`
  const tail: string[] = []
  if (speedBps && speedBps > 0)        tail.push(fmtSpeed(speedBps))
  if (etaSeconds && etaSeconds > 0)    tail.push(fmtEta(etaSeconds))
  return tail.length ? `${head} · ${tail.join(' · ')}` : head
}

function fmtSize(b: number): string {
  if (b < 1024)              return `${b}B`
  if (b < 1024 * 1024)       return `${(b / 1024).toFixed(0)}KB`
  return `${(b / 1024 / 1024).toFixed(1)}MB`
}

function fmtSpeed(bps: number): string {
  if (bps < 1024 * 1024) return `${(bps / 1024).toFixed(0)}KB/s`
  return `${(bps / 1024 / 1024).toFixed(1)}MB/s`
}

function fmtEta(s: number): string {
  if (s < 60) return `${s}s`
  return `${Math.floor(s / 60)}m${(s % 60).toString().padStart(2, '0')}s`
}
