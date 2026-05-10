// Popup-driven import flow with a background upload queue.
//
// Layout: 3-zone sticky shell (header / body / footer) — same pattern
// as the SPA modal, adapted for a 380 px popup. The footer pins the
// primary action so the user never has to scroll to find "Tải lên",
// even with a long thumb strip. The queue lives below the form in
// the body (compact rows, hover-reveal actions).

import { useEffect, useMemo, useState } from 'react'
import { ChevronDown, LogOut, MoreHorizontal, RefreshCw, X } from 'lucide-react'
import { Button } from '@shared/ui/Button'
import { Field, input, Spinner } from '@shared/ui/Field'
import { JobPhasePill } from '@shared/ui/JobPhasePill'
import { JobProgressBar } from '@shared/ui/JobProgressBar'
import { InlineConfirm } from '@shared/ui/InlineConfirm'
import { cn } from '@shared/lib/cn'
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
  // Track whether the auto-suggest already ran for the current
  // project; user clearing the field after that should NOT re-trigger
  // the suggestion (would override their intent).
  const [autoSuggested, setAutoSuggested] = useState<number | null>(null)

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

  // Auto-suggest the next chapter number once per project change.
  // Effect deps deliberately exclude `number` — re-running when the
  // user clears the input would clobber their intent. State machine:
  //   project changes → autoSuggested != project.id → fetch + fill
  //   project unchanged → never re-fetch
  useEffect(() => {
    if (!project) return
    if (autoSuggested === project.project_id) return
    let alive = true
    void suggestNextNumber(config.token, project.project_id).then(n => {
      if (!alive || !n) return
      const inFlight = queue.jobs
        .filter(j => j.job.projectId === project.project_id)
        .map(j => j.job.number)
        .filter((s): s is string => Boolean(s))
        .map(parseFloat)
        .filter(n => !isNaN(n))
      const max = inFlight.length ? Math.max(...inFlight, parseFloat(n) - 1) : parseFloat(n) - 1
      setNumber(String(Math.floor(max) + 1))
      setAutoSuggested(project.project_id)
    })
    return () => { alive = false }
  }, [project, config.token, autoSuggested, queue.jobs])

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
      setAutoSuggested(null)
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
    <div className="w-[380px] max-h-[600px] flex flex-col bg-bg">
      {/* Sticky header — brand + profile menu */}
      <Header
        profileName={profile?.display_name ?? null}
        avatarUrl={profile?.avatar_url ?? null}
        onLogout={() => save({ token: '' })}
      />

      {/* Scrollable body */}
      <div className="flex-1 overflow-y-auto overscroll-contain">
        <div className="px-3 py-3 space-y-3">
          {local.kind === 'loading' && <LoadingPanel />}
          {local.kind === 'idle'    && <IdlePanel onPick={() => startPicker(false)} />}

          {local.kind === 'ready' && (
            <Form
              selection={local.selection}
              project={project}
              onProject={(p) => { setProject(p); setAutoSuggested(null) }}
              onCreate={setCreating}
              number={number}
              onNumber={setNumber}
              title={title}
              onTitle={setTitle}
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
        </div>
      </div>

      {/* Sticky footer — primary action, only when there's a selection
          ready to submit. Hidden in idle/loading so the popup isn't
          a wall of buttons. */}
      {local.kind === 'ready' && (
        <Footer
          selection={local.selection}
          number={number}
          canSubmit={canSubmit}
          onSubmit={() => submit(local.selection)}
          onCancel={() => setLocal({ kind: 'idle' })}
        />
      )}

      {creating !== null && (
        <CreateProjectModal
          initialTitle={creating}
          onCreated={(p) => { setProject(p); setAutoSuggested(null); setCreating(null) }}
          onCancel={() => setCreating(null)}
        />
      )}
    </div>
  )
}


// ── Header ─────────────────────────────────────────────────────────


function Header({
  profileName, avatarUrl, onLogout,
}: {
  profileName: string | null
  avatarUrl:   string | null
  onLogout:    () => void
}) {
  const [open, setOpen] = useState(false)
  return (
    <header className="flex items-center justify-between gap-2 px-3 h-11 bg-bg/95 backdrop-blur border-b border-border-soft shrink-0">
      <div className="flex items-center gap-2 min-w-0">
        <span className="text-[13px] font-semibold text-text tracking-tight">Typoon</span>
        <span className="text-[11px] text-text-subtle">Importer</span>
      </div>

      <div className="relative">
        <button
          type="button"
          onClick={() => setOpen(o => !o)}
          className="flex items-center gap-1.5 h-7 pl-1 pr-1.5 rounded-sm hover:bg-hover transition-colors cursor-pointer"
          title={profileName ?? ''}
          aria-label="Tài khoản"
        >
          {avatarUrl ? (
            <img
              src={avatarUrl}
              alt=""
              className="size-5 rounded-full"
              referrerPolicy="no-referrer"
            />
          ) : (
            <span className="size-5 rounded-full bg-surface-2" />
          )}
          <ChevronDown size={11} className="text-text-subtle" />
        </button>
        {open && (
          <>
            <button
              type="button"
              aria-label="Đóng menu"
              className="fixed inset-0 z-10 cursor-default"
              onClick={() => setOpen(false)}
            />
            <div className="absolute right-0 top-full mt-1 z-20 w-48 rounded-md bg-surface shadow-[0_8px_24px_rgb(0,0,0,0.4)] overflow-hidden">
              <div className="px-3 py-2.5 border-b border-border-soft">
                <p className="text-xs font-medium text-text truncate">
                  {profileName ?? 'Đang tải…'}
                </p>
              </div>
              <button
                type="button"
                onClick={() => { setOpen(false); onLogout() }}
                className="w-full flex items-center gap-2 px-3 py-2 text-xs text-text hover:bg-hover cursor-pointer"
              >
                <LogOut size={11} className="text-text-subtle" />
                Đăng xuất
              </button>
            </div>
          </>
        )}
      </div>
    </header>
  )
}


// ── Body panels ────────────────────────────────────────────────────


function LoadingPanel() {
  return (
    <div className="bg-surface rounded-md p-3 text-xs text-text-muted flex items-center gap-2">
      <Spinner size={12} /> Đang đọc trang…
    </div>
  )
}

function IdlePanel({ onPick }: { onPick: () => void }) {
  return (
    <div className="bg-surface rounded-md p-4 text-center space-y-3">
      <div className="space-y-1">
        <p className="text-sm font-medium text-text">Bắt đầu</p>
        <p className="text-xs text-text-subtle leading-relaxed">
          Mở chương cần upload, sau đó chọn ảnh trên trang.
        </p>
      </div>
      <Button variant="primary" size="md" className="w-full" onClick={onPick}>
        Chọn ảnh trên trang này
      </Button>
    </div>
  )
}


// ── Form ───────────────────────────────────────────────────────────


function Form({
  selection, project, onProject, onCreate,
  number, onNumber, title, onTitle, onRepick, onRepickFull,
}: {
  selection:    PickedSelection
  project:      ApiMeProject | null
  onProject:    (p: ApiMeProject) => void
  onCreate:     (suggested: string) => void
  number:       string
  onNumber:     (s: string) => void
  title:        string
  onTitle:      (s: string) => void
  onRepick:     () => void
  onRepickFull: () => void
}) {
  return (
    <div className="space-y-3">
      <ThumbStrip selection={selection} onRepick={onRepick} onRepickFull={onRepickFull} />

      <Field label="Project">
        <ProjectPicker
          value={project}
          onChange={onProject}
          onCreate={onCreate}
        />
      </Field>

      <div className="grid grid-cols-[5rem_1fr] gap-2">
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
    </div>
  )
}


function ThumbStrip({
  selection, onRepick, onRepickFull,
}: {
  selection:    PickedSelection
  onRepick:     () => void
  onRepickFull: () => void
}) {
  const MAX = 8
  const sample = selection.images.slice(0, MAX)
  const overflow = selection.images.length - sample.length

  return (
    <div className="bg-surface rounded-md p-2 space-y-2">
      <div className="flex items-center justify-between gap-2 text-xs">
        <span className="min-w-0 truncate">
          <span className="text-accent-text font-medium tabular">{selection.images.length} ảnh</span>
          <span className="text-text-subtle"> · {selection.domain}</span>
        </span>
        <button
          type="button"
          onClick={onRepick}
          className="inline-flex items-center gap-1 text-text-subtle hover:text-text"
          title="Chọn lại"
        >
          <RefreshCw size={10} />
          <span>Chọn lại</span>
        </button>
      </div>

      <div className="flex gap-1 overflow-x-auto -mx-0.5 px-0.5">
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
        {overflow > 0 && (
          <div className="size-12 grid place-items-center bg-surface-2 rounded-sm flex-none text-[10px] font-medium text-text-subtle">
            +{overflow}
          </div>
        )}
      </div>

      <button
        type="button"
        onClick={onRepickFull}
        className="text-[11px] text-text-subtle hover:text-text-muted underline-offset-2 hover:underline"
        title="Cuộn trang để load các ảnh chưa render rồi chọn lại"
      >
        Thiếu ảnh? Tự động cuộn rồi chọn lại
      </button>
    </div>
  )
}


// ── Footer ─────────────────────────────────────────────────────────


function Footer({
  selection, number, canSubmit, onSubmit, onCancel,
}: {
  selection: PickedSelection
  number:    string
  canSubmit: boolean
  onSubmit:  () => void
  onCancel:  () => void
}) {
  const trimmed = number.trim()
  return (
    <footer className="flex items-center gap-2 px-3 py-2.5 border-t border-border-soft bg-bg/95 backdrop-blur shrink-0">
      <div className="flex-1 min-w-0 text-[11px] text-text-subtle truncate">
        <span className="text-text font-medium tabular">{selection.images.length}</span> ảnh
        {trimmed && (
          <>
            <span className="mx-1.5 text-text-subtle/60">·</span>
            <span className="tabular">Ch.{trimmed}</span>
          </>
        )}
      </div>
      <Button variant="ghost" size="sm" onClick={onCancel}>
        Huỷ
      </Button>
      <Button
        variant="primary"
        size="sm"
        disabled={!canSubmit}
        onClick={onSubmit}
      >
        Tải lên
      </Button>
    </footer>
  )
}


// ── Queue panel ────────────────────────────────────────────────────


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
  const idleCount     = jobs.filter(
    j => j.phase === 'queued' || j.phase === 'done' || j.phase === 'error',
  ).length

  return (
    <section className="space-y-1.5">
      <div className="flex items-center justify-between px-1">
        <h2 className="text-[10px] uppercase tracking-wider text-text-subtle font-semibold">
          Hàng đợi <span className="text-text-muted tabular">{jobs.length}</span>
        </h2>
        <div className="flex items-center gap-3">
          {finishedCount > 0 && (
            <button
              type="button"
              onClick={onClearFinished}
              className="text-[11px] text-text-subtle hover:text-text-muted underline-offset-2 hover:underline"
            >
              Xoá đã xong
            </button>
          )}
          {idleCount > 0 && (
            <InlineConfirm
              label="Xoá tất cả"
              confirmLabel="Xác nhận?"
              onConfirm={onClearAll}
              className="text-[11px]"
            />
          )}
        </div>
      </div>
      <ul className="space-y-1">
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
    </section>
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

  // Determinate phases:
  //   fetching → fraction of pages pulled from the source CDN
  //   uploading → fraction of bytes pushed to the inbox
  // Indeterminate phases:
  //   packing / finalizing → shimmer, no real fraction
  //   queued → empty bar
  const determinate = phase === 'fetching' || phase === 'uploading'
  const indeterminate = phase === 'packing' || phase === 'finalizing'
  const showBar = determinate || indeterminate

  const pct = phase === 'fetching' && total > 0
    ? (fetched / total) * 100
    : phase === 'uploading' && bytesTotal && bytesTotal > 0
      ? ((bytesSent ?? 0) / bytesTotal) * 100
      : indeterminate ? 100 : 0

  const labelText = job.job.title?.trim()
    ? job.job.title
    : job.job.number
      ? `Chương ${job.job.number}`
      : 'Chương'

  const detail = phase === 'fetching'
    ? `${fetched}/${total}`
    : phase === 'uploading' && bytesTotal
      ? fmtUploadDetail(bytesSent ?? 0, bytesTotal, speedBps, etaSeconds)
      : phase === 'done' && job.chapterNumber
        ? `Ch.${job.chapterNumber}`
        : null

  return (
    <li
      className={cn(
        'group relative bg-surface rounded-md px-2.5 py-2',
        'transition-colors hover:bg-surface-2',
      )}
    >
      <div className="flex items-center gap-2">
        <JobPhasePill phase={phase} />
        <span className="flex-1 min-w-0 text-xs">
          <span className="text-text font-medium truncate">{labelText}</span>
          {job.job.projectTitle && (
            <span className="text-text-subtle"> · {job.job.projectTitle}</span>
          )}
        </span>
        {detail && (
          <span className="text-[10px] tabular text-text-muted shrink-0">
            {detail}
          </span>
        )}
        <RowActions
          phase={phase}
          onDismiss={onDismiss}
          onCancel={onCancel}
          onRetry={onRetry}
        />
      </div>

      {showBar && (
        <div className="mt-1.5">
          <JobProgressBar pct={pct} indeterminate={indeterminate && !determinate} />
        </div>
      )}

      {phase === 'error' && job.error && (
        <p className="mt-1 text-[10px] text-error-text/90 leading-snug truncate" title={job.error}>
          {job.error}
        </p>
      )}
    </li>
  )
}


// ── Row actions — hover-revealed kebab menu ───────────────────────


function RowActions({
  phase, onDismiss, onCancel, onRetry,
}: {
  phase: JobPhase
  onDismiss: () => void
  onCancel:  () => void
  onRetry:   () => void
}) {
  const [open, setOpen] = useState(false)

  // Pick the action set per phase. Empty set → no kebab at all.
  const items = useMemo(() => {
    const out: Array<{ label: string; icon: typeof X; onClick: () => void; danger?: boolean }> = []
    if (phase === 'queued') {
      out.push({ label: 'Huỷ', icon: X, onClick: onCancel, danger: true })
    } else if (phase === 'error') {
      out.push({ label: 'Thử lại', icon: RefreshCw, onClick: onRetry })
      out.push({ label: 'Bỏ',      icon: X,         onClick: onDismiss, danger: true })
    } else if (phase === 'done') {
      out.push({ label: 'Xoá khỏi danh sách', icon: X, onClick: onDismiss, danger: true })
    }
    return out
  }, [phase, onCancel, onDismiss, onRetry])

  if (items.length === 0) return null

  return (
    <div className="relative shrink-0">
      <button
        type="button"
        onClick={() => setOpen(o => !o)}
        className={cn(
          'size-5 rounded-sm flex items-center justify-center',
          'text-text-subtle hover:text-text hover:bg-hover',
          'opacity-0 group-hover:opacity-100 focus:opacity-100',
          'transition-opacity cursor-pointer',
          open && 'opacity-100',
        )}
        aria-label="Thao tác"
      >
        <MoreHorizontal size={12} />
      </button>
      {open && (
        <>
          <button
            type="button"
            aria-label="Đóng menu"
            className="fixed inset-0 z-10 cursor-default"
            onClick={() => setOpen(false)}
          />
          <div className="absolute right-0 top-full mt-1 z-20 w-44 rounded-md bg-surface shadow-[0_8px_24px_rgb(0,0,0,0.4)] overflow-hidden py-1">
            {items.map(({ label, icon: Icon, onClick, danger }) => (
              <button
                key={label}
                type="button"
                onClick={() => { setOpen(false); onClick() }}
                className={cn(
                  'w-full flex items-center gap-2 px-3 py-1.5 text-xs cursor-pointer',
                  danger ? 'text-error-text hover:bg-error/10' : 'text-text hover:bg-hover',
                )}
              >
                <Icon size={11} className="text-text-subtle" />
                {label}
              </button>
            ))}
          </div>
        </>
      )}
    </div>
  )
}


// ── Formatters ─────────────────────────────────────────────────────


function fmtUploadDetail(
  sent: number, total: number,
  speedBps: number | undefined, etaSeconds: number | undefined,
): string {
  const head = `${fmtSize(sent)}/${fmtSize(total)}`
  const tail: string[] = []
  if (speedBps && speedBps > 0)     tail.push(fmtSpeed(speedBps))
  if (etaSeconds && etaSeconds > 0) tail.push(fmtEta(etaSeconds))
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
