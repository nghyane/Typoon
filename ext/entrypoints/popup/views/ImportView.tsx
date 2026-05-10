// Popup-driven import flow with a lightweight "Recent" panel.
//
// Pro-design north star: show progress of the system, not its
// internals. The user opens the popup to do one thing — turn the
// chapter they're viewing into a Typoon chapter — and reopens it to
// answer one question — "did the last upload land?". Everything else
// (per-phase pills, MB/s, ETA, bulk delete) was visual noise and is
// gone. Three states only: running / done / error. Errors get an
// inline retry button; done rows show a relative timestamp and
// auto-vanish after 24h (the SW garbage-collects on every read).

import { useEffect, useMemo, useState } from 'react'
import { AlertTriangle, Check, ChevronDown, Loader2, LogOut, RefreshCw, X } from 'lucide-react'
import { Button } from '@shared/ui/Button'
import { Field, input, Spinner } from '@shared/ui/Field'
import { cn } from '@shared/lib/cn'
import { ProjectPicker } from './ProjectPicker'
import { CreateProjectModal } from './CreateProjectModal'
import { useConfig } from '@shell/hooks/useConfig'
import { useMyProjects } from '@shell/hooks/useMyProjects'
import { useProfile } from '@shell/hooks/useProfile'
import { useQueue } from '@shell/hooks/useQueue'
import {
  cancelJob, consumePendingPick,
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

  // Restore the user's last-used project, falling back to the first
  // entry when storage doesn't have one yet (fresh install, cleared
  // settings). `useMyProjects` returns the list ordered by recency
  // server-side, so projects[0] is "the most recent project this
  // user touched" — a sane default that lets the form render
  // submit-ready instead of greeting users with a blank picker.
  useEffect(() => {
    if (project || !projects.length) return
    const last = projects.find(p => p.project_id === config.lastProjectId)
    setProject(last ?? projects[0]!)
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
            <RecentPanel
              jobs={queue.jobs}
              onDismiss={dismissJob}
              onCancel={cancelJob}
              onRetry={retryJob}
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
  // Minimal header — no wordmark, no tagline. Browser already shows
  // the extension name in the popup's chrome (icon + Action title);
  // duplicating it inside the panel just steals vertical space.
  // Avatar + chevron is the only chrome we need: account context +
  // logout, accessible from any sub-view.
  const [open, setOpen] = useState(false)
  return (
    <header className="flex items-center justify-end px-2 h-9 shrink-0">
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
                  {profileName ?? 'Tài khoản'}
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



// ── Recent panel ───────────────────────────────────────────────────
//
// One-line rows, three states: running / done / error. No bulk
// actions, no phase pills, no per-row progress bars, no kebab menus.
// Done jobs auto-evict after 24h server-side (see SW readQueue).


function RecentPanel({
  jobs, onDismiss, onCancel, onRetry,
}: {
  jobs:      QueuedJob[]
  onDismiss: (id: string) => void
  onCancel:  (id: string) => void
  onRetry:   (id: string) => void
}) {
  return (
    <section className="space-y-1">
      <h2 className="px-1 text-[10px] uppercase tracking-wider text-text-subtle font-semibold">
        Gần đây
      </h2>
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
  job:       QueuedJob
  onDismiss: () => void
  onCancel:  () => void
  onRetry:   () => void
}) {
  const state = bucketize(job.phase)

  // Title chosen with the user's mental model in mind:
  //   "Ch.328 — Title" if both available
  //   "Ch.328"         if only number
  //   project title    fallback (rare; very early in the pipeline)
  const num   = job.chapterNumber ?? job.job.number?.trim()
  const title = job.job.title?.trim()
  const head  = num
    ? (title ? `Ch.${num} — ${title}` : `Ch.${num}`)
    : (title ?? 'Chương')

  return (
    <li
      className={cn(
        'group relative bg-surface rounded-md px-2.5 py-2',
        'flex items-center gap-2',
        'transition-colors hover:bg-surface-2',
      )}
    >
      <StateIcon state={state} />

      <div className="flex-1 min-w-0">
        <p className="text-xs text-text font-medium truncate leading-snug">
          {head}
        </p>
        <p className="text-[11px] text-text-subtle truncate leading-snug">
          <SecondaryLine job={job} state={state} />
        </p>
      </div>

      <RowAction
        state={state}
        phase={job.phase}
        onCancel={onCancel}
        onRetry={onRetry}
        onDismiss={onDismiss}
      />
    </li>
  )
}


// Three buckets — that's the whole status vocabulary the user sees.
type RowState = 'running' | 'done' | 'error'

function bucketize(phase: JobPhase): RowState {
  if (phase === 'done')  return 'done'
  if (phase === 'error') return 'error'
  return 'running'      // queued + fetching + packing + uploading + finalizing
}


function StateIcon({ state }: { state: RowState }) {
  if (state === 'done') {
    return (
      <span className="size-5 rounded-full bg-success-bg text-success-text grid place-items-center shrink-0">
        <Check size={11} strokeWidth={3} />
      </span>
    )
  }
  if (state === 'error') {
    return (
      <span className="size-5 rounded-full bg-error-bg text-error-text grid place-items-center shrink-0">
        <AlertTriangle size={10} strokeWidth={2.5} />
      </span>
    )
  }
  return (
    <span className="size-5 rounded-full bg-info-bg text-info-text grid place-items-center shrink-0">
      <Loader2 size={11} className="animate-spin" />
    </span>
  )
}


function SecondaryLine({ job, state }: { job: QueuedJob; state: RowState }) {
  const project = job.job.projectTitle ?? ''

  if (state === 'error') {
    return (
      <span className="text-error-text/90" title={job.error ?? ''}>
        {job.error ?? 'Lỗi không rõ'}
      </span>
    )
  }
  if (state === 'done') {
    const ts = job.finishedAt ?? job.enqueuedAt
    return (
      <>
        <span title={new Date(ts).toLocaleString()}>
          {fmtRelative(Date.now() - ts)}
        </span>
        {project && <> · <span title={project}>{project}</span></>}
      </>
    )
  }
  return (
    <>
      <span>{liveLabel(job.phase)}</span>
      {project && <> · <span title={project}>{project}</span></>}
    </>
  )
}


function liveLabel(phase: JobPhase): string {
  if (phase === 'queued')   return 'Trong hàng đợi'
  if (phase === 'fetching') return 'Đang tải ảnh'
  return 'Đang upload'      // packing / uploading / finalizing all read the same to the user
}


// One affordance per row, picked by state. Errors get a primary
// "Thử lại"; running shows nothing (cancel is rare and lives in
// the kebab-less past — users wait or close the popup); done shows
// a tiny X to dismiss before the 24h eviction.
function RowAction({
  state, phase, onCancel, onRetry, onDismiss,
}: {
  state:   RowState
  phase:   JobPhase
  onCancel:  () => void
  onRetry:   () => void
  onDismiss: () => void
}) {
  if (state === 'error') {
    return (
      <button
        type="button"
        onClick={onRetry}
        className="text-[11px] font-medium text-accent-text hover:brightness-110 px-1.5 h-6 rounded-sm cursor-pointer shrink-0"
      >
        Thử lại
      </button>
    )
  }
  if (state === 'done') {
    return (
      <button
        type="button"
        onClick={onDismiss}
        title="Xoá khỏi danh sách"
        aria-label="Xoá"
        className="size-5 rounded-sm text-text-subtle hover:text-text hover:bg-hover grid place-items-center cursor-pointer opacity-0 group-hover:opacity-100 focus:opacity-100 transition-opacity shrink-0"
      >
        <X size={11} />
      </button>
    )
  }
  // Cancel only makes sense before the worker grabs the job.
  if (phase === 'queued') {
    return (
      <button
        type="button"
        onClick={onCancel}
        className="text-[11px] text-text-subtle hover:text-text-muted px-1.5 h-6 rounded-sm cursor-pointer shrink-0 opacity-0 group-hover:opacity-100 focus:opacity-100 transition-opacity"
      >
        Huỷ
      </button>
    )
  }
  return null
}


// ── Formatters ─────────────────────────────────────────────────────


function fmtRelative(deltaMs: number): string {
  if (deltaMs < 60_000)         return 'vừa xong'
  const m = Math.floor(deltaMs / 60_000)
  if (m < 60)                   return `${m} phút trước`
  const h = Math.floor(m / 60)
  if (h < 24)                   return `${h} giờ trước`
  const d = Math.floor(h / 24)
  return `${d} ngày trước`
}
