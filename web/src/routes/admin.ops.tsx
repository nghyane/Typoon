// =============================================================================
// /admin/ops — Pipeline ops dashboard.
//
// Single scrollable page, three sections stacked top-to-bottom so an on-call
// admin glances at one URL during an incident:
//
//   ┌─────────────────────────────────────────────────────────────────┐
//   │  Stages  — 4 cards. Paused stages call out reason + Resume btn. │
//   │  Tasks   — filterable queue snapshot. Per-row Requeue/Release.  │
//   │  Audit   — reverse-chronological mutations + expandable diff.   │
//   └─────────────────────────────────────────────────────────────────┘
//
// State conventions:
//   • Polls every 4s (`refetchInterval`) so the dashboard self-heals while
//     workers churn the queue — same cadence as WorkersIndicator.
//   • Mutations carry the snapshot's (attempts, claimed_by) for
//     optimistic concurrency; a 409 from the server surfaces as a toast
//     "Trạng thái đã thay đổi" and re-fetches both Tasks + Audit.
//   • Idempotency-Key minted per click via crypto.randomUUID() — a
//     network retry produces one audit row, not two.
// =============================================================================

import { createFileRoute, useNavigate } from '@tanstack/react-router'
import { useMutation, useQuery, useQueryClient } from '@tanstack/react-query'
import { useEffect, useMemo, useState } from 'react'
import {
  AlertTriangle, ChevronDown, ChevronRight, Hourglass, Loader2,
  PauseCircle, PlayCircle, RotateCcw, Skull, Unlock, History, Layers,
} from 'lucide-react'
import {
  api,
  OpsConflictError,
  type ApiAdminAction, type ApiPausedStage, type ApiTask,
  type AdminActionKind, type PipelineStage, type TaskState, type TaskTargetKind,
} from '@shared/api/api'
import { qk } from '@shared/api/keys'
import { useSession } from '@features/auth/session'
import { Button } from '@shared/ui/Button'
import { Modal } from '@shared/ui/Modal'
import { Badge, Tag, Spinner, input as inputCls } from '@shared/ui/primitives'
import { EmptyState } from '@shared/ui/EmptyState'
import { toast } from '@shared/ui/Toaster'
import { DataTable, Th } from '@shared/ui/DataTable'
import { cn } from '@shared/lib/cn'

// ── route shell ────────────────────────────────────────────────────────────

export const Route = createFileRoute('/admin/ops')({
  component: AdminOpsPage,
  // Opt into the dedicated admin workspace shell — no manga sidebar,
  // no global search, no pipeline pill duplicating <StagesSection>.
  staticData: { chrome: 'admin' },
})

const STAGES: PipelineStage[] = ['prepare', 'scan', 'translate', 'render']
const STAGE_LABEL: Record<PipelineStage, string> = {
  prepare:   'Chuẩn bị',
  scan:      'Quét bong bóng',
  translate: 'Dịch',
  render:    'Render',
}
const STATE_LABEL: Record<TaskState, string> = {
  pending: 'Chờ', running: 'Đang chạy', stale: 'Treo',
  blocked: 'Tạm ngưng', failed: 'Lỗi',
}
const ACTION_LABEL: Record<AdminActionKind, string> = {
  'stage.pause':     'Tạm ngưng stage',
  'stage.resume':    'Mở lại stage',
  'task.requeue':    'Đưa lại vào queue',
  'task.release':    'Giải phóng claim',
  'task.force_fail': 'Buộc dead-letter',
}

const POLL_MS = 4000

function AdminOpsPage() {
  const { user, status } = useSession()
  const navigate = useNavigate()

  // RBAC gate at the leaf — server enforces too (403 from every endpoint),
  // but a non-admin user landing here would just see a wall of errors.
  // Redirect to /library so they get back to known territory.
  useEffect(() => {
    if (status === 'authenticated' && !user?.is_admin) {
      void navigate({ to: '/library', replace: true })
    }
  }, [status, user?.is_admin, navigate])

  if (status === 'loading' || !user?.is_admin) {
    return (
      <div className="min-h-[60vh] flex items-center justify-center">
        <Spinner />
      </div>
    )
  }

  return (
    <div className="mx-auto max-w-6xl px-4 py-6 space-y-8">
      {/* Page title sits in <AdminTopBar>. The body owns sections only. */}
      <p className="text-sm text-text-subtle">
        Theo dõi và can thiệp khi worker dừng. Mọi hành động được ghi
        vào audit log với <code>reason</code> bắt buộc.
      </p>

      <StagesSection />
      <TasksSection />
      <AuditSection />
    </div>
  )
}

// ── shared modal: action + reason ──────────────────────────────────────────

interface ActionModalProps {
  open:        boolean
  onClose:     () => void
  title:       string
  description: string
  confirmLabel:string
  destructive?: boolean
  /** Returns a promise so the modal can show pending state + dismiss
   *  on success / keep open on error. */
  onConfirm:   (reason: string) => Promise<void>
}

function ActionModal({
  open, onClose, title, description, confirmLabel,
  destructive, onConfirm,
}: ActionModalProps) {
  const [reason, setReason] = useState('')
  const [pending, setPending] = useState(false)

  useEffect(() => {
    if (open) {
      setReason('')
      setPending(false)
    }
  }, [open])

  const submit = async () => {
    if (reason.trim().length < 3 || pending) return
    setPending(true)
    try {
      await onConfirm(reason.trim())
      onClose()
    } catch {
      // Error toast is raised by the mutation hook. Keep modal open so
      // the operator can adjust reason / re-read state.
    } finally {
      setPending(false)
    }
  }

  return (
    <Modal
      open={open}
      onClose={pending ? () => {} : onClose}
      title={title}
      size="sm"
      footer={
        <>
          <Button variant="ghost" onClick={onClose} disabled={pending}>
            Huỷ
          </Button>
          <Button
            variant={destructive ? 'danger' : 'primary'}
            onClick={submit}
            disabled={reason.trim().length < 3 || pending}
          >
            {pending ? <Spinner size={12} /> : null}
            {confirmLabel}
          </Button>
        </>
      }
    >
      <div className="p-5 space-y-3">
        <p className="text-sm text-text-muted">{description}</p>
        <div>
          <label className="block text-xs font-medium text-text-subtle mb-1.5">
            Lý do (bắt buộc, sẽ vào audit log)
          </label>
          <textarea
            className={cn(inputCls, 'h-20 resize-none')}
            value={reason}
            onChange={(e) => setReason(e.target.value)}
            placeholder="vd: Packy đã đổi group, model đã có distributor"
            autoFocus
            disabled={pending}
            maxLength={500}
          />
          <p className="mt-1 text-[11px] text-text-subtle">
            {reason.trim().length}/500 — tối thiểu 3 ký tự
          </p>
        </div>
      </div>
    </Modal>
  )
}

// ── Stages section ─────────────────────────────────────────────────────────

function StagesSection() {
  const qc = useQueryClient()
  const stagesQ = useQuery({
    queryKey: qk.adminOps.stages(),
    queryFn:  () => api.adminOps.listStages(),
    refetchInterval: POLL_MS,
  })
  const queueQ = useQuery({
    queryKey: qk.workers(),
    queryFn:  api.workers,
    refetchInterval: POLL_MS,
  })

  const [pauseTarget, setPauseTarget]   = useState<PipelineStage | null>(null)
  const [resumeTarget, setResumeTarget] = useState<ApiPausedStage | null>(null)

  const pausedMap = useMemo(() => {
    const m = new Map<string, ApiPausedStage>()
    for (const s of stagesQ.data ?? []) m.set(s.stage, s)
    return m
  }, [stagesQ.data])

  const pauseMut = useMutation({
    mutationFn: (v: { stage: PipelineStage; reason: string }) =>
      api.adminOps.pauseStage(v.stage, v.reason, { idemKey: crypto.randomUUID() }),
    onSuccess: () => {
      void qc.invalidateQueries({ queryKey: ['admin-ops'] })
      void qc.invalidateQueries({ queryKey: qk.workers() })
      toast.success('Đã tạm ngưng stage')
    },
    onError: (e) => toast.error(messageOf(e, 'Không tạm ngưng được')),
  })
  const resumeMut = useMutation({
    mutationFn: (v: { stage: PipelineStage; reason: string }) =>
      api.adminOps.resumeStage(v.stage, v.reason, { idemKey: crypto.randomUUID() }),
    onSuccess: () => {
      void qc.invalidateQueries({ queryKey: ['admin-ops'] })
      void qc.invalidateQueries({ queryKey: qk.workers() })
      toast.success('Đã mở lại stage')
    },
    onError: (e) => toast.error(messageOf(e, 'Không mở lại được')),
  })

  return (
    <section className="space-y-3">
      <SectionHeader
        icon={<Layers size={14} />}
        title="Stages"
        hint="4 stage của pipeline. Stage tạm ngưng = worker không claim task mới."
      />

      <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-3">
        {STAGES.map((stage) => {
          const paused  = pausedMap.get(stage)
          const counts  = queueQ.data?.stages[stage]
          const total   = counts
            ? counts.pending + counts.running + counts.stale + counts.blocked + counts.failed
            : 0
          return (
            <div
              key={stage}
              className={cn(
                'rounded-md bg-surface p-4 space-y-3',
                paused && 'ring-1 ring-amber-500/40',
              )}
            >
              <div className="flex items-center justify-between">
                <div>
                  <p className="text-sm font-semibold text-text">{STAGE_LABEL[stage]}</p>
                  <p className="text-[11px] text-text-subtle font-mono">{stage}</p>
                </div>
                {paused ? (
                  <Badge tone="warning">Tạm ngưng</Badge>
                ) : counts && counts.running > 0 ? (
                  <Badge tone="info">Đang chạy</Badge>
                ) : total === 0 ? (
                  <Badge tone="neutral">Trống</Badge>
                ) : (
                  <Badge tone="success">Khoẻ</Badge>
                )}
              </div>

              {counts && total > 0 && (
                <div className="flex flex-wrap gap-1.5">
                  {counts.running > 0 && (
                    <Tag tone="info"><Loader2 size={10} className="animate-spin" />{counts.running}</Tag>
                  )}
                  {counts.pending > 0 && (
                    <Tag tone="warning"><Hourglass size={10} />{counts.pending}</Tag>
                  )}
                  {counts.blocked > 0 && (
                    <Tag tone="warning"><PauseCircle size={10} />{counts.blocked}</Tag>
                  )}
                  {counts.stale > 0 && (
                    <Tag tone="error"><AlertTriangle size={10} />{counts.stale}</Tag>
                  )}
                  {counts.failed > 0 && (
                    <Tag tone="error"><Skull size={10} />{counts.failed}</Tag>
                  )}
                </div>
              )}

              {paused ? (
                <div className="space-y-2 pt-1 border-t border-border-soft">
                  <p className="text-xs text-amber-300 font-medium">Lý do</p>
                  <pre className="text-[11px] leading-snug text-text-muted whitespace-pre-wrap break-words font-mono max-h-32 overflow-auto">
                    {paused.reason}
                  </pre>
                  <p className="text-[11px] text-text-subtle">
                    bởi <code>{paused.paused_by ?? 'unknown'}</code>
                  </p>
                  <Button
                    size="sm"
                    onClick={() => setResumeTarget(paused)}
                    className="w-full"
                  >
                    <PlayCircle size={12} />
                    Mở lại stage
                  </Button>
                </div>
              ) : (
                <Button
                  variant="ghost"
                  size="sm"
                  onClick={() => setPauseTarget(stage)}
                  className="w-full"
                >
                  <PauseCircle size={12} />
                  Tạm ngưng
                </Button>
              )}
            </div>
          )
        })}
      </div>

      <ActionModal
        open={pauseTarget !== null}
        onClose={() => setPauseTarget(null)}
        title={`Tạm ngưng ${pauseTarget ? STAGE_LABEL[pauseTarget] : ''}`}
        description="Worker sẽ ngừng claim task mới trên stage này. Task đang chạy vẫn hoàn thành."
        confirmLabel="Tạm ngưng"
        destructive
        onConfirm={async (reason) => {
          if (!pauseTarget) return
          await pauseMut.mutateAsync({ stage: pauseTarget, reason })
        }}
      />
      <ActionModal
        open={resumeTarget !== null}
        onClose={() => setResumeTarget(null)}
        title={`Mở lại ${resumeTarget ? STAGE_LABEL[resumeTarget.stage] : ''}`}
        description="Worker sẽ wake up qua NOTIFY và bắt đầu claim lại ngay. Task đang ở attempts=0 tiếp tục từ đầu."
        confirmLabel="Mở lại"
        onConfirm={async (reason) => {
          if (!resumeTarget) return
          await resumeMut.mutateAsync({ stage: resumeTarget.stage, reason })
        }}
      />
    </section>
  )
}

// ── Tasks section ──────────────────────────────────────────────────────────

function TasksSection() {
  const qc = useQueryClient()
  const [filters, setFilters] = useState<{
    stage:       PipelineStage    | ''
    state:       TaskState        | ''
    target_kind: TaskTargetKind   | ''
  }>({ stage: '', state: '', target_kind: '' })

  const tasksQ = useQuery({
    queryKey: qk.adminOps.tasks(filters),
    queryFn:  () => api.adminOps.listTasks({
      stage:       filters.stage       || undefined,
      state:       filters.state       || undefined,
      target_kind: filters.target_kind || undefined,
      limit: 100,
    }),
    refetchInterval: POLL_MS,
  })

  const [mutTarget, setMutTarget] = useState<
    { kind: 'requeue' | 'release' | 'fail'; task: ApiTask } | null
  >(null)

  const requeueMut = useMutation({
    mutationFn: (v: { task: ApiTask; reason: string }) =>
      api.adminOps.requeueTask(
        { stage: v.task.stage, target_kind: v.task.target_kind, target_id: v.task.target_id },
        {
          reason: v.reason,
          expected_attempts:   v.task.attempts,
          expected_claimed_by: v.task.claimed_by,
        },
        { idemKey: crypto.randomUUID() },
      ),
    onSuccess: () => {
      void qc.invalidateQueries({ queryKey: ['admin-ops'] })
      void qc.invalidateQueries({ queryKey: qk.workers() })
      toast.success('Đã đưa task lại vào queue')
    },
    onError: (e) => handleOpsError(e, qc),
  })

  const releaseMut = useMutation({
    mutationFn: (v: { task: ApiTask; reason: string }) =>
      api.adminOps.releaseTask(
        { stage: v.task.stage, target_kind: v.task.target_kind, target_id: v.task.target_id },
        { reason: v.reason, expected_claimed_by: v.task.claimed_by ?? '' },
        { idemKey: crypto.randomUUID() },
      ),
    onSuccess: () => {
      void qc.invalidateQueries({ queryKey: ['admin-ops'] })
      void qc.invalidateQueries({ queryKey: qk.workers() })
      toast.success('Đã giải phóng claim')
    },
    onError: (e) => handleOpsError(e, qc),
  })

  const failMut = useMutation({
    mutationFn: (v: { task: ApiTask; reason: string }) =>
      api.adminOps.forceFailTask(
        { stage: v.task.stage, target_kind: v.task.target_kind, target_id: v.task.target_id },
        {
          reason: v.reason,
          expected_attempts:   v.task.attempts,
          expected_claimed_by: v.task.claimed_by,
        },
        { idemKey: crypto.randomUUID() },
      ),
    onSuccess: () => {
      void qc.invalidateQueries({ queryKey: ['admin-ops'] })
      void qc.invalidateQueries({ queryKey: qk.workers() })
      toast.success('Đã buộc dead-letter')
    },
    onError: (e) => handleOpsError(e, qc),
  })

  const items = tasksQ.data?.items ?? []

  return (
    <section className="space-y-3">
      <SectionHeader
        icon={<Loader2 size={14} />}
        title="Hàng đợi"
        hint="Snapshot live của bảng tasks. Lifecycle state đã được derive từ DB."
      />

      <div className="flex flex-wrap gap-2 items-center">
        <FilterSelect
          label="Stage"
          value={filters.stage}
          onChange={(v) => setFilters({ ...filters, stage: v as PipelineStage | '' })}
          options={[
            { value: '', label: 'Tất cả' },
            ...STAGES.map((s) => ({ value: s, label: STAGE_LABEL[s] })),
          ]}
        />
        <FilterSelect
          label="Trạng thái"
          value={filters.state}
          onChange={(v) => setFilters({ ...filters, state: v as TaskState | '' })}
          options={[
            { value: '', label: 'Tất cả' },
            { value: 'pending', label: 'Chờ' },
            { value: 'running', label: 'Đang chạy' },
            { value: 'stale',   label: 'Treo' },
            { value: 'blocked', label: 'Tạm ngưng' },
            { value: 'failed',  label: 'Lỗi' },
          ]}
        />
        <FilterSelect
          label="Đối tượng"
          value={filters.target_kind}
          onChange={(v) => setFilters({ ...filters, target_kind: v as TaskTargetKind | '' })}
          options={[
            { value: '', label: 'Tất cả' },
            { value: 'chapter',     label: 'Chapter' },
            { value: 'draft',       label: 'Draft' },
            { value: 'translation', label: 'Translation' },
          ]}
        />
        <div className="ml-auto text-xs text-text-subtle">
          {tasksQ.isFetching ? <Spinner size={11} /> : null} {items.length} task
        </div>
      </div>

      {tasksQ.isLoading ? (
        <div className="bg-surface rounded-md py-10 flex items-center justify-center">
          <Spinner />
        </div>
      ) : items.length === 0 ? (
        <EmptyState
          icon={Hourglass}
          title="Hàng đợi trống"
          hint="Không có task nào khớp bộ lọc hiện tại."
        />
      ) : (
        <DataTable>
          <thead className="bg-surface-2">
            <tr>
              <Th>Stage</Th>
              <Th>Target</Th>
              <Th>Trạng thái</Th>
              <Th className="text-right">Attempts</Th>
              <Th>Worker / Tuổi claim</Th>
              <Th>Last error</Th>
              <Th className="text-right">Hành động</Th>
            </tr>
          </thead>
          <tbody className="divide-y divide-border-soft">
            {items.map((t) => (
              <tr key={`${t.stage}-${t.target_kind}-${t.target_id}`} className="hover:bg-hover/40">
                <td className="px-3 py-2 text-sm">{STAGE_LABEL[t.stage]}</td>
                <td className="px-3 py-2 text-sm font-mono text-text-muted">
                  {t.target_kind}/{t.target_id}
                </td>
                <td className="px-3 py-2"><StateBadge state={t.lifecycle_state} /></td>
                <td className="px-3 py-2 text-sm text-right tabular-nums">{t.attempts}</td>
                <td className="px-3 py-2 text-xs text-text-subtle">
                  {t.claimed_by ? (
                    <span className="font-mono">
                      {truncate(t.claimed_by, 28)}
                      {t.claim_age_seconds != null && (
                        <> · {formatAge(t.claim_age_seconds)}</>
                      )}
                    </span>
                  ) : <span className="text-text-subtle">—</span>}
                </td>
                <td className="px-3 py-2 text-xs text-text-muted max-w-md">
                  {t.last_error
                    ? <pre className="font-mono text-[11px] whitespace-pre-wrap break-words line-clamp-2">{t.last_error}</pre>
                    : <span className="text-text-subtle">—</span>}
                </td>
                <td className="px-3 py-2">
                  <div className="flex justify-end gap-1">
                    {/* Requeue: useful when state=failed (reset attempts) */}
                    {t.lifecycle_state === 'failed' && (
                      <IconButton
                        title="Đưa lại queue"
                        onClick={() => setMutTarget({ kind: 'requeue', task: t })}
                      >
                        <RotateCcw size={12} />
                      </IconButton>
                    )}
                    {/* Release: only when there's a claim to release */}
                    {t.claimed_by && (
                      <IconButton
                        title="Giải phóng claim"
                        onClick={() => setMutTarget({ kind: 'release', task: t })}
                      >
                        <Unlock size={12} />
                      </IconButton>
                    )}
                    {/* Fail: any non-failed row can be force-failed */}
                    {t.lifecycle_state !== 'failed' && (
                      <IconButton
                        title="Buộc dead-letter"
                        tone="destructive"
                        onClick={() => setMutTarget({ kind: 'fail', task: t })}
                      >
                        <Skull size={12} />
                      </IconButton>
                    )}
                  </div>
                </td>
              </tr>
            ))}
          </tbody>
        </DataTable>
      )}

      <ActionModal
        open={mutTarget !== null}
        onClose={() => setMutTarget(null)}
        title={mutTarget ? actionTitle(mutTarget.kind, mutTarget.task) : ''}
        description={mutTarget ? actionDescription(mutTarget.kind) : ''}
        confirmLabel={mutTarget ? actionLabel(mutTarget.kind) : ''}
        destructive={mutTarget?.kind === 'fail'}
        onConfirm={async (reason) => {
          if (!mutTarget) return
          const { kind, task } = mutTarget
          if (kind === 'requeue') await requeueMut.mutateAsync({ task, reason })
          if (kind === 'release') await releaseMut.mutateAsync({ task, reason })
          if (kind === 'fail')    await failMut.mutateAsync({ task, reason })
        }}
      />
    </section>
  )
}

function actionTitle(kind: 'requeue' | 'release' | 'fail', task: ApiTask) {
  const label = `${STAGE_LABEL[task.stage]} · ${task.target_kind}/${task.target_id}`
  if (kind === 'requeue') return `Đưa lại queue — ${label}`
  if (kind === 'release') return `Giải phóng claim — ${label}`
  return `Buộc dead-letter — ${label}`
}
function actionDescription(kind: 'requeue' | 'release' | 'fail') {
  if (kind === 'requeue') return 'Reset attempts = 0, xoá last_error. Worker sẽ thử lại từ đầu.'
  if (kind === 'release') return 'Xoá claim hiện tại (worker chết, không phản hồi). Attempts giữ nguyên.'
  return 'Đánh dấu dead-letter (attempts = MAX). Task không tự retry; admin có thể requeue sau.'
}
function actionLabel(kind: 'requeue' | 'release' | 'fail') {
  if (kind === 'requeue') return 'Đưa lại queue'
  if (kind === 'release') return 'Giải phóng'
  return 'Buộc dead-letter'
}

// ── Audit section ──────────────────────────────────────────────────────────

function AuditSection() {
  const auditQ = useQuery({
    queryKey: qk.adminOps.actions({ limit: 50 }),
    queryFn:  () => api.adminOps.listActions({ limit: 50 }),
    refetchInterval: POLL_MS,
  })

  const [expanded, setExpanded] = useState<number | null>(null)
  const rows = auditQ.data ?? []

  return (
    <section className="space-y-3">
      <SectionHeader
        icon={<History size={14} />}
        title="Audit log"
        hint="50 hành động gần nhất. Bấm row để xem prev_state."
      />
      {auditQ.isLoading ? (
        <div className="bg-surface rounded-md py-10 flex items-center justify-center">
          <Spinner />
        </div>
      ) : rows.length === 0 ? (
        <EmptyState
          icon={History}
          title="Chưa có hành động nào"
          hint="Mọi pause/resume/requeue/fail sẽ xuất hiện ở đây với prev_state để forensic."
        />
      ) : (
        <div className="bg-surface rounded-md divide-y divide-border-soft">
          {rows.map((a) => (
            <AuditRow
              key={a.id}
              action={a}
              open={expanded === a.id}
              onToggle={() => setExpanded(expanded === a.id ? null : a.id)}
            />
          ))}
        </div>
      )}
    </section>
  )
}

function AuditRow({
  action: a, open, onToggle,
}: {
  action: ApiAdminAction
  open: boolean
  onToggle: () => void
}) {
  const target = a.target_ref.target_id != null
    ? `${a.target_ref.stage}/${a.target_ref.target_kind}/${a.target_ref.target_id}`
    : a.target_ref.stage
  return (
    <div>
      <button
        onClick={onToggle}
        className="w-full text-left px-3.5 py-2.5 hover:bg-hover/40 transition-colors flex items-start gap-3"
      >
        {open
          ? <ChevronDown size={13} className="text-text-subtle mt-0.5 flex-none" />
          : <ChevronRight size={13} className="text-text-subtle mt-0.5 flex-none" />}
        <div className="flex-1 min-w-0">
          <div className="flex items-center gap-2 flex-wrap">
            <Tag tone={a.action.startsWith('stage') ? 'warning' : 'info'} size="sm">
              {ACTION_LABEL[a.action]}
            </Tag>
            <span className="font-mono text-xs text-text-muted">{target}</span>
            <span className="text-xs text-text-subtle ml-auto">{formatTime(a.at)}</span>
          </div>
          <p className="mt-1 text-sm text-text-muted truncate">{a.reason}</p>
          <p className="mt-0.5 text-[11px] text-text-subtle font-mono">
            {a.target_ref.source ?? (a.actor_id ? `user:${a.actor_id}` : 'unknown')}
          </p>
        </div>
      </button>
      {open && a.prev_state && (
        <div className="px-10 pb-3 -mt-1">
          <p className="text-[11px] uppercase tracking-wider text-text-subtle mb-1.5">
            Prev state
          </p>
          <pre className="bg-bg/60 rounded p-2.5 text-[11px] font-mono text-text-muted whitespace-pre-wrap break-words">
            {JSON.stringify(a.prev_state, null, 2)}
          </pre>
        </div>
      )}
    </div>
  )
}

// ── Tiny shared bits ───────────────────────────────────────────────────────

function SectionHeader({
  icon, title, hint,
}: { icon: React.ReactNode; title: string; hint: string }) {
  return (
    <div>
      <h2 className="text-sm font-semibold tracking-tight inline-flex items-center gap-2">
        <span className="text-text-subtle">{icon}</span>
        {title}
      </h2>
      <p className="text-xs text-text-subtle mt-0.5">{hint}</p>
    </div>
  )
}

function StateBadge({ state }: { state: TaskState }) {
  const tone =
    state === 'failed'  ? 'error'   :
    state === 'stale'   ? 'error'   :
    state === 'blocked' ? 'warning' :
    state === 'running' ? 'info'    :
                          'neutral'
  return <Badge tone={tone}>{STATE_LABEL[state]}</Badge>
}

function FilterSelect<T extends string>({
  label, value, onChange, options,
}: {
  label:    string
  value:    T
  onChange: (v: T) => void
  options:  Array<{ value: T; label: string }>
}) {
  return (
    <label className="text-xs text-text-subtle inline-flex items-center gap-1.5">
      {label}
      <select
        value={value}
        onChange={(e) => onChange(e.target.value as T)}
        className="h-7 px-2 rounded-sm bg-surface text-text border border-border-soft text-xs"
      >
        {options.map((o) => (
          <option key={o.value} value={o.value}>{o.label}</option>
        ))}
      </select>
    </label>
  )
}

function IconButton({
  children, onClick, title, tone,
}: {
  children: React.ReactNode
  onClick:  () => void
  title:    string
  tone?:    'destructive'
}) {
  return (
    <button
      onClick={onClick}
      title={title}
      className={cn(
        'inline-flex items-center justify-center size-7 rounded-sm transition-colors',
        'hover:bg-hover text-text-muted',
        tone === 'destructive' && 'hover:text-rose-300',
      )}
    >
      {children}
    </button>
  )
}

// ── Helpers ────────────────────────────────────────────────────────────────

function truncate(s: string, n: number) {
  return s.length > n ? s.slice(0, n - 1) + '…' : s
}

function formatAge(sec: number) {
  if (sec < 60)     return `${sec}s`
  if (sec < 3600)   return `${Math.floor(sec / 60)}m`
  if (sec < 86400)  return `${Math.floor(sec / 3600)}h`
  return `${Math.floor(sec / 86400)}d`
}

function formatTime(iso: string) {
  const d = new Date(iso)
  const now = new Date()
  const sameDay = d.toDateString() === now.toDateString()
  if (sameDay) return d.toLocaleTimeString('vi-VN', { hour: '2-digit', minute: '2-digit', second: '2-digit' })
  return d.toLocaleString('vi-VN', { day: '2-digit', month: '2-digit', hour: '2-digit', minute: '2-digit' })
}

function messageOf(e: unknown, fallback: string): string {
  if (e instanceof OpsConflictError) return 'Trạng thái đã thay đổi — đã tải lại'
  if (e instanceof Error) return e.message
  return fallback
}

function handleOpsError(e: unknown, qc: ReturnType<typeof useQueryClient>) {
  if (e instanceof OpsConflictError) {
    toast.info('Trạng thái đã thay đổi — đã tải lại')
    void qc.invalidateQueries({ queryKey: ['admin-ops'] })
    return
  }
  toast.error(e instanceof Error ? e.message : 'Mutation thất bại')
}
