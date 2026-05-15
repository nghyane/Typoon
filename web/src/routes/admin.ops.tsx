// =============================================================================
// /admin/ops — Pipeline ops cockpit.
//
// Layout (≥ lg):
//
//   ┌─────────────────────────────────────────────────────────────────┐
//   │  Summary band — 1 dòng: pipeline · queue · workers · stuck     │
//   ├──────────────────┬──────────────────────────────────────────────┤
//   │  Stages (sticky) │  Hàng đợi (DataTable)                       │
//   │  4 stage rows    │  filter + table chiếm phần lớn không gian   │
//   │                  │                                              │
//   ├──────────────────┴──────────────────────────────────────────────┤
//   │  Lịch sử thao tác — timeline expand inline                      │
//   └─────────────────────────────────────────────────────────────────┘
//
// State conventions:
//   • Polls every 4s (`refetchInterval`) — same cadence as WorkersIndicator.
//   • Mutations carry (attempts, claimed_by) snapshot for optimistic
//     concurrency; 409 → toast "Trạng thái đã thay đổi" + refetch.
//   • Idempotency-Key minted per click via crypto.randomUUID().
// =============================================================================

import { createFileRoute, useNavigate } from '@tanstack/react-router'
import { useMutation, useQuery, useQueryClient } from '@tanstack/react-query'
import { useEffect, useMemo, useState } from 'react'
import {
  AlertTriangle, ChevronDown, ChevronRight, Hourglass, Loader2,
  PauseCircle, PlayCircle, RotateCcw, Skull, Unlock, History, Layers,
  Activity, Users, Inbox,
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
import { Badge, Tag, Spinner, input as inputCls, type BadgeTone } from '@shared/ui/primitives'
import { EmptyState } from '@shared/ui/EmptyState'
import { toast } from '@shared/ui/Toaster'
import { DataTable, Th } from '@shared/ui/DataTable'
import { cn } from '@shared/lib/cn'
import { languageName, languageCode } from '@shared/lib/lang'

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
  scan:      'Quét bóng thoại',
  translate: 'Dịch',
  render:    'Render',
}

// Task lifecycle states. Wording is deliberately distinct from stage
// pause: a task that's `blocked` is waiting because its stage is
// paused — the *stage* is paused, the *task* is just blocked by that.
const STATE_LABEL: Record<TaskState, string> = {
  pending: 'Đang chờ',
  running: 'Đang chạy',
  stale:   'Quá hạn',
  blocked: 'Đang chặn',
  failed:  'Thất bại',
}

const STATE_TONE: Record<TaskState, BadgeTone> = {
  pending: 'neutral',
  running: 'info',
  stale:   'error',
  blocked: 'warning',
  failed:  'error',
}

const TARGET_LABEL: Record<TaskTargetKind, string> = {
  chapter:     'Chương',
  draft:       'Bản nháp',
  translation: 'Bản dịch',
}

const ACTION_LABEL: Record<AdminActionKind, string> = {
  'stage.pause':     'Tạm dừng stage',
  'stage.resume':    'Mở lại stage',
  'task.requeue':    'Đưa lại hàng đợi',
  'task.release':    'Gỡ claim',
  'task.force_fail': 'Đẩy dead-letter',
  'draft.restart':   'Khởi động lại bản nháp',
  'draft.takedown':  'Gỡ bản nháp',
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
    <div className="mx-auto max-w-[1400px] px-4 sm:px-6 py-5 space-y-5">
      <SummaryBand />

      <div className="grid grid-cols-1 lg:grid-cols-[320px_minmax(0,1fr)] gap-5">
        <div className="lg:sticky lg:top-4 lg:self-start space-y-5">
          <StagesPanel />
        </div>
        <TasksPanel />
      </div>

      <AuditPanel />
    </div>
  )
}

// ── Summary band ───────────────────────────────────────────────────────────

function SummaryBand() {
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

  const totals = useMemo(() => {
    const z = { pending: 0, running: 0, stale: 0, blocked: 0, failed: 0 }
    for (const stage of Object.values(queueQ.data?.stages ?? {})) {
      z.pending += stage.pending
      z.running += stage.running
      z.stale   += stage.stale
      z.blocked += stage.blocked
      z.failed  += stage.failed
    }
    return z
  }, [queueQ.data])

  const pausedCount = stagesQ.data?.length ?? 0
  const workersCount = queueQ.data?.active_workers.length ?? 0
  const stuck = totals.stale + totals.failed

  // Overall pipeline health — derived from worst stage status.
  const health: BadgeTone =
    pausedCount > 0 || stuck > 0 ? 'warning' :
    totals.running > 0           ? 'info'    :
    workersCount === 0           ? 'neutral' :
                                   'success'

  const healthLabel =
    pausedCount > 0 ? `${pausedCount} stage tạm dừng` :
    stuck > 0       ? `${stuck} task cần xử lý`       :
    totals.running > 0 ? 'Đang chạy'                  :
    workersCount === 0 ? 'Không có worker'            :
                         'Bình thường'

  return (
    <div className="bg-surface rounded-md p-4 grid grid-cols-2 sm:grid-cols-4 gap-4">
      <SummaryCell
        icon={<Activity size={14} />}
        label="Pipeline"
        value={<Badge tone={health}>{healthLabel}</Badge>}
      />
      <SummaryCell
        icon={<Inbox size={14} />}
        label="Hàng đợi"
        value={
          <span className="text-lg font-semibold tabular-nums">
            {totals.pending + totals.running + totals.blocked}
          </span>
        }
        sub={`${totals.running} đang chạy · ${totals.pending} chờ`}
      />
      <SummaryCell
        icon={<Users size={14} />}
        label="Worker hoạt động"
        value={
          <span className="text-lg font-semibold tabular-nums">
            {workersCount}
          </span>
        }
        sub={workersCount === 0 ? 'Không có worker nào kết nối' : 'Đang nhận task'}
      />
      <SummaryCell
        icon={<AlertTriangle size={14} />}
        label="Cần can thiệp"
        value={
          <span className={cn('text-lg font-semibold tabular-nums', stuck > 0 && 'text-error-text')}>
            {stuck}
          </span>
        }
        sub={
          stuck > 0
            ? `${totals.stale} quá hạn · ${totals.failed} thất bại`
            : 'Không có task lỗi'
        }
      />
    </div>
  )
}

function SummaryCell({
  icon, label, value, sub,
}: {
  icon:  React.ReactNode
  label: string
  value: React.ReactNode
  sub?:  string
}) {
  return (
    <div>
      <p className="text-[11px] uppercase tracking-wider text-text-subtle font-medium inline-flex items-center gap-1.5">
        <span className="text-text-subtle">{icon}</span>
        {label}
      </p>
      <div className="mt-2 leading-none">{value}</div>
      {sub && <p className="mt-1.5 text-[11px] text-text-subtle">{sub}</p>}
    </div>
  )
}

// ── Stages panel ───────────────────────────────────────────────────────────

function StagesPanel() {
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
      toast.success('Đã tạm dừng stage')
    },
    onError: (e) => toast.error(messageOf(e, 'Không tạm dừng được')),
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
    <section className="bg-surface rounded-md overflow-hidden">
      <PanelHeader
        icon={<Layers size={13} />}
        title="Stages"
        hint="Tạm dừng = worker không nhận task mới ở stage đó. Task đang chạy vẫn hoàn thành."
      />

      <div className="divide-y divide-border-soft">
        {STAGES.map((stage) => (
          <StageRow
            key={stage}
            stage={stage}
            paused={pausedMap.get(stage) ?? null}
            counts={queueQ.data?.stages[stage]}
            onPause={() => setPauseTarget(stage)}
            onResume={(p) => setResumeTarget(p)}
          />
        ))}
      </div>

      <ActionModal
        open={pauseTarget !== null}
        onClose={() => setPauseTarget(null)}
        title={`Tạm dừng ${pauseTarget ? STAGE_LABEL[pauseTarget] : ''}`}
        description="Worker sẽ ngừng nhận task mới trên stage này. Task đang chạy vẫn hoàn thành."
        confirmLabel="Tạm dừng"
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
        description="Worker sẽ wake up qua NOTIFY và bắt đầu nhận task ngay. Task đang ở attempts=0 tiếp tục từ đầu."
        confirmLabel="Mở lại"
        onConfirm={async (reason) => {
          if (!resumeTarget) return
          await resumeMut.mutateAsync({ stage: resumeTarget.stage, reason })
        }}
      />
    </section>
  )
}

function StageRow({
  stage, paused, counts, onPause, onResume,
}: {
  stage:    PipelineStage
  paused:   ApiPausedStage | null
  counts?:  { pending: number; running: number; stale: number; blocked: number; failed: number }
  onPause:  () => void
  onResume: (p: ApiPausedStage) => void
}) {
  const total = counts
    ? counts.pending + counts.running + counts.stale + counts.blocked + counts.failed
    : 0

  const tone: BadgeTone =
    paused                              ? 'warning' :
    counts && counts.stale + counts.failed > 0 ? 'error'   :
    counts && counts.running > 0        ? 'info'    :
    total === 0                         ? 'neutral' :
                                          'success'

  return (
    <div className={cn('px-3.5 py-3', paused && 'bg-warning-bg/10')}>
      <div className="flex items-center gap-2.5">
        <span className={cn(
          'size-2 rounded-full flex-none',
          tone === 'success' && 'bg-success',
          tone === 'info'    && 'bg-info',
          tone === 'warning' && 'bg-warning',
          tone === 'error'   && 'bg-error',
          tone === 'neutral' && 'bg-text-subtle',
        )} />
        <div className="flex-1 min-w-0">
          <p className="text-sm font-medium text-text">{STAGE_LABEL[stage]}</p>
          <p className="text-[10px] text-text-subtle font-mono leading-tight">{stage}</p>
        </div>
        {paused ? (
          <Button size="sm" onClick={() => onResume(paused)}>
            <PlayCircle size={12} />
            Mở lại
          </Button>
        ) : (
          <button
            onClick={onPause}
            title="Tạm dừng stage"
            className="inline-flex items-center justify-center size-7 rounded-sm hover:bg-hover text-text-subtle hover:text-text transition-colors cursor-pointer"
          >
            <PauseCircle size={13} />
          </button>
        )}
      </div>

      {counts && total > 0 && (
        <div className="mt-2 flex flex-wrap gap-1">
          {counts.running > 0 && (
            <Tag size="sm" tone="info"><Loader2 size={9} className="animate-spin" />{counts.running}</Tag>
          )}
          {counts.pending > 0 && (
            <Tag size="sm" tone="neutral"><Hourglass size={9} />{counts.pending}</Tag>
          )}
          {counts.blocked > 0 && (
            <Tag size="sm" tone="warning"><PauseCircle size={9} />{counts.blocked}</Tag>
          )}
          {counts.stale > 0 && (
            <Tag size="sm" tone="error"><AlertTriangle size={9} />{counts.stale}</Tag>
          )}
          {counts.failed > 0 && (
            <Tag size="sm" tone="error"><Skull size={9} />{counts.failed}</Tag>
          )}
        </div>
      )}

      {paused && (
        <div className="mt-2.5 rounded-sm bg-bg/40 border border-warning/20 p-2">
          <p className="text-[10px] uppercase tracking-wider text-warning-text font-medium">
            Lý do tạm dừng
          </p>
          <pre className="mt-1 text-[11px] leading-snug text-text-muted whitespace-pre-wrap break-words font-mono max-h-24 overflow-auto">
            {paused.reason}
          </pre>
          <p className="mt-1.5 text-[10px] text-text-subtle">
            bởi <code className="font-mono">{paused.paused_by ?? 'unknown'}</code>
          </p>
        </div>
      )}
    </div>
  )
}

// ── Tasks panel ────────────────────────────────────────────────────────────

function TasksPanel() {
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
      toast.success('Đã đưa lại hàng đợi')
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
      toast.success('Đã gỡ claim')
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
      toast.success('Đã đẩy dead-letter')
    },
    onError: (e) => handleOpsError(e, qc),
  })

  const items = tasksQ.data?.items ?? []
  const hasFilter = !!(filters.stage || filters.state || filters.target_kind)
  const [expanded, setExpanded] = useState<string | null>(null)

  return (
    <section className="space-y-3 min-w-0">
      <div className="flex items-end justify-between gap-3 flex-wrap">
        <div>
          <h2 className="text-sm font-semibold tracking-tight inline-flex items-center gap-2">
            <Inbox size={13} className="text-text-subtle" />
            Hàng đợi
            <span className="text-text-subtle font-normal tabular-nums">· {items.length}</span>
          </h2>
          <p className="text-xs text-text-subtle mt-0.5">
            Ảnh chụp bảng tasks theo thời gian thực. Trạng thái lấy từ DB.
          </p>
        </div>
        <div className="text-xs text-text-subtle inline-flex items-center gap-2">
          {tasksQ.isFetching && <Spinner size={11} />}
          {hasFilter && (
            <button
              onClick={() => setFilters({ stage: '', state: '', target_kind: '' })}
              className="hover:text-text transition-colors cursor-pointer"
            >
              Xoá bộ lọc
            </button>
          )}
        </div>
      </div>

      <div className="bg-surface rounded-md p-3 flex flex-wrap gap-2">
        <FilterSelect
          label="Stage"
          value={filters.stage}
          onChange={(v) => setFilters({ ...filters, stage: v as PipelineStage | '' })}
          options={[
            { value: '', label: 'Tất cả stage' },
            ...STAGES.map((s) => ({ value: s, label: STAGE_LABEL[s] })),
          ]}
        />
        <FilterSelect
          label="Trạng thái"
          value={filters.state}
          onChange={(v) => setFilters({ ...filters, state: v as TaskState | '' })}
          options={[
            { value: '', label: 'Tất cả trạng thái' },
            ...(Object.entries(STATE_LABEL) as Array<[TaskState, string]>)
              .map(([v, label]) => ({ value: v, label })),
          ]}
        />
        <FilterSelect
          label="Loại mục tiêu"
          value={filters.target_kind}
          onChange={(v) => setFilters({ ...filters, target_kind: v as TaskTargetKind | '' })}
          options={[
            { value: '', label: 'Tất cả loại' },
            ...(Object.entries(TARGET_LABEL) as Array<[TaskTargetKind, string]>)
              .map(([v, label]) => ({ value: v, label })),
          ]}
        />
      </div>

      {tasksQ.isLoading ? (
        <div className="bg-surface rounded-md py-10 flex items-center justify-center">
          <Spinner />
        </div>
      ) : items.length === 0 ? (
        <EmptyState
          icon={Hourglass}
          title={hasFilter ? 'Không có task khớp bộ lọc' : 'Hàng đợi trống'}
          hint={hasFilter ? 'Thử xoá bớt bộ lọc.' : 'Worker đã xử lý hết — không có gì cần làm.'}
        />
      ) : (
        <DataTable>
          <thead className="bg-surface-2">
            <tr>
              <Th className="w-6"><span className="sr-only">Mở</span></Th>
              <Th>Mục tiêu</Th>
              <Th>Stage</Th>
              <Th>Ngôn ngữ</Th>
              <Th>Trạng thái</Th>
              <Th className="text-right">Lượt thử</Th>
              <Th>Claim</Th>
              <Th className="text-right">Thao tác</Th>
            </tr>
          </thead>
          <tbody className="divide-y divide-border-soft">
            {items.map((t) => {
              const rowKey = `${t.stage}-${t.target_kind}-${t.target_id}`
              return (
                <TaskRow
                  key={rowKey}
                  task={t}
                  open={expanded === rowKey}
                  onToggle={() => setExpanded(expanded === rowKey ? null : rowKey)}
                  onRequeue={() => setMutTarget({ kind: 'requeue', task: t })}
                  onRelease={() => setMutTarget({ kind: 'release', task: t })}
                  onFail={() => setMutTarget({ kind: 'fail', task: t })}
                />
              )
            })}
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

function TaskRow({
  task: t, open, onToggle, onRequeue, onRelease, onFail,
}: {
  task:      ApiTask
  open:      boolean
  onToggle:  () => void
  onRequeue: () => void
  onRelease: () => void
  onFail:    () => void
}) {
  const hasLangPair = !!(t.source_lang && t.target_lang)
  return (
    <>
      <tr
        className={cn(
          'hover:bg-hover/40 align-top cursor-pointer transition-colors',
          open && 'bg-hover/30',
        )}
        onClick={onToggle}
      >
        <td className="px-2 py-2.5 align-middle">
          {open
            ? <ChevronDown size={13} className="text-text-subtle" />
            : <ChevronRight size={13} className="text-text-subtle" />}
        </td>
        <td className="px-3 py-2.5">
          <div className="text-sm text-text leading-tight">
            {TARGET_LABEL[t.target_kind]}
            <span className="text-text-subtle font-mono ml-1">#{t.target_id}</span>
          </div>
          <div className="text-[11px] text-text-subtle mt-0.5 truncate max-w-[260px]">
            {t.chapter_label
              ? <>Chương <span className="text-text-muted">{t.chapter_label}</span></>
              : <span className="font-mono">{t.target_kind}/{t.target_id}</span>}
          </div>
        </td>
        <td className="px-3 py-2.5 text-sm text-text-muted">{STAGE_LABEL[t.stage]}</td>
        <td className="px-3 py-2.5 text-xs">
          {hasLangPair ? (
            <LangPair source={t.source_lang!} target={t.target_lang!} />
          ) : t.source_lang ? (
            <Tag size="sm" tone="outline" uppercase>{languageCode(t.source_lang)}</Tag>
          ) : (
            <span className="text-text-subtle">—</span>
          )}
        </td>
        <td className="px-3 py-2.5">
          <Badge tone={STATE_TONE[t.lifecycle_state]}>{STATE_LABEL[t.lifecycle_state]}</Badge>
        </td>
        <td className="px-3 py-2.5 text-sm text-right tabular-nums">{t.attempts}</td>
        <td className="px-3 py-2.5 text-xs">
          {t.claimed_by ? (
            <div className="font-mono text-text-muted leading-tight">
              <div className="truncate max-w-[180px]">{t.claimed_by}</div>
              {t.claim_age_seconds != null && (
                <div className="text-text-subtle mt-0.5">giữ {formatAge(t.claim_age_seconds)}</div>
              )}
            </div>
          ) : (
            <span className="text-text-subtle">—</span>
          )}
        </td>
        <td className="px-3 py-2.5" onClick={(e) => e.stopPropagation()}>
          <div className="flex justify-end gap-1">
            {t.lifecycle_state === 'failed' && (
              <IconButton title="Đưa lại hàng đợi" onClick={onRequeue}>
                <RotateCcw size={12} />
              </IconButton>
            )}
            {t.claimed_by && (
              <IconButton title="Gỡ claim" onClick={onRelease}>
                <Unlock size={12} />
              </IconButton>
            )}
            {t.lifecycle_state !== 'failed' && (
              <IconButton title="Đẩy dead-letter" tone="destructive" onClick={onFail}>
                <Skull size={12} />
              </IconButton>
            )}
          </div>
        </td>
      </tr>
      {open && <TaskDetailRow task={t} />}
    </>
  )
}

function TaskDetailRow({ task: t }: { task: ApiTask }) {
  return (
    <tr className="bg-bg/40">
      <td />
      <td colSpan={7} className="px-3 py-3">
        <div className="grid grid-cols-1 md:grid-cols-2 gap-x-6 gap-y-3">
          <DetailGrid task={t} />
          <div>
            <DetailLabel>Lỗi gần nhất</DetailLabel>
            {t.last_error ? (
              <pre className="mt-1 bg-bg/60 rounded-sm p-2.5 text-[11px] font-mono text-text-muted whitespace-pre-wrap break-words max-h-64 overflow-auto">
                {t.last_error}
              </pre>
            ) : (
              <p className="mt-1 text-xs text-text-subtle">Chưa có lỗi nào được ghi nhận.</p>
            )}
          </div>
        </div>
      </td>
    </tr>
  )
}

function DetailGrid({ task: t }: { task: ApiTask }) {
  // Render the joined-context block as a definition list. Items
  // hide themselves when their data is null — `chapter` tasks won't
  // show "Bản dịch" rows, etc. This keeps the panel dense without
  // forcing every kind to render the same shape.
  const items: Array<[string, React.ReactNode]> = []

  if (t.work_id != null) {
    items.push([
      'Tác phẩm',
      <a
        key="w"
        href={`/w/${t.work_id}`}
        className="text-accent-text hover:underline font-mono text-xs"
      >
        work/{t.work_id}
      </a>,
    ])
  }
  if (t.chapter_label || t.chapter_id != null) {
    items.push([
      'Chương',
      <span key="c" className="text-text-muted text-xs">
        {t.chapter_label ?? '—'}
        {t.chapter_id != null && (
          <span className="text-text-subtle font-mono ml-2">chapter/{t.chapter_id}</span>
        )}
      </span>,
    ])
  }
  if (t.source_lang || t.target_lang) {
    items.push([
      'Ngôn ngữ',
      t.source_lang && t.target_lang ? (
        <span key="lp" className="text-xs">
          <span className="text-text-muted">{languageName(t.source_lang)}</span>
          <span className="text-text-subtle mx-1.5">→</span>
          <span className="text-text-muted">{languageName(t.target_lang)}</span>
        </span>
      ) : (
        <span key="lp" className="text-xs text-text-muted">
          {t.source_lang ? languageName(t.source_lang) : '—'}
        </span>
      ),
    ])
  }
  if (t.llm_model) {
    items.push([
      'Model',
      <span key="m" className="text-xs font-mono text-text-muted">{t.llm_model}</span>,
    ])
  }
  if (t.owner_id != null) {
    items.push([
      'Người tạo',
      <span key="o" className="text-xs font-mono text-text-muted">user:{t.owner_id}</span>,
    ])
  }
  items.push([
    'Task ID',
    <span key="t" className="text-xs font-mono text-text-subtle">
      {t.target_kind}/{t.target_id} · {t.stage}
    </span>,
  ])
  if (t.claimed_at) {
    items.push([
      'Claim từ',
      <span key="ca" className="text-xs text-text-muted">
        {formatTime(t.claimed_at)}
      </span>,
    ])
  }

  return (
    <dl className="grid grid-cols-[110px_minmax(0,1fr)] gap-x-3 gap-y-1.5 self-start">
      {items.map(([label, value], i) => (
        <div key={i} className="contents">
          <dt className="text-[11px] uppercase tracking-wider text-text-subtle font-medium py-0.5">
            {label}
          </dt>
          <dd className="min-w-0 py-0.5">{value}</dd>
        </div>
      ))}
    </dl>
  )
}

function DetailLabel({ children }: { children: React.ReactNode }) {
  return (
    <p className="text-[11px] uppercase tracking-wider text-text-subtle font-medium">
      {children}
    </p>
  )
}

function LangPair({ source, target }: { source: string; target: string }) {
  return (
    <span className="inline-flex items-center gap-1">
      <Tag size="sm" tone="outline" uppercase>{languageCode(source)}</Tag>
      <span className="text-text-subtle">→</span>
      <Tag size="sm" tone="accent" uppercase>{languageCode(target)}</Tag>
    </span>
  )
}

function actionTitle(kind: 'requeue' | 'release' | 'fail', task: ApiTask) {
  const label = `${STAGE_LABEL[task.stage]} · ${TARGET_LABEL[task.target_kind]} #${task.target_id}`
  if (kind === 'requeue') return `Đưa lại hàng đợi — ${label}`
  if (kind === 'release') return `Gỡ claim — ${label}`
  return `Đẩy dead-letter — ${label}`
}
function actionDescription(kind: 'requeue' | 'release' | 'fail') {
  if (kind === 'requeue') return 'Reset lượt thử = 0, xoá lỗi gần nhất. Worker sẽ thử lại từ đầu.'
  if (kind === 'release') return 'Xoá claim hiện tại (worker chết / không phản hồi). Lượt thử giữ nguyên.'
  return 'Đánh dấu dead-letter (lượt thử = MAX). Task không tự thử lại; admin có thể đưa lại sau.'
}
function actionLabel(kind: 'requeue' | 'release' | 'fail') {
  if (kind === 'requeue') return 'Đưa lại hàng đợi'
  if (kind === 'release') return 'Gỡ claim'
  return 'Đẩy dead-letter'
}

// ── Audit panel ────────────────────────────────────────────────────────────

function AuditPanel() {
  const auditQ = useQuery({
    queryKey: qk.adminOps.actions({ limit: 50 }),
    queryFn:  () => api.adminOps.listActions({ limit: 50 }),
    refetchInterval: POLL_MS,
  })

  const [expanded, setExpanded] = useState<number | null>(null)
  const rows = auditQ.data ?? []

  return (
    <section className="space-y-3">
      <div>
        <h2 className="text-sm font-semibold tracking-tight inline-flex items-center gap-2">
          <History size={13} className="text-text-subtle" />
          Lịch sử thao tác
          <span className="text-text-subtle font-normal tabular-nums">· {rows.length}</span>
        </h2>
        <p className="text-xs text-text-subtle mt-0.5">
          50 thao tác gần nhất. Bấm vào dòng để xem trạng thái trước khi can thiệp.
        </p>
      </div>
      {auditQ.isLoading ? (
        <div className="bg-surface rounded-md py-10 flex items-center justify-center">
          <Spinner />
        </div>
      ) : rows.length === 0 ? (
        <EmptyState
          icon={History}
          title="Chưa có thao tác nào"
          hint="Mọi pause/resume/requeue/dead-letter sẽ xuất hiện ở đây."
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
  const isStageAction = a.action.startsWith('stage')
  return (
    <div>
      <button
        onClick={onToggle}
        className="w-full text-left px-3.5 py-2.5 hover:bg-hover/40 transition-colors flex items-start gap-3 cursor-pointer"
      >
        {open
          ? <ChevronDown size={13} className="text-text-subtle mt-1 flex-none" />
          : <ChevronRight size={13} className="text-text-subtle mt-1 flex-none" />}
        <div className="flex-1 min-w-0">
          <div className="flex items-center gap-2 flex-wrap">
            <Tag tone={isStageAction ? 'warning' : 'info'} size="sm">
              {ACTION_LABEL[a.action]}
            </Tag>
            <span className="font-mono text-xs text-text-muted truncate">{target}</span>
            <span className="text-xs text-text-subtle ml-auto tabular-nums">{formatTime(a.at)}</span>
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
            Trạng thái trước
          </p>
          <pre className="bg-bg/60 rounded p-2.5 text-[11px] font-mono text-text-muted whitespace-pre-wrap break-words">
            {JSON.stringify(a.prev_state, null, 2)}
          </pre>
        </div>
      )}
    </div>
  )
}

// ── Shared modal: action + reason ──────────────────────────────────────────

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
            Lý do (bắt buộc, lưu vào lịch sử thao tác)
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

// ── Tiny shared bits ───────────────────────────────────────────────────────

function PanelHeader({
  icon, title, hint,
}: { icon: React.ReactNode; title: string; hint: string }) {
  return (
    <div className="px-3.5 py-2.5 border-b border-border-soft bg-surface-2/40">
      <h2 className="text-sm font-semibold tracking-tight inline-flex items-center gap-2">
        <span className="text-text-subtle">{icon}</span>
        {title}
      </h2>
      <p className="text-[11px] text-text-subtle mt-0.5 leading-snug">{hint}</p>
    </div>
  )
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
    <label className="inline-flex flex-col gap-1 min-w-[140px]">
      <span className="text-[10px] uppercase tracking-wider text-text-subtle font-medium">
        {label}
      </span>
      <select
        value={value}
        onChange={(e) => onChange(e.target.value as T)}
        className="h-8 px-2 rounded-sm bg-surface-2 text-text text-sm border border-transparent hover:bg-hover focus:border-accent focus:outline-none cursor-pointer transition-colors"
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
        'inline-flex items-center justify-center size-7 rounded-sm transition-colors cursor-pointer',
        'hover:bg-hover text-text-muted',
        tone === 'destructive' && 'hover:text-rose-300',
      )}
    >
      {children}
    </button>
  )
}

// ── Helpers ────────────────────────────────────────────────────────────────

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
  toast.error(e instanceof Error ? e.message : 'Thao tác thất bại')
}
