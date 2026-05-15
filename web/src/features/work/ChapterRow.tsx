// ChapterRow — single row per chapter in the work-list.
//
// Mental model: one chapter = one row. The row is clickable to READ
// whatever's available (raw if untranslated, translation if done).
// A single chip on the right surfaces the action / state:
//
//   raw, no translation               → row=read-raw, chip="Dịch"
//   client downloading 12/40          → row=read-raw, chip="12/40" + cancel
//   client uploading 70%              → row=read-raw, chip="70%"  + cancel
//   server pending/running            → row=read-raw, chip="Đang dịch"
//   translation done                  → row=read-translation, chip=Check
//   translation error                 → row=read-raw, chip="Thử lại"
//   translation blocked               → row=read-raw, chip="Tạm ngưng"
//   raw at target lang (no AI needed) → row=read-raw, chip=Check
//   no readable source                → disabled
//
// Click row = read (the dominant action — the user almost always
// wants to read, never to fiddle with state). Click chip = state
// action (spawn / cancel / retry). The two never trigger the same
// thing so a misclick has predictable behavior.
//
// All transitions happen in-place. No new row appears next to the
// old one — the React key is `chapter.number`.

import { memo } from 'react'
import {
  AlertTriangle, Check, Hourglass,
  PauseCircle, RotateCcw, Sparkles,
} from 'lucide-react'

import { cn } from '@shared/lib/cn'
import { languageName } from '@shared/lib/lang'
import { timeAgo } from '@shared/lib/time'

import type { HubVersion } from '@features/title/mergeChapters'
import type { ChapterRow as ChapterRowModel } from '@features/title/chapterRow'
import type { SpawnProgress } from '@features/title/useSpawnChapter'


export interface ChapterRowProps {
  row:           ChapterRowModel
  /** Local pipeline state for this chapter — null when no client-
   *  side spawn has been kicked off in this session. */
  spawn:         SpawnProgress | null
  onRead:        (version: HubVersion) => void
  onSpawn:       (raw: HubVersion) => void
  /** Re-kick a server-side `error` translation via the redo endpoint.
   *  Distinct from client-side spawn retry, which goes through
   *  `onSpawn` because the pipeline has cached pages. */
  onRetryServer: (translationId: number) => void
  /** Cancel an in-flight client-side spawn. */
  onAbort:       (chapterNumber: string) => void
}


/** Unified state derived from `row.status` × `spawn`. Drives the chip
 *  + tone selection. Each variant maps to exactly one chip rendering
 *  + one row click target, so the component body is a flat switch. */
type UnifiedState =
  | { kind: 'read';            via: HubVersion }
  | { kind: 'translatable';    from: HubVersion; read: HubVersion | null }
  | { kind: 'client-progress'; label: string; read: HubVersion | null }
  | { kind: 'client-error';    error: string; resumable: boolean; from: HubVersion; read: HubVersion | null }
  | { kind: 'server-pending';  read: HubVersion | null }
  | { kind: 'server-error';    translation: HubVersion; read: HubVersion | null }
  | { kind: 'server-blocked';  read: HubVersion | null }
  | { kind: 'unavailable' }


function unify(row: ChapterRowModel, spawn: SpawnProgress | null): UnifiedState {
  // Client-side pipeline is the freshest signal — when it's running,
  // chip shows progress regardless of what the server thinks.
  if (spawn && spawn.phase !== 'idle' && spawn.phase !== 'done' && spawn.phase !== 'error') {
    return {
      kind:  'client-progress',
      label: progressLabel(spawn),
      read:  row.read,
    }
  }
  if (spawn?.phase === 'error') {
    const from = row.spawnFrom ?? (
      row.status.kind === 'translatable' ? row.status.from : null
    )
    if (from) {
      return {
        kind:      'client-error',
        error:     spawn.error ?? 'Có lỗi xảy ra.',
        resumable: spawn.resumable,
        from,
        read:      row.read,
      }
    }
  }

  switch (row.status.kind) {
    case 'read-translation':
    case 'read-raw-target':
      return { kind: 'read', via: row.status.via }
    case 'translatable':
      // Row click reads the raw verbatim (untranslated); chip kicks
      // the spawn. Two different verbs on two different targets.
      return { kind: 'translatable', from: row.status.from, read: row.read }
    case 'translating-server':
      return { kind: 'server-pending', read: row.read }
    case 'translation-error':
      return {
        kind:        'server-error',
        translation: row.status.via,
        read:        row.read,
      }
    case 'translation-blocked':
      return { kind: 'server-blocked', read: row.read }
    case 'unavailable':
      return { kind: 'unavailable' }
  }
}


function progressLabel(p: SpawnProgress): string {
  switch (p.phase) {
    case 'fetching':    return 'Lấy trang…'
    case 'downloading': return p.total > 0 ? `${p.current}/${p.total}` : 'Đang tải…'
    case 'packing':     return 'Đóng gói…'
    case 'uploading':   return `${p.pct}%`
    case 'spawning':    return 'Khởi tạo…'
    default:            return 'Đang dịch…'
  }
}


export const ChapterRow = memo(function ChapterRow({
  row, spawn, onRead, onSpawn, onRetryServer, onAbort,
}: ChapterRowProps) {
  const state = unify(row, spawn)
  const { chapter } = row

  // Row click — read the most appropriate version. We never wire the
  // row to a destructive verb (spawn/retry/cancel); those are chip-
  // only so a misclick can't trigger them.
  const readTarget = pickReadTarget(state)
  const interactive = readTarget !== null

  const handleRowClick = () => {
    if (readTarget) onRead(readTarget)
  }

  // Subtle row tint so failing states stand out on a list scan.
  const tone =
      state.kind === 'client-error'   ? 'text-rose-300'
    : state.kind === 'server-error'   ? 'text-rose-300'
    : state.kind === 'server-blocked' ? 'text-amber-300'
    : undefined

  const disabled = state.kind === 'unavailable'

  return (
    <div
      role="button"
      tabIndex={interactive ? 0 : -1}
      onClick={interactive ? handleRowClick : undefined}
      onKeyDown={(e) => {
        if (!interactive) return
        if (e.key === 'Enter' || e.key === ' ') {
          e.preventDefault()
          handleRowClick()
        }
      }}
      aria-disabled={!interactive}
      className={cn(
        'group w-full text-left text-sm transition-colors',
        'px-3 py-2.5 min-h-[3.25rem] sm:min-h-11',
        'flex flex-col gap-1',
        'sm:flex-row sm:items-center sm:gap-2.5',
        'focus-visible:outline-none focus-visible:ring-1 focus-visible:ring-accent/50',
        disabled        ? 'opacity-50 cursor-not-allowed' :
        !interactive    ? 'cursor-default' :
                          'cursor-pointer hover:bg-hover/50 active:bg-hover',
        tone,
      )}
    >
      {/* Row 1 — identity + chip. Same layout as the previous
          VersionLine so visual density stays familiar. */}
      <div className="flex items-center gap-2 sm:gap-2.5 min-w-0 sm:flex-1">
        <span className="tabular font-medium text-accent shrink-0 w-12 sm:w-14">
          Ch.{chapter.number || '?'}
        </span>
        <Identity row={row} state={state} className="hidden sm:flex sm:flex-1 sm:min-w-0" />

        <span className="ml-auto sm:ml-0 inline-flex items-center gap-2.5 shrink-0">
          <span
            className="hidden sm:inline whitespace-nowrap text-text-subtle tabular shrink-0 text-xs"
            title={chapter.updatedAt ?? undefined}
          >
            {chapter.updatedAt ? timeAgo(chapter.updatedAt) : '—'}
          </span>
          <span className="sm:w-20 sm:inline-flex sm:justify-end">
            <ActionChip
              state={state}
              onSpawn={() => {
                if (state.kind === 'translatable') onSpawn(state.from)
                else if (state.kind === 'client-error') onSpawn(state.from)
              }}
              onAbort={() => onAbort(chapter.number)}
              onRetryServer={() => {
                if (state.kind === 'server-error' && state.translation.translationId != null) {
                  onRetryServer(state.translation.translationId)
                }
              }}
            />
          </span>
        </span>
      </div>

      {/* Row 2 — mobile only. Identity line drops below the chip on
          narrow viewports so the chip column stays unbroken. */}
      <div className="flex items-center justify-between gap-2 min-w-0 sm:hidden">
        <Identity row={row} state={state} className="text-xs min-w-0" />
        <span
          className="text-xs text-text-subtle tabular shrink-0"
          title={chapter.updatedAt ?? undefined}
        >
          {chapter.updatedAt ? timeAgoShort(chapter.updatedAt) : '—'}
        </span>
      </div>
    </div>
  )
})


/** Read target for a row click. Returns null when the row isn't
 *  interactive (no readable source). */
function pickReadTarget(state: UnifiedState): HubVersion | null {
  switch (state.kind) {
    case 'read':              return state.via
    case 'translatable':      return state.read  // raw verbatim
    case 'client-progress':   return state.read
    case 'client-error':      return state.read
    case 'server-pending':    return state.read
    case 'server-error':      return state.read
    case 'server-blocked':    return state.read
    case 'unavailable':       return null
  }
}


// ── Identity line ─────────────────────────────────────────────


function Identity({
  row, state, className,
}: {
  row:       ChapterRowModel
  state:     UnifiedState
  className: string
}) {
  const v = identityVersion(row, state)
  if (!v) {
    return (
      <span className={cn('truncate text-text-muted', className)}>
        {row.chapter.label ?? 'Không có nguồn'}
      </span>
    )
  }
  return (
    <span className={cn('flex items-center gap-2 min-w-0', className)}>
      <LangChip lang={v.lang} />
      {v.kind === 'translation' && <AiChip />}
      <Label version={v} />
    </span>
  )
}


function identityVersion(row: ChapterRowModel, state: UnifiedState): HubVersion | null {
  switch (state.kind) {
    case 'read':              return state.via
    case 'translatable':      return state.from
    case 'client-progress':
    case 'client-error':
    case 'server-pending':
    case 'server-error':
    case 'server-blocked':
      return row.translation ?? row.read ?? row.spawnFrom ?? null
    case 'unavailable':       return null
  }
}


function Label({ version }: { version: HubVersion }) {
  if (version.kind === 'raw') {
    const creator = version.creatorName
    const source  = version.sourceName
    return (
      <span className="truncate">
        {creator ? (
          <>
            <span className="text-text">@{creator}</span>
            {source && <span className="text-text-subtle"> · {source}</span>}
          </>
        ) : (
          <span className="text-text-muted">{source ?? 'Raw'}</span>
        )}
      </span>
    )
  }
  const creator = version.creatorName ?? 'unknown'
  const via = version.sourceLang ? languageName(version.sourceLang) : null
  const source = version.sourceName ?? null
  return (
    <span className="truncate text-text">
      @{creator}
      {(via || source) && (
        <span className="text-text-subtle">
          {' · từ '}
          {via}
          {via && source ? ' ' : ''}
          {source}
        </span>
      )}
    </span>
  )
}


// ── Action chip ───────────────────────────────────────────────


function ActionChip({
  state, onSpawn, onAbort, onRetryServer,
}: {
  state:         UnifiedState
  onSpawn:       () => void
  onAbort:       () => void
  onRetryServer: () => void
}) {
  switch (state.kind) {
    case 'read':
      // Read-only marker — keeps the chip column aligned with action
      // rows but doesn't compete with the row click.
      return (
        <Chip
          tone="muted"
          icon={<Check size={12} />}
          label="Đọc"
          hideLabelOnMobile
        />
      )
    case 'translatable':
      return (
        <Chip
          tone="accent"
          icon={<Sparkles size={12} />}
          label="Dịch"
          hideLabelOnMobile
          onClick={onSpawn}
        />
      )
    case 'client-progress':
      // Click cancels. Label is the live progress hint so the user
      // can see download/upload progress without a second affordance.
      return (
        <Chip
          tone="progress"
          icon={<Spinner />}
          label={state.label}
          onClick={onAbort}
          title="Bấm để huỷ"
        />
      )
    case 'client-error':
      return (
        <Chip
          tone="error"
          icon={<RotateCcw size={12} />}
          label={state.resumable ? 'Tiếp tục' : 'Thử lại'}
          hideLabelOnMobile
          onClick={onSpawn}
          title={state.error}
        />
      )
    case 'server-pending':
      return (
        <Chip
          tone="info"
          icon={<Hourglass size={12} />}
          label="Đang dịch"
          hideLabelOnMobile
        />
      )
    case 'server-error':
      return (
        <Chip
          tone="error"
          icon={<RotateCcw size={12} />}
          label="Thử lại"
          hideLabelOnMobile
          onClick={onRetryServer}
          title={state.translation.errorMessage ?? undefined}
        />
      )
    case 'server-blocked':
      return (
        <Chip
          tone="warning"
          icon={<PauseCircle size={12} />}
          label="Tạm ngưng"
          hideLabelOnMobile
        />
      )
    case 'unavailable':
      return (
        <Chip
          tone="muted"
          icon={<AlertTriangle size={12} />}
          label="—"
        />
      )
  }
}


// ── Chip primitive ────────────────────────────────────────────


type ChipTone = 'accent' | 'info' | 'warning' | 'error' | 'muted' | 'progress'


function Chip({
  tone, icon, label, hideLabelOnMobile, onClick, title,
}: {
  tone:               ChipTone
  icon:               React.ReactNode
  label:              string
  hideLabelOnMobile?: boolean
  onClick?:           () => void
  title?:             string
}) {
  const palette: Record<ChipTone, string> = {
    accent:   'bg-accent/10 text-accent group-hover:bg-accent/15',
    info:     'bg-info-bg text-info-text',
    warning:  'bg-amber-500/10 text-amber-400',
    error:    'bg-rose-500/10 text-rose-400',
    progress: 'bg-accent/10 text-accent',
    muted:    'text-text-subtle',
  }
  const cls = cn(
    'shrink-0 inline-flex items-center gap-1 text-xs tabular',
    'px-2 h-6 rounded-sm transition-colors',
    palette[tone],
    onClick && 'cursor-pointer hover:brightness-110 active:brightness-90',
  )
  if (onClick) {
    return (
      <button
        type="button"
        onClick={(e) => { e.stopPropagation(); onClick() }}
        className={cls}
        title={title}
      >
        {icon}
        <span className={hideLabelOnMobile ? 'hidden sm:inline' : undefined}>
          {label}
        </span>
      </button>
    )
  }
  return (
    <span className={cls} title={title}>
      {icon}
      <span className={hideLabelOnMobile ? 'hidden sm:inline' : undefined}>
        {label}
      </span>
    </span>
  )
}


// ── Decoration chips ──────────────────────────────────────────


function LangChip({ lang }: { lang: string }) {
  return (
    <span
      className={cn(
        'shrink-0 text-[10px] sm:text-xs uppercase tabular',
        'px-1.5 h-5 inline-flex items-center rounded-sm',
        'bg-surface-2 text-text-muted',
      )}
      title={languageName(lang)}
    >
      {lang}
    </span>
  )
}


function AiChip() {
  return (
    <span
      className={cn(
        'shrink-0 text-[10px] sm:text-xs uppercase tabular',
        'px-1.5 h-5 inline-flex items-center rounded-sm',
        'bg-accent/15 text-accent',
      )}
      title="Bản dịch AI (Typoon)"
    >
      AI
    </span>
  )
}


function Spinner() {
  return (
    <span
      className={cn(
        'inline-block h-3 w-3 rounded-full',
        'border-2 border-current border-r-transparent',
        'animate-spin',
      )}
    />
  )
}


// ── Compact relative time ─────────────────────────────────────


/** '2d', '3h', '5m' — for the mobile meta strip. */
function timeAgoShort(s: string): string {
  const t = new Date(s.includes('T') ? s : s.replace(' ', 'T') + 'Z').getTime()
  const diff = Date.now() - t
  if (!Number.isFinite(diff)) return ''
  const sec = Math.max(1, Math.round(diff / 1000))
  if (sec <       60) return `${sec}s`
  const min = Math.round(sec / 60)
  if (min <       60) return `${min}m`
  const hr  = Math.round(min / 60)
  if (hr  <       24) return `${hr}h`
  const day = Math.round(hr / 24)
  if (day <       30) return `${day}d`
  const mo  = Math.round(day / 30)
  if (mo  <       12) return `${mo}th`
  return `${Math.round(mo / 12)}y`
}
