// VersionLine — one readable option in a chapter group.
//
// Single-line on desktop, two-line stack on mobile so meta has
// breathing room on narrow viewports. The whole row is one tap:
//
//   • Typoon translation row: clicking opens the reader.
//   • Target-lang raw row:    clicking opens the raw reader.
//   • Non-target raw row:     clicking starts a translation spawn
//                             (target_lang from the viewer's library
//                             entry).
//
// State chips are written from the user's POV — "Đang chờ",
// "Đang dịch 12/40", "Tạm ngưng" — not internal jargon. When a
// translation has failed or is blocked, the worker-stamped cause is
// humanized and rendered INLINE on row 2 so mobile users see the
// reason without hover.

import {
  Check, Hourglass, PauseCircle,
  RotateCcw, Sparkles,
} from 'lucide-react'

import { cn } from '@shared/lib/cn'
import { languageName } from '@shared/lib/lang'
import { timeAgo } from '@shared/lib/time'
import type { HubVersion } from '@features/title/mergeChapters'


/** Action this row triggers when clicked. Computed by the parent
 *  (`WorkChapterList`) since it depends on target_lang + readiness
 *  semantics outside the row's own version.
 *
 *  When a chapter has an in-flight or failed translation, the row is
 *  identified by the translation (creator/date metadata) but we still
 *  want the user to be able to OPEN something — either the existing
 *  raw or, for done translations, the translation itself. The state
 *  chip carries the progress/error/blocked indicator; `rawFallback`
 *  on those states tells the row whether clicking opens a raw to read
 *  while waiting (true) or no-ops (false). For `spawn-error` the
 *  chip is itself clickable (retry); the row's own click goes to the
 *  raw fallback if any. */
export type VersionAction =
  | { kind: 'read-translation' }
  | { kind: 'read-raw' }
  | { kind: 'read-raw-with-spawn'; spawnState: 'idle' | 'progress' | 'error' }
  | { kind: 'spawn-translate' }
  | { kind: 'spawn-pending';  rawFallback: boolean }
  | { kind: 'spawn-progress'; rawFallback: boolean }
  | { kind: 'spawn-blocked';  rawFallback: boolean }
  | { kind: 'spawn-error';    rawFallback: boolean }
  | { kind: 'disabled'; reason: string }


export interface VersionLineProps {
  chapterNumber: string
  version:       HubVersion
  action:        VersionAction
  /** Worker progress hint ("12/40", "Đang tải…") when running. */
  progressLabel?: string
  /** Human-readable failure cause stamped by the worker. Surfaced
   *  inline on row 2 for blocked / error rows; null otherwise. */
  errorMessage?:  string | null
  onClick:        () => void
  /** Triggered by the chip on `read-raw-with-spawn` rows. Lets the
   *  user click "Dịch" without first reading the raw chapter. */
  onSpawn?:       () => void
  /** Triggered by the chip on `spawn-error` rows. Row click opens
   *  the raw fallback (if any); chip click retries the failed
   *  translation. Tách 2 affordance ra để user không phải chọn giữa
   *  "đọc tạm" và "thử lại" bằng cùng 1 chuyển động. */
  onRetry?:       () => void
}


export function VersionLine({
  chapterNumber, version, action,
  progressLabel, errorMessage, onClick, onSpawn, onRetry,
}: VersionLineProps) {
  const disabled = action.kind === 'disabled'
  const failed   = action.kind === 'spawn-error'
                || action.kind === 'spawn-blocked'
  // `busy` no longer disables the row outright — a chapter being
  // translated may still have a raw the user wants to read while
  // waiting. The row falls back to a read-raw click whenever
  // `rawFallback` is true on a progress/error/blocked/pending state.
  const stateWithFallback =
       action.kind === 'spawn-pending'
    || action.kind === 'spawn-progress'
    || action.kind === 'spawn-blocked'
    || action.kind === 'spawn-error'
  const rowReadable = stateWithFallback ? action.rawFallback : false
  const interactive =
       !disabled
    && (action.kind !== 'spawn-pending'  || rowReadable)
    && (action.kind !== 'spawn-progress' || rowReadable)
    && (action.kind !== 'spawn-blocked'  || rowReadable)
    && (action.kind !== 'spawn-error'    || rowReadable)

  // Humanized failure reason — null when row isn't in an error /
  // blocked state. Renders on row 2 (mobile) and as the button title.
  const reasonInline = failed
    ? humanizeFailure(action.kind, errorMessage)
    : null

  // Root is a div role=button (not a real <button>) so the chip on
  // raw-with-spawn rows can be its own <button> without producing
  // invalid nested-button HTML. Keyboard activation via Enter / Space
  // mirrors a native button.
  return (
    <div
      role="button"
      tabIndex={interactive ? 0 : -1}
      onClick={interactive ? onClick : undefined}
      onKeyDown={(e) => {
        if (!interactive) return
        if (e.key === 'Enter' || e.key === ' ') {
          e.preventDefault()
          onClick()
        }
      }}
      aria-disabled={!interactive}
      title={
        disabled       ? (action as { reason: string }).reason :
        reasonInline   ? reasonInline.full :
        undefined
      }
      className={cn(
        'group w-full text-left text-sm transition-colors',
        'px-3 py-2.5 min-h-[3.25rem] sm:min-h-11',
        // Mobile: 2-line stack. Desktop (sm+): collapse to single row.
        'flex flex-col gap-1',
        'sm:flex-row sm:items-center sm:gap-2.5',
        'focus-visible:outline-none focus-visible:ring-1 focus-visible:ring-accent/50',
        disabled        ? 'opacity-50 cursor-not-allowed' :
        !interactive    ? 'cursor-default' :
                          'cursor-pointer hover:bg-hover/50 active:bg-hover',
        action.kind === 'spawn-error'   && 'text-rose-300',
        action.kind === 'spawn-blocked' && 'text-amber-300',
      )}
    >
      {/* Row 1 — identity + action. */}
      <div className="flex items-center gap-2 sm:gap-2.5 min-w-0 sm:flex-1">
        <span className="tabular font-medium text-accent shrink-0 w-12 sm:w-14">
          Ch.{chapterNumber || '?'}
        </span>
        <LangChip lang={version.lang} />

        {/* Label inline on desktop; mobile renders on row 2. */}
        <Label
          version={version}
          className="hidden sm:inline-flex sm:flex-1 sm:min-w-0"
        />

        {/* Right cluster — order: date (metadata) → chip (action).
            Chip ngoài cùng phải, sát ngón cái mobile, dễ với chuột
            desktop. Date đứng trước chip nên không cần cột cố định
            → không wrap. Chip slot fixed-width để align dọc thành
            một cột thẳng giữa các row. */}
        <span className="ml-auto sm:ml-0 inline-flex items-center gap-2.5 shrink-0">
          {version.date && (
            <span
              className="hidden sm:inline whitespace-nowrap text-text-subtle tabular shrink-0"
              title={version.date}
            >
              {timeAgo(version.date)}
            </span>
          )}
          <span className="sm:w-20 sm:inline-flex sm:justify-end">
            <ActionChip
              action={action}
              progressLabel={progressLabel}
              onSpawn={onSpawn}
              onRetry={onRetry}
            />
          </span>
        </span>
      </div>

      {/* Row 2 — mobile only. Failure rows show the reason in place
          of the meta strip; healthy rows show creator + date. */}
      <div className="flex items-center justify-between gap-2 min-w-0 sm:hidden">
        {reasonInline ? (
          <span
            className={cn(
              'truncate text-xs',
              action.kind === 'spawn-error'   && 'text-rose-300',
              action.kind === 'spawn-blocked' && 'text-amber-300',
            )}
          >
            {reasonInline.short}
          </span>
        ) : (
          <Label version={version} className="text-xs min-w-0" />
        )}
        {version.date && (
          <span
            className="text-xs text-text-subtle tabular shrink-0"
            title={version.date}
          >
            {timeAgoShort(version.date)}
          </span>
        )}
      </div>
    </div>
  )
}


// ── Subcomponents ──────────────────────────────────────────────


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


function Label({
  version, className,
}: {
  version:   HubVersion
  className: string
}) {
  if (version.kind === 'raw') {
    const creator = version.creatorName
    const source  = version.sourceName
    return (
      <span className={cn('truncate', className)}>
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
  // translation
  const creator = version.creatorName ?? 'unknown'
  const viaLang = version.sourceLang ? languageName(version.sourceLang) : null
  const source  = version.sourceName ?? null
  return (
    <span className={cn('truncate text-text', className)}>
      @{creator}
      {(viaLang || source) && (
        <span className="text-text-subtle">
          {' · từ '}
          {viaLang}
          {viaLang && source ? ' ' : ''}
          {source}
        </span>
      )}
    </span>
  )
}


/** Action affordance. Read actions render nothing (the whole row is
 *  the affordance). Every non-read state gets its own chip with a
 *  distinct icon + color so the user can scan the list and instantly
 *  see which chapters are healthy, queued, working, blocked, or
 *  errored.
 *
 *  `read-raw-with-spawn` is the only kind whose chip is INTERACTIVE
 *  separately from the row click: the row reads the raw verbatim, the
 *  chip kicks off a translation spawn. `onSpawn` is wired by the
 *  parent — when null, the chip degrades to a non-clickable hint.
 */
function ActionChip({
  action, progressLabel, onSpawn, onRetry,
}: {
  action:         VersionAction
  progressLabel?: string
  onSpawn?:       () => void
  onRetry?:       () => void
}) {
  if (action.kind === 'spawn-translate') {
    return (
      <Chip
        tone="accent"
        icon={<Sparkles size={12} />}
        label="Dịch"
        hideLabelOnMobile
      />
    )
  }
  if (action.kind === 'read-raw-with-spawn') {
    // Spawn affordance riding on a raw row. The row stays a
    // read-raw click; this chip is its own button that intercepts
    // the click and triggers a spawn instead.
    if (action.spawnState === 'progress') {
      return <Chip tone="accent" label={progressLabel ?? 'Đang dịch…'} />
    }
    if (action.spawnState === 'error') {
      return (
        <Chip
          tone="error"
          icon={<RotateCcw size={12} />}
          label="Thử lại"
          hideLabelOnMobile
          onClick={onSpawn}
        />
      )
    }
    return (
      <Chip
        tone="accent"
        icon={<Sparkles size={12} />}
        label="Dịch"
        hideLabelOnMobile
        onClick={onSpawn}
      />
    )
  }
  if (action.kind === 'spawn-pending') {
    return (
      <Chip
        tone="info"
        icon={<Hourglass size={12} />}
        label="Đang chờ"
        hideLabelOnMobile
      />
    )
  }
  if (action.kind === 'spawn-progress') {
    // Worker is actively processing — show whatever progress hint
    // the parent has (e.g. "Đang tải · 12/40"). Falls back to a
    // generic "Đang dịch…" when the worker hasn't reported a step.
    return (
      <Chip
        tone="accent"
        label={progressLabel ?? 'Đang dịch…'}
      />
    )
  }
  if (action.kind === 'spawn-blocked') {
    return (
      <Chip
        tone="warning"
        icon={<PauseCircle size={12} />}
        label="Tạm ngưng"
        hideLabelOnMobile
      />
    )
  }
  if (action.kind === 'spawn-error') {
    // Chip is the retry affordance — row click falls through to the
    // raw fallback (if any) so the user can still read while we redo.
    return (
      <Chip
        tone="error"
        icon={<RotateCcw size={12} />}
        label="Thử lại"
        hideLabelOnMobile
        onClick={onRetry}
      />
    )
  }
  if (action.kind === 'read-translation') {
    // Already translated — row is clickable to OPEN the reader, but
    // the chip itself is a read-only marker (mọi row có cùng cột
    // chip thẳng hàng, không có ô trống lạc lõng giữa các row đang
    // có chip).
    return (
      <Chip
        tone="muted"
        icon={<Check size={12} />}
        label="Đã dịch"
        hideLabelOnMobile
      />
    )
  }
  return null
}


function Chip({
  tone, icon, label, hideLabelOnMobile, onClick,
}: {
  tone:               'accent' | 'info' | 'warning' | 'error' | 'muted'
  icon?:              React.ReactNode
  label:              string
  hideLabelOnMobile?: boolean
  /** When provided, the chip becomes a real button that swallows the
   *  click before it reaches the parent row. */
  onClick?:           () => void
}) {
  const palette = {
    accent:  'bg-accent/10 text-accent group-hover:bg-accent/15',
    info:    'bg-info/15 text-info-text',
    warning: 'bg-amber-500/10 text-amber-400',
    error:   'bg-rose-500/10 text-rose-400',
    // muted = read-only marker. No background — đỡ chiếm thị giác
    // so với chip action; user vẫn thấy "đây là row đã xong" mà
    // không nhầm là nút bấm.
    muted:   'text-text-subtle',
  }[tone]
  const cls = cn(
    'shrink-0 inline-flex items-center gap-1 text-xs tabular',
    'px-2 h-6 rounded-sm transition-colors',
    palette,
    onClick && 'cursor-pointer hover:brightness-110 active:brightness-90',
  )
  if (onClick) {
    return (
      <button
        type="button"
        onClick={(e) => { e.stopPropagation(); onClick() }}
        className={cls}
      >
        {icon}
        <span className={hideLabelOnMobile ? 'hidden sm:inline' : undefined}>
          {label}
        </span>
      </button>
    )
  }
  return (
    <span className={cls}>
      {icon}
      <span className={hideLabelOnMobile ? 'hidden sm:inline' : undefined}>
        {label}
      </span>
    </span>
  )
}


// ── Failure humanization ──────────────────────────────────────
//
// Worker stamps `error_message` with whatever the upstream said —
// raw OpenAI exception strings, asyncpg tracebacks, etc. Surfacing
// those verbatim is hostile to readers; we map known patterns into
// short Vietnamese phrases. `full` keeps the original text for the
// title attribute / log so debugging stays possible.


interface HumanReason {
  /** One-line phrase for the mobile row + visible UI. */
  short: string
  /** Original message — title attribute, panel detail. */
  full:  string
}


function humanizeFailure(
  kind: 'spawn-error' | 'spawn-blocked',
  raw:  string | null | undefined,
): HumanReason {
  const full = (raw ?? '').trim() || (
    kind === 'spawn-blocked'
      ? 'Pipeline tạm ngưng để quản trị xử lý.'
      : 'Có lỗi xảy ra. Bấm để thử lại.'
  )
  return { short: classifyReason(kind, full), full }
}


function classifyReason(
  kind: 'spawn-error' | 'spawn-blocked',
  msg:  string,
): string {
  const low = msg.toLowerCase()

  // Operator-level patterns — these match the OpenAI classifier's
  // OPERATOR_5XX_HINTS so the UI text stays consistent with the
  // backend's reasoning for pausing.
  if (low.includes('model_not_found') || low.includes('model not found')) {
    return 'Model dịch chưa khả dụng — đang chờ quản trị cập nhật cấu hình.'
  }
  if (low.includes('no available credential')
      || low.includes('credential has been invalidated')
      || low.includes('token_invalidated')
      || low.includes('authentication token has been invalidated')) {
    return 'Khoá API hết hạn — đang chờ quản trị thay khoá mới.'
  }
  if (low.includes('insufficient_quota')
      || low.includes('billing_hard_limit')
      || low.includes('account is suspended')) {
    return 'Tài khoản dịch hết hạn mức — đang chờ quản trị nạp lại.'
  }
  if (low.includes('region not supported')) {
    return 'Khu vực không hỗ trợ — đang chờ quản trị đổi tuyến.'
  }

  // Transient / user-facing patterns.
  if (low.includes('timeout') || low.includes('timed out')) {
    return 'Quá thời gian chờ — bấm để thử lại.'
  }
  if (low.includes('rate limit') || low.includes('rate_limit')) {
    return 'Vượt giới hạn tốc độ — bấm để thử lại sau.'
  }
  if (low.includes('upstream') || low.includes('502')
      || low.includes('503') || low.includes('504')) {
    return 'Dịch vụ tạm gián đoạn — bấm để thử lại.'
  }
  if (low.includes('chapter deleted')) {
    return 'Chương đã bị xoá.'
  }
  if (low.includes('page count') && low.includes('0')) {
    return 'Không tải được ảnh chương — bấm để thử lại.'
  }

  // Generic fallback. Keep it actionable so the user knows what to do.
  return kind === 'spawn-blocked'
    ? 'Pipeline tạm ngưng — đang chờ quản trị xử lý.'
    : 'Có lỗi xảy ra. Bấm để thử lại.'
}


/** Compact relative time for mobile meta strip — '2d', '3h', '5m'. */
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
