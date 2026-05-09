import type { ApiChapter, ChapterState } from '@shared/api/api'
import type { BadgeTone } from '@shared/ui/primitives'

// =============================================================================
// Chapter status mapping — pure data → presentation hint.
// Two separate maps:
//   STATE_LABEL — Vietnamese label for the state (used in badges, sub-meta)
//   STATE_TONE  — semantic tone for color (badge tone, progress bar tone)
// Old code packed both + a tailwind class string into one record; that
// coupled style with display and made the bar class unusable for new layers
// (sidebar, mini-cards). Tone is the single source of truth — anything that
// needs a colour derives from it.
// =============================================================================

export const STATE_LABEL: Record<ChapterState, string> = {
  done:    'Hoàn thành',
  running: 'Đang xử lý',
  error:   'Lỗi',
  // Generic fallback labels used only when the stage is unknown. With a
  // stage hint, stageLabelFor() below picks the proper "Đang …"/"Chờ …"
  // wording so the user can distinguish queued from in-flight work.
  pending: 'Chờ xử lý',
  idle:    'Chưa bắt đầu',
}

export const STATE_TONE: Record<ChapterState, BadgeTone> = {
  done:    'success',
  running: 'info',
  error:   'error',
  // pending = enqueued, no worker has claimed yet → neutral so a row
  // sitting in the queue doesn't look like it's actively running.
  pending: 'neutral',
  idle:    'neutral',
}

const STAGE_RUNNING_LABEL: Record<string, string> = {
  prepare:   'Đang chuẩn bị',
  scan:      'Đang quét bong bóng',
  translate: 'Đang dịch',
  render:    'Đang render',
}

const STAGE_PENDING_LABEL: Record<string, string> = {
  prepare:   'Chờ chuẩn bị',
  scan:      'Chờ quét',
  translate: 'Chờ dịch',
  render:    'Chờ render',
}

export function stageLabelFor(state: ChapterState, stage: string): string {
  if (state === 'running') return STAGE_RUNNING_LABEL[stage] ?? stage
  if (state === 'pending') return STAGE_PENDING_LABEL[stage] ?? `Chờ ${stage}`
  return stage
}

// ── stats ──────────────────────────────────────────────────────────────────

export interface ChapterStats { total: number; done: number; running: number; error: number }

export function chapterStats(chs: readonly ApiChapter[]): ChapterStats {
  let done = 0, running = 0, error = 0
  for (const c of chs) {
    if (c.state === 'done')                                  done++
    if (c.state === 'running' || c.state === 'pending')      running++
    if (c.state === 'error')                                 error++
  }
  return { total: chs.length, done, running, error }
}

// 0..100. Errors count toward "settled" so a chapter doesn't pin progress
// at < 100% forever; the error dot still flags them.
export function progressPct(s: ChapterStats): number {
  return s.total === 0 ? 0 : Math.round(((s.done + s.error) / s.total) * 100)
}

// Per-chapter completion. Only meaningful while a worker is actively
// running: done = 100, idle/pending/error = 0 (no progress to show).
// Pending rows still display a label ("Chờ quét") but no progress bar —
// nothing has started, percentage would be misleading.
export function chapterPct(ch: ApiChapter): number {
  if (ch.state === 'done') return 100
  if (ch.state === 'running' && ch.progress && ch.progress.page_total > 0) {
    return Math.round((ch.progress.page_index / ch.progress.page_total) * 100)
  }
  return 0
}
