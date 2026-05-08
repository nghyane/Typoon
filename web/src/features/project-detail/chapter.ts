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
  // pending = enqueued, waiting for worker. Treat it identically to
  // running for user-facing display: same colour, same wording, the
  // stage label still applies. The difference (claimed vs unclaimed)
  // is a backend concern, not a story for the user — and surfacing it
  // produced a "running → pending → running" flicker on every stage
  // handoff.
  pending: 'Đang xử lý',
  idle:    'Chờ xử lý',
}

export const STATE_TONE: Record<ChapterState, BadgeTone> = {
  done:    'success',
  running: 'info',
  error:   'error',
  pending: 'info',
  idle:    'neutral',
}

const STAGE_LABEL: Record<string, string> = {
  scan:      'Quét bong bóng',
  translate: 'Đang dịch',
  render:    'Đang render',
}

export function stageLabel(stage: string): string {
  return STAGE_LABEL[stage] ?? stage
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

// Per-chapter completion. Only meaningful while in flight — done = 100,
// idle = 0, error stays at last known progress (0 here). Pending is
// treated like running: backend distinguishes them but the user does not.
export function chapterPct(ch: ApiChapter): number {
  if (ch.state === 'done') return 100
  const inFlight = ch.state === 'running' || ch.state === 'pending'
  if (inFlight && ch.progress && ch.progress.page_total > 0) {
    return Math.round((ch.progress.page_index / ch.progress.page_total) * 100)
  }
  return 0
}
