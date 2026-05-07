import type { ApiChapter, ChapterState } from './api'

// Single source of truth for chapter status presentation. Routes import
// these — no inline ad-hoc switches anywhere.

export interface StateStyle {
  label: string
  dot:   string  // tailwind bg-* class
  text:  string  // tailwind text-* class
  bar:  string   // tailwind bg-* class for progress fill
}

export const STATE: Record<ChapterState, StateStyle> = {
  done:    { label: 'Hoàn thành', dot: 'bg-emerald-500', text: 'text-zinc-700', bar: 'bg-emerald-500' },
  running: { label: 'Đang xử lý', dot: 'bg-blue-500',    text: 'text-zinc-700', bar: 'bg-blue-500'    },
  error:   { label: 'Lỗi',        dot: 'bg-red-500',     text: 'text-red-600',  bar: 'bg-red-500'     },
  pending: { label: 'Đang chờ',   dot: 'bg-amber-400',   text: 'text-zinc-500', bar: 'bg-amber-400'   },
  idle:    { label: 'Chờ xử lý',  dot: 'bg-zinc-300',    text: 'text-zinc-400', bar: 'bg-zinc-300'    },
}

const STAGE_LABEL: Record<string, string> = {
  scan:      'Quét bong bóng',
  translate: 'Dịch',
  render:    'Render',
}

export function stageLabel(stage: string): string {
  return STAGE_LABEL[stage] ?? stage
}

export interface ChapterStats { total: number; done: number; running: number; error: number }

export function chapterStats(chs: readonly ApiChapter[]): ChapterStats {
  let done = 0, running = 0, error = 0
  for (const c of chs) {
    if (c.state === 'done')    done++
    if (c.state === 'running') running++
    if (c.state === 'error')   error++
  }
  return { total: chs.length, done, running, error }
}

// 0..100. Errors count toward "settled" so a chapter doesn't pin progress
// at < 100% forever; the error dot still flags them.
export function progressPct(s: ChapterStats): number {
  return s.total === 0 ? 0 : Math.round(((s.done + s.error) / s.total) * 100)
}

// Per-chapter completion. While running: page_index / page_total; otherwise
// 0 (idle/pending) or 100 (done) — error stays at last known progress.
export function chapterPct(ch: ApiChapter): number {
  if (ch.state === 'done')    return 100
  if (ch.state === 'running' && ch.progress && ch.progress.page_total > 0) {
    return Math.round((ch.progress.page_index / ch.progress.page_total) * 100)
  }
  return 0
}
