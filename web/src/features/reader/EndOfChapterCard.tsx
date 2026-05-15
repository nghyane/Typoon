// EndOfChapterCard — surfaced after the last page (single mode) and
// at the bottom of the strip (TTB mode), plus as the body of an
// `empty`-status chapter. One affordance path: keep reading. Other
// actions live as smaller links.
//
// The card resolves its own next-step plan via the pure 7-step
// fallback in `./resolvers`. The component owns nothing beyond
// rendering — no IO, no derived state. Spawn callback is passed in
// so the route can wire the same multi-key spawn hook used on the
// work hub.
//
// Never auto-navigates. Auto-jump on scroll-end would break the
// "đã đọc xong" moment and trap users who scroll past the last
// page by accident.

import { useMemo } from 'react'
import { Link } from '@tanstack/react-router'
import { ArrowRight, Sparkles, ChevronLeft, SkipForward } from 'lucide-react'

import { cn } from '@shared/lib/cn'
import { languageName } from '@shared/lib/lang'
import type { HubChapter, HubVersion } from '@features/title/mergeChapters'

import { resolveNextPlan, type NextChapterPlan } from './resolvers'


interface Props {
  workId:            number
  currentLabel:      string
  readingSourceLang: string | null
  targetLang:        string | null
  chapters:          HubChapter[]
  currentNum:        string
  onSpawn:           (chapter: HubChapter, raw: HubVersion) => void
  onBackToList:      () => void
}


export function EndOfChapterCard({
  workId, currentLabel, readingSourceLang, targetLang,
  chapters, currentNum, onSpawn, onBackToList,
}: Props) {
  const plan = useMemo(
    () => resolveNextPlan(chapters, currentNum, targetLang, readingSourceLang),
    [chapters, currentNum, targetLang, readingSourceLang],
  )

  return (
    <div className="px-4 py-12 sm:py-16 max-w-md mx-auto">
      <div className="text-center mb-6">
        <p className="text-xs uppercase tracking-wide text-text-subtle">
          Đã đọc xong
        </p>
        <p className="text-base font-medium text-text mt-0.5">
          {currentLabel}
        </p>
      </div>

      <NextCallToAction
        workId={workId}
        plan={plan}
        onSpawn={onSpawn}
      />

      <div className="mt-6 flex items-center justify-center gap-4 text-xs text-text-subtle">
        <button
          onClick={onBackToList}
          className="inline-flex items-center gap-1 hover:text-text transition-colors cursor-pointer"
        >
          <ChevronLeft size={12} />
          Danh sách chương
        </button>
      </div>
    </div>
  )
}


function NextCallToAction({
  workId, plan, onSpawn,
}: {
  workId:  number
  plan:    NextChapterPlan
  onSpawn: (chapter: HubChapter, raw: HubVersion) => void
}) {
  if (plan.kind === 'end-of-spine') {
    return (
      <div className="rounded-md bg-surface border border-border-soft px-4 py-6 text-center">
        <p className="text-sm font-medium text-text">
          Đây là chương mới nhất.
        </p>
        <p className="text-xs text-text-subtle mt-1">
          Chờ scanlator / nguồn upload chương kế tiếp.
        </p>
      </div>
    )
  }

  if (plan.kind === 'empty') {
    return (
      <div className="rounded-md bg-surface border border-border-soft px-4 py-5">
        <p className="text-sm font-medium text-text mb-1">
          Ch.{plan.chapter.number}
          {plan.chapter.label && (
            <span className="text-text-muted font-normal"> · {plan.chapter.label}</span>
          )}
        </p>
        <p className="text-xs text-text-subtle mb-4">
          Chương này chưa có nguồn nào. Có thể là filler chưa được scan.
        </p>
        <button
          onClick={() => history.back()}
          className={cn(
            'w-full inline-flex items-center justify-center gap-2',
            'rounded-md bg-surface-2 text-text px-4 py-2.5',
            'text-sm font-medium hover:bg-hover transition-colors cursor-pointer',
          )}
        >
          <ChevronLeft size={14} />
          Quay lại
        </button>
      </div>
    )
  }

  if (plan.kind === 'open-translation' || plan.kind === 'open-raw') {
    const ch = plan.chapter
    const v  = plan.version
    return (
      <Link
        to="/r/$workId/$numberNorm"
        params={{ workId: String(workId), numberNorm: ch.number }}
        className={cn(
          'block rounded-md bg-accent text-accent-fg px-4 py-3',
          'hover:bg-accent-strong active:scale-[0.98] transition-all',
          'shadow-sm',
        )}
      >
        <div className="flex items-center gap-3">
          <ArrowRight size={18} />
          <div className="min-w-0 flex-1">
            <p className="text-xs uppercase tracking-wide opacity-60">
              Tiếp theo
            </p>
            <p className="text-sm font-medium truncate">
              Ch.{ch.number}
              {ch.label && <span className="opacity-60"> · {ch.label}</span>}
            </p>
            <p className="text-xs opacity-60 truncate">
              {plan.kind === 'open-translation'
                ? `AI ${languageName(v.lang)}${v.sourceLang ? ` · từ ${languageName(v.sourceLang)}` : ''}`
                : `${languageName(v.lang)} · ${v.sourceName ?? 'raw'}`}
            </p>
          </div>
        </div>
      </Link>
    )
  }

  // spawn-from-source | spawn-from-any
  const ch  = plan.chapter
  const raw = plan.raw
  const srcLabel = languageName(raw.lang)
  return (
    <div className="rounded-md bg-surface border border-border-soft px-4 py-4">
      <p className="text-sm font-medium text-text mb-1">
        Ch.{ch.number}
        {ch.label && (
          <span className="text-text-muted font-normal"> · {ch.label}</span>
        )}
      </p>
      <p className="text-xs text-text-subtle mb-4">
        Chưa có bản dịch AI. Dịch từ {srcLabel}
        {raw.sourceName ? ` · ${raw.sourceName}` : ''}?
      </p>
      <button
        onClick={() => onSpawn(ch, raw)}
        className={cn(
          'w-full inline-flex items-center justify-center gap-2',
          'rounded-md bg-accent text-accent-fg px-4 py-2.5',
          'text-sm font-medium',
          'hover:bg-accent-strong active:scale-[0.98] transition-all',
          'cursor-pointer',
        )}
      >
        <Sparkles size={16} />
        Dịch ngay từ {srcLabel}
      </button>

      <div className="mt-2 flex items-center justify-center">
        <Link
          to="/r/$workId/$numberNorm"
          params={{ workId: String(workId), numberNorm: ch.number }}
          className="text-xs text-text-subtle hover:text-text-muted transition-colors inline-flex items-center gap-1"
        >
          <SkipForward size={12} />
          Đọc raw tạm
        </Link>
      </div>
    </div>
  )
}
