// ReaderTopBar — fixed top chrome.
//
// Layout:
//   ←  Tên truyện              [Ch.5 · 5/93 ▾] [VI · Otruyen ▾]
//
// Uses design-system tokens only: bg-surface, border-divider, h-bar,
// gap-2 / px-3, Button component for the chips. No backdrop-blur
// glass effects — the rest of the app uses solid surfaces and so
// should the reader.

import { ChevronLeft, ChevronDown, BookOpen, Languages, Sparkles } from 'lucide-react'
import { Link } from '@tanstack/react-router'

import { Button } from '@shared/ui/Button'
import { cn } from '@shared/lib/cn'
import { useWorkIdentity } from '@features/work/contexts/WorkIdentityContext'
import { useWorkChapters } from '@features/work/contexts/WorkChaptersContext'
import { useReader } from '../ReaderContext'
import { useSourcePref } from '../hooks/useSourcePref'
import { useChapterSources } from '../data/queries/useChapterSources'
import { versionKeyOf } from '../data/selectors/resolveSource'
import { pickBestVersion } from '@features/work/data/selectors/mergeChapters'


interface Props {
  onOpenChapters: () => void
  onOpenSources:  () => void
  totalPages:     number
  chapterTriggerRef: React.RefObject<HTMLButtonElement | null>
  sourceTriggerRef:  React.RefObject<HTMLButtonElement | null>
}


export function ReaderTopBar({
  onOpenChapters, onOpenSources,
  chapterTriggerRef, sourceTriggerRef,
}: Props) {
  const { work }   = useWorkIdentity()
  const { merged } = useWorkChapters()
  const { workId, chapterRef, progress, chromeVisible } = useReader()
  const pref     = useSourcePref(workId)
  const sources  = useChapterSources(workId, chapterRef)

  // Chapter position
  const idx = merged.findIndex(c => c.numberNorm === chapterRef)
  const chapterIndex = idx >= 0 ? merged.length - idx : 0
  const currentNumber = merged[idx]?.number ?? chapterRef

  // Source label for the chip
  const chapter = merged[idx] ?? null
  const sourceLabel = computeSourceLabel({
    chapter,
    pref,
    sources,
    targetLang: work.target_lang,
  })

  // Progress strip — clamped to [0, 1]
  const pct = Math.max(0, Math.min(1, progress)) * 100

  return (
    <header
      className={cn(
        'fixed top-0 inset-x-0 z-30 h-bar',
        'bg-surface border-b border-border-soft',
        'pt-[var(--sait)] pl-[var(--sail)] pr-[var(--sair)]',
        'flex items-center gap-2 px-3',
        'transition-transform duration-200 ease-out',
        !chromeVisible && '-translate-y-[calc(100%+var(--sait))]',
      )}
    >
      <Link
        to="/w/$workId"
        params={{ workId }}
        search={{ tab: undefined }}
        className={cn(
          'inline-flex items-center justify-center size-8 shrink-0',
          'rounded-sm text-text-muted',
          'hover:text-text hover:bg-hover',
          'transition-colors duration-150',
        )}
        aria-label="Quay lại"
      >
        <ChevronLeft size={16} />
      </Link>

      <Link
        to="/w/$workId"
        params={{ workId }}
        search={{ tab: undefined }}
        className="min-w-0 flex-1 text-sm font-medium text-text truncate hover:text-accent-text transition-colors"
        title={work.title}
      >
        {work.title}
      </Link>

      <Button
        ref={chapterTriggerRef}
        variant="ghost"
        size="sm"
        onClick={onOpenChapters}
        aria-label="Danh sách chương"
      >
        <BookOpen size={14} className="text-text-subtle" />
        <span className="tabular-nums">Ch.{currentNumber}</span>
        {merged.length > 0 && (
          <span className="text-xs text-text-subtle tabular-nums font-normal">
            {chapterIndex}/{merged.length}
          </span>
        )}
        <ChevronDown size={12} className="text-text-subtle" />
      </Button>

      {sourceLabel && (
        <>
          {/* Desktop chip — icon + language + source name */}
          <Button
            ref={sourceTriggerRef}
            variant="ghost"
            size="sm"
            onClick={onOpenSources}
            aria-label="Nguồn đọc"
            className="hidden sm:inline-flex"
          >
            {sourceLabel.kind === 'translated' ? (
              <Sparkles size={14} className="text-accent" />
            ) : (
              <Languages size={14} className="text-text-subtle" />
            )}
            {sourceLabel.lang && (
              <span className={cn(
                'text-xs uppercase font-semibold tabular-nums',
                sourceLabel.kind === 'translated' ? 'text-accent-text' : 'text-text-subtle',
              )}>
                {sourceLabel.lang}
              </span>
            )}
            <span className="text-text max-w-[7rem] truncate">
              {sourceLabel.name}
            </span>
            <ChevronDown size={12} className="text-text-subtle" />
          </Button>

          {/* Mobile chip — language code only */}
          <Button
            variant="ghost"
            size="sm"
            onClick={onOpenSources}
            aria-label="Nguồn đọc"
            className="sm:hidden"
          >
            {sourceLabel.kind === 'translated' ? (
              <Sparkles size={14} className="text-accent" />
            ) : (
              <Languages size={14} className="text-text-subtle" />
            )}
            <span className={cn(
              'text-xs uppercase font-semibold tabular-nums',
              sourceLabel.kind === 'translated' && 'text-accent-text',
            )}>
              {sourceLabel.lang ?? '—'}
            </span>
            <ChevronDown size={12} className="text-text-subtle" />
          </Button>
        </>
      )}

      {/* Progress strip — sits on the bottom hairline */}
      <div
        className="pointer-events-none absolute inset-x-0 bottom-0 h-px overflow-hidden"
        aria-hidden
      >
        <div
          className="h-full bg-accent transition-[width] duration-150 ease-out"
          style={{ width: `${pct}%` }}
        />
      </div>
    </header>
  )
}


// ── Source label ──────────────────────────────────────────────


interface LabelInput {
  chapter:    import('@features/work/data/types').MergedChapter | null
  pref:       import('../data/types').SourcePref
  sources:    import('../data/types').ChapterSources
  targetLang: string
}


type SourceLabel = {
  /** 'translated' = AI/Typoon bundle. 'raw' = source manifest version. */
  kind: 'translated' | 'raw'
  lang: string | null
  name: string
}


function computeSourceLabel(input: LabelInput): SourceLabel | null {
  const { chapter, pref, sources, targetLang } = input
  if (!chapter) return null

  // Helper — does this chapter actually have a translated path?
  const hasTranslated =
    sources.saved?.kind === 'translated' ||
    (sources.job && (sources.archiveLive || sources.job.archive_url))

  // Explicit translated pick — show as AI bundle if it actually resolves
  if (pref.kind === 'translated' && hasTranslated) {
    return { kind: 'translated', lang: targetLang.toUpperCase(), name: 'Bản dịch của bạn' }
  }

  // Explicit raw pick
  if (pref.kind === 'raw') {
    const v = chapter.sourceVersions.find(x => versionKeyOf(x) === pref.versionKey)
    if (v) return { kind: 'raw', lang: v.lang.toUpperCase(), name: v.source.manifest.name }
  }

  // Auto:
  //   - if a translated bundle is available, it's what the resolver
  //     will return — surface that to the user
  //   - otherwise the best raw version for target lang
  if (pref.kind === 'auto' && hasTranslated) {
    return { kind: 'translated', lang: targetLang.toUpperCase(), name: 'Bản dịch của bạn' }
  }

  const best = pickBestVersion(chapter, targetLang.toLowerCase())
  if (best) {
    return { kind: 'raw', lang: best.lang.toUpperCase(), name: best.source.manifest.name }
  }
  return null
}
