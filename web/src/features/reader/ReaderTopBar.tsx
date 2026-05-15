// ReaderTopBar — context strip. Three logical zones:
//
//   ┌─────────────────────────────────────────────────────────┐
//   │ ←   Work Title (truncate)        Ch.5/93 ▾   [VI][AI] ▾│
//   └─────────────────────────────────────────────────────────┘
//        back        info                action       action
//
// The two ▾ triggers each anchor a popover (desktop) or bottom
// sheet (mobile). Triggers expose their DOM nodes via refs so
// the parent can position the popovers against them.
//
// Visibility driven by the parent so this bar peeks / hides in
// lockstep with the bottom pill — the chrome behaves as one layer.

import { forwardRef } from 'react'
import { ChevronLeft, ChevronDown } from 'lucide-react'

import { cn } from '@shared/lib/cn'
import { LangChip, AiChip, IconButton } from './atoms'
import type { HubVersion } from '@features/title/mergeChapters'


interface Props {
  hidden:           boolean
  workTitle:        string
  /** "Ch.5" — the position string for the chapter chip. */
  chapterLabel:     string
  totalChapters:    number
  chapterIndex:     number          // 1-based
  /** Currently-rendering version. Drives the source chip's
   *  inline lang/AI/source-name display so the user always
   *  sees what they're reading from without opening the menu. */
  picked:           HubVersion | null
  /** Refs to the trigger buttons so the parent can anchor
   *  popovers against them. */
  chapterTriggerRef: React.RefObject<HTMLButtonElement | null>
  sourceTriggerRef:  React.RefObject<HTMLButtonElement | null>
  onBackWork:       () => void
  onOpenChapters:   () => void
  onOpenSources:    () => void
}


export function ReaderTopBar({
  hidden, workTitle, chapterLabel, totalChapters, chapterIndex,
  picked, chapterTriggerRef, sourceTriggerRef,
  onBackWork, onOpenChapters, onOpenSources,
}: Props) {
  return (
    <header
      className={cn(
        'fixed top-0 inset-x-0 z-30',
        'bg-bg/80 backdrop-blur-md border-b border-border-soft',
        'pt-[var(--sait)]',
        'transition-transform duration-200 ease-out',
        hidden && '-translate-y-[calc(100%+var(--sait))]',
      )}
    >
      <div
        className={cn(
          'flex items-center gap-2 h-bar',
          'pl-[max(0.5rem,var(--sail))]',
          'pr-[max(0.5rem,var(--sair))]',
        )}
      >
        <IconButton
          onClick={onBackWork}
          aria-label="Quay lại work"
          title="Quay lại work"
        >
          <ChevronLeft size={16} />
        </IconButton>

        <button
          onClick={onBackWork}
          className={cn(
            'min-w-0 flex-1 text-left',
            'text-sm font-medium text-text truncate',
            'hover:text-accent-text transition-colors duration-150 cursor-pointer',
          )}
          title={workTitle}
        >
          {workTitle}
        </button>

        <ChapterTrigger
          ref={chapterTriggerRef}
          label={chapterLabel}
          totalChapters={totalChapters}
          chapterIndex={chapterIndex}
          onClick={onOpenChapters}
        />

        <SourceTrigger
          ref={sourceTriggerRef}
          picked={picked}
          onClick={onOpenSources}
        />
      </div>
    </header>
  )
}


// ── Triggers ──────────────────────────────────────────────────


/** Chapter chip — tabular position display, opens the chapter
 *  picker dropdown. */
const ChapterTrigger = forwardRef<
  HTMLButtonElement,
  {
    label:         string
    totalChapters: number
    chapterIndex:  number
    onClick:       () => void
  }
>(function ChapterTrigger({ label, totalChapters, chapterIndex, onClick }, ref) {
  return (
    <button
      ref={ref}
      onClick={onClick}
      className={cn(
        'shrink-0 inline-flex items-center gap-1',
        'h-8 px-2.5 rounded-md',
        'text-sm font-medium text-text',
        'hover:bg-hover transition-colors duration-150 cursor-pointer',
      )}
      title="Danh sách chương"
      aria-label="Danh sách chương"
    >
      <span className="tabular">{label || 'Ch.?'}</span>
      {totalChapters > 0 && (
        <span className="text-xs text-text-subtle tabular font-normal">
          {chapterIndex}/{totalChapters}
        </span>
      )}
      <ChevronDown size={12} className="text-text-subtle" />
    </button>
  )
})


/** Source chip — lang + AI marker + source-name token, opens the
 *  source picker dropdown. Loading state (`picked === null`)
 *  renders a placeholder so the bar's right edge doesn't reflow
 *  when pages land. */
const SourceTrigger = forwardRef<
  HTMLButtonElement,
  {
    picked:  HubVersion | null
    onClick: () => void
  }
>(function SourceTrigger({ picked, onClick }, ref) {
  if (!picked) {
    return (
      <button
        ref={ref}
        onClick={onClick}
        className={cn(
          'shrink-0 inline-flex items-center gap-1',
          'h-8 px-2 rounded-md',
          'text-xs text-text-subtle',
          'hover:bg-hover transition-colors duration-150 cursor-pointer',
        )}
        aria-label="Chọn nguồn đọc"
      >
        Nguồn
        <ChevronDown size={12} />
      </button>
    )
  }

  const isTranslation = picked.kind === 'translation'
  const sub = isTranslation
    ? (picked.sourceLang ? `từ ${picked.sourceLang.toUpperCase()}` : null)
    : (picked.sourceName ?? null)

  return (
    <button
      ref={ref}
      onClick={onClick}
      className={cn(
        'shrink-0 inline-flex items-center gap-1.5',
        'h-8 px-2 rounded-md',
        'hover:bg-hover transition-colors duration-150 cursor-pointer',
      )}
      title="Đổi nguồn đọc"
      aria-label="Đổi nguồn đọc"
    >
      <LangChip lang={picked.lang} />
      {isTranslation && <AiChip />}
      {sub && (
        <span
          className={cn(
            'text-xs text-text-muted font-medium',
            'max-w-[7rem] truncate',
            'hidden sm:inline',
          )}
        >
          {sub}
        </span>
      )}
      <ChevronDown size={12} className="text-text-subtle" />
    </button>
  )
})
