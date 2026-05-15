// Reader atoms — 1 mini design system. Every chip / button / chip-
// like element across the reader (SourcePicker, ChapterListSheet,
// BottomPill, TopBar) imports from here so visual drift between
// surfaces is impossible.
//
// Rules:
//   - 2 button sizes: sm (h-7) and md (h-9). No h-8 / h-10.
//   - 2 chip sizes:   xs (h-4) inline, sm (h-5) standalone.
//   - 2 elevations:   `surface` (panels) and `floating` (pill / sheet).
//   - 1 transition:   colors-only, 150ms.
//   - Radius:         rounded-md for cards, rounded-sm for chips,
//                     rounded-full for pill / pill-button.
//   - Colors:         only design tokens. No arbitrary opacity.
//                     `bg-row-active` for active state (token).
//                     `bg-hover` for hover (token).

import { forwardRef } from 'react'
import { Sparkles } from 'lucide-react'

import { cn } from '@shared/lib/cn'


// ── Chips ─────────────────────────────────────────────────────


/** Compact language chip. Surface tone by default; accent TEXT
 *  only when the row this chip lives in is the "active" pick —
 *  background stays neutral so it doesn't compound with the row's
 *  active surface elevation. */
export function LangChip({
  lang, active = false,
}: {
  lang:    string
  active?: boolean
}) {
  return (
    <span
      className={cn(
        'shrink-0 inline-flex items-center justify-center',
        'h-5 px-1.5 rounded-sm',
        'text-xs uppercase tabular font-medium',
        'bg-surface-2',
        active ? 'text-accent' : 'text-text-muted',
      )}
    >
      {lang}
    </span>
  )
}


/** AI marker chip — paired with a LangChip on translation rows so
 *  the user can tell AI-rendered apart from human scanlation at a
 *  glance. Info-tone palette (Discord-blue) keeps it distinct from
 *  the surface lang chip without competing with accent CTAs. */
export function AiChip({ active = false }: { active?: boolean }) {
  return (
    <span
      className={cn(
        'shrink-0 inline-flex items-center gap-0.5',
        'h-5 px-1.5 rounded-sm',
        'text-xs uppercase tabular font-medium',
        'bg-info-bg',
        active ? 'text-accent' : 'text-info-text',
      )}
      title="Bản dịch AI"
    >
      <Sparkles size={9} />
      AI
    </span>
  )
}


/** State indicator chip used in the chapter list dropdown. Each
 *  chapter row carries one StateChip on the right summarising the
 *  default version's readiness:
 *
 *    `done`     — readable now (raw target lang OR AI done).
 *    `running`  — AI mid-pipeline.
 *    `pending`  — queued.
 *    `error`    — AI failed, needs retry.
 *    `raw-only` — only non-target raws exist; user can spawn.
 *    `none`     — nothing readable at all (filler).
 *
 *  Compact (h-5) so it doesn't push chapter row heights past the
 *  density target. Colors map to design tokens so the visual is
 *  consistent with the rest of the reader's state system. */
export type ChapterState =
  | 'done' | 'running' | 'pending' | 'error' | 'raw-only' | 'none'


const STATE_STYLE: Record<ChapterState, string> = {
  done:     'bg-surface-2 text-text-muted',
  running:  'bg-info-bg text-info-text',
  pending:  'bg-info-bg text-info-text',
  error:    'bg-error-bg text-error-text',
  'raw-only': 'bg-warning-bg text-warning-text',
  none:     'bg-surface-2 text-text-subtle',
}


export function StateChip({
  state, label,
}: {
  state: ChapterState
  label: string
}) {
  return (
    <span
      className={cn(
        'shrink-0 inline-flex items-center justify-center',
        'h-5 px-1.5 rounded-sm',
        'text-xs uppercase tabular font-medium tracking-wide',
        STATE_STYLE[state],
      )}
    >
      {label}
    </span>
  )
}


// ── Buttons ───────────────────────────────────────────────────


type IconBtnVariant = 'ghost' | 'soft'

export const IconButton = forwardRef<
  HTMLButtonElement,
  React.ButtonHTMLAttributes<HTMLButtonElement> & {
    variant?: IconBtnVariant
  }
>(function IconButton({ variant = 'ghost', className, ...rest }, ref) {
  return (
    <button
      ref={ref}
      {...rest}
      className={cn(
        'shrink-0 inline-flex items-center justify-center',
        'h-9 w-9 rounded-full',
        'transition-colors duration-150',
        'disabled:opacity-30 disabled:cursor-not-allowed',
        'cursor-pointer',
        variant === 'ghost'
          ? 'text-text-muted hover:text-text hover:bg-hover'
          : 'bg-surface-2 text-text hover:bg-hover',
        className,
      )}
    />
  )
})


/** A small "pill button" that hosts text + optional icon. Same
 *  height as IconButton so the bottom pill rhythm stays uniform. */
export const PillButton = forwardRef<
  HTMLButtonElement,
  React.ButtonHTMLAttributes<HTMLButtonElement> & {
    variant?: 'ghost' | 'soft' | 'accent'
  }
>(function PillButton({ variant = 'ghost', className, ...rest }, ref) {
  return (
    <button
      ref={ref}
      {...rest}
      className={cn(
        'shrink-0 inline-flex items-center gap-1.5',
        'h-9 px-3 rounded-full',
        'text-sm font-medium',
        'transition-colors duration-150',
        'cursor-pointer',
        variant === 'ghost'
          ? 'text-text hover:bg-hover'
          : variant === 'soft'
            ? 'bg-surface-2 text-text hover:bg-hover'
            : 'bg-accent text-accent-fg hover:bg-accent-strong',
        className,
      )}
    />
  )
})


/** Compact action button used inside row lists (SourcePicker rows,
 *  ChapterListSheet inline actions). Smaller than PillButton so it
 *  doesn't dominate the row height. */
export const RowButton = forwardRef<
  HTMLButtonElement,
  React.ButtonHTMLAttributes<HTMLButtonElement> & {
    variant?: 'soft' | 'accent'
  }
>(function RowButton({ variant = 'soft', className, ...rest }, ref) {
  return (
    <button
      ref={ref}
      {...rest}
      className={cn(
        'shrink-0 inline-flex items-center gap-1',
        'h-7 px-2.5 rounded-md',
        'text-xs font-medium',
        'transition-colors duration-150',
        'disabled:opacity-30 disabled:cursor-not-allowed',
        'cursor-pointer',
        variant === 'soft'
          ? 'bg-surface-2 text-text hover:bg-hover'
          : 'bg-accent text-accent-fg hover:bg-accent-strong',
        className,
      )}
    />
  )
})


// ── Slider ────────────────────────────────────────────────────


/** Native range slider styled to match the reader theme. The
 *  default chrome thumb on dark surface looks alien; we override
 *  via ::-webkit-slider-thumb in `index.css`. Kept as a styled
 *  primitive so every slider (page width, page index, preload
 *  count) shares the same dimensions + accent. */
export function RangeSlider({
  className, ...rest
}: React.InputHTMLAttributes<HTMLInputElement>) {
  return (
    <input
      type="range"
      {...rest}
      className={cn(
        'reader-slider',
        'w-full h-1 appearance-none rounded-full bg-surface-2',
        'cursor-pointer accent-accent',
        'disabled:opacity-30 disabled:cursor-not-allowed',
        className,
      )}
    />
  )
}


// ── Pill divider ──────────────────────────────────────────────


// (no PillDivider — clusters in the bottom pill use spacing alone;
// hairline dividers on dark elevated surfaces tend to look like
// stray artifacts. Keep grouping rhythmic with `gap-1` / `mx-3`.)
