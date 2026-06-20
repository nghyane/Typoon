// ReaderSettingsSheet — minimal layout.
//
//   CHẾ ĐỘ ĐỌC
//     [📖 Tiêu chuẩn] [📕 Phải→Trái] [⬇ Dọc] [📜 Webtoon]
//     ↑ segmented radio, h-14 cards (token: list row card)
//
//   HIỂN THỊ                (desktop only)
//     Bề rộng tối đa                       1040 px
//     ●─────────────
//
// Mobile = BottomSheet, desktop = Modal (sm).

import { type LucideIcon } from 'lucide-react'
import {
  RectangleVertical, BookOpen, ScrollText, ArrowDown,
} from 'lucide-react'

import { Modal } from '@shared/ui/Modal'
import { BottomSheet } from '@shared/ui/BottomSheet'
import { cn } from '@shared/lib/cn'
import { useIsDesktop } from '@shared/lib/useMediaQuery'
import {
  useReaderSettings, usePatchReaderSettings,
  PAGE_WIDTH_MIN, PAGE_WIDTH_MAX,
  type ReadingStyle,
} from '../settings'


interface Props {
  open:    boolean
  onClose: () => void
}


export function ReaderSettingsSheet({ open, onClose }: Props) {
  const isDesktop = useIsDesktop()

  if (isDesktop) {
    return (
      <Modal open={open} onClose={onClose} title="Cài đặt đọc" size="sm">
        <Body isDesktop />
      </Modal>
    )
  }
  return (
    <BottomSheet open={open} onClose={onClose} title="Cài đặt đọc">
      <Body isDesktop={false} />
    </BottomSheet>
  )
}


// ── Body ───────────────────────────────────────────────────────


const STYLE_OPTIONS: Array<{
  value: ReadingStyle
  label: string
  icon:  LucideIcon
}> = [
  { value: 'standard', label: 'Tiêu chuẩn', icon: RectangleVertical },
  { value: 'rtl',      label: 'Phải→Trái',  icon: BookOpen },
  { value: 'vertical', label: 'Dọc',         icon: ArrowDown },
  { value: 'webtoon',  label: 'Webtoon',     icon: ScrollText },
]


function Body({ isDesktop }: { isDesktop: boolean }) {
  const s     = useReaderSettings()
  const patch = usePatchReaderSettings()

  return (
    <div className="px-4 sm:px-5 py-4 space-y-6">
      <Section title="Chế độ đọc">
        <ModeSegmented
          value={s.style}
          onChange={(v) => patch({ style: v })}
        />
      </Section>

      {isDesktop && (
        <>
          <div className="border-t border-border-soft" />
          <Section title="Hiển thị">
            <div className="space-y-3">
              <div className="flex items-baseline justify-between gap-2">
                <span className="text-sm text-text">Bề rộng tối đa</span>
                <span className="text-xs tabular-nums text-text-muted h-6 px-2 inline-flex items-center rounded-full bg-surface-2">
                  {s.pageWidth} px
                </span>
              </div>
              <input
                type="range"
                min={PAGE_WIDTH_MIN}
                max={PAGE_WIDTH_MAX}
                step={20}
                value={s.pageWidth}
                onChange={(e) => patch({ pageWidth: Number(e.target.value) })}
                className="reader-slider w-full h-1 appearance-none rounded-full bg-surface-2 cursor-pointer accent-accent"
              />
            </div>
          </Section>
        </>
      )}
    </div>
  )
}


// ── Segmented mode picker ─────────────────────────────────────


function ModeSegmented({
  value, onChange,
}: {
  value:    ReadingStyle
  onChange: (v: ReadingStyle) => void
}) {
  return (
    <div
      role="radiogroup"
      aria-label="Chế độ đọc"
      className="grid grid-cols-2 sm:grid-cols-4 gap-2"
    >
      {STYLE_OPTIONS.map(opt => {
        const Icon = opt.icon
        const active = value === opt.value
        return (
          <button
            key={opt.value}
            type="button"
            role="radio"
            aria-checked={active}
            onClick={() => onChange(opt.value)}
            className={cn(
              'inline-flex flex-col items-center justify-center gap-2',
               'h-14 px-2 rounded-sm text-xs font-medium',
              'transition-colors duration-150 cursor-pointer',
              active
                ? 'bg-accent-bg text-accent-text'
                : 'bg-surface-2 text-text-muted hover:bg-hover hover:text-text',
            )}
          >
            <Icon size={18} className={active ? 'text-accent' : ''} />
            <span className="truncate w-full text-center leading-tight">
              {opt.label}
            </span>
          </button>
        )
      })}
    </div>
  )
}


// ── Section block ─────────────────────────────────────────────


function Section({
  title, hint, children,
}: {
  title:    string
  hint?:    string
  children: React.ReactNode
}) {
  return (
    <section className="space-y-3">
      <div className="flex items-baseline justify-between gap-2">
        <h3 className="text-xs uppercase tracking-wider text-text-subtle font-semibold">
          {title}
        </h3>
        {hint && (
          <span className="text-xs text-text-subtle">{hint}</span>
        )}
      </div>
      {children}
    </section>
  )
}
