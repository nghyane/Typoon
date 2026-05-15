// ReaderBottomPill — view-control surface. Per-page actions only:
// chapter navigation arrows, page width, image fit, settings.
// Chapter list and source picker live on the top bar.
//
//   ┌─────────────────────────────────────────────────┐
//   │ ◀  ▶       ─●── 720px         ⛶       ⚙        │
//   └─────────────────────────────────────────────────┘
//
// Five controls grouped by gap alone (no dividers). The slider
// hides on mobile because thumb drag on a 32px wide track is
// imprecise; mobile users adjust width from the Settings sheet.

import {
  ChevronLeft, ChevronRight,
  Maximize2, Settings,
} from 'lucide-react'
import { Link } from '@tanstack/react-router'

import { cn } from '@shared/lib/cn'
import {
  PAGE_WIDTH_MIN, PAGE_WIDTH_MAX,
  useReaderSettings, type ImageFit,
} from './store'
import type { NavTarget } from './resolvers'
import { IconButton, RangeSlider } from './atoms'


interface Props {
  hidden:          boolean
  prev:            NavTarget | null
  next:            NavTarget | null
  onOpenSettings:  () => void
}


const FIT_LABEL: Record<ImageFit, string> = {
  width:  'Vừa rộng',
  height: 'Vừa cao',
  free:   'Tự do',
}


export function ReaderBottomPill({
  hidden, prev, next, onOpenSettings,
}: Props) {
  const pageWidth    = useReaderSettings((s) => s.pageWidth)
  const setPageWidth = useReaderSettings((s) => s.setPageWidth)
  const imageFit     = useReaderSettings((s) => s.imageFit)
  const setImageFit  = useReaderSettings((s) => s.setImageFit)

  const cycleFit = () => {
    setImageFit(
      imageFit === 'width'  ? 'height' :
      imageFit === 'height' ? 'free'   :
                              'width',
    )
  }

  return (
    <footer
      className={cn(
        'fixed inset-x-0 bottom-0 z-30',
        'pb-[max(0.5rem,var(--saib))]',
        'pl-[max(0.5rem,var(--sail))]',
        'pr-[max(0.5rem,var(--sair))]',
        'flex justify-center',
        'transition-transform duration-200 ease-out',
        hidden && 'translate-y-[calc(100%+var(--saib))]',
        'pointer-events-none',
      )}
    >
      <div
        className={cn(
          'pointer-events-auto',
          'flex items-center h-12 px-2',
          'rounded-full bg-surface shadow-lg',
          'border border-border-soft',
          'max-w-full',
        )}
      >
        <div className="flex items-center gap-1">
          <NavBtn target={prev} title="Chương trước" aria-label="Chương trước">
            <ChevronLeft size={16} />
          </NavBtn>
          <NavBtn target={next} title="Chương sau" aria-label="Chương sau">
            <ChevronRight size={16} />
          </NavBtn>
        </div>

        <div className="hidden sm:flex items-center gap-2 mx-3">
          <RangeSlider
            min={PAGE_WIDTH_MIN}
            max={PAGE_WIDTH_MAX}
            step={20}
            value={pageWidth}
            onChange={(e) => setPageWidth(Number(e.target.value))}
            aria-label="Bề rộng trang"
            title={`${pageWidth}px`}
            className="w-32"
          />
          <span className="text-xs tabular text-text-subtle min-w-[3.5rem] text-right">
            {pageWidth}px
          </span>
        </div>

        <div className="flex items-center gap-1 sm:ml-0 ml-2">
          <IconButton
            onClick={cycleFit}
            title={`Vừa: ${FIT_LABEL[imageFit]}`}
            aria-label="Đổi kiểu vừa khung"
          >
            <Maximize2 size={14} />
          </IconButton>
          <IconButton
            onClick={onOpenSettings}
            title="Cài đặt"
            aria-label="Cài đặt đọc"
          >
            <Settings size={14} />
          </IconButton>
        </div>
      </div>
    </footer>
  )
}


function NavBtn({
  target, title, children, ...rest
}: {
  target:   NavTarget | null
  title:    string
  children: React.ReactNode
} & React.AriaAttributes) {
  if (!target) {
    return (
      <span
        {...rest}
        className={cn(
          'shrink-0 inline-flex items-center justify-center',
          'h-9 w-9 rounded-full',
          'text-text-subtle opacity-40',
          'cursor-not-allowed',
        )}
        title={title}
        aria-disabled
      >
        {children}
      </span>
    )
  }
  return (
    <Link
      to="/r/$workId/$numberNorm"
      params={{ workId: String(target.workId), numberNorm: target.numberNorm }}
      title={title}
      {...rest}
      className={cn(
        'shrink-0 inline-flex items-center justify-center',
        'h-9 w-9 rounded-full',
        'text-text-muted hover:text-text hover:bg-hover',
        'transition-colors duration-150 cursor-pointer',
      )}
    >
      {children}
    </Link>
  )
}
