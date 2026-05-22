// ReaderBottomPill — fixed bottom chrome.
//
//   ┌────────────────────────────────────────────────┐
//   │  ◀   │   Ch.5 · 12 / 187   │   ▶   │   ⚙       │
//   └────────────────────────────────────────────────┘
//
// Solid surface, hairline border, system shadow. No backdrop blur.
// Atoms use the design system (h-8 ghost buttons, divider hairlines).

import { ChevronLeft, ChevronRight, Settings } from 'lucide-react'
import { Link } from '@tanstack/react-router'

import { Button } from '@shared/ui/Button'
import { cn } from '@shared/lib/cn'
import { useReader } from '../ReaderContext'


interface Props {
  onOpenSettings: () => void
  totalPages:     number
}


export function ReaderBottomPill({ onOpenSettings, totalPages }: Props) {
  const { workId, prev, next, page, chromeVisible } = useReader()

  return (
    <footer
      className={cn(
        'fixed inset-x-0 bottom-0 z-30',
        'pb-[max(0.75rem,var(--saib))]',
        'pl-[max(0.5rem,var(--sail))]',
        'pr-[max(0.5rem,var(--sair))]',
        'flex justify-center pointer-events-none',
        'transition-transform duration-200 ease-out',
        !chromeVisible && 'translate-y-[calc(100%+var(--saib))]',
      )}
    >
      <div
        className={cn(
          'pointer-events-auto inline-flex items-center h-12 max-w-full',
          'rounded-full bg-surface border border-border-soft',
          'shadow-[0_8px_24px_rgb(0,0,0,0.35)]',
          'gap-1 px-2',
        )}
      >
        <NavBtn target={prev} workId={workId} aria-label="Chương trước">
          <ChevronLeft size={16} />
        </NavBtn>

        <Divider />

        <PageIndicator page={page} total={totalPages} />

        <Divider />

        <NavBtn target={next} workId={workId} aria-label="Chương sau">
          <ChevronRight size={16} />
        </NavBtn>

        <Divider />

        <Button
          variant="ghost"
          size="sm"
          icon
          onClick={onOpenSettings}
          aria-label="Cài đặt đọc"
          className="rounded-full"
        >
          <Settings size={14} />
        </Button>
      </div>
    </footer>
  )
}


// ── Atoms ──────────────────────────────────────────────────────


function PageIndicator({ page, total }: { page: number; total: number }) {
  if (total <= 0) {
    return (
      <span className="px-2 text-xs text-text-subtle tabular-nums select-none">
        —
      </span>
    )
  }
  return (
    <span
      className={cn(
        'px-2 text-xs tabular-nums text-text-muted select-none',
        'inline-flex items-baseline gap-1',
      )}
      aria-label={`Trang ${page + 1} trên ${total}`}
    >
      <span className="font-semibold text-text">{page + 1}</span>
      <span className="text-text-subtle">/</span>
      <span>{total}</span>
    </span>
  )
}


function Divider() {
  return <span className="block h-5 w-px bg-divider shrink-0" aria-hidden />
}


function NavBtn({
  target, workId, children, ...rest
}: {
  target:  { ref: string } | null
  workId:  string
  children: React.ReactNode
} & React.HTMLAttributes<HTMLElement>) {
  const base = cn(
    'shrink-0 inline-flex items-center justify-center',
    'size-8 rounded-full',
    'transition-colors duration-150',
  )
  if (!target) {
    return (
      <span
        className={cn(base, 'text-text-subtle opacity-40 cursor-not-allowed')}
        aria-disabled
        {...rest}
      >
        {children}
      </span>
    )
  }
  return (
    <Link
      to="/r/$workId/$numberNorm"
      params={{ workId, numberNorm: target.ref }}
      className={cn(base, 'text-text-muted hover:text-text hover:bg-hover cursor-pointer')}
      {...rest as React.AnchorHTMLAttributes<HTMLAnchorElement>}
    >
      {children}
    </Link>
  )
}
