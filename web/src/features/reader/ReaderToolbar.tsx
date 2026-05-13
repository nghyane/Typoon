// Sticky reader header. Auto-hides on scroll-down past 80px, returns
// on scroll-up. Hosts breadcrumb (manga title), chapter number, mode
// toggle (cuộn/trang) and prev/next chapter buttons.
//
// Navigation targets are pre-resolved `ReaderNavTarget` values — the
// toolbar just builds a `<Link>` URL from them, no source-kind
// branching here.

import { useEffect, useState } from 'react'
import { ChevronLeft, ChevronRight } from 'lucide-react'
import { Link } from '@tanstack/react-router'
import { Button } from '@shared/ui/Button'
import { cn } from '@shared/lib/cn'
import type { ReaderNavTarget } from './types'

export type ViewMode = 'continuous' | 'single'

interface Props {
  workTitle:    string
  chapterText:  string
  chapterSub:   string | null
  prev:         ReaderNavTarget | null
  next:         ReaderNavTarget | null
  page:         number
  totalPages:   number
  mode:         ViewMode
  onModeChange: (m: ViewMode) => void
  onBack:       () => void
}

export function ReaderToolbar({
  workTitle, chapterText, chapterSub,
  prev, next, page, totalPages, mode, onModeChange,
  onBack,
}: Props) {
  const [hidden, setHidden] = useState(false)

  useEffect(() => {
    let lastY = window.scrollY
    let raf: number | null = null
    const onScroll = () => {
      if (raf !== null) return
      raf = requestAnimationFrame(() => {
        const y = window.scrollY
        const dy = y - lastY
        if (Math.abs(dy) > 8) {
          setHidden(dy > 0 && y > 80)
          lastY = y
        }
        raf = null
      })
    }
    window.addEventListener('scroll', onScroll, { passive: true })
    return () => {
      window.removeEventListener('scroll', onScroll)
      if (raf !== null) cancelAnimationFrame(raf)
    }
  }, [])

  return (
    <header
      className={cn(
        'sticky top-0 z-20 bg-bg/85 backdrop-blur-md border-b border-border-soft',
        'pt-[var(--sait)]',
        'transition-transform duration-200',
        hidden && '-translate-y-[calc(100%+var(--sait))]',
      )}
    >
      <div
        className={cn(
          'flex items-center gap-3 h-bar',
          'pl-[max(1.25rem,var(--sail))]',
          'pr-[max(1.25rem,var(--sair))]',
        )}
      >
        <div className="flex items-center gap-2 text-sm min-w-0 flex-1">
          <button
            onClick={onBack}
            className="text-text-subtle hover:text-text truncate transition-colors cursor-pointer"
          >
            {workTitle}
          </button>
          <span className="text-text-subtle/60">/</span>
          <span className="text-text font-medium tabular shrink-0">
            {chapterText}
          </span>
          {chapterSub && (
            <span className="text-text-muted truncate">{chapterSub}</span>
          )}
        </div>

        <div className="flex items-center gap-1 shrink-0">
          <div className="inline-flex items-center bg-surface-2 rounded-sm p-0.5 mr-1">
            <button
              onClick={() => onModeChange('continuous')}
              className={cn(
                'h-6 px-2 text-xs rounded-xs cursor-pointer transition-colors',
                mode === 'continuous'
                  ? 'bg-surface text-text font-medium'
                  : 'text-text-muted hover:text-text',
              )}
            >
              Cuộn
            </button>
            <button
              onClick={() => onModeChange('single')}
              className={cn(
                'h-6 px-2 text-xs rounded-xs cursor-pointer transition-colors',
                mode === 'single'
                  ? 'bg-surface text-text font-medium'
                  : 'text-text-muted hover:text-text',
              )}
            >
              Trang
            </button>
          </div>

          {mode === 'single' && (
            <span className="text-xs text-text-subtle tabular px-2">
              <span className="text-text-muted">{page + 1}</span>
              <span className="opacity-50">/</span>
              {totalPages}
            </span>
          )}

          <NavLink target={prev} title="Chương trước">
            <ChevronLeft size={14} />
          </NavLink>
          <NavLink target={next} title="Chương sau">
            <ChevronRight size={14} />
          </NavLink>
        </div>
      </div>
    </header>
  )
}


function NavLink({
  target, title, children,
}: {
  target:   ReaderNavTarget | null
  title:    string
  children: React.ReactNode
}) {
  if (!target) {
    return (
      <Button variant="ghost" size="sm" icon disabled title={title}>
        {children}
      </Button>
    )
  }
  return (
    <Link
      to="/r/$workId/$numberNorm"
      params={{
        workId:     String(target.workId),
        numberNorm: target.numberNorm,
      }}
      search={{
        lang: target.lang,
        src:  target.src,
      }}
      title={title}
    >
      <Button variant="ghost" size="sm" icon>
        {children}
      </Button>
    </Link>
  )
}
