import { useEffect, useState } from 'react'
import { ChevronLeft, ChevronRight } from 'lucide-react'
import { Link } from '@tanstack/react-router'
import { Button } from '@shared/ui/Button'
import { cn } from '@shared/lib/cn'

// =============================================================================
// ReaderToolbar — sticky top bar inside the reader route.
// Breadcrumb back to the project, chapter title, prev/next chapter,
// view mode (continuous vs single-page).
//
// Auto-hides on scroll down, reappears on scroll up — keeps the page
// content unobstructed for reading. Sticky CSS only, no js scroll listener
// for hide/show beyond a 200ms threshold.
// =============================================================================

export type ViewMode = 'continuous' | 'single'

interface Props {
  projectId:     number
  projectTitle:  string
  chapterNumber: string
  chapterTitle?: string | null
  prevId:        number | null
  nextId:        number | null
  page:          number
  totalPages:    number
  mode:          ViewMode
  onModeChange:  (m: ViewMode) => void
}

export function ReaderToolbar({
  projectId, projectTitle, chapterNumber, chapterTitle,
  prevId, nextId, page, totalPages, mode, onModeChange,
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
        {/* breadcrumb */}
        <div className="flex items-center gap-2 text-[13px] min-w-0 flex-1">
          <Link
            to="/projects/$projectId"
            params={{ projectId: String(projectId) }}
            className="text-text-subtle hover:text-text truncate transition-colors"
          >
            {projectTitle}
          </Link>
          <span className="text-text-subtle/60">/</span>
          <span className="text-text font-medium tabular shrink-0">
            Ch.{chapterNumber}
          </span>
          {chapterTitle && (
            <span className="text-text-muted truncate">{chapterTitle}</span>
          )}
        </div>

        <div className="flex items-center gap-1 shrink-0">
          {/* view mode toggle */}
          <div className="inline-flex items-center bg-surface-2 rounded-sm p-0.5 mr-1">
            <button
              onClick={() => onModeChange('continuous')}
              className={cn(
                'h-6 px-2 text-[11px] rounded-xs cursor-pointer transition-colors',
                mode === 'continuous' ? 'bg-surface text-text font-medium' : 'text-text-muted hover:text-text',
              )}
            >
              Cuộn
            </button>
            <button
              onClick={() => onModeChange('single')}
              className={cn(
                'h-6 px-2 text-[11px] rounded-xs cursor-pointer transition-colors',
                mode === 'single' ? 'bg-surface text-text font-medium' : 'text-text-muted hover:text-text',
              )}
            >
              Trang
            </button>
          </div>

          {/* page counter (single mode) */}
          {mode === 'single' && (
            <span className="text-xs text-text-subtle tabular px-2">
              <span className="text-text-muted">{page + 1}</span>
              <span className="opacity-50">/</span>
              {totalPages}
            </span>
          )}

          {/* prev / next chapter */}
          {prevId !== null ? (
            <Link
              to="/projects/$projectId/chapters/$chapterId"
              params={{ projectId: String(projectId), chapterId: String(prevId) }}
              title="Chương trước"
            >
              <Button variant="ghost" size="sm" icon>
                <ChevronLeft size={14} />
              </Button>
            </Link>
          ) : (
            <Button variant="ghost" size="sm" icon disabled title="Chương trước">
              <ChevronLeft size={14} />
            </Button>
          )}
          {nextId !== null ? (
            <Link
              to="/projects/$projectId/chapters/$chapterId"
              params={{ projectId: String(projectId), chapterId: String(nextId) }}
              title="Chương sau"
            >
              <Button variant="ghost" size="sm" icon>
                <ChevronRight size={14} />
              </Button>
            </Link>
          ) : (
            <Button variant="ghost" size="sm" icon disabled title="Chương sau">
              <ChevronRight size={14} />
            </Button>
          )}
        </div>
      </div>
    </header>
  )
}
