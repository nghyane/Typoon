import { Search, Bell, ChevronLeft } from 'lucide-react'
import { Link, useRouterState } from '@tanstack/react-router'

export function TopBar() {
  const pathname = useRouterState({ select: (s) => s.location.pathname })
  const inProject = pathname.startsWith('/projects/') && pathname !== '/projects'

  return (
    <header
      className="flex items-center gap-3 px-4 h-14 shrink-0"
      style={{ borderBottom: '1px solid var(--color-border-subtle)', background: 'var(--color-bg)' }}
    >
      {/* Left — breadcrumb */}
      <div className="flex items-center gap-1.5 min-w-[120px]">
        {inProject ? (
          <Link
            to="/projects"
            className="flex items-center gap-1 text-sm text-(--color-text-2) hover:text-(--color-text-1) transition-colors"
          >
            <ChevronLeft size={14} />
            Dự án
          </Link>
        ) : (
          <span className="text-sm font-semibold text-(--color-text-1)">
            {pathname === '/projects' ? 'Dự án' : ''}
          </span>
        )}
      </div>

      {/* Center — search */}
      <div className="flex-1 flex justify-center">
        <div
          className="flex items-center gap-2 h-8 px-3 rounded-lg w-64"
          style={{ background: 'var(--color-surface)', border: '1px solid var(--color-border-subtle)' }}
        >
          <Search size={13} className="text-(--color-text-3) shrink-0" />
          <input
            placeholder="Tìm nhanh..."
            className="flex-1 text-sm bg-transparent outline-none placeholder:text-(--color-text-3) text-(--color-text-1)"
          />
          <kbd
            className="text-[10px] font-mono text-(--color-text-3) px-1 py-px rounded"
            style={{ background: 'var(--color-border-subtle)' }}
          >
            ⌘K
          </kbd>
        </div>
      </div>

      {/* Right */}
      <div className="flex items-center gap-1.5 min-w-[120px] justify-end">
        <button className="w-8 h-8 flex items-center justify-center rounded-lg text-(--color-text-2) hover:bg-(--color-surface) transition-colors">
          <Bell size={16} />
        </button>

        {/* Avatar — iOS style circular */}
        <div
          className="w-8 h-8 rounded-full flex items-center justify-center text-sm font-semibold text-white cursor-pointer overflow-hidden"
          style={{ background: 'linear-gradient(135deg, #636366, #48484a)' }}
        >
          U
        </div>
      </div>
    </header>
  )
}
