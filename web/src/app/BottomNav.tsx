import { Link, useRouterState } from '@tanstack/react-router'
import { FolderOpen, Library, Compass, Settings } from 'lucide-react'
import { cn } from '../shared/lib/cn'

const ITEMS = [
  { to: '/projects', label: 'Của tôi',   icon: FolderOpen, filter: 'mine'      },
  { to: '/library',  label: 'Thư viện',  icon: Library,    filter: undefined   },
  { to: '/browse',   label: 'Duyệt',     icon: Compass,    filter: undefined   },
  { to: '/settings', label: 'Cài đặt',   icon: Settings,   filter: undefined   },
] as const

export function BottomNav() {
  const { location } = useRouterState()

  const isActive = (to: string, filter?: string) => {
    if (to !== '/projects') return location.pathname.startsWith(to)
    const onProjects =
      location.pathname === '/projects' || location.pathname.startsWith('/projects/')
    if (!onProjects) return false
    const current = (location.search as { filter?: string })?.filter ?? 'mine'
    return current === (filter ?? 'mine')
  }

  return (
    <nav
      className={cn(
        'sm:hidden flex items-stretch bg-surface border-t border-border shrink-0',
        // Bar grows by the iOS home-indicator inset so its surface
        // color reaches the bottom of the viewport (no `bg-bg` gap
        // showing through under the bar). Tap targets stay inside
        // the visible 3.5rem via padding.
        'h-[calc(3.5rem+var(--saib))] pb-[var(--saib)]',
      )}
    >
      {ITEMS.map(({ to, label, icon: Icon, filter }) => {
        const active = isActive(to, filter)
        return (
          <Link
            key={`${to}:${filter ?? ''}`}
            to={to}
            search={filter ? { filter } as never : undefined}
            className={cn(
              'flex-1 flex flex-col items-center justify-center gap-0.5 text-[10px] transition-colors',
              active ? 'text-accent-text' : 'text-text-subtle',
            )}
          >
            <Icon size={18} />
            <span>{label}</span>
          </Link>
        )
      })}
    </nav>
  )
}
