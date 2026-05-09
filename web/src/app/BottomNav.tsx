import { Link, useRouterState } from '@tanstack/react-router'
import { FolderOpen, Star, Globe, Settings } from 'lucide-react'
import { cn } from '../shared/lib/cn'

const ITEMS = [
  { to: '/projects', label: 'Của tôi',   icon: FolderOpen, filter: 'mine'      },
  { to: '/projects', label: 'Đã lưu',    icon: Star,       filter: 'pinned'    },
  { to: '/projects', label: 'Cộng đồng', icon: Globe,      filter: 'community' },
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
    <nav className="sm:hidden flex items-stretch h-14 bg-surface border-t border-border shrink-0">
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
