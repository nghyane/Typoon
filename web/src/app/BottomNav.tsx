import { Link, useRouterState } from '@tanstack/react-router'
import { Home, Library, Compass, Settings, type LucideIcon } from 'lucide-react'
import { cn } from '../shared/lib/cn'

// =============================================================================
// BottomNav — mobile shell mirroring Sidebar. Mobile-only (sm:hidden).
// =============================================================================

interface Item {
  to:    string
  label: string
  icon:  LucideIcon
}

const ITEMS: Item[] = [
  { to: '/',         label: 'Nhà',      icon: Home     },
  { to: '/library',  label: 'Thư viện', icon: Library  },
  { to: '/explore',  label: 'Khám phá', icon: Compass  },
  { to: '/settings', label: 'Cài đặt',  icon: Settings },
]

export function BottomNav() {
  const { location } = useRouterState()
  const isActive = (to: string) =>
    to === '/' ? location.pathname === '/' : location.pathname.startsWith(to)

  return (
    <nav
      className={cn(
        'sm:hidden flex items-stretch bg-surface border-t border-border-soft shrink-0',
        // Bar grows by the iOS home-indicator inset so its surface
        // color reaches the bottom of the viewport (no `bg-bg` gap
        // showing through under the bar). Tap targets stay inside
        // the visible 3.5rem via padding.
        'h-[calc(3.5rem+var(--saib))] pb-[var(--saib)]',
      )}
    >
      {ITEMS.map(({ to, label, icon: Icon }) => {
        const active = isActive(to)
        return (
          <Link
            key={to}
            to={to}
            className={cn(
              'flex-1 flex flex-col items-center justify-center gap-0.5 text-xs transition-colors',
              active ? 'text-accent-text' : 'text-text-subtle hover:text-text',
            )}
          >
            <span className={cn(
              'h-6 min-w-9 px-2 rounded-full inline-flex items-center justify-center',
              active && 'bg-accent-bg',
            )}>
              <Icon size={17} />
            </span>
            <span>{label}</span>
          </Link>
        )
      })}
    </nav>
  )
}
