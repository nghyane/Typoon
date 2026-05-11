import { Link, useRouterState } from '@tanstack/react-router'
import { Library, Compass, Settings } from 'lucide-react'
import { cn } from '../shared/lib/cn'

const ITEMS = [
  { to: '/library',  label: 'Thư viện', icon: Library  },
  { to: '/browse',   label: 'Duyệt',    icon: Compass  },
  { to: '/settings', label: 'Cài đặt',  icon: Settings },
] as const

export function BottomNav() {
  const { location } = useRouterState()
  const isActive = (to: string) => location.pathname.startsWith(to)

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
      {ITEMS.map(({ to, label, icon: Icon }) => {
        const active = isActive(to)
        return (
          <Link
            key={to}
            to={to}
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
