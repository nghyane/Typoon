import { Link, useRouterState } from '@tanstack/react-router'
import { BookOpen, Activity, Settings } from 'lucide-react'
import { cn } from '../../lib/cn'

interface NavItem {
  to: string
  icon: React.ReactNode
  label: string
}

const NAV: NavItem[] = [
  { to: '/projects', icon: <BookOpen size={16} />,  label: 'Projects' },
  { to: '/pipeline', icon: <Activity  size={16} />, label: 'Pipeline' },
]

function NavItem({ to, icon, label }: NavItem) {
  const pathname = useRouterState({ select: (s) => s.location.pathname })
  const isActive = pathname.startsWith(to)

  return (
    <Link
      to={to}
      className={cn(
        'flex items-center gap-2.5 px-3 py-2 rounded-lg text-sm transition-colors',
        isActive ? 'font-medium' : 'hover:opacity-80',
      )}
      style={{
        background: isActive ? 'var(--color-accent-muted)' : 'transparent',
        color: isActive ? 'var(--color-accent-hover)' : 'var(--color-text-2)',
      }}
    >
      {icon}
      <span>{label}</span>
    </Link>
  )
}

export function Sidebar() {
  return (
    <aside
      className="flex flex-col shrink-0 border-r"
      style={{
        width: 'var(--sidebar-width)',
        background: 'var(--color-surface-1)',
        borderColor: 'var(--color-border)',
      }}
    >
      {/* Logo */}
      <div
        className="flex items-center gap-2.5 px-4 h-14 border-b shrink-0"
        style={{ borderColor: 'var(--color-border)' }}
      >
        <div
          className="w-7 h-7 rounded-md flex items-center justify-center text-xs font-bold shrink-0"
          style={{ background: 'var(--color-accent)', color: '#fff' }}
        >
          T
        </div>
        <div className="min-w-0">
          <div className="text-sm font-semibold leading-tight truncate" style={{ color: 'var(--color-text-1)' }}>
            Typoon
          </div>
          <div className="text-[10px] leading-tight tracking-wide uppercase" style={{ color: 'var(--color-text-3)' }}>
            Manga Translation
          </div>
        </div>
      </div>

      {/* Nav */}
      <nav className="flex-1 px-2 py-3 flex flex-col gap-0.5">
        {NAV.map((item) => (
          <NavItem key={item.to} {...item} />
        ))}
      </nav>

      {/* Settings */}
      <div className="px-2 pb-4 border-t pt-2" style={{ borderColor: 'var(--color-border)' }}>
        <NavItem to="/settings" icon={<Settings size={16} />} label="Settings" />
      </div>
    </aside>
  )
}
