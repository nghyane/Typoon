import { Link, useRouterState } from '@tanstack/react-router'
import {
  LayoutDashboard, FolderOpen, Library,
  BookText, Users, BarChart2, Settings,
} from 'lucide-react'
import { cn } from '../../lib/cn'

const NAV = [
  { to: '/overview', icon: LayoutDashboard, label: 'Tổng quan' },
  { to: '/projects', icon: FolderOpen,      label: 'Dự án' },
  { to: '/library',  icon: Library,         label: 'Thư viện' },
  { to: '/glossary', icon: BookText,        label: 'Thuật ngữ' },
  { to: '/teams',    icon: Users,           label: 'Nhóm' },
  { to: '/reports',  icon: BarChart2,       label: 'Báo cáo' },
  { to: '/settings', icon: Settings,        label: 'Cài đặt' },
]

function NavItem({ to, icon: Icon, label }: { to: string; icon: React.ElementType; label: string }) {
  const pathname = useRouterState({ select: (s) => s.location.pathname })
  const isActive = pathname === to || (to !== '/' && pathname.startsWith(to))

  return (
    <Link
      to={to}
      className={cn(
        'flex items-center gap-3 px-3 py-2.5 rounded-xl text-sm transition-colors select-none',
        isActive
          ? 'bg-(--color-nav-active-bg) text-(--color-nav-active-text) font-semibold'
          : 'text-(--color-text-1) hover:bg-(--color-nav-hover-bg)',
      )}
    >
      <Icon
        size={20}
        strokeWidth={isActive ? 2 : 1.5}
        className={isActive ? 'text-(--color-text-1)' : 'text-(--color-text-3)'}
      />
      <span>{label}</span>
    </Link>
  )
}

export function Sidebar() {
  return (
    <aside
      className="flex flex-col shrink-0 h-full bg-(--color-bg) pt-4"
      style={{ width: 'var(--sidebar-width)', borderRight: '1px solid var(--color-border-subtle)' }}
    >
      <nav className="px-2 flex flex-col gap-0.5">
        {NAV.map((item) => <NavItem key={item.to} {...item} />)}
      </nav>
    </aside>
  )
}
