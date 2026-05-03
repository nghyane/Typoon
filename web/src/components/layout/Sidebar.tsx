import { Link, useRouterState } from '@tanstack/react-router'
import {
  LayoutDashboard, FolderOpen, Library,
  BookText, Users, BarChart2, Settings,
} from 'lucide-react'

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
  const pathname = useRouterState({ select: s => s.location.pathname })
  const active = pathname === to || (to !== '/' && pathname.startsWith(to))
  return (
    <Link
      to={to}
      className={[
        'flex items-center gap-3 px-3 py-2.5 rounded-xl text-sm transition-colors select-none',
        active
          ? 'bg-(--color-nav-active) text-(--color-text) font-semibold'
          : 'text-(--color-text-2) hover:bg-(--color-surface) hover:text-(--color-text)',
      ].join(' ')}
    >
      <Icon size={20} strokeWidth={active ? 2 : 1.5} className={active ? 'text-(--color-text)' : 'text-(--color-text-3)'} />
      <span>{label}</span>
    </Link>
  )
}

export function Sidebar() {
  return (
    <aside className="flex flex-col shrink-0 h-full w-[var(--sidebar-width)] bg-(--color-bg) border-r border-(--color-border) pt-3">
      {/* Logo */}
      <div className="flex items-center gap-2 px-4 pb-3 mb-1">
        <div className="w-7 h-7 rounded-lg bg-(--color-text) flex items-center justify-center shrink-0">
          <span className="text-xs font-bold text-white">T</span>
        </div>
        <span className="text-sm font-bold text-(--color-text)">Typoon</span>
        <button className="ml-auto text-(--color-text-3) hover:text-(--color-text-2) transition-colors">
          <svg width="14" height="14" viewBox="0 0 16 16" fill="none">
            <path d="M2 3.5h5M2 8h5M2 12.5h5M9 3.5h5M9 8h5M9 12.5h5" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round"/>
          </svg>
        </button>
      </div>

      <nav className="px-2 flex flex-col gap-0.5 flex-1">
        {NAV.map(item => <NavItem key={item.to} {...item} />)}
      </nav>

      {/* Recent projects — section */}
      <div className="px-4 py-3 border-t border-(--color-border)">
        <p className="text-[10px] font-semibold uppercase tracking-wider text-(--color-text-3) mb-2">Gần đây</p>
        {/* populated by parent if needed */}
      </div>

      {/* Storage */}
      <div className="px-4 pb-4">
        <div className="flex items-center gap-2 mb-1">
          <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.5" className="text-(--color-text-3)">
            <path d="M2 9.5v5A2.5 2.5 0 004.5 17h15a2.5 2.5 0 002.5-2.5v-5M2 9.5A2.5 2.5 0 014.5 7h15A2.5 2.5 0 0122 9.5M2 9.5h20"/>
          </svg>
          <span className="text-xs text-(--color-text-2)">Dung lượng</span>
        </div>
        <div className="h-1 rounded-full bg-(--color-surface-2) mb-1 overflow-hidden">
          <div className="h-full w-1/4 rounded-full bg-(--color-blue)" />
        </div>
        <p className="text-[11px] text-(--color-text-3)">128.6 GB / 500 GB</p>
      </div>
    </aside>
  )
}
