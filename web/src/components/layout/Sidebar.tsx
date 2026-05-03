import { Link, useRouterState } from '@tanstack/react-router'
import {
  LayoutDashboard, FolderOpen, Library,
  BookText, Users, BarChart2, Settings, Cloud,
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
      className="flex flex-col shrink-0 h-full bg-(--color-bg)"
      style={{ width: 'var(--sidebar-width)', borderRight: '1px solid var(--color-border-subtle)' }}
    >
      {/* Logo */}
      <div className="px-4 pt-4 pb-3 shrink-0">
        <div className="flex items-center justify-between">
          <div className="flex items-center gap-2.5">
            {/* GitHub-style circle logo */}
            <div className="w-8 h-8 rounded-full bg-(--color-text-1) flex items-center justify-center shrink-0">
              <svg width="18" height="18" viewBox="0 0 24 24" fill="white">
                <path d="M12 2C6.477 2 2 6.484 2 12.017c0 4.425 2.865 8.18 6.839 9.504.5.092.682-.217.682-.483 0-.237-.008-.868-.013-1.703-2.782.605-3.369-1.343-3.369-1.343-.454-1.158-1.11-1.466-1.11-1.466-.908-.62.069-.608.069-.608 1.003.07 1.531 1.032 1.531 1.032.892 1.53 2.341 1.088 2.91.832.092-.647.35-1.088.636-1.338-2.22-.253-4.555-1.113-4.555-4.951 0-1.093.39-1.988 1.029-2.688-.103-.253-.446-1.272.098-2.65 0 0 .84-.27 2.75 1.026A9.564 9.564 0 0112 6.844c.85.004 1.705.115 2.504.337 1.909-1.296 2.747-1.027 2.747-1.027.546 1.379.202 2.398.1 2.651.64.7 1.028 1.595 1.028 2.688 0 3.848-2.339 4.695-4.566 4.943.359.309.678.92.678 1.855 0 1.338-.012 2.419-.012 2.747 0 .268.18.58.688.482A10.019 10.019 0 0022 12.017C22 6.484 17.522 2 12 2z"/>
              </svg>
            </div>
            <span className="text-base font-bold text-(--color-text-1)">Typoon</span>
          </div>
          <button className="text-(--color-text-3) hover:text-(--color-text-2) transition-colors text-sm font-mono">
            ×&lt;
          </button>
        </div>
      </div>

      {/* Separator */}
      <div className="mx-3 mb-3 h-px bg-(--color-border-subtle)" />

      {/* Nav */}
      <nav className="px-2 flex flex-col gap-0.5">
        {NAV.map((item) => <NavItem key={item.to} {...item} />)}
      </nav>

      {/* Storage */}
      <div className="mt-auto px-4 pb-4">
        <div className="flex items-center gap-2 mb-1.5">
          <Cloud size={13} className="text-(--color-text-3)" />
          <span className="text-xs text-(--color-text-2)">Dung lượng</span>
        </div>
        <div className="h-1 rounded-full bg-(--color-border-subtle) overflow-hidden mb-1">
          <div className="h-full w-1/4 rounded-full bg-(--color-running)" />
        </div>
        <p className="text-[11px] text-(--color-text-3)">128.6 GB / 500 GB</p>
      </div>
    </aside>
  )
}
