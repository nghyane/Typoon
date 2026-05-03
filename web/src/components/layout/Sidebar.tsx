import { Link, useRouterState } from '@tanstack/react-router'
import { useQuery } from '@tanstack/react-query'
import {
  LayoutDashboard, FolderOpen, Library,
  BookText, Users, BarChart2, Settings, Cloud,
} from 'lucide-react'
import { cn } from '../../lib/cn'
import { projectsApi, projectKeys } from '../../api/projects'

const NAV = [
  { to: '/overview', icon: <LayoutDashboard size={16} />, label: 'Tổng quan' },
  { to: '/projects', icon: <FolderOpen size={16} />,      label: 'Dự án' },
  { to: '/library',  icon: <Library size={16} />,         label: 'Thư viện' },
  { to: '/glossary', icon: <BookText size={16} />,        label: 'Thuật ngữ' },
  { to: '/teams',    icon: <Users size={16} />,           label: 'Nhóm' },
  { to: '/reports',  icon: <BarChart2 size={16} />,       label: 'Báo cáo' },
]

function NavItem({ to, icon, label }: { to: string; icon: React.ReactNode; label: string }) {
  const pathname = useRouterState({ select: (s) => s.location.pathname })
  const isActive = pathname.startsWith(to)

  return (
    <Link
      to={to}
      className={cn(
        'flex items-center gap-2.5 px-3 py-2 rounded-lg text-sm transition-colors',
        isActive
          ? 'bg-(--color-nav-active-bg) text-(--color-nav-active-text) font-medium'
          : 'text-(--color-text-2) hover:bg-(--color-nav-hover-bg)',
      )}
    >
      {icon}
      <span>{label}</span>
    </Link>
  )
}

export function Sidebar() {
  const { data: projects = [] } = useQuery({
    queryKey: projectKeys.all(),
    queryFn: projectsApi.list,
  })

  const recent = projects.slice(0, 4)

  return (
    <aside className="flex flex-col shrink-0 border-r border-(--color-border) bg-(--color-bg) h-full w-(--sidebar-width)">
      {/* Logo */}
      <div className="flex items-center justify-between px-4 h-14 shrink-0">
        <div className="flex items-center gap-2.5">
          <div className="w-8 h-8 rounded-lg flex items-center justify-center text-sm font-bold bg-(--color-accent) text-(--color-accent-text)">
            T
          </div>
          <span className="text-sm font-bold text-(--color-text-1)">Typoon</span>
        </div>
      </div>

      {/* Main nav */}
      <nav className="px-2 flex flex-col gap-0.5">
        {NAV.map((item) => <NavItem key={item.to} {...item} />)}
      </nav>

      <div className="px-2 mt-1">
        <NavItem to="/settings" icon={<Settings size={16} />} label="Cài đặt" />
      </div>

      {/* Recent */}
      {recent.length > 0 && (
        <div className="px-4 mt-5">
          <p className="text-[10px] font-semibold uppercase tracking-widest mb-2 text-(--color-text-3)">
            Gần đây
          </p>
          <div className="flex flex-col gap-0.5">
            {recent.map((p) => (
              <Link
                key={p.project_id}
                to="/projects/$id"
                params={{ id: String(p.project_id) }}
                className="flex items-center gap-2.5 px-1 py-1.5 rounded-lg hover:bg-(--color-surface-1) transition-colors"
              >
                <div className="w-7 h-7 rounded-md shrink-0 flex items-center justify-center text-xs font-bold bg-(--color-surface-2) text-(--color-text-2)">
                  {p.title[0]}
                </div>
                <div className="min-w-0">
                  <p className="text-xs font-medium truncate text-(--color-text-1)">{p.title}</p>
                  <p className="text-[10px] text-(--color-text-3)">{p.source_lang} → {p.target_lang}</p>
                </div>
              </Link>
            ))}
            <Link to="/projects" className="px-1 py-1 text-xs text-(--color-text-3) hover:underline">
              Xem tất cả
            </Link>
          </div>
        </div>
      )}

      {/* Storage */}
      <div className="mt-auto px-4 py-4 border-t border-(--color-border)">
        <div className="flex items-center gap-2 mb-1.5">
          <Cloud size={14} className="text-(--color-text-3)" />
          <span className="text-xs font-medium text-(--color-text-2)">Dung lượng</span>
        </div>
        <div className="h-1.5 rounded-full overflow-hidden bg-(--color-surface-2) mb-1">
          <div className="h-full w-1/4 rounded-full bg-(--color-running)" />
        </div>
        <p className="text-[10px] text-(--color-text-3)">128.6 GB / 500 GB</p>
      </div>
    </aside>
  )
}
