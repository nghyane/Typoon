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
  { to: '/projects', icon: <FolderOpen     size={16} />, label: 'Dự án' },
  { to: '/library',  icon: <Library        size={16} />, label: 'Thư viện' },
  { to: '/glossary', icon: <BookText       size={16} />, label: 'Thuật ngữ' },
  { to: '/teams',    icon: <Users          size={16} />, label: 'Nhóm' },
  { to: '/reports',  icon: <BarChart2      size={16} />, label: 'Báo cáo' },
  { to: '/settings', icon: <Settings      size={16} />, label: 'Cài đặt' },
]

function NavItem({ to, icon, label }: { to: string; icon: React.ReactNode; label: string }) {
  const pathname = useRouterState({ select: (s) => s.location.pathname })
  const isActive = pathname === to || (to !== '/' && pathname.startsWith(to))

  return (
    <Link
      to={to}
      className={cn(
        'flex items-center gap-2.5 px-3 py-1.5 rounded-lg text-sm transition-colors select-none',
        isActive
          ? 'bg-(--color-nav-active-bg) text-(--color-nav-active-text) font-medium'
          : 'text-(--color-text-1) hover:bg-(--color-nav-hover-bg)',
      )}
    >
      {/* icon inherits color from parent */}
      <span className="shrink-0 opacity-70">{icon}</span>
      <span>{label}</span>
    </Link>
  )
}

export function Sidebar() {
  const { data: projects = [] } = useQuery({
    queryKey: projectKeys.all(),
    queryFn:  projectsApi.list,
    staleTime: 60_000,
  })

  return (
    <aside
      className="flex flex-col shrink-0 h-full bg-(--color-bg)"
      style={{ width: 'var(--sidebar-width)', borderRight: '1px solid var(--color-border-subtle)' }}
    >
      {/* Logo + collapse */}
      <div className="flex items-center justify-between px-4 py-3 shrink-0">
        <div className="flex items-center gap-2">
          {/* iOS-style app icon */}
          <div
            className="w-8 h-8 rounded-lg flex items-center justify-center text-sm font-bold text-white shrink-0"
            style={{ background: 'linear-gradient(145deg, #1c1c1e 0%, #3a3a3c 100%)' }}
          >
            T
          </div>
          <span className="text-sm font-semibold text-(--color-text-1)">Typoon</span>
        </div>
        <button className="w-6 h-6 flex items-center justify-center rounded text-(--color-text-3) hover:text-(--color-text-2) transition-colors">
          {/* collapse icon — ⊠ style */}
          <svg width="14" height="14" viewBox="0 0 14 14" fill="none">
            <rect x="1" y="1" width="5" height="12" rx="1.5" stroke="currentColor" strokeWidth="1.2"/>
            <rect x="8" y="1" width="5" height="12" rx="1.5" stroke="currentColor" strokeWidth="1.2"/>
          </svg>
        </button>
      </div>

      {/* Nav */}
      <nav className="px-2 flex flex-col gap-0.5">
        {NAV.map((item) => <NavItem key={item.to} {...item} />)}
      </nav>

      {/* GẦN ĐÂY */}
      {projects.length > 0 && (
        <div className="mt-5 px-3">
          <p className="text-[10px] font-semibold uppercase tracking-widest text-(--color-text-3) mb-2 px-1">
            Gần đây
          </p>
          <div className="flex flex-col gap-0.5">
            {projects.slice(0, 4).map((p) => (
              <Link
                key={p.project_id}
                to="/projects/$id"
                params={{ id: String(p.project_id) }}
                className="flex items-center gap-2.5 px-2 py-1.5 rounded-lg hover:bg-(--color-nav-hover-bg) transition-colors group"
              >
                {/* Thumbnail placeholder — square rounded */}
                <div
                  className="w-8 h-8 rounded-lg shrink-0 flex items-center justify-center text-xs font-bold text-white"
                  style={{ background: 'linear-gradient(135deg, #636366, #48484a)' }}
                >
                  {p.title[0]}
                </div>
                <div className="min-w-0">
                  <p className="text-sm font-medium text-(--color-text-1) truncate leading-tight">{p.title}</p>
                  <p className="text-xs text-(--color-text-3) leading-tight mt-px">
                    {/* chapter count not in ProjectOut yet, show lang */}
                    {p.source_lang} → {p.target_lang}
                  </p>
                </div>
              </Link>
            ))}
            <Link
              to="/projects"
              className="px-2 py-1 text-xs text-(--color-text-3) hover:text-(--color-accent) transition-colors"
            >
              Xem tất cả
            </Link>
          </div>
        </div>
      )}

      {/* Storage */}
      <div className="mt-auto px-3 pb-4">
        <div className="flex items-center gap-2 mb-1.5">
          <Cloud size={13} className="text-(--color-text-3)" />
          <span className="text-xs text-(--color-text-2)">Dung lượng</span>
        </div>
        <div className="h-1 rounded-full bg-(--color-border-subtle) overflow-hidden mb-1">
          <div
            className="h-full rounded-full"
            style={{ width: '25%', background: 'var(--color-running)' }}
          />
        </div>
        <p className="text-[11px] text-(--color-text-3)">128.6 GB / 500 GB</p>
      </div>
    </aside>
  )
}
