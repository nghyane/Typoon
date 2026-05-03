import { Link, useRouterState } from '@tanstack/react-router'
import { useQuery } from '@tanstack/react-query'
import { FolderOpen, Activity, Settings } from 'lucide-react'
import { cn } from '../../lib/cn'
import { projectsApi, projectKeys } from '../../api/projects'

const NAV = [
  { to: '/projects', icon: <FolderOpen size={15} />, label: 'Projects' },
  { to: '/pipeline', icon: <Activity  size={15} />, label: 'Pipeline' },
]

function NavItem({ to, icon, label }: { to: string; icon: React.ReactNode; label: string }) {
  const pathname = useRouterState({ select: (s) => s.location.pathname })
  const isActive = pathname.startsWith(to)
  return (
    <Link
      to={to}
      className={cn(
        'flex items-center gap-2 px-2 py-1.5 rounded-md text-sm transition-colors',
        isActive
          ? 'bg-(--color-nav-active-bg) text-(--color-nav-active-text) font-medium'
          : 'text-(--color-text-2) hover:bg-(--color-nav-hover-bg) hover:text-(--color-text-1)',
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
    queryFn:  projectsApi.list,
    staleTime: 60_000,
  })

  return (
    <aside className="flex flex-col shrink-0 h-full border-r border-(--color-border) bg-(--color-bg)" style={{ width: 'var(--sidebar-width)' }}>
      {/* Logo */}
      <div className="flex items-center gap-2 px-3 h-14 shrink-0 border-b border-(--color-border)">
        <div className="w-6 h-6 rounded-md flex items-center justify-center text-xs font-bold bg-(--color-text-1) text-white">
          T
        </div>
        <span className="text-sm font-semibold text-(--color-text-1)">Typoon</span>
      </div>

      {/* Nav */}
      <div className="px-2 pt-3 flex flex-col gap-0.5">
        {NAV.map((item) => <NavItem key={item.to} {...item} />)}
      </div>

      {/* Recent */}
      {projects.length > 0 && (
        <div className="px-3 mt-5">
          <p className="text-xs font-semibold text-(--color-text-3) mb-1 px-1">Recent</p>
          <div className="flex flex-col gap-0.5">
            {projects.slice(0, 5).map((p) => (
              <Link
                key={p.project_id}
                to="/projects/$id"
                params={{ id: String(p.project_id) }}
                className="flex items-center gap-2 px-2 py-1.5 rounded-md text-sm text-(--color-text-2) hover:bg-(--color-nav-hover-bg) hover:text-(--color-text-1) transition-colors truncate"
              >
                <span className="w-4 h-4 rounded text-[10px] font-bold flex items-center justify-center bg-(--color-surface) text-(--color-text-3) shrink-0">
                  {p.title[0]}
                </span>
                <span className="truncate">{p.title}</span>
              </Link>
            ))}
          </div>
        </div>
      )}

      {/* Bottom */}
      <div className="mt-auto px-2 pb-3 border-t border-(--color-border) pt-2">
        <NavItem to="/settings" icon={<Settings size={15} />} label="Settings" />
      </div>
    </aside>
  )
}
