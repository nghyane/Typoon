import { Link, useRouterState } from '@tanstack/react-router'
import { useSidebar } from '../store/sidebar'
import { cn } from '../lib/cn'
import {
  LayoutDashboard, FolderOpen, Library, BookOpen,
  Users, BarChart2, Settings, Cloud,
  ChevronLeft, ChevronRight,
} from 'lucide-react'

const NAV = [
  { to: '/',         label: 'Tổng quan', icon: LayoutDashboard },
  { to: '/projects', label: 'Dự án',     icon: FolderOpen },
  { to: '/library',  label: 'Thư viện',  icon: Library },
  { to: '/glossary', label: 'Thuật ngữ', icon: BookOpen },
  { to: '/groups',   label: 'Nhóm',      icon: Users },
  { to: '/reports',  label: 'Báo cáo',   icon: BarChart2 },
] as const

const NAV_FOOT = [
  { to: '/settings', label: 'Cài đặt', icon: Settings },
] as const

const STORAGE_USED  = 128.6
const STORAGE_TOTAL = 500
const STORAGE_PCT   = Math.round((STORAGE_USED / STORAGE_TOTAL) * 100)

export function Sidebar() {
  const { collapsed, toggle } = useSidebar()
  const { location } = useRouterState()

  function active(to: string) {
    return to === '/' ? location.pathname === '/' : location.pathname.startsWith(to)
  }

  function linkCls(isActive: boolean) {
    return cn(
      'flex items-center gap-2.5 rounded-md text-sm transition-colors select-none cursor-pointer',
      collapsed ? 'justify-center w-9 h-9 mx-auto px-0' : 'px-2.5 h-9',
      isActive
        ? 'bg-zinc-900/[0.06] text-zinc-900 font-medium'
        : 'text-zinc-500 hover:bg-zinc-900/[0.04] hover:text-zinc-900',
    )
  }

  return (
    <aside
      className={cn(
        'flex flex-col h-full shrink-0 overflow-hidden bg-zinc-50 border-r border-zinc-200',
        'transition-[width] duration-200 ease-in-out',
        collapsed ? 'w-[var(--spacing-nav)]' : 'w-[var(--spacing-sidebar)]',
      )}
    >
      {/* brand */}
      <div className="flex items-center h-[var(--spacing-bar)] px-3 shrink-0 gap-2.5">
        <span className="size-7 rounded-lg bg-zinc-900 flex items-center justify-center shrink-0">
          <svg width="13" height="13" viewBox="0 0 13 13" fill="none">
            <path d="M2 3h9M2 6.5h5.5M2 10h7" stroke="white" strokeWidth="1.6" strokeLinecap="round" />
          </svg>
        </span>
        {!collapsed && (
          <span className="font-semibold text-sm tracking-tight text-zinc-900 truncate flex-1">
            Typoon
          </span>
        )}
        <button
          onClick={toggle}
          title={collapsed ? 'Mở rộng' : 'Thu gọn'}
          className="ml-auto size-7 rounded-md flex items-center justify-center text-zinc-400 hover:text-zinc-600 hover:bg-zinc-200 transition-colors cursor-pointer shrink-0"
        >
          {collapsed ? <ChevronRight size={14} /> : <ChevronLeft size={14} />}
        </button>
      </div>

      {/* main nav */}
      <nav className="py-2 px-2 space-y-0.5">
        {NAV.map(({ to, label, icon: Icon }) => (
          <Link key={to} to={to} title={collapsed ? label : undefined} className={linkCls(active(to))}>
            <Icon size={15} className="shrink-0" />
            {!collapsed && <span className="truncate">{label}</span>}
          </Link>
        ))}
      </nav>

      <div className="flex-1" />

      {/* storage */}
      {!collapsed && (
        <div className="px-3 mb-3">
          <div className="flex items-center gap-2 mb-1.5">
            <Cloud size={12} className="text-zinc-400" />
            <span className="text-xs text-zinc-500">{STORAGE_USED} GB / {STORAGE_TOTAL} GB</span>
          </div>
          <div className="h-1 rounded-full bg-zinc-200 overflow-hidden">
            <div className="h-full rounded-full bg-zinc-400 transition-[width]" style={{ width: `${STORAGE_PCT}%` }} />
          </div>
        </div>
      )}

      {/* footer nav */}
      <div className="px-2 pb-2 border-t border-zinc-200 pt-2 space-y-0.5">
        {NAV_FOOT.map(({ to, label, icon: Icon }) => (
          <Link key={to} to={to} title={collapsed ? label : undefined} className={linkCls(active(to))}>
            <Icon size={15} className="shrink-0" />
            {!collapsed && <span className="truncate">{label}</span>}
          </Link>
        ))}
      </div>
    </aside>
  )
}
