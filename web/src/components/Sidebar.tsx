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

// Sidebar widths. Icon lane widths are derived so every icon center sits at
// sidebar_x = 30 in both states — no horizontal motion while width animates.
const W_COLLAPSED = 60
const W_EXPANDED  = 252
const NAV_PAD_X   = 8                            // matches `px-2` on <nav>
const NAV_LANE    = W_COLLAPSED - NAV_PAD_X * 2  // 44 — icon lane inside a nav link

export function Sidebar() {
  const { collapsed, toggle } = useSidebar()
  const { location } = useRouterState()

  function active(to: string) {
    return to === '/' ? location.pathname === '/' : location.pathname.startsWith(to)
  }

  function linkCls(isActive: boolean) {
    return cn(
      'flex items-center h-9 w-full rounded-md text-sm select-none cursor-pointer transition-colors duration-200',
      isActive
        ? 'bg-zinc-900/[0.06] text-zinc-900 font-medium'
        : 'text-zinc-500 hover:bg-zinc-900/[0.04] hover:text-zinc-900',
    )
  }

  return (
    <aside
      style={{ width: collapsed ? W_COLLAPSED : W_EXPANDED, transition: 'width 200ms ease-in-out' }}
      className="flex flex-col h-full shrink-0 overflow-hidden bg-zinc-50 border-r border-zinc-200"
    >
      {/* brand — icon sits in a 60px lane so it stays centered when sidebar = 60 */}
      <div className="flex items-center h-bar shrink-0">
        <div
          style={{ width: W_COLLAPSED }}
          className="h-full flex items-center justify-center shrink-0"
        >
          <button
            onClick={collapsed ? toggle : undefined}
            title={collapsed ? 'Mở rộng' : undefined}
            className={cn(
              'group relative size-6 rounded-md bg-zinc-900 flex items-center justify-center',
              collapsed ? 'cursor-pointer' : 'cursor-default',
            )}
          >
            <svg width="11" height="11" viewBox="0 0 13 13" fill="none"
              className={cn(collapsed && 'group-hover:opacity-0 transition-opacity')}>
              <path d="M2 3h9M2 6.5h5.5M2 10h7" stroke="white" strokeWidth="1.6" strokeLinecap="round" />
            </svg>
            {collapsed && (
              <ChevronRight size={11} className="absolute text-white opacity-0 group-hover:opacity-100 transition-opacity" />
            )}
          </button>
        </div>

        {/* text: takes remaining width, fades out without affecting brand position */}
        <span
          className="flex-1 min-w-0 font-semibold text-sm tracking-tight text-zinc-900 truncate transition-opacity duration-200"
          style={{ opacity: collapsed ? 0 : 1 }}
        >
          Typoon
        </span>

        {/* chevron: fades + slides off-canvas when collapsed (overflow:hidden clips it) */}
        <button
          onClick={toggle}
          title="Thu gọn"
          aria-hidden={collapsed}
          tabIndex={collapsed ? -1 : 0}
          className="size-6 mr-2.5 rounded-md flex items-center justify-center text-zinc-400 hover:text-zinc-600 hover:bg-zinc-200 cursor-pointer shrink-0 transition-opacity duration-200"
          style={{ opacity: collapsed ? 0 : 1, pointerEvents: collapsed ? 'none' : 'auto' }}
        >
          <ChevronLeft size={13} />
        </button>
      </div>

      {/* main nav */}
      <nav className="px-2 py-2 flex flex-col gap-0.5">
        {NAV.map(({ to, label, icon: Icon }) => (
          <Link key={to} to={to} title={collapsed ? label : undefined} className={linkCls(active(to))}>
            <span
              style={{ width: NAV_LANE }}
              className="h-full flex items-center justify-center shrink-0"
            >
              <Icon size={18} />
            </span>
            <span
              className="flex-1 min-w-0 truncate pr-2.5 transition-opacity duration-200"
              style={{ opacity: collapsed ? 0 : 1 }}
            >
              {label}
            </span>
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
      <div className="px-2 pb-2 border-t border-zinc-200 pt-2 flex flex-col gap-0.5">
        {NAV_FOOT.map(({ to, label, icon: Icon }) => (
          <Link key={to} to={to} title={collapsed ? label : undefined} className={linkCls(active(to))}>
            <span
              style={{ width: NAV_LANE }}
              className="h-full flex items-center justify-center shrink-0"
            >
              <Icon size={18} />
            </span>
            <span
              className="flex-1 min-w-0 truncate pr-2.5 transition-opacity duration-200"
              style={{ opacity: collapsed ? 0 : 1 }}
            >
              {label}
            </span>
          </Link>
        ))}
      </div>
    </aside>
  )
}
