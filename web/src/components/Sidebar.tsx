import { Link, useRouterState } from '@tanstack/react-router'
import { useSidebar } from '../store/sidebar'
import { cn } from '../lib/cn'
import { FolderOpen, Settings, ChevronLeft, ChevronRight } from 'lucide-react'

const NAV = [
  { to: '/projects', label: 'Dự án', icon: FolderOpen },
] as const

const NAV_FOOT = [
  { to: '/settings', label: 'Cài đặt', icon: Settings },
] as const

// Sidebar widths. Icon lane widths are derived so every icon center sits at
// sidebar_x = 30 in both states — no horizontal motion while width animates.
const W_COLLAPSED = 60
const W_EXPANDED  = 240
const NAV_PAD_X   = 8                            // matches `px-2` on <nav>
const NAV_LANE    = W_COLLAPSED - NAV_PAD_X * 2  // 44 — icon lane inside a nav link

interface Props { brandName: string }

export function Sidebar({ brandName }: Props) {
  const { collapsed, toggle } = useSidebar()
  const { location } = useRouterState()

  const active = (to: string) =>
    to === '/' ? location.pathname === '/' : location.pathname.startsWith(to)

  const linkCls = (isActive: boolean) =>
    cn(
      'flex items-center h-9 w-full rounded-md text-sm select-none cursor-pointer transition-colors duration-150',
      isActive
        ? 'bg-zinc-900/[0.06] text-zinc-900 font-medium'
        : 'text-zinc-500 hover:bg-zinc-900/[0.04] hover:text-zinc-900',
    )

  const renderLink = (to: string, label: string, Icon: typeof FolderOpen) => (
    <Link key={to} to={to} title={collapsed ? label : undefined} className={linkCls(active(to))}>
      <span style={{ width: NAV_LANE }} className="h-full flex items-center justify-center shrink-0">
        <Icon size={17} />
      </span>
      <span
        className="flex-1 min-w-0 truncate pr-2.5 transition-opacity duration-150"
        style={{ opacity: collapsed ? 0 : 1 }}
      >
        {label}
      </span>
    </Link>
  )

  return (
    <aside
      style={{ width: collapsed ? W_COLLAPSED : W_EXPANDED, transition: 'width 180ms ease-in-out' }}
      className="flex flex-col h-full shrink-0 overflow-hidden bg-zinc-50 border-r border-zinc-200"
    >
      {/* brand */}
      <div className="flex items-center h-bar shrink-0">
        <div
          style={{ width: W_COLLAPSED }}
          className="h-full flex items-center justify-center shrink-0"
        >
          <button
            onClick={collapsed ? toggle : undefined}
            title={collapsed ? 'Mở rộng' : undefined}
            className={cn(
              'group relative size-7 rounded-md bg-zinc-900 flex items-center justify-center',
              collapsed ? 'cursor-pointer' : 'cursor-default',
            )}
          >
            <svg
              width="12" height="12" viewBox="0 0 13 13" fill="none"
              className={cn('transition-opacity', collapsed && 'group-hover:opacity-0')}
            >
              <path d="M2 3h9M2 6.5h5.5M2 10h7" stroke="white" strokeWidth="1.6" strokeLinecap="round" />
            </svg>
            {collapsed && (
              <ChevronRight
                size={12}
                className="absolute text-white opacity-0 group-hover:opacity-100 transition-opacity"
              />
            )}
          </button>
        </div>

        <span
          className="flex-1 min-w-0 font-semibold text-sm tracking-tight text-zinc-900 truncate transition-opacity duration-150"
          style={{ opacity: collapsed ? 0 : 1 }}
          title={brandName}
        >
          {brandName}
        </span>

        <button
          onClick={toggle}
          title="Thu gọn"
          aria-hidden={collapsed}
          tabIndex={collapsed ? -1 : 0}
          className="size-7 mr-2 rounded-md flex items-center justify-center text-zinc-400 hover:text-zinc-600 hover:bg-zinc-200 cursor-pointer shrink-0 transition-opacity duration-150"
          style={{ opacity: collapsed ? 0 : 1, pointerEvents: collapsed ? 'none' : 'auto' }}
        >
          <ChevronLeft size={14} />
        </button>
      </div>

      {/* main nav */}
      <nav className="px-2 py-2 flex flex-col gap-0.5">
        {NAV.map(({ to, label, icon }) => renderLink(to, label, icon))}
      </nav>

      <div className="flex-1" />

      {/* footer nav */}
      <div className="px-2 pb-2 pt-2 border-t border-zinc-200 flex flex-col gap-0.5">
        {NAV_FOOT.map(({ to, label, icon }) => renderLink(to, label, icon))}
      </div>
    </aside>
  )
}
