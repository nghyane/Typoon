import { Link, useRouterState } from '@tanstack/react-router'
import { useSidebar } from '../store/sidebar'
import { cn } from '../shared/lib/cn'
import { FolderOpen, Star, Globe, Settings, ChevronLeft, ChevronRight } from 'lucide-react'
import { SidebarQuota } from './SidebarQuota'

const NAV = [
  { to: '/projects', label: 'Của tôi',   icon: FolderOpen, search: { filter: 'mine'      } },
  { to: '/projects', label: 'Đã lưu',    icon: Star,       search: { filter: 'pinned'    } },
  { to: '/projects', label: 'Cộng đồng', icon: Globe,      search: { filter: 'community' } },
] as const

const NAV_FOOT = [
  { to: '/settings', label: 'Cài đặt', icon: Settings, search: undefined },
] as const

const W_COLLAPSED = 60
const W_EXPANDED  = 240
const NAV_PAD_X   = 8
const NAV_LANE    = W_COLLAPSED - NAV_PAD_X * 2  // 44px — icon column

interface Props {
  brandName: string | null
  brandIcon: string | null
}

export function Sidebar({ brandName, brandIcon }: Props) {
  const { collapsed, toggle } = useSidebar()
  const { location } = useRouterState()

  const isActive = (to: string, filter?: string) => {
    if (to !== '/projects') return location.pathname.startsWith(to)
    const onProjects =
      location.pathname === '/projects' || location.pathname.startsWith('/projects/')
    if (!onProjects) return false
    const current = (location.search as { filter?: string })?.filter ?? 'mine'
    return current === (filter ?? 'mine')
  }

  // Active item: surface-2 fill only. No floating bar — that pattern was
  // fragile (absolute positioning depending on container padding) and
  // conflicted with the tab underline at the page level.
  const linkCls = (active: boolean) =>
    cn(
      'group flex items-center h-8 w-full rounded-sm select-none cursor-pointer',
      'transition-colors duration-150',
      active
        ? 'bg-surface-2 text-text font-medium'
        : 'text-text-muted hover:bg-hover hover:text-text',
    )

  const renderLink = (
    to: string,
    label: string,
    Icon: typeof FolderOpen,
    search?: Record<string, string>,
  ) => {
    const active = isActive(to, search?.filter)
    return (
      <Link
        key={`${to}:${search?.filter ?? ''}:${label}`}
        to={to}
        search={search as never}
        title={collapsed ? label : undefined}
        className={linkCls(active)}
      >
        <span style={{ width: NAV_LANE }} className="h-full flex items-center justify-center shrink-0">
          <Icon size={16} className={cn(active && 'text-accent-text')} />
        </span>
        <span
          className="flex-1 min-w-0 truncate pr-2.5 text-[13px] transition-opacity duration-150"
          style={{ opacity: collapsed ? 0 : 1 }}
        >
          {label}
        </span>
      </Link>
    )
  }

  return (
    <aside
      style={{ width: collapsed ? W_COLLAPSED : W_EXPANDED, transition: 'width 180ms ease-in-out' }}
      className="hidden sm:flex flex-col h-full shrink-0 overflow-hidden bg-surface"
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
              'group relative size-7 rounded-sm flex items-center justify-center overflow-hidden',
              brandIcon
                ? 'bg-surface-2'
                : 'bg-accent text-accent-fg text-[13px] font-bold',
              collapsed ? 'cursor-pointer' : 'cursor-default',
            )}
          >
            {brandIcon ? (
              <img
                src={brandIcon}
                alt={brandName ?? ''}
                className={cn(
                  'w-full h-full object-cover',
                  collapsed && 'group-hover:opacity-0 transition-opacity',
                )}
              />
            ) : (
              <span className={cn('transition-opacity', collapsed && 'group-hover:opacity-0')}>
                {brandName ? brandName.charAt(0).toUpperCase() : 'T'}
              </span>
            )}
            {collapsed && (
              <ChevronRight
                size={12}
                className="absolute opacity-0 group-hover:opacity-100 transition-opacity text-text"
              />
            )}
          </button>
        </div>

        {brandName && (
          <span
            className="flex-1 min-w-0 font-semibold text-sm tracking-tight text-text truncate transition-opacity duration-150"
            style={{ opacity: collapsed ? 0 : 1 }}
            title={brandName}
          >
            {brandName}
          </span>
        )}
        {!brandName && <div className="flex-1" />}

        <button
          onClick={toggle}
          title="Thu gọn"
          aria-hidden={collapsed}
          tabIndex={collapsed ? -1 : 0}
          className="size-7 mr-2 rounded-sm flex items-center justify-center text-text-subtle hover:text-text hover:bg-hover cursor-pointer shrink-0 transition-opacity duration-150"
          style={{ opacity: collapsed ? 0 : 1, pointerEvents: collapsed ? 'none' : 'auto' }}
        >
          <ChevronLeft size={14} />
        </button>
      </div>

      {/* main nav */}
      <nav className="px-2 py-2 flex flex-col gap-0.5">
        {NAV.map(({ to, label, icon, search }) => renderLink(to, label, icon, search))}
      </nav>

      <div className="flex-1" />

      {/* footer nav */}
      <div className="px-2 pb-2 pt-2 flex flex-col gap-0.5">
        <SidebarQuota collapsed={collapsed} />
        {NAV_FOOT.map(({ to, label, icon, search }) => renderLink(to, label, icon, search))}
      </div>
    </aside>
  )
}
