import { useState, useEffect, useRef } from 'react'
import { Bell, Search, ArrowLeft, LogOut, Shield } from 'lucide-react'
import { Link } from '@tanstack/react-router'
import { useHeaderStore } from '../store/header'
import { WorkersIndicator } from './WorkersIndicator'
import { cn } from '../lib/cn'
import { useLogout, type AuthUser } from '../lib/auth'

interface Props { user: AuthUser }

export function Header({ user }: Props) {
  const { crumbs, title } = useHeaderStore()
  const back = crumbs[0]
  const [_searchOpen, setSearchOpen] = useState(false)

  return (
    <header className="flex items-center gap-3 px-5 h-bar bg-white shrink-0 border-b border-zinc-100">
      <div className="flex-1 min-w-0">
        {back ? (
          <Link
            to={back.to}
            className="inline-flex items-center gap-1.5 text-sm text-zinc-500 hover:text-zinc-900 transition-colors"
          >
            <ArrowLeft size={14} />
            {back.label}
          </Link>
        ) : title ? (
          <span className="text-sm font-semibold text-zinc-900 tracking-tight truncate">
            {title}
          </span>
        ) : null}
      </div>

      <button
        onClick={() => setSearchOpen(true)}
        title="Tìm nhanh (⌘K)"
        className="hidden sm:flex items-center gap-2 h-8 px-3 w-56 rounded-lg border border-zinc-200 text-zinc-400 hover:border-zinc-300 hover:text-zinc-600 transition-colors cursor-pointer"
      >
        <Search size={13} className="shrink-0" />
        <span className="flex-1 text-sm text-left select-none">Tìm nhanh...</span>
        <kbd className="text-[11px] font-mono bg-zinc-100 border border-zinc-200 rounded px-1.5 text-zinc-400 leading-none py-0.5">
          ⌘K
        </kbd>
      </button>

      <WorkersIndicator />

      <button
        title="Thông báo"
        className="size-8 rounded-lg flex items-center justify-center text-zinc-400 hover:text-zinc-600 hover:bg-zinc-100 transition-colors cursor-pointer"
      >
        <Bell size={15} />
      </button>

      <UserMenu user={user} />
    </header>
  )
}

function UserMenu({ user }: { user: AuthUser }) {
  const [open, setOpen] = useState(false)
  const ref = useRef<HTMLDivElement>(null)
  const logout = useLogout()

  useEffect(() => {
    if (!open) return
    const onClick = (e: MouseEvent) => {
      if (ref.current && !ref.current.contains(e.target as Node)) setOpen(false)
    }
    document.addEventListener('mousedown', onClick)
    return () => document.removeEventListener('mousedown', onClick)
  }, [open])

  return (
    <div className="relative" ref={ref}>
      <button
        onClick={() => setOpen(!open)}
        className="flex items-center gap-2 pl-1 pr-2 h-8 rounded-lg hover:bg-zinc-100 transition-colors cursor-pointer"
      >
        <Avatar user={user} />
        <span className="text-sm text-zinc-700 max-w-32 truncate hidden md:inline">
          {user.display_name}
        </span>
      </button>

      {open && (
        <div className="absolute right-0 top-full mt-1.5 w-56 rounded-xl border border-zinc-200 bg-white shadow-[0_8px_24px_rgb(0,0,0,0.08)] overflow-hidden z-50">
          <div className="px-3.5 py-3 border-b border-zinc-100">
            <p className="text-sm font-medium text-zinc-900 truncate">{user.display_name}</p>
            {user.email && (
              <p className="text-xs text-zinc-400 truncate mt-0.5">{user.email}</p>
            )}
            {user.tier === 'admin' && (
              <span className="inline-flex items-center gap-1 mt-1.5 text-[10px] font-semibold uppercase tracking-wider text-emerald-700 bg-emerald-50 border border-emerald-100 rounded px-1.5 py-0.5">
                <Shield size={9} />
                Admin
              </span>
            )}
          </div>
          <button
            onClick={() => { setOpen(false); logout() }}
            className="w-full flex items-center gap-2 px-3.5 py-2.5 text-sm text-zinc-700 hover:bg-zinc-50 cursor-pointer transition-colors"
          >
            <LogOut size={13} className="text-zinc-400" />
            Đăng xuất
          </button>
        </div>
      )}
    </div>
  )
}

function Avatar({ user }: { user: AuthUser }) {
  const [failed, setFailed] = useState(false)
  const showImg = user.avatar_url && !failed
  const initial = user.display_name.charAt(0).toUpperCase()
  return (
    <span className={cn(
      'size-7 rounded-full overflow-hidden border border-zinc-200 shrink-0 flex items-center justify-center',
      'bg-zinc-100 text-xs font-semibold text-zinc-500',
    )}>
      {showImg ? (
        <img
          src={user.avatar_url!}
          alt={user.display_name}
          className="w-full h-full object-cover"
          onError={() => setFailed(true)}
        />
      ) : (
        initial
      )}
    </span>
  )
}
