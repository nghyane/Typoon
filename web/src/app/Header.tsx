import { useState, useEffect, useRef } from 'react'
import { Search, ArrowLeft, LogOut, Shield } from 'lucide-react'
import { Link } from '@tanstack/react-router'
import { useHeaderStore } from '../store/header'
import { WorkersIndicator } from './WorkersIndicator'
import { cn } from '@shared/lib/cn'
import { useLogout, type AuthUser } from '@features/auth/auth'
import { Monogram } from '@shared/ui/primitives'

interface Props { user: AuthUser }

export function Header({ user }: Props) {
  const { crumbs, title } = useHeaderStore()
  const back = crumbs[0]
  const [, setSearchOpen] = useState(false)

  return (
    <header className="flex items-center gap-3 px-5 h-bar bg-bg shrink-0">
      <div className="flex-1 min-w-0">
        {back ? (
          <Link
            to={back.to}
            className="inline-flex items-center gap-1.5 text-sm text-text-subtle hover:text-text transition-colors"
          >
            <ArrowLeft size={14} />
            {back.label}
          </Link>
        ) : title ? (
          <span className="text-sm font-semibold text-text tracking-tight truncate">
            {title}
          </span>
        ) : null}
      </div>

      {/* search — input shape (not fake button) */}
      <button
        onClick={() => setSearchOpen(true)}
        title="Tìm nhanh (⌘K)"
        className="hidden sm:flex items-center gap-2 h-8 px-2.5 w-56 rounded-sm bg-surface-2 text-text-subtle hover:bg-hover transition-colors cursor-pointer"
      >
        <Search size={13} className="shrink-0" />
        <span className="flex-1 text-left text-[13px] select-none">Tìm nhanh…</span>
        <kbd className="text-[10px] font-mono bg-black/30 rounded-xs px-1.5 py-0.5 text-text-subtle leading-none">
          ⌘K
        </kbd>
      </button>

      <WorkersIndicator />

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
        className="flex items-center gap-2 pl-1 pr-2 h-8 rounded-sm hover:bg-hover transition-colors cursor-pointer"
      >
        <Avatar user={user} />
        <span className="text-[13px] text-text max-w-32 truncate hidden md:inline">
          {user.display_name}
        </span>
      </button>

      {open && (
        <div className="absolute right-0 top-full mt-1.5 w-56 rounded-md bg-surface shadow-[0_8px_24px_rgb(0,0,0,0.4)] overflow-hidden z-50">
          <div className="px-3.5 py-3 border-b border-border-soft">
            <p className="text-sm font-medium text-text truncate">{user.display_name}</p>
            {user.email && (
              <p className="text-xs text-text-subtle truncate mt-0.5">{user.email}</p>
            )}
            {user.is_admin && (
              <span className="inline-flex items-center gap-1 mt-1.5 text-[10px] font-semibold uppercase tracking-wider text-success-text bg-success-bg rounded px-1.5 py-0.5">
                <Shield size={9} />
                Admin
              </span>
            )}
          </div>
          <button
            onClick={() => { setOpen(false); logout() }}
            className="w-full flex items-center gap-2 px-3.5 py-2.5 text-sm text-text hover:bg-hover cursor-pointer transition-colors"
          >
            <LogOut size={13} className="text-text-subtle" />
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
  if (showImg) {
    return (
      <span className={cn('size-7 rounded-full overflow-hidden shrink-0 flex items-center justify-center bg-surface-2')}>
        <img
          src={user.avatar_url!}
          alt={user.display_name}
          className="w-full h-full object-cover"
          onError={() => setFailed(true)}
        />
      </span>
    )
  }
  return <Monogram name={user.display_name} size={28} />
}
