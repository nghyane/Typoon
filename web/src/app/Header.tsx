import { useState, useEffect, useRef } from 'react'
import {
  ArrowLeft, ChevronLeft, ChevronRight, Check, Globe, LogOut,
  Search, Settings, Shield,
} from 'lucide-react'
import { Link, useNavigate } from '@tanstack/react-router'

import { useHeaderStore } from '../store/header'
import { cn } from '@shared/lib/cn'
import {
  useSession, useSignOut, useUpdatePreferredLang,
  type SessionUser,
} from '@features/auth/session'
import { LANG_OPTIONS } from '@features/auth/readingLang'
import { Monogram } from '@shared/ui/primitives'


interface Props { user: SessionUser }


export function Header({ user }: Props) {
  const { crumbs, title, slot, actions } = useHeaderStore()
  const back = crumbs[0]
  const [, setSearchOpen] = useState(false)

  return (
    <header className="flex items-center gap-2 px-3 sm:px-5 h-bar bg-bg shrink-0">
      {/* left — back / title */}
      <div className={cn('min-w-0', slot ? 'shrink-0' : 'flex-1')}>
        {back ? (
          <Link
            to={back.to}
            className="inline-flex items-center gap-2 text-sm text-text-subtle hover:text-text transition-colors"
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

      {/* center slot — route-injected (e.g. search input on /explore) */}
      {slot && <div className="flex-1 min-w-0">{slot}</div>}

      {/* Desktop global search; hidden when the center slot is active. */}
      {!slot && (
        <button
          onClick={() => setSearchOpen(true)}
          title="Tìm nhanh (⌘K)"
          className="hidden sm:flex items-center gap-2 h-8 px-2.5 w-56 rounded-sm bg-surface-2 text-text-subtle hover:bg-hover transition-colors cursor-pointer"
        >
          <Search size={14} className="shrink-0" />
          <span className="flex-1 text-left text-sm select-none">Tìm nhanh…</span>
          <kbd className="text-xs font-mono bg-black/30 rounded-xs px-1.5 py-0.5 text-text-subtle leading-none">
            ⌘K
          </kbd>
        </button>
      )}

      {/* right slot — route-injected page actions (e.g. upload on /w/) */}
      {actions && <div className="flex items-center gap-1 shrink-0">{actions}</div>}

      <UserMenu user={user} />
    </header>
  )
}


// ── User menu ──────────────────────────────────────────────────────
//
// Two-pane dropdown (iOS Settings pattern): a root pane with row-style
// entries, and named sub-panes that slide in over the root. Each pane
// fills the full dropdown width so layouts stay legible without
// nesting popovers or wrapping chips. Add a new pane by extending
// `Pane` + branching the body.

type Pane = 'root' | 'lang'


function UserMenu({ user }: { user: SessionUser }) {
  const [open, setOpen] = useState(false)
  const [pane, setPane] = useState<Pane>('root')
  const ref = useRef<HTMLDivElement>(null)
  const signOut = useSignOut()
  const nav = useNavigate()

  // Reset to root pane every time the menu closes so the next open
  // starts fresh — keeping pane state stale would surprise the user.
  useEffect(() => {
    if (!open) setPane('root')
  }, [open])

  useEffect(() => {
    if (!open) return
    const onClick = (e: MouseEvent) => {
      if (ref.current && !ref.current.contains(e.target as Node)) setOpen(false)
    }
    const onEsc = (e: KeyboardEvent) => {
      if (e.key === 'Escape') {
        if (pane !== 'root') setPane('root')
        else setOpen(false)
      }
    }
    document.addEventListener('mousedown', onClick)
    document.addEventListener('keydown', onEsc)
    return () => {
      document.removeEventListener('mousedown', onClick)
      document.removeEventListener('keydown', onEsc)
    }
  }, [open, pane])

  return (
    <div className="relative" ref={ref}>
      <button
        onClick={() => setOpen(!open)}
        className="flex items-center gap-2 pl-1 pr-2 h-8 rounded-sm hover:bg-hover transition-colors cursor-pointer"
      >
        <Avatar user={user} />
        <span className="text-sm text-text max-w-32 truncate hidden md:inline">
          {user.display_name}
        </span>
      </button>

      {open && (
        <div className="absolute right-0 top-full mt-1.5 w-64 rounded-md bg-surface shadow-[0_8px_24px_rgb(0,0,0,0.4)] overflow-hidden z-50">
          {pane === 'root' ? (
            <RootPane
              user={user}
              onOpenLang={() => setPane('lang')}
              onOpenSettings={() => {
                setOpen(false)
                nav({ to: '/settings', search: { section: 'account' } })
              }}
              onLogout={() => { setOpen(false); void signOut().then(() => nav({ to: '/login' })) }}
            />
          ) : (
            <LangPane onBack={() => setPane('root')} />
          )}
        </div>
      )}
    </div>
  )
}


function RootPane({
  user, onOpenLang, onOpenSettings, onLogout,
}: {
  user:           SessionUser
  onOpenLang:     () => void
  onOpenSettings: () => void
  onLogout:       () => void
}) {
  const currentLang = LANG_OPTIONS.find(
    (o) => o.code === user.preferred_target_lang,
  )?.label ?? '—'

  return (
    <div>
      <header className="px-3.5 py-3 border-b border-border-soft">
        <p className="text-sm font-medium text-text truncate">
          {user.display_name}
        </p>
        {user.is_admin && (
          <span className="inline-flex items-center gap-1 mt-1.5 text-xs font-semibold uppercase tracking-wider text-success-text bg-success-bg rounded px-1.5 py-0.5">
            <Shield size={9} />
            Admin
          </span>
        )}
      </header>

      <nav className="py-1">
        <MenuRow
          icon={<Globe size={14} />}
          label="Đọc bằng"
          value={currentLang}
          onClick={onOpenLang}
          trailing={<ChevronRight size={12} className="text-text-subtle" />}
        />
        <MenuRow
          icon={<Settings size={14} />}
          label="Cài đặt"
          onClick={onOpenSettings}
          trailing={<ChevronRight size={12} className="text-text-subtle" />}
        />
      </nav>

      <div className="border-t border-border-soft py-1">
        <MenuRow
          icon={<LogOut size={14} />}
          label="Đăng xuất"
          onClick={onLogout}
          tone="destructive"
        />
      </div>
    </div>
  )
}


function LangPane({ onBack }: { onBack: () => void }) {
  const { user } = useSession()
  const update = useUpdatePreferredLang()
  const current = user?.preferred_target_lang ?? ''

  return (
    <div>
      <header className="flex items-center gap-1 px-2 py-2 border-b border-border-soft">
        <button
          type="button"
          onClick={onBack}
          className="inline-flex items-center gap-1 h-7 px-2 rounded-sm text-sm text-text-muted hover:bg-hover hover:text-text cursor-pointer transition-colors"
        >
          <ChevronLeft size={14} />
          Quay lại
        </button>
      </header>

      <ul role="listbox" className="py-1">
        {LANG_OPTIONS.map((opt) => {
          const active = current === opt.code
          return (
            <li key={opt.code}>
              <button
                type="button"
                role="option"
                aria-selected={active}
                onClick={() => update(active ? null : opt.code)}
                className={cn(
                  'w-full flex items-center gap-2 px-3.5 py-2 text-sm text-left cursor-pointer transition-colors',
                  active ? 'text-text' : 'text-text-muted hover:bg-hover hover:text-text',
                )}
              >
                <span className="flex-1">{opt.label}</span>
                {active && <Check size={14} className="text-accent" />}
              </button>
            </li>
          )
        })}
      </ul>
    </div>
  )
}


function MenuRow({
  icon, label, value, trailing, onClick, tone = 'default',
}: {
  icon:      React.ReactNode
  label:     string
  value?:    string
  trailing?: React.ReactNode
  onClick:   () => void
  tone?:     'default' | 'destructive'
}) {
  return (
    <button
      type="button"
      onClick={onClick}
      className={cn(
        'w-full flex items-center gap-2.5 px-3.5 py-2 text-sm text-left cursor-pointer transition-colors',
        tone === 'destructive'
          ? 'text-text hover:bg-error/10 hover:text-error-text'
          : 'text-text hover:bg-hover',
      )}
    >
      <span className="text-text-subtle">{icon}</span>
      <span className="flex-1">{label}</span>
      {value && (
        <span className="text-xs text-text-subtle truncate max-w-[80px]">
          {value}
        </span>
      )}
      {trailing}
    </button>
  )
}


function Avatar({ user }: { user: SessionUser }) {
  const [failed, setFailed] = useState(false)
  const showImg = user.avatar_url && !failed
  if (showImg) {
    return (
      <span className="size-7 rounded-full overflow-hidden shrink-0 flex items-center justify-center bg-surface-2">
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
