import { Bell, Search, ArrowLeft } from 'lucide-react'
import { Link } from '@tanstack/react-router'
import { useHeaderStore } from '../store/header'

export function Header() {
  const { crumbs, title } = useHeaderStore()
  const back = crumbs[0]

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
        title="Tìm nhanh (⌘K)"
        className="hidden sm:flex items-center gap-2 h-8 px-3 w-56 rounded-lg border border-zinc-200 text-zinc-400 hover:border-zinc-300 hover:text-zinc-600 transition-colors cursor-pointer"
      >
        <Search size={13} className="shrink-0" />
        <span className="flex-1 text-sm text-left select-none">Tìm nhanh...</span>
        <kbd className="text-[11px] font-mono bg-zinc-100 border border-zinc-200 rounded px-1.5 text-zinc-400 leading-none py-0.5">
          ⌘K
        </kbd>
      </button>

      <button
        title="Thông báo"
        className="size-8 rounded-lg flex items-center justify-center text-zinc-400 hover:text-zinc-600 hover:bg-zinc-100 transition-colors cursor-pointer"
      >
        <Bell size={15} />
      </button>
    </header>
  )
}
