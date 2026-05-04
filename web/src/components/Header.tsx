import { Bell, Search, ArrowLeft, ChevronDown } from 'lucide-react'
import { Link } from '@tanstack/react-router'
import { useHeaderStore } from '../store/header'

export function Header() {
  const { crumbs, title } = useHeaderStore()
  const back = crumbs[0]

  return (
    <header className="flex items-center gap-2 px-5 h-[var(--spacing-bar)] bg-white shrink-0">

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
          <span className="text-sm font-semibold text-zinc-900 tracking-tight">{title}</span>
        ) : null}
      </div>

      <div className="flex items-center gap-2 h-8 px-3 w-48 rounded-lg border border-zinc-200 text-zinc-400 cursor-text hover:border-zinc-300 transition-colors">
        <Search size={13} className="shrink-0" />
        <span className="flex-1 text-sm select-none">Tìm nhanh...</span>
        <kbd className="text-[11px] font-mono bg-zinc-100 border border-zinc-200 rounded px-1.5 text-zinc-400 leading-none py-0.5">⌘K</kbd>
      </div>

      <button
        title="Thông báo"
        className="size-8 rounded-lg flex items-center justify-center text-zinc-400 hover:text-zinc-600 hover:bg-zinc-100 transition-colors cursor-pointer"
      >
        <Bell size={15} />
      </button>

      <button className="flex items-center gap-1.5 pl-1 pr-2 h-8 rounded-lg hover:bg-zinc-100 transition-colors cursor-pointer">
        <span className="size-6 rounded-full overflow-hidden bg-zinc-100 border border-zinc-200 shrink-0">
          <img src="https://api.dicebear.com/9.x/adventurer/svg?seed=typoon" alt="" className="size-full" />
        </span>
        <ChevronDown size={11} className="text-zinc-400" />
      </button>

    </header>
  )
}
