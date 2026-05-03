import { Search, Bell, ChevronDown } from 'lucide-react'

export function TopBar() {
  return (
    <header className="flex items-center justify-end gap-3 px-6 h-[57px] shrink-0 bg-(--color-bg) border-b border-(--color-border)">
      {/* Search pill */}
      <div className="flex items-center gap-2 h-9 px-3 rounded-lg bg-(--color-surface) min-w-52">
        <Search size={13} className="text-(--color-text-3) shrink-0" />
        <input
          placeholder="Tìm nhanh..."
          className="flex-1 text-sm bg-transparent outline-none placeholder:text-(--color-text-3) text-(--color-text)"
        />
        <kbd className="text-[11px] text-(--color-text-3)">⌘</kbd>
        <kbd className="text-[11px] text-(--color-text-3)">K</kbd>
      </div>

      {/* Bell */}
      <button className="w-9 h-9 flex items-center justify-center rounded-lg hover:bg-(--color-surface) transition-colors text-(--color-text-2)">
        <Bell size={17} />
      </button>

      {/* Avatar */}
      <div className="flex items-center gap-1.5 cursor-pointer">
        <img
          src="https://api.dicebear.com/7.x/adventurer/svg?seed=typoon"
          className="w-8 h-8 rounded-full bg-(--color-surface)"
          alt="avatar"
        />
        <ChevronDown size={13} className="text-(--color-text-3)" />
      </div>
    </header>
  )
}
