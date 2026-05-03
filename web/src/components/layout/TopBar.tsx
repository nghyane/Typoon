import { Search, Bell } from 'lucide-react'

export function TopBar() {
  return (
    <header className="flex items-center gap-4 px-6 h-14 border-b border-(--color-border) bg-(--color-bg) shrink-0">
      <div className="flex items-center gap-2 flex-1 max-w-xs px-3 h-8 rounded-lg border border-(--color-border) bg-(--color-surface-1)">
        <Search size={13} className="text-(--color-text-3) shrink-0" />
        <input
          placeholder="Tìm nhanh..."
          className="flex-1 text-sm bg-transparent outline-none placeholder:text-(--color-text-3) text-(--color-text-1)"
        />
        <kbd className="text-[10px] px-1 rounded bg-(--color-surface-2) text-(--color-text-3)">⌘K</kbd>
      </div>

      <div className="flex-1" />

      <button className="w-8 h-8 rounded-lg flex items-center justify-center hover:bg-(--color-surface-1) transition-colors text-(--color-text-2)">
        <Bell size={16} />
      </button>

      <div className="w-8 h-8 rounded-full flex items-center justify-center text-xs font-bold bg-(--color-surface-2) text-(--color-text-2) cursor-pointer shrink-0">
        U
      </div>
    </header>
  )
}
