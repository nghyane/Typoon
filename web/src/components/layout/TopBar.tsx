import { Search, Bell } from 'lucide-react'

export function TopBar() {
  return (
    <header className="flex items-center gap-3 px-4 h-14 border-b border-(--color-border) bg-(--color-bg) shrink-0">
      <div className="flex items-center gap-2 flex-1 max-w-sm h-8 px-3 rounded-md border border-(--color-border) bg-(--color-surface) focus-within:ring-2 focus-within:ring-(--color-accent) focus-within:border-(--color-accent) transition-all">
        <Search size={13} className="text-(--color-text-3) shrink-0" />
        <input
          placeholder="Search or jump to..."
          className="flex-1 text-sm bg-transparent outline-none placeholder:text-(--color-text-3) text-(--color-text-1)"
        />
        <kbd className="text-[11px] text-(--color-text-3) font-mono">⌘K</kbd>
      </div>

      <div className="flex-1" />

      <button className="w-8 h-8 flex items-center justify-center rounded-md hover:bg-(--color-surface) transition-colors text-(--color-text-2)">
        <Bell size={16} />
      </button>

      <div className="w-7 h-7 rounded-full bg-(--color-surface) border border-(--color-border) flex items-center justify-center text-xs font-semibold text-(--color-text-2) cursor-pointer">
        U
      </div>
    </header>
  )
}
