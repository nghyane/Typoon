import { Search, Bell, ChevronDown } from 'lucide-react'

export function TopBar() {
  return (
    <header
      className="flex items-center justify-end gap-3 px-6 h-14 shrink-0"
      style={{ borderBottom: '1px solid var(--color-border-subtle)', background: 'var(--color-bg)' }}
    >
      {/* Search pill */}
      <div
        className="flex items-center gap-2 h-9 px-3 rounded-full"
        style={{ background: 'var(--color-surface)', border: '1px solid var(--color-border-subtle)', minWidth: 220 }}
      >
        <Search size={14} className="text-(--color-text-3) shrink-0" />
        <input
          placeholder="Tìm nhanh..."
          className="flex-1 text-sm bg-transparent outline-none placeholder:text-(--color-text-3) text-(--color-text-1)"
        />
        <div className="flex items-center gap-0.5 shrink-0">
          <kbd className="text-[11px] text-(--color-text-3)">⌘</kbd>
          <kbd className="text-[11px] text-(--color-text-3)">K</kbd>
        </div>
      </div>

      {/* Bell */}
      <button className="w-9 h-9 flex items-center justify-center rounded-full hover:bg-(--color-surface) transition-colors text-(--color-text-2)">
        <Bell size={18} />
      </button>

      {/* Avatar + chevron */}
      <div className="flex items-center gap-1 cursor-pointer">
        <div
          className="w-8 h-8 rounded-full overflow-hidden flex items-center justify-center text-sm font-semibold text-white shrink-0"
          style={{ background: 'linear-gradient(135deg, #636366, #48484a)' }}
        >
          U
        </div>
        <ChevronDown size={14} className="text-(--color-text-3)" />
      </div>
    </header>
  )
}
