import { create } from 'zustand'
import { persist } from 'zustand/middleware'

interface SidebarStore {
  collapsed: boolean
  toggle: () => void
  setCollapsed: (v: boolean) => void
}

// Default-collapsed on small screens (mobile, Activity sidebar mode).
// `< 640px` = Tailwind `sm` breakpoint.
const isSmallScreen = () =>
  typeof window !== 'undefined' && window.innerWidth < 640

export const useSidebar = create<SidebarStore>()(
  persist(
    (set) => ({
      collapsed: isSmallScreen(),
      toggle:    () => set((s) => ({ collapsed: !s.collapsed })),
      setCollapsed: (v) => set({ collapsed: v }),
    }),
    { name: 'typoon_sidebar' },
  ),
)
