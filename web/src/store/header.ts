import { create } from 'zustand'

interface Crumb { label: string; to: string }

interface HeaderStore {
  crumbs: Crumb[]
  title: string
  set: (title: string, crumbs?: Crumb[]) => void
  clear: () => void
}

export const useHeaderStore = create<HeaderStore>((set) => ({
  crumbs: [],
  title: '',
  set: (title, crumbs = []) => set({ title, crumbs }),
  clear: () => set({ title: '', crumbs: [] }),
}))
