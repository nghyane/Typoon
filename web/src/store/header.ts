import { create } from 'zustand'
import type { ReactNode } from 'react'

interface Crumb { label: string; to: string }

interface HeaderStore {
  crumbs: Crumb[]
  title:  string
  /** Optional center slot — route can inject any node (e.g. search input). */
  slot:   ReactNode
  set:    (title: string, crumbs?: Crumb[]) => void
  setSlot:(node: ReactNode) => void
  clear:  () => void
}

export const useHeaderStore = create<HeaderStore>((set) => ({
  crumbs: [],
  title:  '',
  slot:   null,
  set:    (title, crumbs = []) => set({ title, crumbs }),
  setSlot:(node) => set({ slot: node }),
  clear:  () => set({ title: '', crumbs: [], slot: null }),
}))
