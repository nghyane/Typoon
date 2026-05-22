import { create } from 'zustand'
import type { ReactNode } from 'react'

interface Crumb { label: string; to: string }

interface HeaderStore {
  crumbs: Crumb[]
  title:  string
  /** Optional center slot — route can inject any node (e.g. search input). */
  slot:   ReactNode
  /** Optional right slot — route can inject page-level actions
   *  (e.g. upload + overflow on /w/...). Renders before the Avatar
   *  so account controls stay rightmost. */
  actions: ReactNode
  set:    (title: string, crumbs?: Crumb[]) => void
  setSlot:(node: ReactNode) => void
  setActions: (node: ReactNode) => void
  clear:  () => void
}

export const useHeaderStore = create<HeaderStore>((set) => ({
  crumbs:     [],
  title:      '',
  slot:       null,
  actions:    null,
  set:        (title, crumbs = []) => set({ title, crumbs }),
  setSlot:    (node) => set({ slot: node }),
  setActions: (node) => set({ actions: node }),
  clear:      () => set({ title: '', crumbs: [], slot: null, actions: null }),
}))
