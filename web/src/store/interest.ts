import { useEffect } from 'react'
import { create } from 'zustand'

/**
 * Per-tab set of project IDs the user is currently looking at.
 *
 * The SSE hook reads this to narrow its server-side subscription so a
 * tab opened on /projects/9 doesn't receive PageDone bursts from every
 * other project still rendering. Routes register interest on mount and
 * clear it on unmount via `useProjectInterest(id)`.
 */
interface InterestStore {
  ids: ReadonlySet<number>
  add: (id: number) => void
  remove: (id: number) => void
}

const useInterestStore = create<InterestStore>((set, get) => ({
  ids: new Set(),
  add: (id) => {
    if (get().ids.has(id)) return
    const next = new Set(get().ids)
    next.add(id)
    set({ ids: next })
  },
  remove: (id) => {
    if (!get().ids.has(id)) return
    const next = new Set(get().ids)
    next.delete(id)
    set({ ids: next })
  },
}))

export function useProjectInterest(id: number | null | undefined) {
  const add    = useInterestStore((s) => s.add)
  const remove = useInterestStore((s) => s.remove)
  useEffect(() => {
    if (id == null || isNaN(id)) return
    add(id)
    return () => remove(id)
  }, [id, add, remove])
}

/** Snapshot of currently-tracked project IDs for the SSE subscription. */
export function useProjectInterestList(): readonly number[] {
  const ids = useInterestStore((s) => s.ids)
  // Stable order so the SSE hook's effect doesn't re-fire on every
  // render. Sorted numbers; identity changes only when membership does.
  return [...ids].sort((a, b) => a - b)
}
