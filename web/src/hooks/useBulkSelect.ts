import { useState, useCallback } from 'react'

export function useBulkSelect<T extends { chapter_id: number }>(items: T[]) {
  const [selected, setSelected] = useState<Set<number>>(new Set())

  const toggle = useCallback((id: number) => {
    setSelected((s) => {
      const next = new Set(s)
      next.has(id) ? next.delete(id) : next.add(id)
      return next
    })
  }, [])

  const toggleAll = useCallback(() => {
    setSelected((s) =>
      s.size === items.length
        ? new Set()
        : new Set(items.map((i) => i.chapter_id))
    )
  }, [items])

  const clear = useCallback(() => setSelected(new Set()), [])

  return { selected, toggle, toggleAll, clear, isAllSelected: selected.size === items.length }
}
