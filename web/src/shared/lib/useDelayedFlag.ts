import { useEffect, useState } from 'react'

/**
 * Returns true only after `value` has been true continuously for `delayMs`.
 * Use to gate skeleton placeholders so a fast network response doesn't
 * cause a single-frame skeleton flash:
 *
 *   const showSkeleton = useDelayedFlag(isPending, 250)
 *
 * - 0–250ms after fetch starts: nothing shown (page keeps previous data
 *   thanks to `placeholderData: keepPreviousData`, or stays empty).
 * - >250ms: skeleton appears. Slow networks show progress, fast ones
 *   never trigger.
 *
 * The flag also resets to false the moment `value` flips back to false,
 * so the skeleton unmounts immediately when data arrives.
 */
export function useDelayedFlag(value: boolean, delayMs = 250): boolean {
  const [delayed, setDelayed] = useState(false)

  useEffect(() => {
    if (!value) {
      setDelayed(false)
      return
    }
    const t = setTimeout(() => setDelayed(true), delayMs)
    return () => clearTimeout(t)
  }, [value, delayMs])

  return delayed
}
