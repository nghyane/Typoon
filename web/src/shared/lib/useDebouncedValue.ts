import { useEffect, useState } from 'react'

/** Debounce a fast-changing value. Returns the value after `delay`
 *  ms of stability. Use for search inputs where each keystroke
 *  shouldn't fire a network request. */
export function useDebouncedValue<T>(value: T, delay = 250): T {
  const [debounced, setDebounced] = useState(value)
  useEffect(() => {
    const t = setTimeout(() => setDebounced(value), delay)
    return () => clearTimeout(t)
  }, [value, delay])
  return debounced
}
