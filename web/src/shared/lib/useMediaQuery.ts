// useMediaQuery — subscribe to a CSS media query. Returns the
// current match boolean. Re-renders on transitions across the
// breakpoint.
//
// Pattern: SSR-safe (`typeof window` guard at the entry point),
// uses the modern `addEventListener('change', …)` API (no
// `addListener` fallback — we don't ship to browsers old enough
// to need it).

import { useEffect, useState } from 'react'


export function useMediaQuery(query: string): boolean {
  const [matches, setMatches] = useState<boolean>(() => {
    if (typeof window === 'undefined') return false
    return window.matchMedia(query).matches
  })

  useEffect(() => {
    if (typeof window === 'undefined') return
    const mq = window.matchMedia(query)
    const onChange = (e: MediaQueryListEvent) => setMatches(e.matches)
    // Sync once on subscribe — query may have changed between SSR
    // initial state and effect attach.
    setMatches(mq.matches)
    mq.addEventListener('change', onChange)
    return () => mq.removeEventListener('change', onChange)
  }, [query])

  return matches
}


/** Convenience: matches Tailwind's `sm:` breakpoint (≥640px). */
export function useIsDesktop(): boolean {
  return useMediaQuery('(min-width: 640px)')
}
