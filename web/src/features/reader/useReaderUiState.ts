// useReaderUiState — transient UI state owned by the reader route.
// Settings sheet open, chapter list open, chrome hidden, peek
// toggle. Single source of truth so different surfaces never
// disagree on visibility.
//
// Not persisted: every value resets on route mount. Persist-worthy
// settings live in `useReaderSettings` (zustand-persist).

import { useCallback, useEffect, useRef, useState } from 'react'


export function useReaderUiState() {
  // Chrome visibility — top bar + bottom pill share this.
  const [chromeHidden, setChromeHidden] = useState(false)
  const lastYRef = useRef(0)
  const rafRef   = useRef<number | null>(null)

  useEffect(() => {
    lastYRef.current = window.scrollY
    const onScroll = () => {
      if (rafRef.current !== null) return
      rafRef.current = requestAnimationFrame(() => {
        const y = window.scrollY
        const dy = y - lastYRef.current
        if (Math.abs(dy) > 8) {
          setChromeHidden(dy > 0 && y > 80)
          lastYRef.current = y
        }
        rafRef.current = null
      })
    }
    window.addEventListener('scroll', onScroll, { passive: true })
    return () => {
      window.removeEventListener('scroll', onScroll)
      if (rafRef.current !== null) cancelAnimationFrame(rafRef.current)
    }
  }, [])

  const toggleChrome = useCallback(() => {
    setChromeHidden((h) => {
      lastYRef.current = window.scrollY
      return !h
    })
  }, [])

  // Sheets — open / close. Mutually exclusive at the surface level:
  // opening one closes the other so two sheets never overlap.
  const [settingsOpen,    setSettingsOpen]    = useState(false)
  const [chapterListOpen, setChapterListOpen] = useState(false)
  const [sourcesOpen,     setSourcesOpen]     = useState(false)

  const closeAll = useCallback(() => {
    setSettingsOpen(false)
    setChapterListOpen(false)
    setSourcesOpen(false)
  }, [])

  const openSettings = useCallback(() => {
    setChapterListOpen(false)
    setSourcesOpen(false)
    setSettingsOpen(true)
  }, [])
  const closeSettings = useCallback(() => setSettingsOpen(false), [])

  const openChapterList = useCallback(() => {
    setSettingsOpen(false)
    setSourcesOpen(false)
    setChapterListOpen(true)
  }, [])
  const closeChapterList = useCallback(() => setChapterListOpen(false), [])

  const openSources = useCallback(() => {
    setSettingsOpen(false)
    setChapterListOpen(false)
    setSourcesOpen(true)
  }, [])
  const closeSources = useCallback(() => setSourcesOpen(false), [])

  const anySheetOpen = settingsOpen || chapterListOpen || sourcesOpen

  return {
    chromeHidden,
    toggleChrome,
    settingsOpen,
    chapterListOpen,
    sourcesOpen,
    openSettings,
    closeSettings,
    openChapterList,
    closeChapterList,
    openSources,
    closeSources,
    closeAll,
    anySheetOpen,
  }
}
