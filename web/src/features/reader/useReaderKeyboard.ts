// useReaderKeyboard — global keyboard shortcuts active while the
// reader route is mounted. Bound to window so the user doesn't
// have to click the page first.
//
//   ←        prev page / prev chapter (rtl-aware)
//   →        next page / next chapter
//   ↑ / k    scroll up one viewport (ttb only)
//   ↓ / j    scroll down one viewport (ttb only)
//   PgUp / PgDn  same as arrows but always step
//   [        prev chapter
//   ]        next chapter
//   t        toggle chrome
//   ,        open settings
//   c        open chapter list
//   Esc      close top-most sheet
//
// Skips when focus is in an input/textarea/contenteditable so the
// user can search / type comments without page-flipping by accident.

import { useEffect } from 'react'


export interface KeyboardHandlers {
  onPrev:           () => void
  onNext:           () => void
  onPrevChapter:    () => void
  onNextChapter:    () => void
  onToggleChrome:   () => void
  onOpenSettings:   () => void
  onOpenChapterList:() => void
}


export function useReaderKeyboard(h: KeyboardHandlers) {
  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      // Ignore when typing.
      const t = e.target as HTMLElement | null
      const tag = t?.tagName?.toLowerCase()
      if (tag === 'input' || tag === 'textarea' || t?.isContentEditable) return
      if (e.metaKey || e.ctrlKey || e.altKey) return

      switch (e.key) {
        case 'ArrowLeft':
        case 'PageUp':
          e.preventDefault()
          h.onPrev()
          break
        case 'ArrowRight':
        case 'PageDown':
        case ' ':            // space = page forward, common reader convention
          e.preventDefault()
          h.onNext()
          break
        case '[':
          e.preventDefault()
          h.onPrevChapter()
          break
        case ']':
          e.preventDefault()
          h.onNextChapter()
          break
        case 't':
          e.preventDefault()
          h.onToggleChrome()
          break
        case ',':
          e.preventDefault()
          h.onOpenSettings()
          break
        case 'c':
          e.preventDefault()
          h.onOpenChapterList()
          break
      }
    }
    window.addEventListener('keydown', onKey)
    return () => window.removeEventListener('keydown', onKey)
  }, [h])
}
