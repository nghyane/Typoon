// TapZones — overlay div for the pager view that captures left/right
// taps to turn pages, center tap to toggle chrome.
//
// Respects reader direction (rtl flips left/right semantics) and the
// `clickTurn` setting (disables tap-to-turn but keeps center toggle).

import { useCallback } from 'react'

import { useReader } from '../ReaderContext'
import { useReaderSettings, styleToLayout } from '../settings'


interface Props {
  /** Source page count — used to clamp navigation. */
  pageCount: number
}


export function TapZones({ pageCount }: Props) {
  const { page, setPage, toggleChrome } = useReader()
  const { direction } = styleToLayout(useReaderSettings().style)
  const clickTurn = true

  const handleClick = useCallback((e: React.MouseEvent<HTMLDivElement>) => {
    const rect = e.currentTarget.getBoundingClientRect()
    const xPct = (e.clientX - rect.left) / rect.width

    // Center zone — toggle chrome regardless of clickTurn
    if (xPct > 0.33 && xPct < 0.67) {
      toggleChrome()
      return
    }
    if (!clickTurn) return

    const leftTap = xPct <= 0.33
    // RTL: left tap goes forward (manga style)
    const goNext = direction === 'rtl' ? leftTap : !leftTap

    const max = Math.max(0, pageCount - 1)
    if (goNext && page < max) setPage(page + 1)
    if (!goNext && page > 0)  setPage(page - 1)
  }, [page, pageCount, direction, clickTurn, setPage, toggleChrome])

  return (
    <div
      className="absolute inset-0 z-10"
      onClick={handleClick}
    />
  )
}
