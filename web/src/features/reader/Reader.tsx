// ReaderBody — page surface. Strip (TTB) or Pager (LTR/RTL) based
// on the work's saved direction. Tap zones + hover edges sit on
// top of the body as overlays; sheets / modals stack above.

import type { ReactNode } from 'react'

import { StripView } from './StripView'
import { PagerView } from './PagerView'
import { TapZones } from './TapZones'
import { HoverEdges } from './HoverEdges'
import { useReaderSettings, type Direction } from './store'
import type { ReaderSource } from './types'


interface Props {
  source:        ReaderSource
  direction:     Direction
  page:          number
  pastEnd:       boolean
  endSlot:       ReactNode
  onVisiblePageChange: (idx: number) => void
  onPastEnd:     () => void
  /** Tap zones / hover edges callbacks. */
  onPrev:        () => void
  onNext:        () => void
  onTogglePeek:  () => void
  /** True when a sheet is open — TapZones and HoverEdges pause so
   *  clicks fall through to the sheet's own overlay. */
  inputDisabled: boolean
}


export function ReaderBody({
  source, direction, page, pastEnd, endSlot,
  onVisiblePageChange, onPastEnd,
  onPrev, onNext, onTogglePeek,
  inputDisabled,
}: Props) {
  const clickTurn = useReaderSettings((s) => s.clickTurnPage)

  const isTTB = direction === 'ttb'
  const isRTL = direction === 'rtl'

  const body = isTTB
    ? (
      <StripView
        pages={source.pages}
        urls={source.urls}
        rawSource={source.rawSource}
        onVisiblePageChange={onVisiblePageChange}
        endSlot={endSlot}
      />
    )
    : pastEnd
      ? <div className="max-w-3xl mx-auto pt-4">{endSlot}</div>
      : (
        <PagerView
          pages={source.pages}
          urls={source.urls}
          rawSource={source.rawSource}
          page={page}
          onPastEnd={onPastEnd}
        />
      )

  return (
    <div className="relative min-h-dvh">
      {body}
      {clickTurn && (
        <TapZones
          onPrev={onPrev}
          onNext={onNext}
          onTogglePeek={onTogglePeek}
          rtl={isRTL}
          disabled={inputDisabled}
        />
      )}
      <HoverEdges
        onLeft={isRTL ? onNext : onPrev}
        onRight={isRTL ? onPrev : onNext}
        rtl={isRTL}
        disabled={inputDisabled}
      />
    </div>
  )
}
