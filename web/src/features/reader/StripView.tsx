// StripView — vertical strip (TTB direction). Webtoon / continuous
// reading. Each page is a slot that reserves its aspect ratio so
// the layout never jumps; only slots within the viewport ± buffer
// mount their <img> via IntersectionObserver.
//
// Settings-aware: respects `pageWidth`, `pageGap`, `stripMargin` so
// the user can tune density without prop drilling through every
// page. The wrapper width is the slider's source of truth — page
// images use `width: 100%` to follow it.

import { useEffect, useMemo, useRef, useState } from 'react'

import { PageImage } from './PageImage'
import { useReaderSettings } from './store'
import type { ReaderPage } from './types'


interface Props {
  pages: ReaderPage[]
  urls?: ReadonlyMap<number, string>
  /** Notified when the topmost intersecting page changes. Drives
   *  the page slider and resume-position writer. */
  onVisiblePageChange?: (index: number) => void
  /** Rendered after the last page slot. Reader uses this slot to
   *  inject the end-of-chapter CTA card. */
  endSlot?: React.ReactNode
}


const BUFFER = 3   // slots above + below the viewport that pre-mount <img>


export function StripView({
  pages, urls, onVisiblePageChange, endSlot,
}: Props) {
  const { pageWidth, pageGap, stripMargin } = useReaderSettings()

  const refs = useRef<Array<HTMLDivElement | null>>([])
  const [visible, setVisible] = useState<Set<number>>(() => new Set([0, 1, 2]))

  useEffect(() => {
    const io = new IntersectionObserver(
      (entries) => {
        setVisible((prev) => {
          const next = new Set(prev)
          let dirty = false
          for (const e of entries) {
            const i = Number((e.target as HTMLElement).dataset.idx)
            if (e.isIntersecting) {
              if (!next.has(i)) { next.add(i); dirty = true }
            } else {
              if (next.has(i))  { next.delete(i); dirty = true }
            }
          }
          return dirty ? next : prev
        })
      },
      { rootMargin: '200% 0px' },
    )
    for (const el of refs.current) if (el) io.observe(el)
    return () => io.disconnect()
  }, [pages.length])

  // Expand visible with neighbours so a slot just past the IO root
  // margin still mounts when its neighbour scrolls in.
  const window = useMemo(() => {
    const w = new Set<number>()
    for (const i of visible) {
      for (let k = i - BUFFER; k <= i + BUFFER; k++) {
        if (k >= 0 && k < pages.length) w.add(k)
      }
    }
    return w
  }, [visible, pages.length])

  const topVisible = useMemo(() => {
    let min = Infinity
    for (const i of visible) if (i < min) min = i
    return Number.isFinite(min) ? min : 0
  }, [visible])

  useEffect(() => {
    onVisiblePageChange?.(topVisible)
  }, [topVisible, onVisiblePageChange])

  return (
    <div
      className="mx-auto"
      style={{
        maxWidth: pageWidth,
        paddingTop: stripMargin,
        paddingBottom: stripMargin,
      }}
    >
      {pages.map((p, i) => (
        <div
          key={p.index}
          ref={(el) => { refs.current[i] = el }}
          data-idx={i}
          style={i < pages.length - 1 ? { marginBottom: pageGap } : undefined}
        >
          <PageImage
            page={p}
            src={urls?.get(p.index) ?? p.url}
            inWindow={window.has(i)}
          />
        </div>
      ))}
      {endSlot}
    </div>
  )
}
