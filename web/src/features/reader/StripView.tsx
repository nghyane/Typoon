// StripView — vertical strip (TTB direction). Webtoon / continuous
// reading. Each page is a slot that reserves its aspect ratio so
// the layout never jumps; only slots within the viewport ± buffer
// mount their <img> via IntersectionObserver.
//
// Settings-aware: respects `pageWidth`, `pageGap`, `stripMargin` so
// the user can tune density without prop drilling through every
// page. The wrapper width is the slider's source of truth — page
// images use `width: 100%` to follow it.

import { useEffect, useLayoutEffect, useMemo, useRef, useState } from 'react'

import { PageImage } from './PageImage'
import { LazyPageImage } from './LazyPageImage'
import { useReaderSettings } from './store'
import type { ReaderPage } from './types'
import type { InstalledSource } from '@features/browse/manifest/types'


interface Props {
  pages: ReaderPage[]
  urls?: ReadonlyMap<number, string>
  /** Raw source for lazy token resolution. Set when pages carry tokens. */
  rawSource?: InstalledSource
  /** Notified when the topmost intersecting page changes. Drives
   *  the page slider and resume-position writer. */
  onVisiblePageChange?: (index: number) => void
  /** Rendered after the last page slot. Reader uses this slot to
   *  inject the end-of-chapter CTA card. */
  endSlot?: React.ReactNode
}


const BUFFER = 3   // slots above + below the viewport that pre-mount <img>


export function StripView({
  pages, urls, rawSource, onVisiblePageChange, endSlot,
}: Props) {
  const { pageWidth, pageGap, stripMargin } = useReaderSettings()

  const refs = useRef<Array<HTMLDivElement | null>>([])

  // Stable key that changes whenever the page list is replaced.
  // Keying on indices (not just length) catches chapter switches that
  // happen to have the same number of pages — otherwise the IO effect
  // wouldn't re-attach and stale `visible` indices would survive into
  // the new chapter.
  const pagesKey = pages.map((p) => p.index).join(',')

  const [visible, setVisible] = useState<Set<number>>(() => new Set([0, 1, 2]))

  // Reset to the first few slots synchronously before the browser
  // paints whenever the chapter changes, so the strip always starts
  // at the top and never shows stale page indices from a previous chapter.
  useLayoutEffect(() => {
    setVisible(new Set([0, 1, 2]))
  // pagesKey is the only trigger — exhaustive-deps would want pages
  // which is referentially new every render.
  // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [pagesKey])

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
      // 150% gives ~1–1.5 screen of pre-load buffer without firing
      // all slots at once on chapters whose pages have no aspect ratio
      // (raw pages without width/height start at height 0, so 200%
      // would make every slot intersect immediately and mount all images).
      { rootMargin: '150% 0px' },
    )
    for (const el of refs.current) if (el) io.observe(el)
    return () => io.disconnect()
  // Re-attach whenever the page list identity changes.
  // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [pagesKey])

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
          {p.token && rawSource
            ? <LazyPageImage page={p} source={rawSource} inWindow={window.has(i)} />
            : <PageImage page={p} src={urls?.get(p.index) ?? p.url} inWindow={window.has(i)} />
          }
        </div>
      ))}
      {endSlot}
    </div>
  )
}
