// Windowed continuous reader. Every page has a slot reserving its
// aspect ratio (zero CLS, instant scrollbar), but only slots within
// the visible viewport ± a few-page buffer mount the <img>. Far-away
// slots stay as empty divs — DOM stays bounded by viewport size.
//
// IntersectionObserver tracks which slots are visible; we expand the
// visible set with a buffer so scrolling never lands on an empty page
// for more than ~1 frame.

import { useEffect, useMemo, useRef, useState } from 'react'
import { PageImage } from './PageImage'
import type { ReaderPage } from './types'

interface Props {
  pages: ReaderPage[]
  /** When present, blob URLs from a streaming source (BNL). Falls
   *  back to `page.url` on each entry. */
  urls?: ReadonlyMap<number, string>
}

const BUFFER = 3   // slots above + below the viewport that pre-mount <img>

export function ContinuousView({ pages, urls }: Props) {
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
      // 200% rootMargin: start mounting before the slot enters the
      // viewport so the bitmap is ready by the time it scrolls in.
      { rootMargin: '200% 0px' },
    )
    for (const el of refs.current) if (el) io.observe(el)
    return () => io.disconnect()
  }, [pages.length])

  // Expand visible set with neighbours so a slot just outside the IO
  // window still mounts when its neighbour scrolls in.
  const window = useMemo(() => {
    const w = new Set<number>()
    for (const i of visible) {
      for (let k = i - BUFFER; k <= i + BUFFER; k++) {
        if (k >= 0 && k < pages.length) w.add(k)
      }
    }
    return w
  }, [visible, pages.length])

  return (
    <div className="max-w-3xl mx-auto">
      {pages.map((p, i) => (
        <div
          key={p.index}
          ref={(el) => { refs.current[i] = el }}
          data-idx={i}
        >
          <PageImage
            page={p}
            src={urls?.get(p.index) ?? p.url}
            inWindow={window.has(i)}
          />
        </div>
      ))}
    </div>
  )
}
