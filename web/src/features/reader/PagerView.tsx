// PagerView — single-page reader for LTR/RTL directions. One page
// visible at a time; user advances with tap zones, keyboard, swipe,
// or the bottom-bar slider.
//
// Image-fit aware (`width` / `height` / `free`):
//   - width:  page fits viewport width, scroll vertically inside.
//             Best for tall manga pages on portrait.
//   - height: page fits viewport height, scroll horizontally inside
//             a single page when too wide (spread pages, wide art).
//   - free:   intrinsic image size, both axes scrollable. Power-user
//             fallback for zoom.
//
// RTL is purely a navigation semantic (tap-zone wiring) — the image
// itself is not mirrored. Same applies to keyboard: parent owns the
// directional callbacks.

import { PageImage } from './PageImage'
import { LazyPageImage } from './LazyPageImage'
import { useReaderSettings } from './store'
import { cn } from '@shared/lib/cn'
import type { ReaderPage } from './types'
import type { InstalledSource } from '@features/browse/manifest/types'


interface Props {
  pages:    ReaderPage[]
  urls?:    ReadonlyMap<number, string>
  /** Raw source for lazy token resolution. Set when pages carry tokens. */
  rawSource?: InstalledSource
  page:     number
  /** Triggered when the user attempts to advance past the last page.
   *  Caller surfaces the end-of-chapter card. */
  onPastEnd?: () => void
}


export function PagerView({ pages, urls, rawSource, page, onPastEnd }: Props) {
  const { pageWidth, imageFit } = useReaderSettings()

  const total = pages.length
  if (total === 0) return null
  const safe = Math.min(Math.max(0, page), total - 1)
  const p    = pages[safe]!

  // Container sizing per fit mode. `width` constrains the page-image
  // wrapper to `pageWidth`; `height` constrains to viewport height;
  // `free` removes both constraints. Wrapping in a flex centre so
  // the page is always visually anchored regardless of fit.
  const wrapperStyle: React.CSSProperties =
    imageFit === 'width'
      ? { maxWidth: pageWidth, margin: '0 auto' }
    : imageFit === 'height'
      ? {
          // Subtract the floating-pill safe gap. The pill is
          // ~3.5rem tall + safe-area; using `100dvh` plus a fixed
          // 64px subtraction keeps the page centred in the visible
          // area without overlap on mobile DA.
          maxHeight: 'calc(100dvh - 64px)',
          width: 'auto',
          margin: '0 auto',
        }
    : { margin: '0 auto' }   // free

  return (
    <div
      className={cn(
        'flex justify-center',
        // Vertical scroll inside the wrapper when image is taller
        // than the viewport. Horizontal scroll handled by the inner
        // page wrapper when `height` fit produces a wide spread.
        imageFit === 'height' ? 'overflow-x-auto' : '',
      )}
      onClick={(e) => {
        // Detect past-end intent on the last page: a tap that comes
        // in via parent gestures already routes through tap zones,
        // so this handler is purely the fallback for keyboard-only
        // navigation that arrived here without going through zones.
        if (safe >= total - 1 && e.detail === 0) {
          onPastEnd?.()
        }
      }}
    >
      <div style={wrapperStyle}>
        {p.token && rawSource
          ? <LazyPageImage page={p} source={rawSource} inWindow />
          : <PageImage page={p} src={urls?.get(p.index) ?? p.url} inWindow />
        }
      </div>
    </div>
  )
}
