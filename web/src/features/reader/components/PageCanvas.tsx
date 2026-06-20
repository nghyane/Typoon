// PageCanvas — render a Blob to <canvas>, accept overlay.
//
// No <img>, no blob URLs.  Canvas pixel data is the source of truth
// for both display and future OCR input.

import { memo, type CSSProperties } from 'react'
import { usePageCanvas } from '../hooks/usePageCanvas'

interface Props {
  blob: Blob | null
  style?: CSSProperties
  className?: string
  children?: React.ReactNode
}

export const PageCanvas = memo(function PageCanvas({ blob, style, className, children }: Props) {
  const { ref } = usePageCanvas(blob)

  return (
    <div style={{ position: 'relative', ...style }} className={className}>
      <canvas
        ref={ref}
        style={{ display: 'block', width: '100%', height: '100%', objectFit: 'contain' }}
      />
      {children}
    </div>
  )
})
