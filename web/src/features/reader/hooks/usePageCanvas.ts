// usePageCanvas — draw a Blob onto a <canvas>, keep ImageBitmap alive.

import { useEffect, useRef, useState } from 'react'

export interface CanvasHandle {
  ref: React.RefObject<HTMLCanvasElement | null>
  bitmap: ImageBitmap | null
  ready: boolean
}

export function usePageCanvas(blob: Blob | null): CanvasHandle {
  const canvasRef = useRef<HTMLCanvasElement>(null)
  const bitmapRef = useRef<ImageBitmap | null>(null)
  const [ready, setReady] = useState(false)
  const [bitmap, setBitmap] = useState<ImageBitmap | null>(null)

  useEffect(() => {
    if (!blob || !canvasRef.current) return

    let cancelled = false
    const canvas = canvasRef.current

    createImageBitmap(blob).then(bmp => {
      if (cancelled) { bmp.close(); return }

      canvas.width = bmp.width
      canvas.height = bmp.height
      const ctx = canvas.getContext('2d')
      if (ctx) ctx.drawImage(bmp, 0, 0)

      bitmapRef.current?.close()
      bitmapRef.current = bmp
      setBitmap(bmp)
      setReady(true)
    })

    return () => {
      cancelled = true
      bitmapRef.current?.close()
      bitmapRef.current = null
      setReady(false)
      setBitmap(null)
    }
  }, [blob])

  return { ref: canvasRef, bitmap, ready }
}
