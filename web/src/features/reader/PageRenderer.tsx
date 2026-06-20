// PageRenderer — image surface with optional translation overlay.
//
// When `blob` is provided, renders via <canvas> (no re-fetch, no CORS).
// Falls back to <img> + source.getUrl() when `source` is used.

import { useCallback, useState, type SyntheticEvent } from 'react'
import { cn } from '@shared/lib/cn'
import { usePageUrl } from './hooks/usePageUrl'
import { usePageCanvas } from './hooks/usePageCanvas'
import { PageErrorState } from './components/PageErrorState'
import { PagePlaceholder } from './components/PagePlaceholder'
import type { ReaderSource } from './sources'

interface Props {
  source?: ReaderSource
  blob?:   Blob | null
  index:   number
  className?: string
}

export function PageRenderer({ source, blob, index, className }: Props) {
  // Canvas path — when blob is available directly.
  if (blob !== undefined) {
    return <CanvasPage blob={blob} index={index} className={className} />
  }

  // Legacy <img> path — uses source.getUrl().
  if (!source) return null
  return <ImagePage source={source} index={index} className={className} />
}


function CanvasPage({ blob, index, className }: { blob: Blob | null; index: number; className?: string }) {
  const { ref, ready, bitmap } = usePageCanvas(blob)

  const aspect = bitmap
    ? { paddingTop: `${(bitmap.height / bitmap.width) * 100}%` }
    : { paddingTop: '150%' }

  if (!blob) {
    return <PagePlaceholder page={{ index, width: null, height: null }} aspectRatio={null} className={className} busy />
  }

  return (
    <div className={cn('relative w-full', className)} style={aspect}>
      <canvas
        ref={ref}
        className="absolute inset-0 w-full h-full object-contain"
        style={{ display: ready ? 'block' : 'none' }}
      />
      {!ready && <PagePlaceholder page={{ index, width: null, height: null }} aspectRatio={null} className={className} busy />}
    </div>
  )
}


function ImagePage({ source, index, className }: { source: ReaderSource; index: number; className?: string }) {
  const { status, url, error, retry } = usePageUrl(source, index)
  const page = source.pages[index]

  const [naturalAspect, setNaturalAspect] = useState<number | null>(null)
  const onLoad = useCallback((e: SyntheticEvent<HTMLImageElement>) => {
    const img = e.currentTarget
    if (img.naturalWidth > 0 && img.naturalHeight > 0) {
      setNaturalAspect(img.naturalHeight / img.naturalWidth)
    }
  }, [])

  const aspectRatio = naturalAspect
    ?? (page?.width && page?.height ? page.height / page.width : null)
  const aspect = aspectRatio
    ? { paddingTop: `${aspectRatio * 100}%` }
    : { paddingTop: '150%' }

  if (status === 'error') {
    return <PageErrorState page={page} aspectRatio={aspectRatio} error={error} onRetry={retry} className={className} />
  }
  if (status !== 'ready' || !url) {
    return <PagePlaceholder page={page} aspectRatio={aspectRatio} className={className} busy />
  }

  return (
    <div className={cn('relative w-full', className)} style={aspect}>
      <img src={url} alt={`Trang ${index + 1}`}
        className="absolute inset-0 w-full h-full object-contain select-none"
        draggable={false} onLoad={onLoad} />
    </div>
  )
}
