// HeroDropZone — primary CTA on the home page.
//
// Accepts:
//   .zip / .cbz       passed through as-is
//   loose images      packed into a stored .zip in natural sort order
//   folder            traversed → all images packed
//
// File picker mode opens with `multiple` + `accept=image/*,.zip,.cbz`
// so a single dialog covers all three.

import { useCallback, useRef, useState } from 'react'
import { Upload, FileArchive } from 'lucide-react'
import { cn } from '@shared/lib/cn'

interface Props {
  /** Receives a DataTransfer (drop) OR a synthetic with `files` (picker). */
  onDrop:     (dt: DataTransfer | null, files: FileList | null) => void
  disabled?:  boolean
  hint?:      string
  className?: string
}

const ACCEPT = '.zip,.cbz,image/*,application/zip'

export function HeroDropZone({ onDrop, disabled, hint, className }: Props) {
  const inputRef = useRef<HTMLInputElement | null>(null)
  const [over, setOver] = useState(false)

  const handleDragOver = useCallback((e: React.DragEvent<HTMLDivElement>) => {
    e.preventDefault()
    if (!disabled) setOver(true)
  }, [disabled])

  const handleDragLeave = useCallback((e: React.DragEvent<HTMLDivElement>) => {
    e.preventDefault()
    setOver(false)
  }, [])

  const handleDrop = useCallback((e: React.DragEvent<HTMLDivElement>) => {
    e.preventDefault()
    setOver(false)
    if (disabled) return
    onDrop(e.dataTransfer, null)
  }, [disabled, onDrop])

  const handleFile = useCallback((e: React.ChangeEvent<HTMLInputElement>) => {
    if (disabled) return
    onDrop(null, e.target.files)
    e.target.value = ''
  }, [disabled, onDrop])

  return (
    <div
      role="button"
      tabIndex={disabled ? -1 : 0}
      aria-disabled={disabled}
      onClick={() => !disabled && inputRef.current?.click()}
      onKeyDown={(e) => {
        if (disabled) return
        if (e.key === 'Enter' || e.key === ' ') {
          e.preventDefault()
          inputRef.current?.click()
        }
      }}
      onDragOver={handleDragOver}
      onDragEnter={handleDragOver}
      onDragLeave={handleDragLeave}
      onDrop={handleDrop}
      className={cn(
        'relative w-full rounded-md border-2 border-dashed transition-colors',
        'px-6 py-12 flex flex-col items-center justify-center gap-3 text-center cursor-pointer',
        over
          ? 'border-accent bg-accent-bg/30'
          : 'border-border bg-surface hover:border-border-strong hover:bg-hover',
        disabled && 'opacity-60 cursor-not-allowed',
        className,
      )}
    >
      <input
        ref={inputRef}
        type="file"
        accept={ACCEPT}
        multiple
        className="hidden"
        onChange={handleFile}
        disabled={disabled}
      />

      <div className="size-12 rounded-full bg-surface-2 flex items-center justify-center">
        {over
          ? <Upload      size={22} className="text-accent" />
          : <FileArchive size={22} className="text-text-subtle" />
        }
      </div>

      <div className="space-y-1">
        <p className="text-sm font-medium text-text">
          {over ? 'Thả vào đây' : 'Thả zip, ảnh, hoặc folder để dịch'}
        </p>
        <p className="text-xs text-text-muted">
          Hỗ trợ <code>.zip</code> · <code>.cbz</code> · ảnh rời · folder
        </p>
        {hint && <p className="text-xs text-text-subtle">{hint}</p>}
      </div>
    </div>
  )
}


// ── Filename heuristics ──────────────────────────────────────────────

/** Best-effort source language detection from the filename. */
export function detectSourceLangFromName(name: string): string | null {
  const lc = name.toLowerCase()
  if (/\b(jp|jpn|jap|jajp|raw|生)\b/i.test(lc)) return 'ja'
  if (/\b(kr|kor|kokr)\b/i.test(lc))             return 'ko'
  if (/\b(cn|zh|chi)\b/i.test(lc))               return 'zh'
  if (/\b(en|eng)\b/i.test(lc))                  return 'en'
  return null
}
