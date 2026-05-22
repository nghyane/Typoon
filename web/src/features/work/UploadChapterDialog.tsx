// UploadChapterDialog — "Tải lên chương" on a Work hub.
//
// Three-zone modal (drop zone → file list → chapter form). The
// design follows the global upload pattern (see
// `shared/ui/UploadProgressFooter`): action footer flips to a
// progress strip while the job uploads.
//
// v3.5 wiring:
//   • Target is a Work nanoid (no server material). The upload
//     becomes a translate job tied to `(work_id, chapter_ref)` via
//     `useSubmitJob` — same path the home-page drop zone uses.
//
//   • `existing` is the set of `numberNorm` already on the Work's
//     chapter spine (across every source AND prior uploads). Drives
//     the "Đã có chương N" warning + next-number suggestion.
//
// The job appears on the chapter list under "Tải lên" immediately
// after start; its state badge updates live via the
// `useLiveQuery`-backed job mirror.

import {
  useEffect, useMemo, useRef, useState, type DragEvent,
} from 'react'
import { useMutation } from '@tanstack/react-query'
import {
  AlertTriangle, Archive, Image as ImageIcon, Upload, X,
} from 'lucide-react'
import {
  packPagesToZip,
  ProgressTracker,
  type UploadProgress,
} from '@typoon/upload-sdk'

import { Modal } from '@shared/ui/Modal'
import { Button } from '@shared/ui/Button'
import { input as inputCls, label as labelCls } from '@shared/ui/primitives'
import { UploadProgressFooter } from '@shared/ui/UploadProgressFooter'
import { toast } from '@shared/ui/Toaster'
import { cn } from '@shared/lib/cn'
import { useSubmitJob } from '@features/jobs/useSubmitJob'


interface Props {
  open:    boolean
  onClose: () => void
  workId:      string
  workTitle:   string
  /** Default source language (work.source_lang). Job inherits this. */
  sourceLang:  string
  /** Chapter `numberNorm` values already on this Work, across every
   *  source AND prior uploads. */
  existing:    Set<string>
}


// Same accept set the pipeline knows how to unpack. PDF intentionally
// dropped (pdf.js bundle too big).
const ACCEPT = '.cbz,.zip,.png,.jpg,.jpeg,.webp,application/zip,image/*'
const IMAGE_EXT   = /\.(png|jpe?g|webp|bmp|tiff?)$/i
const ARCHIVE_EXT = /\.(zip|cbz)$/i


export function UploadChapterDialog({
  open, onClose, workId, workTitle, sourceLang, existing,
}: Props) {
  const { submit, progress } = useSubmitJob()

  const [files,    setFiles]    = useState<File[]>([])
  const [number,   setNumber]   = useState('')
  const [title,    setTitle]    = useState('')
  const [dragOver, setDragOver] = useState(false)
  /** Local phase tracker — `useSubmitJob` only exposes `{loaded,
   *  total}`, so we narrate the lifecycle here for
   *  `<UploadProgressFooter />`. */
  const [phase, setPhase] = useState<UploadProgress['phase']>('packing')
  const fileRef = useRef<HTMLInputElement>(null)

  // Reset state on every fresh open. Suggesting the next chapter
  // number is cheap and is the right default 90% of the time.
  useEffect(() => {
    if (open) {
      setNumber(suggestNextNumber(existing))
      setTitle('')
      setFiles([])
      setPhase('packing')
    }
  }, [open, existing])

  const upload = useMutation({
    mutationFn: async () => {
      const ref = number.trim() || suggestNextNumber(existing)
      setPhase('packing')
      const zip = await buildZip(files)
      setPhase('uploading')
      const out = await submit({
        work_id:     workId,
        chapter_ref: ref,
        source_lang: sourceLang,
        kind:        'translate',
        zip,
      })
      setPhase('finalizing')
      // Title denormalization is a future hook — the `jobs` mirror
      // doesn't have a label column today; the chapter row falls back
      // to "Ch.{ref}" until the reader writes one through
      // `history.chapter_label`. Keep the input alive so a future
      // denorm hook has somewhere to read from.
      void title.trim()
      return { ref, job_id: out.job_id }
    },
    onSuccess: ({ ref }) => {
      toast.success(`Đã gửi Ch.${ref} đi dịch`)
      onClose()
    },
    onError: (e: Error) => toast.error(e.message),
  })

  const isArchive   = files.length === 1 && ARCHIVE_EXT.test(files[0]!.name)
  const totalSize   = useMemo(
    () => files.reduce((s, f) => s + f.size, 0),
    [files],
  )
  const isPending   = upload.isPending
  const trimmedRef  = number.trim()
  const hasConflict = trimmedRef !== '' && existing.has(trimmedRef)
  const valid       = files.length > 0 && !isPending

  const onDrop = (e: DragEvent<HTMLDivElement>) => {
    e.preventDefault()
    setDragOver(false)
    if (isPending) return
    const dropped = Array.from(e.dataTransfer.files)
    if (dropped.length > 0) addFiles(dropped)
  }

  const addFiles = (incoming: File[]) => {
    // One archive → replace whole list. Many images → accumulate so
    // user can drag in additional batches without losing the picks.
    const archive = incoming.find(f => ARCHIVE_EXT.test(f.name))
    if (archive) {
      setFiles([archive])
      return
    }
    setFiles(prev => [
      ...prev,
      ...incoming.filter(f => IMAGE_EXT.test(f.name)),
    ])
  }

  // ProgressTracker shape that `<UploadProgressFooter />` consumes.
  // We synthesize parts from the byte counters — the job-submit hook
  // doesn't expose per-part deltas, but `UploadProgress.partsSent`
  // isn't read by the footer, so 0/1 is fine.
  const uploadProgress: UploadProgress = {
    phase,
    bytesSent:  progress.loaded,
    bytesTotal: progress.total,
    partsSent:  phase === 'finalizing' ? 1 : 0,
    partsTotal: 1,
  }

  const footerCustom = isPending
    ? <UploadProgressFooter progress={uploadProgress} />
    : undefined

  // `ProgressTracker` is imported for parity with the SDK contract
  // (future: switch to per-part deltas); silence the unused warning.
  void ProgressTracker

  return (
    <Modal
      open={open}
      onClose={() => { if (!isPending) onClose() }}
      title={`Tải chương — ${workTitle}`}
      size="md"
      footerLeft={!isPending && files.length > 0 ? (
        <FooterContext
          files={files}
          isArchive={isArchive}
          totalSize={totalSize}
          number={trimmedRef}
        />
      ) : undefined}
      footer={!isPending ? (
        <>
          <Button variant="ghost" onClick={onClose}>Huỷ</Button>
          <Button
            variant="primary"
            onClick={() => upload.mutate()}
            disabled={!valid}
          >
            <Upload size={14} />
            Tải lên
          </Button>
        </>
      ) : undefined}
      footerCustom={footerCustom}
    >
      <div className="px-5 py-4 space-y-4">
        <DropZone
          dragOver={dragOver}
          isPending={isPending}
          hasFiles={files.length > 0}
          onDragOver={(e) => { e.preventDefault(); setDragOver(true) }}
          onDragLeave={() => setDragOver(false)}
          onDrop={onDrop}
          onClick={() => !isPending && fileRef.current?.click()}
        />
        <input
          ref={fileRef}
          type="file"
          accept={ACCEPT}
          multiple
          onChange={(e) => {
            if (e.target.files) addFiles(Array.from(e.target.files))
            e.target.value = ''
          }}
          className="hidden"
        />

        {files.length > 0 && (
          <FileList
            files={files}
            isArchive={isArchive}
            disabled={isPending}
            onClear={() => setFiles([])}
            onRemove={(i) => setFiles(files.filter((_, j) => j !== i))}
          />
        )}

        <div className="grid grid-cols-[7rem_1fr] gap-x-3 gap-y-3">
          <div>
            <label className={labelCls}>Số chương</label>
            <input
              type="text"
              value={number}
              onChange={(e) => setNumber(e.target.value)}
              placeholder="—"
              className={cn(inputCls, hasConflict && 'border-warning-text')}
              disabled={isPending}
            />
          </div>
          <div>
            <div className="flex items-center justify-between mb-1.5">
              <label className={cn(labelCls, 'mb-0')}>Tiêu đề</label>
              <span className="text-xs text-text-subtle">Tuỳ chọn</span>
            </div>
            <input
              type="text"
              value={title}
              onChange={(e) => setTitle(e.target.value)}
              placeholder="VD: Mở đầu"
              className={inputCls}
              disabled={isPending}
            />
          </div>

          {hasConflict && (
            <div className="col-span-2 flex items-start gap-2 px-3 py-2 rounded-sm bg-warning-bg text-warning-text">
              <AlertTriangle size={14} className="mt-0.5 shrink-0" />
              <p className="text-xs leading-relaxed">
                Đã có chương <span className="font-semibold tabular-nums">{trimmedRef} </span>
                trên truyện này. Bản tải lên sẽ thành một bản song song dưới “Tải lên”.
              </p>
            </div>
          )}
        </div>
      </div>
    </Modal>
  )
}


// ── Internals ────────────────────────────────────────────────────────


function DropZone({
  dragOver, isPending, hasFiles,
  onDragOver, onDragLeave, onDrop, onClick,
}: {
  dragOver:    boolean
  isPending:   boolean
  hasFiles:    boolean
  onDragOver:  (e: DragEvent<HTMLDivElement>) => void
  onDragLeave: () => void
  onDrop:      (e: DragEvent<HTMLDivElement>) => void
  onClick:     () => void
}) {
  return (
    <div
      onDragOver={onDragOver}
      onDragLeave={onDragLeave}
      onDrop={onDrop}
      onClick={onClick}
      className={cn(
        'rounded-md border-2 border-dashed flex items-center gap-3 px-4',
        'transition-[background-color,border-color,transform] duration-150 cursor-pointer',
        hasFiles ? 'h-12' : 'h-20',
        dragOver
          ? 'border-accent bg-accent-bg scale-[1.005]'
          : 'border-border-soft bg-surface-2/40 hover:border-text-subtle',
        isPending && 'opacity-60 pointer-events-none',
      )}
    >
      <Upload size={hasFiles ? 14 : 18} className="text-text-subtle shrink-0" />
      <div className="flex-1 min-w-0">
        <p className={cn('font-medium text-text', hasFiles ? 'text-xs' : 'text-sm')}>
          {hasFiles ? 'Thêm tệp khác' : 'Kéo thả tệp vào đây'}
          <span className="text-text-subtle font-normal"> hoặc </span>
          <span className="underline underline-offset-2">chọn tệp</span>
        </p>
        {!hasFiles && (
          <p className="text-xs text-text-subtle mt-0.5">
            CBZ, ZIP, hoặc nhiều ảnh
          </p>
        )}
      </div>
    </div>
  )
}


function FooterContext({
  files, isArchive, totalSize, number,
}: {
  files:     File[]
  isArchive: boolean
  totalSize: number
  number:    string
}) {
  return (
    <span className="flex items-center gap-2 truncate">
      <span className="text-text font-medium tabular-nums">
        {isArchive ? '1 tệp' : `${files.length} ảnh`}
      </span>
      <span className="text-text-subtle/60">·</span>
      <span className="tabular-nums">{fmtSize(totalSize)}</span>
      {number && (
        <>
          <span className="text-text-subtle/60">·</span>
          <span className="tabular-nums">Ch.{number}</span>
        </>
      )}
    </span>
  )
}


function FileList({
  files, isArchive, disabled, onClear, onRemove,
}: {
  files:     File[]
  isArchive: boolean
  disabled:  boolean
  onClear:   () => void
  onRemove:  (i: number) => void
}) {
  if (isArchive) {
    const f = files[0]!
    return (
      <div className="flex items-center gap-3 px-3 py-2 rounded-sm bg-surface-2">
        <div className="size-9 rounded-sm bg-surface flex items-center justify-center shrink-0">
          <Archive size={14} className="text-text-muted" />
        </div>
        <div className="flex-1 min-w-0">
          <p className="text-sm text-text truncate">{f.name}</p>
          <p className="text-xs text-text-subtle tabular-nums">{fmtSize(f.size)}</p>
        </div>
        <Button variant="ghost" size="sm" icon onClick={onClear} disabled={disabled}>
          <X size={14} />
        </Button>
      </div>
    )
  }
  return (
    <div>
      <div className="flex items-center justify-between mb-2">
        <span className="text-xs text-text-muted">
          <span className="text-text font-medium tabular-nums">{files.length}</span> ảnh
          <span className="text-text-subtle"> · sắp xếp theo tên tệp</span>
        </span>
        <Button variant="ghost" size="sm" onClick={onClear} disabled={disabled}>
          Xoá tất cả
        </Button>
      </div>
      <div className="grid grid-cols-[repeat(auto-fill,minmax(72px,1fr))] gap-2 max-h-44 overflow-auto overscroll-contain pr-0.5">
        {files.map((f, i) => (
          <Thumb
            key={`${f.name}-${i}-${f.lastModified}`}
            file={f}
            disabled={disabled}
            onRemove={() => onRemove(i)}
          />
        ))}
      </div>
    </div>
  )
}


function Thumb({
  file, disabled, onRemove,
}: {
  file:     File
  disabled: boolean
  onRemove: () => void
}) {
  const [src, setSrc] = useState<string | null>(null)
  const [failed, setFailed] = useState(false)

  useEffect(() => {
    const url = URL.createObjectURL(file)
    setSrc(url)
    return () => URL.revokeObjectURL(url)
  }, [file])

  return (
    <div
      className="relative group aspect-[3/4] rounded-xs bg-surface-2 overflow-hidden"
      title={file.name}
    >
      {src && !failed ? (
        <img
          src={src}
          alt={file.name}
          loading="lazy"
          className="absolute inset-0 w-full h-full object-cover"
          onError={() => setFailed(true)}
        />
      ) : (
        <div className="absolute inset-0 flex items-center justify-center">
          <ImageIcon size={16} className="text-text-subtle" />
        </div>
      )}
      <div className="absolute inset-x-0 bottom-0 h-6 bg-gradient-to-t from-black/70 to-transparent flex items-end px-1.5 pb-1">
        <span className="text-xs text-white/90 truncate font-medium">
          {file.name.replace(/\.[^.]+$/, '')}
        </span>
      </div>
      {!disabled && (
        <button
          type="button"
          onClick={onRemove}
          className={cn(
            'absolute top-1 right-1 size-5 rounded-full',
            'bg-error-text text-white shadow',
            'flex items-center justify-center',
            'opacity-0 group-hover:opacity-100 transition-opacity cursor-pointer',
          )}
          aria-label={`Xoá ${file.name}`}
        >
          <X size={12} />
        </button>
      )}
    </div>
  )
}


async function buildZip(files: File[]): Promise<Blob> {
  if (files.length === 1 && ARCHIVE_EXT.test(files[0]!.name)) {
    return files[0]!
  }
  const sorted = [...files].sort((a, b) => naturalCompare(a.name, b.name))
  const pages = await Promise.all(sorted.map(async (f) => ({
    source: f.name,
    bytes:  new Uint8Array(await f.arrayBuffer()),
  })))
  return packPagesToZip(pages)
}


function naturalCompare(a: string, b: string): number {
  return a.localeCompare(b, undefined, { numeric: true, sensitivity: 'base' })
}


function fmtSize(b: number): string {
  if (b < 1024)              return `${b} B`
  if (b < 1024 * 1024)       return `${(b / 1024).toFixed(0)} KB`
  if (b < 1024 * 1024 * 1024) return `${(b / 1024 / 1024).toFixed(1)} MB`
  return `${(b / 1024 / 1024 / 1024).toFixed(2)} GB`
}


function suggestNextNumber(existing: Set<string>): string {
  let max = 0
  for (const s of existing) {
    const n = Number(s)
    if (Number.isFinite(n) && n > max) max = n
  }
  if (max === 0) return '1'
  return String(Number.isInteger(max) ? max + 1 : Math.floor(max) + 1)
}
