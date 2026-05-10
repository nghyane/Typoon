import { useState, useRef, useEffect, useMemo, type DragEvent } from 'react'
import { useMutation, useQueryClient } from '@tanstack/react-query'
import { Upload, Image as ImageIcon, Archive, X, AlertTriangle } from 'lucide-react'
import { uploadChapterZip, packPagesToZip, type UploadProgress } from '@typoon/upload-sdk'
import { api, type ApiProject } from '@shared/api/api'
import { cn } from '@shared/lib/cn'
import { Modal } from '@shared/ui/Modal'
import { Button } from '@shared/ui/Button'
import { input, label } from '@shared/ui/primitives'
import { UploadProgressFooter } from '@shared/ui/UploadProgressFooter'
import { toast } from '@shared/ui/Toaster'

interface Props {
  open:    boolean
  onClose: () => void
  project: ApiProject
  /** Chapter numbers already present in the project — used to suggest the next one. */
  existing: Set<string>
}

// Accepts the same set the engine knows how to unzip + the `.zip/.cbz`
// archive shape the SDK can pass through verbatim. PDF is dropped: the
// engine no longer rasterises PDFs, and pdf.js in the browser bundle
// would balloon the SPA. CLI ingest still handles PDF.
const ACCEPT = '.cbz,.zip,.png,.jpg,.jpeg,.webp,application/zip,image/*'

const IMAGE_EXT   = /\.(png|jpe?g|webp|bmp|tiff?)$/i
const ARCHIVE_EXT = /\.(zip|cbz)$/i

export function UploadChapterDialog({ open, onClose, project, existing }: Props) {
  const qc = useQueryClient()

  const [files,    setFiles]    = useState<File[]>([])
  const [number,   setNumber]   = useState<string>('')
  const [title,    setTitle]    = useState('')
  const [dragOver, setDragOver] = useState(false)
  const [progress, setProgress] = useState<UploadProgress | null>(null)
  const fileRef = useRef<HTMLInputElement>(null)

  // Reset on open.
  useEffect(() => {
    if (open) {
      setNumber(suggestNextNumber(existing))
      setTitle('')
      setFiles([])
      setProgress(null)
    }
  }, [open, existing])

  const upload = useMutation({
    mutationFn: async () => {
      // Phase 1: pack into zip. The SDK accepts the zip blob ready-made,
      // so we either pass through a user-supplied zip/cbz or build one
      // here from the picked images.
      setProgress({ phase: 'packing', bytesSent: 0, bytesTotal: 0, partsSent: 0, partsTotal: 0 })
      const zip = await buildZip(files)
      return uploadChapterZip(api, project.project_id, zip, {
        number: number.trim() || undefined,
        title:  title.trim() || undefined,
        onProgress: setProgress,
      })
    },
    onSuccess: (ch) => {
      qc.invalidateQueries({ queryKey: ['projects', project.project_id, 'chapters'] })
      qc.invalidateQueries({ queryKey: ['projects'] })
      // Server always enqueues scan from the multipart finalize path, so
      // workers + quota meters always need a refresh.
      qc.invalidateQueries({ queryKey: ['workers'] })
      qc.invalidateQueries({ queryKey: ['quota'] })
      toast.success(`Đã thêm và bắt đầu Ch.${ch.number} (${ch.page_count} trang)`)
      onClose()
    },
    onError: (e: Error) => toast.error(e.message),
    onSettled: () => setProgress(null),
  })

  const isArchive   = files.length === 1 && ARCHIVE_EXT.test(files[0]!.name)
  const totalSize   = useMemo(() => files.reduce((s, f) => s + f.size, 0), [files])
  const isPending   = upload.isPending
  const hasConflict = number.trim() !== '' && existing.has(number.trim())
  const valid       = files.length > 0 && !isPending

  const onDrop = (e: DragEvent<HTMLDivElement>) => {
    e.preventDefault()
    setDragOver(false)
    if (isPending) return
    const dropped = Array.from(e.dataTransfer.files)
    if (dropped.length > 0) addFiles(dropped)
  }

  const addFiles = (incoming: File[]) => {
    // If user drops a zip/cbz, replace the whole list (single-archive
    // upload). For images, accumulate so users can drag multiple times.
    const archive = incoming.find(f => ARCHIVE_EXT.test(f.name))
    if (archive) {
      setFiles([archive])
    } else {
      setFiles(prev => [...prev, ...incoming.filter(f => IMAGE_EXT.test(f.name))])
    }
  }

  // Footer is replaced wholesale during async upload — phase indicator
  // + dominant progress bar carry far more signal than two greyed-out
  // buttons. Modal X stays the abort path (currently triggers
  // mutation onSettled cleanup; a true cancel-token pass would let us
  // abort the in-flight XHR — TODO when SDK exposes AbortSignal).
  const footerCustom = isPending && progress
    ? <UploadProgressFooter progress={progress} />
    : undefined

  return (
    <Modal
      open={open}
      onClose={() => { if (!isPending) onClose() }}
      title={`Tải chương — ${project.title}`}
      size="md"
      footerLeft={!isPending && files.length > 0 ? (
        <FooterContext
          files={files}
          isArchive={isArchive}
          totalSize={totalSize}
          number={number}
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
            Tải lên & dịch
          </Button>
        </>
      ) : undefined}
      footerCustom={footerCustom}
    >
      <div className="px-5 py-4 space-y-4">
        {/* Compact drop zone — always visible, never replaced by the
            file list. Click anywhere to add more; users can drag in
            additional images on top of an existing selection. */}
        <DropZone
          dragOver={dragOver}
          isPending={isPending}
          hasFiles={files.length > 0}
          onDragOver={e => { e.preventDefault(); setDragOver(true) }}
          onDragLeave={() => setDragOver(false)}
          onDrop={onDrop}
          onClick={() => !isPending && fileRef.current?.click()}
        />
        <input
          ref={fileRef}
          type="file"
          accept={ACCEPT}
          multiple
          onChange={e => {
            if (e.target.files) addFiles(Array.from(e.target.files))
            e.target.value = ''
          }}
          className="hidden"
        />

        {/* File list — separate from the drop zone so the picker stays
            available for adding more. Renders thumbnails for images,
            a single-row card for archives. */}
        {files.length > 0 && (
          <FileList
            files={files}
            isArchive={isArchive}
            disabled={isPending}
            onClear={() => setFiles([])}
            onRemove={i => setFiles(files.filter((_, j) => j !== i))}
          />
        )}

        {/* Form fields. Labels right-aligned to inputs of equal weight
            instead of fighting a hard 120 px sidebar. */}
        <div className="grid grid-cols-[7rem_1fr] gap-x-3 gap-y-3">
          <div>
            <label className={label}>Số chương</label>
            <input
              type="text"
              value={number}
              onChange={e => setNumber(e.target.value)}
              placeholder="—"
              className={cn(input, hasConflict && 'border-warning')}
              disabled={isPending}
            />
          </div>
          <div>
            <div className="flex items-center justify-between mb-1.5">
              <label className={cn(label, 'mb-0')}>Tiêu đề</label>
              <span className="text-[11px] text-text-subtle">Tuỳ chọn</span>
            </div>
            <input
              type="text"
              value={title}
              onChange={e => setTitle(e.target.value)}
              placeholder="VD: Mở đầu"
              className={input}
              disabled={isPending}
            />
          </div>

          {hasConflict && (
            <div className="col-span-2 flex items-start gap-2 px-3 py-2 rounded-sm bg-warning-bg text-warning-text">
              <AlertTriangle size={13} className="mt-0.5 shrink-0" />
              <p className="text-xs leading-relaxed">
                Đã có chương <span className="font-semibold tabular">{number.trim()}</span> trong dự án.
                Bản tải lên sẽ được lưu như một bản dịch song song.
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
  dragOver:  boolean
  isPending: boolean
  hasFiles:  boolean
  onDragOver: (e: DragEvent<HTMLDivElement>) => void
  onDragLeave: () => void
  onDrop:     (e: DragEvent<HTMLDivElement>) => void
  onClick:    () => void
}) {
  return (
    <div
      onDragOver={onDragOver}
      onDragLeave={onDragLeave}
      onDrop={onDrop}
      onClick={onClick}
      className={cn(
        'rounded-md border-2 border-dashed flex items-center gap-3 px-4 cursor-pointer transition-all',
        hasFiles ? 'h-12' : 'h-20',
        dragOver
          ? 'border-accent bg-accent-bg scale-[1.005]'
          : 'border-border-soft bg-surface-2/40 hover:border-text-subtle',
        isPending && 'opacity-60 cursor-not-allowed pointer-events-none',
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
          <p className="text-xs text-text-subtle mt-0.5">CBZ, ZIP, hoặc nhiều ảnh</p>
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
  const trimmed = number.trim()
  return (
    <span className="flex items-center gap-1.5 truncate">
      <span className="text-text font-medium tabular">
        {isArchive ? '1 tệp' : `${files.length} ảnh`}
      </span>
      <span className="text-text-subtle/60">·</span>
      <span className="tabular">{fmtSize(totalSize)}</span>
      {trimmed && (
        <>
          <span className="text-text-subtle/60">·</span>
          <span className="tabular">Ch.{trimmed}</span>
        </>
      )}
    </span>
  )
}


async function buildZip(files: File[]): Promise<Blob> {
  if (files.length === 1 && ARCHIVE_EXT.test(files[0]!.name)) {
    // Already a zip/cbz — pass through. Engine's `unpack_zip` handles
    // it.
    return files[0]!
  }
  // Image set → store-mode zip ordered by filename. Engine natural-
  // sorts after unzip, so the SDK's 0001-prefix filename scheme just
  // preserves the user's drop order.
  const sorted = [...files].sort((a, b) => naturalCompare(a.name, b.name))
  const pages = await Promise.all(sorted.map(async f => ({
    source: f.name,
    bytes:  new Uint8Array(await f.arrayBuffer()),
  })))
  return packPagesToZip(pages)
}

function naturalCompare(a: string, b: string): number {
  return a.localeCompare(b, undefined, { numeric: true, sensitivity: 'base' })
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
          <Archive size={15} className="text-text-muted" />
        </div>
        <div className="flex-1 min-w-0">
          <p className="text-sm text-text truncate">{f.name}</p>
          <p className="text-xs text-text-subtle tabular">{fmtSize(f.size)}</p>
        </div>
        <Button variant="ghost" size="sm" icon onClick={onClear} disabled={disabled} title="Xoá">
          <X size={13} />
        </Button>
      </div>
    )
  }
  return (
    <div>
      <div className="flex items-center justify-between mb-2">
        <span className="text-xs text-text-muted">
          <span className="text-text font-medium tabular">{files.length}</span> ảnh
          <span className="text-text-subtle"> · sắp xếp theo tên tệp</span>
        </span>
        <button
          onClick={onClear}
          disabled={disabled}
          className="text-xs text-text-subtle hover:text-text cursor-pointer disabled:opacity-50"
        >
          Xoá tất cả
        </button>
      </div>
      <div className="grid grid-cols-[repeat(auto-fill,minmax(72px,1fr))] gap-1.5 max-h-44 overflow-auto overscroll-contain pr-0.5">
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
        <span className="text-[10px] text-white/90 truncate font-medium">
          {file.name.replace(/\.[^.]+$/, '')}
        </span>
      </div>
      {!disabled && (
        <button
          onClick={onRemove}
          className="absolute top-1 right-1 size-5 rounded-full bg-error text-white opacity-0 group-hover:opacity-100 flex items-center justify-center transition-opacity shadow"
          title="Xoá ảnh này"
          aria-label={`Xoá ${file.name}`}
        >
          <X size={11} />
        </button>
      )}
    </div>
  )
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
