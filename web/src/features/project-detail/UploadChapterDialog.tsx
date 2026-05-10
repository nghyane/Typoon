import { useState, useRef, useEffect, useMemo, type DragEvent } from 'react'
import { useMutation, useQueryClient } from '@tanstack/react-query'
import { Upload, Image as ImageIcon, Archive, X } from 'lucide-react'
import { uploadChapterZip, packPagesToZip, type UploadProgress } from '@typoon/upload-sdk'
import { api, type ApiProject } from '@shared/api/api'
import { cn } from '@shared/lib/cn'
import { Modal } from '@shared/ui/Modal'
import { Button } from '@shared/ui/Button'
import { input, label, Spinner } from '@shared/ui/primitives'
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

  const isArchive = files.length === 1 && ARCHIVE_EXT.test(files[0]!.name)
  const totalSize = useMemo(() => files.reduce((s, f) => s + f.size, 0), [files])
  const isPending = upload.isPending

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

  const valid = files.length > 0 && !isPending

  return (
    <Modal
      open={open}
      onClose={() => { if (!isPending) onClose() }}
      title={`Tải chương — ${project.title}`}
      size="md"
      footer={
        <>
          <Button onClick={onClose} disabled={isPending}>Huỷ</Button>
          <Button
            variant="primary"
            onClick={() => upload.mutate()}
            disabled={!valid}
          >
            {isPending && <Spinner />}
            <Upload size={14} />
            Tải lên & dịch
          </Button>
        </>
      }
    >
      <div className="px-5 py-4 space-y-4">
        {/* Drop zone */}
        <div
          onDragOver={e => { e.preventDefault(); setDragOver(true) }}
          onDragLeave={() => setDragOver(false)}
          onDrop={onDrop}
          onClick={() => !isPending && fileRef.current?.click()}
          className={cn(
            'rounded-md border-2 border-dashed flex flex-col items-center justify-center cursor-pointer transition-colors',
            'min-h-32 p-6 text-center',
            dragOver
              ? 'border-accent bg-accent-bg'
              : 'border-border-soft bg-surface-2/40 hover:border-text-subtle',
            isPending && 'opacity-60 cursor-not-allowed',
          )}
        >
          {files.length === 0 ? (
            <>
              <Upload size={20} className="text-text-subtle mb-2" />
              <p className="text-sm font-medium text-text">
                Kéo thả tệp vào đây hoặc <span className="underline">chọn tệp</span>
              </p>
              <p className="text-xs text-text-subtle mt-1">Hỗ trợ: CBZ, ZIP, hoặc nhiều ảnh</p>
            </>
          ) : (
            <FileList
              files={files}
              isArchive={isArchive}
              onClear={() => setFiles([])}
              onRemove={i => setFiles(files.filter((_, j) => j !== i))}
            />
          )}
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
        </div>

        <div className="grid grid-cols-[120px_1fr] gap-3">
          <div>
            <label className={label}>Số chương</label>
            <input
              type="text"
              value={number}
              onChange={e => setNumber(e.target.value)}
              placeholder="VD: 4, 4.5, Extra"
              className={input}
              disabled={isPending}
            />
          </div>
          <div>
            <label className={label}>Tiêu đề (tuỳ chọn)</label>
            <input
              type="text"
              value={title}
              onChange={e => setTitle(e.target.value)}
              placeholder="VD: Mở đầu"
              className={input}
              disabled={isPending}
            />
          </div>
        </div>

        {files.length > 0 && !progress && (
          <p className="text-xs text-text-subtle">
            {files.length === 1 ? '1 tệp' : `${files.length} ảnh`} · {fmtSize(totalSize)}
          </p>
        )}

        {progress && <ProgressBar progress={progress} />}

        {existing.has(number.trim()) && (
          <p className="text-xs text-warning-text">
            Chương {number.trim()} đã tồn tại — bản tải lên sẽ chèn vào danh sách như một bản dịch khác.
          </p>
        )}
      </div>
    </Modal>
  )
}


// ── Internals ────────────────────────────────────────────────────────


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

function ProgressBar({ progress }: { progress: UploadProgress }) {
  const { phase, bytesSent, bytesTotal, partsSent, partsTotal, speedBps, etaSeconds } = progress
  const pct = bytesTotal > 0 ? Math.min(100, (bytesSent / bytesTotal) * 100) : 0
  return (
    <div className="space-y-1">
      <div className="h-1.5 rounded-full bg-surface-2 overflow-hidden">
        <div
          className="h-full bg-accent transition-[width] duration-200 ease-out"
          style={{ width: phase === 'packing' ? '5%' : `${pct}%` }}
        />
      </div>
      <div className="flex justify-between text-xs text-text-subtle">
        <span>{phaseLabel(phase, partsSent, partsTotal)}</span>
        <span>
          {phase === 'uploading' ? (
            <>
              {fmtSize(bytesSent)}/{fmtSize(bytesTotal)}
              {speedBps && ` · ${fmtSpeed(speedBps)}`}
              {etaSeconds !== undefined && etaSeconds > 0 && ` · ${fmtEta(etaSeconds)}`}
            </>
          ) : phase === 'finalizing'
            ? 'Đang xử lý…'
            : null}
        </span>
      </div>
    </div>
  )
}

function phaseLabel(phase: UploadProgress['phase'], partsSent: number, partsTotal: number): string {
  if (phase === 'packing')    return 'Đang đóng gói…'
  if (phase === 'uploading')  return `Tải lên (${partsSent}/${partsTotal})`
  return 'Engine đang xử lý'
}

function FileList({
  files, isArchive, onClear, onRemove,
}: {
  files:    File[]
  isArchive: boolean
  onClear:  () => void
  onRemove: (i: number) => void
}) {
  if (isArchive) {
    const f = files[0]!
    return (
      <div
        className="w-full flex items-center gap-3 px-3 py-2 rounded-sm bg-surface-2"
        onClick={e => e.stopPropagation()}
      >
        <div className="size-9 rounded-sm bg-surface flex items-center justify-center shrink-0">
          <Archive size={15} className="text-text-muted" />
        </div>
        <div className="flex-1 min-w-0 text-left">
          <p className="text-sm text-text truncate">{f.name}</p>
          <p className="text-xs text-text-subtle">{fmtSize(f.size)}</p>
        </div>
        <button
          onClick={onClear}
          className="size-7 rounded-sm flex items-center justify-center text-text-subtle hover:text-text hover:bg-hover cursor-pointer"
        >
          <X size={13} />
        </button>
      </div>
    )
  }
  return (
    <div className="w-full" onClick={e => e.stopPropagation()}>
      <div className="flex items-center justify-between mb-2">
        <span className="text-xs text-text-muted">
          {files.length} ảnh — sẽ sắp xếp theo tên tệp
        </span>
        <button
          onClick={onClear}
          className="text-xs text-text-muted hover:text-text cursor-pointer"
        >
          Xoá tất cả
        </button>
      </div>
      <div className="grid grid-cols-[repeat(auto-fill,minmax(80px,1fr))] gap-2 max-h-40 overflow-auto">
        {files.map((f, i) => (
          <div
            key={`${f.name}-${i}`}
            className="relative group aspect-square rounded-xs bg-surface-2 flex items-center justify-center text-xs text-text-muted truncate p-1.5"
            title={f.name}
          >
            <ImageIcon size={14} className="text-text-subtle absolute top-1 left-1" />
            <span className="truncate">{f.name.slice(0, 14)}</span>
            <button
              onClick={() => onRemove(i)}
              className="absolute top-0.5 right-0.5 size-5 rounded-full bg-error text-white opacity-0 group-hover:opacity-100 flex items-center justify-center transition-opacity"
            >
              <X size={10} />
            </button>
          </div>
        ))}
      </div>
    </div>
  )
}


function fmtSize(b: number): string {
  if (b < 1024)              return `${b} B`
  if (b < 1024 * 1024)       return `${(b / 1024).toFixed(0)} KB`
  if (b < 1024 * 1024 * 1024) return `${(b / 1024 / 1024).toFixed(1)} MB`
  return `${(b / 1024 / 1024 / 1024).toFixed(2)} GB`
}

function fmtSpeed(bps: number): string {
  if (bps < 1024)              return `${bps.toFixed(0)} B/s`
  if (bps < 1024 * 1024)       return `${(bps / 1024).toFixed(0)} KB/s`
  return `${(bps / 1024 / 1024).toFixed(1)} MB/s`
}

function fmtEta(seconds: number): string {
  if (seconds < 60) return `còn ${seconds}s`
  const m = Math.floor(seconds / 60)
  const s = seconds % 60
  return `còn ${m}m${s.toString().padStart(2, '0')}s`
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
