import { useState, useRef, useEffect, type DragEvent } from 'react'
import { useMutation, useQueryClient } from '@tanstack/react-query'
import { Upload, FileText, Image as ImageIcon, Archive, X } from 'lucide-react'
import { api, type ApiProject } from '../lib/api'
import { cn } from '../lib/cn'
import { Modal } from './Modal'
import { btn, input, label, Spinner } from './ui'
import { toast } from './Toaster'

interface Props {
  open:    boolean
  onClose: () => void
  project: ApiProject
  /** Chapter idx already in the project — used to suggest the next number. */
  existing: Set<number>
}

const ACCEPT = '.pdf,.cbz,.zip,.png,.jpg,.jpeg,.webp,application/pdf,application/zip,image/*'

// All-image extension regex (lowercase). Anything else is treated as an
// archive/PDF for the single-file branch.
const IMAGE_EXT = /\.(png|jpe?g|webp|bmp|tiff?)$/i

export function UploadChapterDialog({ open, onClose, project, existing }: Props) {
  const qc = useQueryClient()

  const [files,    setFiles]    = useState<File[]>([])
  const [idx,      setIdx]      = useState<string>('')
  const [title,    setTitle]    = useState('')
  const [dragOver, setDragOver] = useState(false)
  const fileRef = useRef<HTMLInputElement>(null)

  // Reset on close. Suggest next chapter number on open.
  useEffect(() => {
    if (open) {
      setIdx(suggestNextIdx(existing))
      setTitle('')
      setFiles([])
    }
  }, [open, existing])

  const upload = useMutation({
    mutationFn: () => api.uploadChapter(project.project_id, files, {
      idx:   idx.trim() ? Number(idx) : undefined,
      title: title.trim() || undefined,
    }),
    onSuccess: (ch) => {
      qc.invalidateQueries({ queryKey: ['projects', project.project_id, 'chapters'] })
      qc.invalidateQueries({ queryKey: ['projects'] })
      toast.success(`Đã thêm Ch.${ch.idx} (${ch.page_count} trang)`)
      onClose()
    },
    onError: (e: Error) => toast.error(e.message),
  })

  const isArchive = files.length === 1 && !IMAGE_EXT.test(files[0].name)
  const totalSize = files.reduce((s, f) => s + f.size, 0)

  const onDrop = (e: DragEvent<HTMLDivElement>) => {
    e.preventDefault()
    setDragOver(false)
    if (upload.isPending) return
    const dropped = Array.from(e.dataTransfer.files)
    if (dropped.length > 0) addFiles(dropped)
  }

  const addFiles = (incoming: File[]) => {
    // If user drops an archive/PDF, replace the whole list (single-chapter
    // upload). For images, accumulate so users can drag multiple times.
    const hasArchive = incoming.some((f) => !IMAGE_EXT.test(f.name))
    if (hasArchive) {
      setFiles([incoming.find((f) => !IMAGE_EXT.test(f.name))!])
    } else {
      setFiles((prev) => [...prev, ...incoming.filter((f) => IMAGE_EXT.test(f.name))])
    }
  }

  const valid = files.length > 0 && idx.trim() !== '' && !Number.isNaN(Number(idx))

  return (
    <Modal
      open={open}
      onClose={() => { if (!upload.isPending) onClose() }}
      title={`Tải chương — ${project.title}`}
      size="md"
      footer={
        <>
          <button
            onClick={onClose}
            disabled={upload.isPending}
            className={btn.secondary}
          >
            Huỷ
          </button>
          <button
            onClick={() => upload.mutate()}
            disabled={!valid || upload.isPending}
            className={btn.primary}
          >
            {upload.isPending && <Spinner />}
            <Upload size={14} />
            Tải lên
          </button>
        </>
      }
    >
      <div className="px-5 py-4 space-y-4">
        {/* Drop zone */}
        <div
          onDragOver={(e) => { e.preventDefault(); setDragOver(true) }}
          onDragLeave={() => setDragOver(false)}
          onDrop={onDrop}
          onClick={() => !upload.isPending && fileRef.current?.click()}
          className={cn(
            'rounded-xl border-2 border-dashed flex flex-col items-center justify-center cursor-pointer transition-colors',
            'min-h-32 p-6 text-center',
            dragOver
              ? 'border-zinc-900 bg-zinc-50'
              : 'border-zinc-200 bg-zinc-50/40 hover:border-zinc-300',
            upload.isPending && 'opacity-60 cursor-not-allowed',
          )}
        >
          {files.length === 0 ? (
            <>
              <Upload size={20} className="text-zinc-400 mb-2" />
              <p className="text-sm font-medium text-zinc-700">
                Kéo thả tệp vào đây hoặc <span className="underline">chọn tệp</span>
              </p>
              <p className="text-xs text-zinc-400 mt-1">
                Hỗ trợ: PDF, CBZ, ZIP, hoặc nhiều ảnh
              </p>
            </>
          ) : (
            <FileList
              files={files}
              isArchive={isArchive}
              onClear={() => setFiles([])}
              onRemove={(i) => setFiles(files.filter((_, j) => j !== i))}
            />
          )}
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
        </div>

        {/* Idx + title */}
        <div className="grid grid-cols-[80px_1fr] gap-3">
          <div>
            <label className={label}>Số chương</label>
            <input
              type="number"
              step="0.1"
              value={idx}
              onChange={(e) => setIdx(e.target.value)}
              className={input}
              disabled={upload.isPending}
            />
          </div>
          <div>
            <label className={label}>Tiêu đề (tuỳ chọn)</label>
            <input
              type="text"
              value={title}
              onChange={(e) => setTitle(e.target.value)}
              placeholder="VD: Mở đầu"
              className={input}
              disabled={upload.isPending}
            />
          </div>
        </div>

        {files.length > 0 && (
          <p className="text-xs text-zinc-400">
            {files.length === 1 ? '1 tệp' : `${files.length} ảnh`} · {fmtSize(totalSize)}
          </p>
        )}

        {existing.has(Number(idx)) && (
          <p className="text-xs text-amber-600">
            Chương {idx} đã tồn tại — sẽ được ghi đè dữ liệu chuẩn bị.
          </p>
        )}
      </div>
    </Modal>
  )
}

// ── Internals ────────────────────────────────────────────────────────────────

function FileList({
  files, isArchive, onClear, onRemove,
}: {
  files:    File[]
  isArchive: boolean
  onClear:  () => void
  onRemove: (i: number) => void
}) {
  if (isArchive) {
    const f = files[0]
    return (
      <div
        className="w-full flex items-center gap-3 px-3 py-2 rounded-lg bg-white border border-zinc-200"
        onClick={(e) => e.stopPropagation()}
      >
        <FileIcon name={f.name} />
        <div className="flex-1 min-w-0 text-left">
          <p className="text-sm text-zinc-900 truncate">{f.name}</p>
          <p className="text-xs text-zinc-400">{fmtSize(f.size)}</p>
        </div>
        <button
          onClick={onClear}
          className="size-7 rounded-md flex items-center justify-center text-zinc-400 hover:text-zinc-700 hover:bg-zinc-100 cursor-pointer"
        >
          <X size={13} />
        </button>
      </div>
    )
  }
  return (
    <div className="w-full" onClick={(e) => e.stopPropagation()}>
      <div className="flex items-center justify-between mb-2">
        <span className="text-xs text-zinc-500">
          {files.length} ảnh — sẽ sắp xếp theo tên tệp
        </span>
        <button
          onClick={onClear}
          className="text-xs text-zinc-500 hover:text-zinc-900 cursor-pointer"
        >
          Xoá tất cả
        </button>
      </div>
      <div className="grid grid-cols-[repeat(auto-fill,minmax(80px,1fr))] gap-2 max-h-40 overflow-auto">
        {files.map((f, i) => (
          <div
            key={`${f.name}-${i}`}
            className="relative group aspect-square rounded-md bg-white border border-zinc-200 flex items-center justify-center text-xs text-zinc-500 truncate p-1.5"
            title={f.name}
          >
            <ImageIcon size={14} className="text-zinc-300 absolute top-1 left-1" />
            <span className="truncate">{f.name.slice(0, 14)}</span>
            <button
              onClick={() => onRemove(i)}
              className="absolute top-0.5 right-0.5 size-5 rounded-full bg-zinc-900 text-white opacity-0 group-hover:opacity-100 flex items-center justify-center transition-opacity"
            >
              <X size={10} />
            </button>
          </div>
        ))}
      </div>
    </div>
  )
}

function FileIcon({ name }: { name: string }) {
  const lower = name.toLowerCase()
  const Icon = lower.endsWith('.pdf') ? FileText : Archive
  return (
    <div className="size-9 rounded-lg bg-zinc-100 flex items-center justify-center shrink-0">
      <Icon size={15} className="text-zinc-500" />
    </div>
  )
}

function fmtSize(b: number): string {
  if (b < 1024)              return `${b} B`
  if (b < 1024 * 1024)       return `${(b / 1024).toFixed(0)} KB`
  if (b < 1024 * 1024 * 1024) return `${(b / 1024 / 1024).toFixed(1)} MB`
  return `${(b / 1024 / 1024 / 1024).toFixed(2)} GB`
}

function suggestNextIdx(existing: Set<number>): string {
  if (existing.size === 0) return '1'
  const max = Math.max(...existing)
  const next = Number.isInteger(max) ? max + 1 : Math.floor(max) + 1
  return String(next)
}
