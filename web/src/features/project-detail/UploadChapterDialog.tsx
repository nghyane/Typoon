import { useState, useRef, useEffect, type DragEvent } from 'react'
import { useMutation, useQueryClient } from '@tanstack/react-query'
import { Upload, FileText, Image as ImageIcon, Archive, X } from 'lucide-react'
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

const ACCEPT = '.pdf,.cbz,.zip,.png,.jpg,.jpeg,.webp,application/pdf,application/zip,image/*'

// All-image extension regex (lowercase). Anything else is treated as an
// archive/PDF for the single-file branch.
const IMAGE_EXT = /\.(png|jpe?g|webp|bmp|tiff?)$/i

export function UploadChapterDialog({ open, onClose, project, existing }: Props) {
  const qc = useQueryClient()

  const [files,    setFiles]    = useState<File[]>([])
  const [number,   setNumber]   = useState<string>('')
  const [title,    setTitle]    = useState('')
  const [start,    setStart]    = useState(false)
  const [dragOver, setDragOver] = useState(false)
  const fileRef = useRef<HTMLInputElement>(null)

  // Reset on close. Suggest next chapter number on open. `start`
  // intentionally does NOT reset on every open — we remember the
  // user's preference for this session: somebody who always wants to
  // dịch ngay won't have to re-tick on every upload.
  useEffect(() => {
    if (open) {
      setNumber(suggestNextNumber(existing))
      setTitle('')
      setFiles([])
    }
  }, [open, existing])

  const upload = useMutation({
    mutationFn: () => api.uploadChapter(project.project_id, files, {
      number: number.trim() || undefined,
      title:  title.trim() || undefined,
      start,
    }),
    onSuccess: (ch) => {
      qc.invalidateQueries({ queryKey: ['projects', project.project_id, 'chapters'] })
      qc.invalidateQueries({ queryKey: ['projects'] })
      // Workers indicator + quota meter refresh only when something
      // actually went into the queue (start=true). A pure upload
      // doesn't move either counter so we skip those invalidations.
      if (start) {
        qc.invalidateQueries({ queryKey: ['workers'] })
        qc.invalidateQueries({ queryKey: ['quota'] })
      }
      toast.success(
        start
          ? `Đã thêm và bắt đầu Ch.${ch.number} (${ch.page_count} trang)`
          : `Đã thêm Ch.${ch.number} (${ch.page_count} trang)`,
      )
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

  // Form is valid as soon as files are picked. `number` is optional —
  // empty falls back to suggestNextNumber on the server side, so we
  // don't gate the upload button on it.
  const valid = files.length > 0

  return (
    <Modal
      open={open}
      onClose={() => { if (!upload.isPending) onClose() }}
      title={`Tải chương — ${project.title}`}
      size="md"
      footer={
        <>
          <Button onClick={onClose} disabled={upload.isPending}>
            Huỷ
          </Button>
          <Button
            variant="primary"
            onClick={() => upload.mutate()}
            disabled={!valid || upload.isPending}
          >
            {upload.isPending && <Spinner />}
            <Upload size={14} />
            {start ? 'Tải lên & dịch' : 'Tải lên'}
          </Button>
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
            'rounded-md border-2 border-dashed flex flex-col items-center justify-center cursor-pointer transition-colors',
            'min-h-32 p-6 text-center',
            dragOver
              ? 'border-accent bg-accent-bg'
              : 'border-border-soft bg-surface-2/40 hover:border-text-subtle',
            upload.isPending && 'opacity-60 cursor-not-allowed',
          )}
        >
          {files.length === 0 ? (
            <>
              <Upload size={20} className="text-text-subtle mb-2" />
              <p className="text-sm font-medium text-text">
                Kéo thả tệp vào đây hoặc <span className="underline">chọn tệp</span>
              </p>
              <p className="text-xs text-text-subtle mt-1">
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

        {/* Number + title */}
        <div className="grid grid-cols-[120px_1fr] gap-3">
          <div>
            <label className={label}>Số chương</label>
            <input
              type="text"
              value={number}
              onChange={(e) => setNumber(e.target.value)}
              placeholder="VD: 4, 4.5, Extra"
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
          <p className="text-xs text-text-subtle">
            {files.length === 1 ? '1 tệp' : `${files.length} ảnh`} · {fmtSize(totalSize)}
          </p>
        )}

        {existing.has(number.trim()) && (
          <p className="text-xs text-warning-text">
            Chương {number.trim()} đã tồn tại — bản tải lên sẽ chèn vào danh sách như một bản dịch khác.
          </p>
        )}

        {/* Default off so a misnamed/wrong upload doesn't auto-burn LLM
            cost. Power users who always want to translate immediately
            tick once and the choice persists for the dialog session. */}
        <label className="flex items-center gap-2 text-sm text-text-muted cursor-pointer select-none">
          <input
            type="checkbox"
            checked={start}
            onChange={(e) => setStart(e.target.checked)}
            disabled={upload.isPending}
            className="size-4 rounded-xs accent-accent cursor-pointer"
          />
          Bắt đầu dịch ngay sau khi tải lên
        </label>
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
        className="w-full flex items-center gap-3 px-3 py-2 rounded-sm bg-surface-2"
        onClick={(e) => e.stopPropagation()}
      >
        <FileIcon name={f.name} />
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
    <div className="w-full" onClick={(e) => e.stopPropagation()}>
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

function FileIcon({ name }: { name: string }) {
  const lower = name.toLowerCase()
  const Icon = lower.endsWith('.pdf') ? FileText : Archive
  return (
    <div className="size-9 rounded-sm bg-surface flex items-center justify-center shrink-0">
      <Icon size={15} className="text-text-muted" />
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
  // Suggest next integer past the largest numeric chapter; non-numeric
  // entries ("Extra", "Oneshot") are ignored. Empty project → "1".
  let max = 0
  for (const s of existing) {
    const n = Number(s)
    if (Number.isFinite(n) && n > max) max = n
  }
  if (max === 0) return '1'
  return String(Number.isInteger(max) ? max + 1 : Math.floor(max) + 1)
}
