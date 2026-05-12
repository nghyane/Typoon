import { useState } from 'react'
import { useMutation, useQueryClient } from '@tanstack/react-query'
import { Wand2, FileText } from 'lucide-react'
import { Button } from '@shared/ui/Button'
import { input, label } from '@shared/ui/primitives'
import { toast } from '@shared/ui/Toaster'
import { cn } from '@shared/lib/cn'
import { api, type LibraryStatus } from '@shared/api/api'

// =============================================================================
// ManualCreateForm — for manga not in any installed source.
//
// Backend: api.createLocalMaterial({origin: 'upload'}) → material row
// with NULL source / upstream_ref. The hub page (slice 13+) lets the
// user upload chapter zips into the entry afterwards.
//
// The form intentionally asks for the minimum that makes a sensible
// library card: title + cover + optional metadata. Author / status /
// nsfw aren't required because a user might be importing 1 chapter
// from a friend's scan with zero metadata.
// =============================================================================

interface Props {
  /** Pre-fill title from the search query or pasted URL. */
  initialTitle?: string
  onCancel:      () => void
  onCreated:     () => void
}

export function ManualCreateForm({ initialTitle, onCancel, onCreated }: Props) {
  const qc = useQueryClient()

  const [title,   setTitle]   = useState(initialTitle ?? '')
  const [cover,   setCover]   = useState('')
  const [author,  setAuthor]  = useState('')
  const [desc,    setDesc]    = useState('')
  const [nsfw,    setNsfw]    = useState(false)

  const [targetLang, setTargetLang] = useState('vi')
  const [autoTr,     setAutoTr]     = useState(false)
  const [status,     setStatus]     = useState<LibraryStatus>('reading')

  const valid = title.trim().length >= 1

  const create = useMutation({
    mutationFn: async () => {
      const material = await api.createLocalMaterial({
        origin:      'upload',
        title:       title.trim(),
        cover_url:   cover.trim() || null,
        description: desc.trim()  || null,
        author:      author.trim() || null,
        nsfw,
      })
      await api.createLibraryEntry({
        material_id:    material.id,
        title:          material.title,
        cover_url:      material.cover_url,
        target_lang:    targetLang,
        auto_translate: autoTr,
        status,
      })
    },
    onSuccess: () => {
      qc.invalidateQueries({ queryKey: ['library'] })
      toast.success(`Đã tạo "${title.trim()}". Tải chương ở trang truyện.`)
      onCreated()
    },
    onError: (e: Error) => toast.error(e.message),
  })

  return (
    <div className="space-y-4">
      <div className="flex items-start gap-2.5 p-3 rounded-md bg-info/10 border border-info/20">
        <Wand2 size={14} className="text-info-text shrink-0 mt-0.5" />
        <div className="flex-1 min-w-0">
          <p className="text-sm text-text">Tạo manga thủ công</p>
          <p className="text-[11px] text-text-subtle mt-0.5">
            Manga sẽ được tạo trống. Vào trang truyện để tải chương từ
            file zip/cbz hoặc ảnh.
          </p>
        </div>
      </div>

      <div className="grid grid-cols-[7rem_1fr] gap-x-3 gap-y-3">
        <div className="col-span-2">
          <label className={label}>Tiêu đề</label>
          <input
            autoFocus
            type="text"
            value={title}
            onChange={(e) => setTitle(e.target.value)}
            placeholder="Tên manga"
            className={input}
            disabled={create.isPending}
          />
        </div>

        <div>
          <label className={label}>Tác giả</label>
          <input
            type="text"
            value={author}
            onChange={(e) => setAuthor(e.target.value)}
            placeholder="—"
            className={input}
            disabled={create.isPending}
          />
        </div>

        <div>
          <div className="flex items-center justify-between mb-1.5">
            <label className={cn(label, 'mb-0')}>Cover (link)</label>
            <span className="text-[11px] text-text-subtle">Tuỳ chọn</span>
          </div>
          <input
            type="url"
            value={cover}
            onChange={(e) => setCover(e.target.value)}
            placeholder="https://…"
            className={input}
            disabled={create.isPending}
          />
        </div>

        <div className="col-span-2">
          <div className="flex items-center justify-between mb-1.5">
            <label className={cn(label, 'mb-0')}>Mô tả</label>
            <span className="text-[11px] text-text-subtle">Tuỳ chọn</span>
          </div>
          <textarea
            value={desc}
            onChange={(e) => setDesc(e.target.value)}
            placeholder="Tóm tắt nội dung…"
            rows={2}
            className={cn(input, 'h-auto py-2 resize-none')}
            disabled={create.isPending}
          />
        </div>

        <div>
          <label className={label}>Đọc bằng</label>
          <select
            value={targetLang}
            onChange={(e) => setTargetLang(e.target.value)}
            className={input}
            disabled={create.isPending}
          >
            <option value="vi">Tiếng Việt</option>
            <option value="en">English</option>
            <option value="ja">日本語</option>
            <option value="ko">한국어</option>
            <option value="zh">中文</option>
          </select>
        </div>

        <div>
          <label className={label}>Tình trạng</label>
          <select
            value={status}
            onChange={(e) => setStatus(e.target.value as LibraryStatus)}
            className={input}
            disabled={create.isPending}
          >
            <option value="reading">Đang đọc</option>
            <option value="plan">Kế hoạch</option>
            <option value="on_hold">Tạm dừng</option>
            <option value="done">Đã xong</option>
          </select>
        </div>

        <div className="col-span-2 flex items-center gap-4">
          <label className="inline-flex items-center gap-2 cursor-pointer select-none">
            <input
              type="checkbox"
              checked={nsfw}
              onChange={(e) => setNsfw(e.target.checked)}
              className="size-4 cursor-pointer accent-accent"
            />
            <span className="text-sm text-text-muted">NSFW</span>
          </label>
          <label className="inline-flex items-center gap-2 cursor-pointer select-none">
            <input
              type="checkbox"
              checked={autoTr}
              onChange={(e) => setAutoTr(e.target.checked)}
              className="size-4 cursor-pointer accent-accent"
            />
            <span className="text-sm text-text-muted">
              Tự dịch khi tải chương
            </span>
          </label>
        </div>
      </div>

      <div className="flex items-center justify-end gap-2 pt-2">
        <Button variant="ghost" onClick={onCancel} disabled={create.isPending}>
          Quay lại
        </Button>
        <Button
          variant="primary"
          onClick={() => create.mutate()}
          disabled={!valid || create.isPending}
        >
          <FileText size={14} />
          Tạo manga
        </Button>
      </div>
    </div>
  )
}
