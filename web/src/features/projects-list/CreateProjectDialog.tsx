import { useState, useRef } from 'react'
import { useMutation, useQueryClient } from '@tanstack/react-query'
import { useNavigate } from '@tanstack/react-router'
import { Image as ImageIcon, X } from 'lucide-react'
import { api } from '@shared/api/api'
import { cn } from '@shared/lib/cn'
import { Modal } from '@shared/ui/Modal'
import { Button } from '@shared/ui/Button'
import { LangPicker } from '@shared/ui/LangPicker'
import { input, label, Spinner } from '@shared/ui/primitives'
import { toast } from '@shared/ui/Toaster'

interface Props { open: boolean; onClose: () => void }

const SOURCE_LANGS = [
  { code: 'en', label: 'EN' },
  { code: 'ja', label: 'JA' },
  { code: 'ko', label: 'KO' },
  { code: 'zh', label: 'ZH' },
]

const TARGET_LANGS = [
  { code: 'vi', label: 'VI' },
  { code: 'en', label: 'EN' },
  { code: 'zh', label: 'ZH' },
  { code: 'ja', label: 'JA' },
]

export function CreateProjectDialog({ open, onClose }: Props) {
  const qc  = useQueryClient()
  const nav = useNavigate()

  const [title,       setTitle]       = useState('')
  const [description, setDescription] = useState('')
  const [sourceLang,  setSourceLang]  = useState('en')
  const [targetLang,  setTargetLang]  = useState('vi')
  const [coverFile,   setCoverFile]   = useState<File | null>(null)
  const fileRef = useRef<HTMLInputElement>(null)

  const reset = () => {
    setTitle(''); setDescription('')
    setSourceLang('en'); setTargetLang('vi')
    setCoverFile(null)
    if (fileRef.current) fileRef.current.value = ''
  }

  const create = useMutation({
    mutationFn: async () => {
      const project = await api.createProject({
        title:       title.trim(),
        description: description.trim() || undefined,
        source_lang: sourceLang,
        target_lang: targetLang,
      })
      if (coverFile) {
        await api.uploadCover(project.project_id, coverFile)
      }
      return project
    },
    onSuccess: (project) => {
      qc.invalidateQueries({ queryKey: ['projects'] })
      toast.success(`Đã tạo dự án: ${project.title}`)
      reset()
      onClose()
      nav({ to: '/projects/$projectId', params: { projectId: String(project.project_id) } })
    },
    onError: (e: Error) => toast.error(e.message),
  })

  const previewUrl = coverFile ? URL.createObjectURL(coverFile) : null

  return (
    <Modal
      open={open}
      onClose={() => { if (!create.isPending) { reset(); onClose() } }}
      title="Tạo dự án mới"
      size="md"
      footer={
        <>
          <Button
            onClick={() => { reset(); onClose() }}
            disabled={create.isPending}
          >
            Huỷ
          </Button>
          <Button
            variant="primary"
            onClick={() => create.mutate()}
            disabled={create.isPending || !title.trim()}
          >
            {create.isPending && <Spinner />}
            Tạo dự án
          </Button>
        </>
      }
    >
      <div className="px-5 py-4 space-y-4">
        <div>
          <label className={label}>Tên dự án <span className="text-error-text">*</span></label>
          <input
            value={title}
            onChange={(e) => setTitle(e.target.value)}
            placeholder="VD: Solo Leveling"
            className={input}
            autoFocus
            disabled={create.isPending}
          />
        </div>

        <div>
          <label className={label}>Mô tả</label>
          <textarea
            value={description}
            onChange={(e) => setDescription(e.target.value)}
            placeholder="Tóm tắt nội dung (tuỳ chọn)"
            rows={3}
            disabled={create.isPending}
            className={cn(input, 'h-auto py-2 resize-y')}
          />
        </div>

        <div className="grid grid-cols-2 gap-4">
          <div>
            <label className={label}>Ngôn ngữ nguồn</label>
            <LangPicker value={sourceLang} onChange={setSourceLang} options={SOURCE_LANGS} />
          </div>
          <div>
            <label className={label}>Ngôn ngữ đích</label>
            <LangPicker value={targetLang} onChange={setTargetLang} options={TARGET_LANGS} />
          </div>
        </div>

        <div>
          <label className={label}>Cover (tuỳ chọn)</label>
          <div className="flex items-start gap-3">
            <div
              onClick={() => !create.isPending && fileRef.current?.click()}
              className={cn(
                'w-20 h-28 rounded-md border-2 border-dashed flex items-center justify-center shrink-0 overflow-hidden bg-surface-2 cursor-pointer transition-colors',
                previewUrl ? 'border-transparent' : 'border-border-soft hover:border-text-subtle',
              )}
            >
              {previewUrl ? (
                <img src={previewUrl} alt="cover preview" className="w-full h-full object-cover" />
              ) : (
                <ImageIcon size={20} className="text-text-subtle" />
              )}
            </div>
            <div className="flex-1 text-xs text-text-subtle pt-1">
              {coverFile ? (
                <div className="flex items-center gap-2">
                  <span className="truncate text-text-muted">{coverFile.name}</span>
                  <button
                    onClick={() => { setCoverFile(null); if (fileRef.current) fileRef.current.value = '' }}
                    className="size-5 rounded-xs hover:bg-hover flex items-center justify-center cursor-pointer text-text-subtle hover:text-text"
                  >
                    <X size={12} />
                  </button>
                </div>
              ) : (
                <>
                  Kéo thả hoặc <button
                    onClick={() => fileRef.current?.click()}
                    className="underline cursor-pointer text-text-muted hover:text-text"
                  >chọn ảnh</button>.
                  <br />
                  Sẽ được tự động cắt theo tỷ lệ 2:3.
                </>
              )}
            </div>
            <input
              ref={fileRef}
              type="file"
              accept="image/*"
              onChange={(e) => setCoverFile(e.target.files?.[0] ?? null)}
              className="hidden"
            />
          </div>
        </div>
      </div>
    </Modal>
  )
}
