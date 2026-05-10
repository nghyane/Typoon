import { useState, useEffect, useRef } from 'react'
import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query'
import { useNavigate } from '@tanstack/react-router'
import { Save, Trash2, ImagePlus } from 'lucide-react'
import { api, type ApiProject } from '@shared/api/api'
import { Button } from '@shared/ui/Button'
import { Cover } from '@shared/ui/Cover'
import { cn } from '@shared/lib/cn'
import { input, Spinner } from '@shared/ui/primitives'
import { LangPicker, type LangOption } from '@shared/ui/LangPicker'
import {
  SettingsSection, SettingsRow, SettingsValue, SettingsToggle, SettingsAction,
  SettingsDivider, Textarea,
} from '@shared/ui/SettingsForm'
import { toast } from '@shared/ui/Toaster'
import { confirm } from '@shared/ui/Confirm'

interface Props { project: ApiProject }

const TARGET_LANGS: readonly LangOption[] = [
  { code: 'vi', label: 'VI' },
  { code: 'en', label: 'EN' },
  { code: 'zh', label: 'ZH' },
  { code: 'ja', label: 'JA' },
]

// =============================================================================
// Project settings — Linear/Stripe pattern: 2-col rows, divider sections,
// sticky save bar when dirty. Sharing toggle is INSTANT (mutation per click)
// — not part of the form draft, to avoid two competing UIs (this form vs
// "Đang chia sẻ" button in Hero).
// =============================================================================

export function SettingsPanel({ project }: Props) {
  const qc  = useQueryClient()
  const nav = useNavigate()

  const { data: settings } = useQuery({
    queryKey: ['projects', project.project_id, 'settings'],
    queryFn:  () => api.getSettings(project.project_id),
  })

  // Form draft — only fields that need an explicit Save action.
  // `shared` is NOT in here (instant toggle).
  const [draft, setDraft] = useState({
    title:       project.title,
    description: project.description ?? '',
    target_lang: project.target_lang,
  })

  useEffect(() => {
    if (!settings) return
    setDraft({
      title:       settings.title,
      description: settings.description ?? '',
      target_lang: settings.target_lang,
    })
  }, [settings])

  const isOwner = settings?.is_owner ?? project.is_owner

  const dirty =
    draft.title       !== project.title ||
    draft.description !== (project.description ?? '') ||
    draft.target_lang !== project.target_lang

  const reset = () => setDraft({
    title:       project.title,
    description: project.description ?? '',
    target_lang: project.target_lang,
  })

  const save = useMutation({
    mutationFn: () => api.patchSettings(project.project_id, {
      title:       draft.title.trim() || project.title,
      description: draft.description,
      target_lang: draft.target_lang,
    }),
    onSuccess: () => {
      qc.invalidateQueries({ queryKey: ['projects'] })
      qc.invalidateQueries({ queryKey: ['projects', project.project_id] })
      qc.invalidateQueries({ queryKey: ['projects', project.project_id, 'settings'] })
      toast.success('Đã lưu thay đổi')
    },
    onError: (e: Error) => toast.error(e.message),
  })

  const uploadCover = useMutation({
    mutationFn: (file: File) => api.uploadCover(project.project_id, file),
    onSuccess: () => {
      qc.invalidateQueries({ queryKey: ['projects'] })
      qc.invalidateQueries({ queryKey: ['projects', project.project_id] })
      toast.success('Đã cập nhật ảnh bìa')
    },
    onError: (e: Error) => toast.error(e.message),
  })

  const onPickCover = (file: File | null) => {
    if (!file) return
    if (!file.type.startsWith('image/')) {
      toast.error('Tệp phải là ảnh')
      return
    }
    uploadCover.mutate(file)
  }

  const toggleShare = useMutation({
    mutationFn: (next: boolean) => api.patchSettings(project.project_id, { shared: next }),
    onSuccess: (_, next) => {
      qc.invalidateQueries({ queryKey: ['projects'] })
      qc.invalidateQueries({ queryKey: ['projects', project.project_id] })
      toast.success(next ? 'Đã bật chia sẻ' : 'Đã tắt chia sẻ')
    },
    onError: (e: Error) => toast.error(e.message),
  })

  const del = useMutation({
    mutationFn: () => api.deleteProject(project.project_id),
    onSuccess: () => {
      qc.invalidateQueries({ queryKey: ['projects'] })
      toast.success('Đã xoá dự án')
      nav({ to: '/projects' })
    },
    onError: (e: Error) => toast.error(e.message),
  })

  return (
    <div className="pb-20">
      <SettingsSection
        title="Thông tin chung"
        description="Hiển thị tên, mô tả và ngôn ngữ đích của bản dịch."
      >
        <SettingsRow label="Tên dự án">
          <input
            value={draft.title}
            onChange={(e) => setDraft({ ...draft, title: e.target.value })}
            disabled={!isOwner}
            className={input}
          />
        </SettingsRow>

        <SettingsRow label="Mô tả" hint="Hiển thị trong danh sách dự án và mục Cộng đồng.">
          <Textarea
            value={draft.description}
            onChange={(e) => setDraft({ ...draft, description: e.target.value })}
            disabled={!isOwner}
            rows={3}
          />
        </SettingsRow>

        <SettingsRow
          label="Ngôn ngữ đích"
          hint="Render và bản dịch sẽ tạo bằng ngôn ngữ này."
        >
          <LangPicker
            value={draft.target_lang}
            onChange={(v) => setDraft({ ...draft, target_lang: v })}
            options={TARGET_LANGS}
            disabled={!isOwner}
          />
        </SettingsRow>

        <SettingsRow
          label="Ảnh bìa"
          hint="Hiển thị trong danh sách dự án và trang chi tiết. Khuyến nghị tỷ lệ 2:3, JPG/PNG."
        >
          <CoverDropzone
            project={project}
            disabled={!isOwner}
            uploading={uploadCover.isPending}
            onPick={onPickCover}
          />
        </SettingsRow>
      </SettingsSection>

      <SettingsDivider />

      {/* Source — read-only, only when present */}
      {project.source_url && (
        <>
          <SettingsSection
            title="Nguồn"
            description="Thông tin bản gốc — không thể chỉnh sửa."
          >
            <SettingsRow label="URL">
              <SettingsValue>
                <a
                  href={project.source_url}
                  target="_blank"
                  rel="noreferrer"
                  className="truncate hover:text-accent-text underline-offset-2 hover:underline"
                >
                  {project.source_url}
                </a>
              </SettingsValue>
            </SettingsRow>
            <SettingsRow label="Ngôn ngữ nguồn">
              <SettingsValue>{project.source_lang.toUpperCase()}</SettingsValue>
            </SettingsRow>
          </SettingsSection>
          <SettingsDivider />
        </>
      )}

      {/* Sharing — instant toggle, owner only */}
      {isOwner && (
        <>
          <SettingsSection
            title="Chia sẻ"
            description="Cho phép thành viên khác trong cộng đồng xem dự án này."
          >
            <SettingsRow
              label="Chia sẻ với cộng đồng"
              hint="Bật để dự án xuất hiện trong mục Cộng đồng. Chỉ chủ dự án có quyền sửa."
            >
              <SettingsToggle
                checked={project.shared}
                disabled={toggleShare.isPending}
                onChange={(v) => toggleShare.mutate(v)}
              />
            </SettingsRow>
          </SettingsSection>
          <SettingsDivider />
        </>
      )}

      {/* Danger zone — visually distinct: error border + bg tint */}
      {isOwner && (
        <SettingsSection
          title="Vùng nguy hiểm"
          description="Hành động không thể khôi phục."
          danger
        >
          <SettingsAction
            title="Xoá dự án"
            description="Xoá toàn bộ chương, bản dịch, bong bóng, thuật ngữ và file render đã tạo."
            action={
              <Button
                variant="danger"
                onClick={async () => {
                  const ok = await confirm({
                    title:       `Xoá dự án "${project.title}"?`,
                    description: 'Toàn bộ chương, bản dịch, bong bóng, thuật ngữ và file render sẽ bị xoá vĩnh viễn. Hành động không thể hoàn tác.',
                    confirmText: 'Xoá dự án',
                    tone:        'danger',
                  })
                  if (ok) del.mutate()
                }}
                disabled={del.isPending}
              >
                {del.isPending ? <Spinner /> : <Trash2 size={14} />}
                Xoá dự án
              </Button>
            }
          />
        </SettingsSection>
      )}

      {/* Sticky save bar — only when form is dirty */}
      {dirty && isOwner && (
        <div className="fixed bottom-[calc(3.5rem+0.75rem)] sm:bottom-5 left-1/2 -translate-x-1/2 z-40 flex items-center gap-3 bg-surface rounded-md pl-4 pr-2 py-2 shadow-[0_8px_32px_rgb(0,0,0,0.4)]">
          <span className="text-sm text-text-muted">Có thay đổi chưa lưu</span>
          <Button onClick={reset} disabled={save.isPending}>
            Huỷ
          </Button>
          <Button
            variant="primary"
            onClick={() => save.mutate()}
            disabled={save.isPending}
          >
            {save.isPending ? <Spinner /> : <Save size={14} />}
            Lưu thay đổi
          </Button>
        </div>
      )}
    </div>
  )
}

// =============================================================================
// CoverDropzone — preview + click-to-pick + drag-and-drop. Uploading state
// shows a spinner overlay; drag-over highlights the drop region. Designed
// to read as a single tile so the user understands "the image IS the input".
// =============================================================================

function CoverDropzone({
  project, disabled, uploading, onPick,
}: {
  project:   ApiProject
  disabled:  boolean
  uploading: boolean
  onPick:    (file: File | null) => void
}) {
  const fileRef = useRef<HTMLInputElement>(null)
  const [dragOver, setDragOver] = useState(false)

  const interactive = !disabled && !uploading
  const hasCover    = !!project.cover_url

  const open = () => {
    if (!interactive) return
    fileRef.current?.click()
  }

  const onDrop = (e: React.DragEvent) => {
    e.preventDefault()
    setDragOver(false)
    if (!interactive) return
    const file = e.dataTransfer.files?.[0] ?? null
    onPick(file)
  }

  return (
    <div className="flex items-start gap-4">
      <button
        type="button"
        onClick={open}
        onDragOver={(e) => { e.preventDefault(); if (interactive) setDragOver(true) }}
        onDragLeave={() => setDragOver(false)}
        onDrop={onDrop}
        disabled={!interactive}
        aria-label={hasCover ? 'Đổi ảnh bìa' : 'Tải ảnh bìa'}
        className={cn(
          'group relative w-24 aspect-[2/3] rounded-md overflow-hidden shrink-0',
          'bg-surface-2 ring-1 ring-inset ring-border-soft transition-all',
          interactive && 'cursor-pointer hover:ring-text-subtle',
          dragOver && 'ring-2 ring-accent ring-offset-2 ring-offset-surface',
          !interactive && 'opacity-60 cursor-not-allowed',
        )}
      >
        <Cover
          src={project.cover_url}
          title={project.title}
          fontSize="text-xl"
          version={project.updated_at}
          className="absolute inset-0"
        />

        {/* Hover/drag overlay — shows the affordance without cluttering the tile */}
        {interactive && (
          <div className={cn(
            'absolute inset-0 flex flex-col items-center justify-center gap-1.5',
            'bg-black/55 text-white text-[11px] font-medium',
            'opacity-0 transition-opacity duration-150',
            'group-hover:opacity-100 group-focus-visible:opacity-100',
            dragOver && 'opacity-100',
          )}>
            <ImagePlus size={18} />
            <span>{hasCover ? 'Đổi ảnh' : 'Tải ảnh'}</span>
          </div>
        )}

        {/* Uploading overlay — wins over hover, blocks pointer */}
        {uploading && (
          <div className="absolute inset-0 flex items-center justify-center bg-black/60">
            <Spinner />
          </div>
        )}
      </button>

      <div className="min-w-0 pt-1 space-y-1">
        <p className="text-[13px] text-text-muted">
          {disabled
            ? 'Chỉ chủ dự án có thể đổi ảnh bìa.'
            : <>Nhấn vào ảnh hoặc <span className="text-text">kéo thả</span> tệp để tải lên.</>}
        </p>
        <p className="text-xs text-text-subtle leading-relaxed">
          JPG hoặc PNG. Khuyến nghị tỷ lệ 2:3, tối thiểu 600×900.
        </p>
      </div>

      <input
        ref={fileRef}
        type="file"
        accept="image/*"
        onChange={(e) => {
          const file = e.target.files?.[0] ?? null
          onPick(file)
          // Reset so picking the same file again still fires onChange.
          e.target.value = ''
        }}
        className="hidden"
        disabled={!interactive}
      />
    </div>
  )
}