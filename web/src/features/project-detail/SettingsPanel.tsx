import { useState, useEffect } from 'react'
import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query'
import { useNavigate } from '@tanstack/react-router'
import { Save, Trash2 } from 'lucide-react'
import { api, type ApiProject } from '@shared/api/api'
import { Button } from '@shared/ui/Button'
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