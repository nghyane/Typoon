import { useState, useEffect } from 'react'
import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query'
import { useNavigate } from '@tanstack/react-router'
import { Save, Trash2 } from 'lucide-react'
import { api, type ApiProject } from '../lib/api'
import { cn } from '../lib/cn'
import { btn, input, label, Spinner } from './ui'
import { toast } from './Toaster'

interface Props { project: ApiProject }

const TARGET_LANGS = [
  { code: 'vi', label: 'Tiếng Việt' },
  { code: 'en', label: 'English'    },
  { code: 'zh', label: '中文'        },
  { code: 'ja', label: '日本語'      },
]

export function SettingsPanel({ project }: Props) {
  const qc  = useQueryClient()
  const nav = useNavigate()

  const { data: settings } = useQuery({
    queryKey: ['projects', project.project_id, 'settings'],
    queryFn:  () => api.getSettings(project.project_id),
  })

  const [draft, setDraft] = useState({
    title:       project.title,
    description: project.description ?? '',
    target_lang: project.target_lang,
    shared:      project.shared,
  })

  // Sync draft with settings (loaded async).
  useEffect(() => {
    if (!settings) return
    setDraft({
      title:       settings.title,
      description: settings.description ?? '',
      target_lang: settings.target_lang,
      shared:      settings.shared,
    })
  }, [settings])

  const isOwner = settings?.is_owner ?? project.is_owner

  const dirty =
    draft.title       !== project.title ||
    draft.description !== (project.description ?? '') ||
    draft.target_lang !== project.target_lang ||
    draft.shared      !== project.shared

  const save = useMutation({
    mutationFn: () => api.patchSettings(project.project_id, {
      title:       draft.title.trim() || project.title,
      description: draft.description,
      target_lang: draft.target_lang,
      shared:      draft.shared,
    }),
    onSuccess: () => {
      qc.invalidateQueries({ queryKey: ['projects'] })
      qc.invalidateQueries({ queryKey: ['projects', project.project_id] })
      qc.invalidateQueries({ queryKey: ['projects', project.project_id, 'settings'] })
      toast.success('Đã lưu thay đổi')
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
    <div className="space-y-6 max-w-2xl">
      {/* General */}
      <Section title="Thông tin chung">
        <Field label="Tên dự án">
          <input
            value={draft.title}
            onChange={(e) => setDraft({ ...draft, title: e.target.value })}
            className={input}
          />
        </Field>
        <Field label="Mô tả">
          <textarea
            value={draft.description}
            onChange={(e) => setDraft({ ...draft, description: e.target.value })}
            rows={4}
            className={cn(input, 'h-auto py-2 resize-y')}
          />
        </Field>
        <Field label="Ngôn ngữ đích">
          <div className="flex gap-1.5">
            {TARGET_LANGS.map((l) => (
              <button
                key={l.code}
                onClick={() => setDraft({ ...draft, target_lang: l.code })}
                className={cn(
                  'h-8 px-3 rounded-lg text-xs cursor-pointer transition-colors border',
                  draft.target_lang === l.code
                    ? 'bg-zinc-900 text-white border-zinc-900 font-medium'
                    : 'bg-white text-zinc-600 border-zinc-200 hover:border-zinc-300',
                )}
              >
                {l.label}
              </button>
            ))}
          </div>
        </Field>
        <div className="flex justify-end gap-2 pt-1">
          <button
            onClick={() => save.mutate()}
            disabled={!dirty || save.isPending}
            className={btn.primary}
          >
            {save.isPending ? <Spinner /> : <Save size={14} />}
            Lưu
          </button>
        </div>
      </Section>

      {/* Sharing — owner only */}
      {isOwner && (
        <Section title="Chia sẻ">
          <label className="flex items-start gap-3 cursor-pointer">
            <input
              type="checkbox"
              checked={draft.shared}
              onChange={(e) => setDraft({ ...draft, shared: e.target.checked })}
              className="mt-1 size-4 accent-zinc-900"
            />
            <div>
              <p className="text-sm font-medium text-zinc-900">Chia sẻ với cộng đồng</p>
              <p className="text-xs text-zinc-500 mt-0.5">
                Bật để các thành viên khác xem được dự án trong mục “Cộng đồng”. Chỉ chủ dự án mới có quyền sửa.
              </p>
            </div>
          </label>
        </Section>
      )}

      {/* Source */}
      {project.source_url && (
        <Section title="Nguồn">
          <Field label="URL gốc">
            <input value={project.source_url} readOnly className={cn(input, 'text-zinc-500')} />
          </Field>
          <Field label="Ngôn ngữ nguồn">
            <input value={project.source_lang.toUpperCase()} readOnly className={cn(input, 'text-zinc-500 w-32')} />
          </Field>
        </Section>
      )}

      {/* Danger zone */}
      {isOwner && (
      <Section title="Vùng nguy hiểm" tone="danger">
        <div className="flex items-start justify-between gap-4">
          <div>
            <p className="text-sm font-medium text-zinc-900">Xoá dự án</p>
            <p className="text-xs text-zinc-500 mt-0.5">
              Xoá toàn bộ dữ liệu — chương, bản dịch, bong bóng, glossary, render. Không thể khôi phục.
            </p>
          </div>
          <button
            onClick={() => {
              if (!confirm(`Xoá dự án "${project.title}" và toàn bộ dữ liệu?`)) return
              del.mutate()
            }}
            disabled={del.isPending}
            className={btn.danger}
          >
            {del.isPending ? <Spinner /> : <Trash2 size={14} />}
            Xoá dự án
          </button>
        </div>
      </Section>
      )}
    </div>
  )
}

function Section({
  title, tone = 'default', children,
}: {
  title:    string
  tone?:    'default' | 'danger'
  children: React.ReactNode
}) {
  return (
    <section
      className={cn(
        'rounded-xl border bg-white',
        tone === 'danger' ? 'border-red-200' : 'border-zinc-200',
      )}
    >
      <header
        className={cn(
          'px-4 py-2.5 border-b text-xs font-semibold uppercase tracking-wider',
          tone === 'danger' ? 'border-red-100 text-red-600' : 'border-zinc-100 text-zinc-500',
        )}
      >
        {title}
      </header>
      <div className="p-4 space-y-3">{children}</div>
    </section>
  )
}

function Field({ label: lbl, children }: { label: string; children: React.ReactNode }) {
  return (
    <div>
      <label className={label}>{lbl}</label>
      {children}
    </div>
  )
}
