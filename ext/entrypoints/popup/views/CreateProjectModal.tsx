// Inline modal for creating a project from the popup. Minimal — title
// + 2 language pickers. The SPA has a richer "new project" experience;
// here we ship the smallest set the engine actually requires.

import { useState } from 'react'
import { useMutation, useQueryClient } from '@tanstack/react-query'
import { Button } from '@shared/ui/Button'
import { Field, input } from '@shared/ui/Field'
import { TypoonClient, type ApiMeProject, type ApiProject } from '@core/typoon'
import { API_URL } from '@core/config'
import { useConfig } from '@shell/hooks/useConfig'

interface Props {
  initialTitle: string
  onCreated:    (p: ApiMeProject) => void
  onCancel:     () => void
}

const COMMON_LANGS: Array<[code: string, label: string]> = [
  ['ja', 'Tiếng Nhật'],
  ['ko', 'Tiếng Hàn'],
  ['zh', 'Tiếng Trung'],
  ['en', 'Tiếng Anh'],
  ['vi', 'Tiếng Việt'],
]

export function CreateProjectModal({ initialTitle, onCreated, onCancel }: Props) {
  const { config } = useConfig()
  const qc = useQueryClient()

  const [title,  setTitle]  = useState(initialTitle)
  const [source, setSource] = useState('ja')
  const [target, setTarget] = useState('vi')

  const m = useMutation({
    mutationFn: (): Promise<ApiProject> => new TypoonClient({
      apiUrl: API_URL, token: config.token,
    }).createProject({
      title, source_lang: source, target_lang: target,
    }),
    onSuccess: (p) => {
      qc.invalidateQueries({ queryKey: ['me', 'projects'] })
      onCreated({
        project_id:  p.project_id,
        slug:        p.slug,
        title:       p.title,
        cover_url:   p.cover_url,
        source_lang: p.source_lang,
        target_lang: p.target_lang,
        shared:      p.shared,
      })
    },
  })

  return (
    <div className="fixed inset-0 z-20 flex items-center justify-center bg-bg/80">
      <div className="w-[320px] bg-surface rounded-md p-4 space-y-3">
        <h2 className="text-sm font-semibold">Tạo project mới</h2>

        <Field label="Tên project">
          <input
            className={input}
            value={title}
            onChange={e => setTitle(e.target.value)}
            disabled={m.isPending}
          />
        </Field>

        <Field label="Ngôn ngữ gốc">
          <select
            className={input}
            value={source}
            onChange={e => setSource(e.target.value)}
            disabled={m.isPending}
          >
            {COMMON_LANGS.map(([code, label]) =>
              <option key={code} value={code}>{label}</option>
            )}
          </select>
        </Field>

        <Field label="Dịch sang">
          <select
            className={input}
            value={target}
            onChange={e => setTarget(e.target.value)}
            disabled={m.isPending}
          >
            {COMMON_LANGS.map(([code, label]) =>
              <option key={code} value={code}>{label}</option>
            )}
          </select>
        </Field>

        {m.error && (
          <p className="text-xs text-error-text">{(m.error as Error).message}</p>
        )}

        <div className="flex justify-end gap-2 pt-1">
          <Button variant="ghost" size="sm" onClick={onCancel} disabled={m.isPending}>Hủy</Button>
          <Button
            variant="primary"
            size="sm"
            disabled={!title.trim() || m.isPending}
            onClick={() => m.mutate()}
          >
            {m.isPending ? 'Đang tạo…' : 'Tạo'}
          </Button>
        </div>
      </div>
    </div>
  )
}
