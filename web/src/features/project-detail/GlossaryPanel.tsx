import { useState, useMemo } from 'react'
import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query'
import { Plus, Pencil, Trash2, BookText } from 'lucide-react'
import { api, type ApiGlossaryTerm } from '@shared/api/api'
import { Button } from '@shared/ui/Button'
import { Modal } from '@shared/ui/Modal'
import { input, label as labelCls, Spinner } from '@shared/ui/primitives'
import { DataToolbar, SearchInput } from '@shared/ui/DataToolbar'
import { DataTable, Th } from '@shared/ui/DataTable'
import { EmptyState } from '@shared/ui/EmptyState'
import { useDelayedFlag } from '@shared/lib/useDelayedFlag'
import { toast } from '@shared/ui/Toaster'

interface Props { projectId: number }

type Editing =
  | { mode: 'create' }
  | { mode: 'edit'; term: ApiGlossaryTerm }
  | null

export function GlossaryPanel({ projectId }: Props) {
  const qc = useQueryClient()
  const [q,       setQ]       = useState('')
  const [editing, setEditing] = useState<Editing>(null)

  const { data: terms = [], isPending } = useQuery({
    queryKey: ['projects', projectId, 'glossary'],
    queryFn:  () => api.listGlossary(projectId),
  })

  const filtered = useMemo(() => {
    const needle = q.trim().toLowerCase()
    if (!needle) return terms
    return terms.filter(
      (t) =>
        t.source_term.toLowerCase().includes(needle) ||
        t.target_term.toLowerCase().includes(needle),
    )
  }, [terms, q])

  const showSkeleton = useDelayedFlag(isPending, 250)
  const showEmpty    = !isPending && filtered.length === 0

  const del = useMutation({
    mutationFn: (id: number) => api.deleteTerm(projectId, id),
    onSuccess: () => {
      qc.invalidateQueries({ queryKey: ['projects', projectId, 'glossary'] })
      toast.success('Đã xoá')
    },
    onError: (e: Error) => toast.error(e.message),
  })

  return (
    <>
      <DataToolbar
        right={
          <Button variant="primary" onClick={() => setEditing({ mode: 'create' })}>
            <Plus size={14} />
            Thêm thuật ngữ
          </Button>
        }
      >
        <SearchInput
          value={q}
          onChange={setQ}
          placeholder={`Tìm trong ${terms.length} thuật ngữ…`}
          className="w-72"
        />
      </DataToolbar>

      <DataTable>
        <thead>
          <tr className="bg-surface-2">
            <Th>Source</Th>
            <Th>Target</Th>
            <Th>Ghi chú</Th>
            <Th className="w-24 text-right pr-3">Thao tác</Th>
          </tr>
        </thead>
        <tbody>
          {showSkeleton && Array.from({ length: 4 }).map((_, i) => (
            <tr key={i} className="border-b border-border-soft last:border-0">
              <td colSpan={4} className="px-4 py-3.5">
                <div className="h-3 rounded bg-surface-2 animate-pulse" />
              </td>
            </tr>
          ))}

          {showEmpty && (
            <tr>
              <td colSpan={4}>
                <EmptyState
                  icon={BookText}
                  title={q ? 'Không tìm thấy' : 'Chưa có thuật ngữ nào'}
                  hint={
                    q
                      ? 'Thử từ khoá khác.'
                      : 'Thuật ngữ giúp model dịch thống nhất tên riêng, biệt danh, chiêu thức.'
                  }
                  action={!q && (
                    <Button variant="primary" onClick={() => setEditing({ mode: 'create' })}>
                      <Plus size={12} />
                      Thêm thuật ngữ
                    </Button>
                  )}
                />
              </td>
            </tr>
          )}

          {!isPending && filtered.map((t) => (
            <tr
              key={t.id}
              className="border-b border-border-soft last:border-0 group hover:bg-hover transition-colors"
            >
              <td className="px-3 py-2.5 text-text font-medium">{t.source_term}</td>
              <td className="px-3 py-2.5 text-text-muted">{t.target_term}</td>
              <td className="px-3 py-2.5 text-xs text-text-subtle truncate max-w-md">{t.notes}</td>
              <td className="px-3 py-2.5">
                <div className="flex items-center gap-1 justify-end opacity-0 group-hover:opacity-100 transition-opacity">
                  <Button
                    variant="ghost"
                    size="sm"
                    icon
                    onClick={() => setEditing({ mode: 'edit', term: t })}
                    title="Sửa"
                  >
                    <Pencil size={13} />
                  </Button>
                  <Button
                    variant="ghost"
                    size="sm"
                    icon
                    onClick={() => confirm(`Xoá "${t.source_term}"?`) && del.mutate(t.id)}
                    className="hover:text-error-text"
                    title="Xoá"
                  >
                    <Trash2 size={13} />
                  </Button>
                </div>
              </td>
            </tr>
          ))}
        </tbody>
      </DataTable>

      {filtered.length > 0 && (
        <p className="text-xs text-text-subtle mt-3 tabular">
          {q
            ? <>Hiển thị <span className="text-text-muted">{filtered.length}</span> trong <span className="text-text-muted">{terms.length}</span> thuật ngữ</>
            : <><span className="text-text-muted">{terms.length}</span> thuật ngữ</>
          }
        </p>
      )}

      <TermDialog
        editing={editing}
        projectId={projectId}
        onClose={() => setEditing(null)}
      />
    </>
  )
}

// ── dialog ─────────────────────────────────────────────────────────────────

function TermDialog({
  editing, projectId, onClose,
}: {
  editing:   Editing
  projectId: number
  onClose:   () => void
}) {
  const qc = useQueryClient()
  const term = editing?.mode === 'edit' ? editing.term : null
  const open = editing !== null

  const [draft, setDraft] = useState({
    source_term: term?.source_term ?? '',
    target_term: term?.target_term ?? '',
    notes:       term?.notes ?? '',
  })

  // Re-seed draft when target term changes (open another edit).
  useMemo(() => {
    setDraft({
      source_term: term?.source_term ?? '',
      target_term: term?.target_term ?? '',
      notes:       term?.notes ?? '',
    })
  }, [term?.id])

  const save = useMutation({
    mutationFn: async () => {
      const body = {
        source_term: draft.source_term.trim(),
        target_term: draft.target_term.trim(),
        notes:       draft.notes.trim() || null,
      }
      if (term) return api.updateTerm(projectId, term.id, body)
      return api.createTerm(projectId, body)
    },
    onSuccess: () => {
      qc.invalidateQueries({ queryKey: ['projects', projectId, 'glossary'] })
      toast.success(term ? 'Đã cập nhật' : 'Đã thêm thuật ngữ')
      onClose()
    },
    onError: (e: Error) => toast.error(e.message),
  })

  const valid = draft.source_term.trim() && draft.target_term.trim()

  return (
    <Modal
      open={open}
      onClose={() => { if (!save.isPending) onClose() }}
      title={term ? 'Sửa thuật ngữ' : 'Thêm thuật ngữ'}
      size="md"
      footer={
        <>
          <Button onClick={onClose} disabled={save.isPending}>
            Huỷ
          </Button>
          <Button
            variant="primary"
            onClick={() => save.mutate()}
            disabled={save.isPending || !valid}
          >
            {save.isPending && <Spinner />}
            {term ? 'Lưu' : 'Thêm'}
          </Button>
        </>
      }
    >
      <div className="px-5 py-4 space-y-4">
        <div className="grid grid-cols-2 gap-3">
          <div>
            <label className={labelCls}>Source <span className="text-error-text">*</span></label>
            <input
              value={draft.source_term}
              onChange={(e) => setDraft({ ...draft, source_term: e.target.value })}
              autoFocus
              className={input}
            />
          </div>
          <div>
            <label className={labelCls}>Target <span className="text-error-text">*</span></label>
            <input
              value={draft.target_term}
              onChange={(e) => setDraft({ ...draft, target_term: e.target.value })}
              className={input}
            />
          </div>
        </div>
        <div>
          <label className={labelCls}>Ghi chú</label>
          <input
            value={draft.notes}
            onChange={(e) => setDraft({ ...draft, notes: e.target.value })}
            placeholder="VD: tên riêng, biệt danh, chiêu thức…"
            className={input}
          />
        </div>
      </div>
    </Modal>
  )
}
