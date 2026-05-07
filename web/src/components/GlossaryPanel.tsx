import { useState, useMemo } from 'react'
import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query'
import { Search, Plus, Pencil, Trash2, Check } from 'lucide-react'
import { api, type ApiGlossaryTerm } from '../lib/api'
import { cn } from '../lib/cn'
import { btn, input } from './ui'
import { toast } from './Toaster'

interface Props { projectId: number }

export function GlossaryPanel({ projectId }: Props) {
  const qc = useQueryClient()
  const [q,         setQ]         = useState('')
  const [editing,   setEditing]   = useState<ApiGlossaryTerm | null>(null)
  const [draft,     setDraft]     = useState({ source_term: '', target_term: '', notes: '' })
  const [creating,  setCreating]  = useState(false)

  const { data: terms = [], isLoading } = useQuery({
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

  const save = useMutation({
    mutationFn: async () => {
      const body = {
        source_term: draft.source_term.trim(),
        target_term: draft.target_term.trim(),
        notes:       draft.notes.trim() || null,
      }
      if (editing) return api.updateTerm(projectId, editing.id, body)
      return api.createTerm(projectId, body)
    },
    onSuccess: () => {
      qc.invalidateQueries({ queryKey: ['projects', projectId, 'glossary'] })
      toast.success(editing ? 'Đã cập nhật' : 'Đã thêm thuật ngữ')
      setEditing(null)
      setCreating(false)
      setDraft({ source_term: '', target_term: '', notes: '' })
    },
    onError: (e: Error) => toast.error(e.message),
  })

  const del = useMutation({
    mutationFn: (id: number) => api.deleteTerm(projectId, id),
    onSuccess: () => {
      qc.invalidateQueries({ queryKey: ['projects', projectId, 'glossary'] })
      toast.success('Đã xoá')
    },
    onError: (e: Error) => toast.error(e.message),
  })

  const startEdit = (t: ApiGlossaryTerm) => {
    setCreating(false)
    setEditing(t)
    setDraft({ source_term: t.source_term, target_term: t.target_term, notes: t.notes ?? '' })
  }

  const startCreate = () => {
    setEditing(null)
    setCreating(true)
    setDraft({ source_term: '', target_term: '', notes: '' })
  }

  const cancelEdit = () => {
    setEditing(null)
    setCreating(false)
    setDraft({ source_term: '', target_term: '', notes: '' })
  }

  return (
    <div className="space-y-4">
      <div className="flex items-center gap-2">
        <label className="flex items-center gap-2 h-9 px-3 flex-1 max-w-sm rounded-lg border border-zinc-200 hover:border-zinc-300 focus-within:border-zinc-400 transition-colors cursor-text">
          <Search size={13} className="text-zinc-400 shrink-0" />
          <input
            placeholder={`Tìm trong ${terms.length} thuật ngữ...`}
            value={q}
            onChange={(e) => setQ(e.target.value)}
            className="flex-1 bg-transparent outline-none text-sm placeholder:text-zinc-300"
          />
        </label>
        <button
          onClick={startCreate}
          className={btn.primary}
        >
          <Plus size={14} />
          Thêm
        </button>
      </div>

      {(creating || editing) && (
        <div className="rounded-xl border border-zinc-200 p-4 bg-zinc-50/40">
          <div className="grid grid-cols-2 gap-3 mb-3">
            <input
              placeholder="Source term"
              value={draft.source_term}
              onChange={(e) => setDraft({ ...draft, source_term: e.target.value })}
              className={input}
              autoFocus
            />
            <input
              placeholder="Target term"
              value={draft.target_term}
              onChange={(e) => setDraft({ ...draft, target_term: e.target.value })}
              className={input}
            />
          </div>
          <input
            placeholder="Ghi chú (tuỳ chọn)"
            value={draft.notes}
            onChange={(e) => setDraft({ ...draft, notes: e.target.value })}
            className={input}
          />
          <div className="flex justify-end gap-2 mt-3">
            <button onClick={cancelEdit} className={btn.secondary}>Huỷ</button>
            <button
              onClick={() => save.mutate()}
              disabled={save.isPending || !draft.source_term.trim() || !draft.target_term.trim()}
              className={btn.primary}
            >
              <Check size={14} />
              {editing ? 'Lưu' : 'Thêm'}
            </button>
          </div>
        </div>
      )}

      <div className="rounded-xl border border-zinc-200 overflow-hidden">
        <table className="w-full text-sm">
          <thead>
            <tr className="border-b border-zinc-100 bg-zinc-50/50">
              <Th>Source</Th>
              <Th>Target</Th>
              <Th>Ghi chú</Th>
              <Th className="w-24 text-right pr-4">Thao tác</Th>
            </tr>
          </thead>
          <tbody>
            {isLoading && (
              <tr><td colSpan={4} className="py-6 text-center text-sm text-zinc-400">Đang tải…</td></tr>
            )}
            {!isLoading && filtered.length === 0 && (
              <tr>
                <td colSpan={4} className="py-12 text-center">
                  <p className="text-sm text-zinc-500 font-medium">
                    {q ? 'Không tìm thấy' : 'Chưa có thuật ngữ nào'}
                  </p>
                  <p className="text-xs text-zinc-400 mt-1">
                    {q ? 'Thử từ khoá khác' : 'Thêm thuật ngữ để dịch thống nhất'}
                  </p>
                </td>
              </tr>
            )}
            {filtered.map((t) => (
              <tr key={t.id} className="border-b border-zinc-100 last:border-0 group hover:bg-zinc-50/60 transition-colors">
                <td className="px-3 py-2.5 text-zinc-900 font-medium">{t.source_term}</td>
                <td className="px-3 py-2.5 text-zinc-700">{t.target_term}</td>
                <td className="px-3 py-2.5 text-xs text-zinc-400 truncate max-w-md">{t.notes}</td>
                <td className="px-3 py-2.5">
                  <div className="flex items-center gap-1 justify-end opacity-0 group-hover:opacity-100 transition-opacity">
                    <button onClick={() => startEdit(t)} className={btn.ghost} title="Sửa"><Pencil size={13} /></button>
                    <button
                      onClick={() => confirm(`Xoá "${t.source_term}"?`) && del.mutate(t.id)}
                      className={cn(btn.ghost, 'hover:text-red-600')}
                      title="Xoá"
                    >
                      <Trash2 size={13} />
                    </button>
                  </div>
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>

      {filtered.length > 0 && (
        <p className="text-xs text-zinc-400">
          {q ? `${filtered.length} / ${terms.length}` : `${terms.length} thuật ngữ`}
        </p>
      )}
    </div>
  )
}

function Th({ children, className }: { children: React.ReactNode; className?: string }) {
  return (
    <th className={cn('px-3 py-2.5 text-left text-xs font-medium text-zinc-400', className)}>
      {children}
    </th>
  )
}
