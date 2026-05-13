// StatusPicker — viewer's reading status on a Work.
//
// Single control covering both states:
//
//   no entry yet     "+ Theo dõi" trigger. Dropdown picks the
//                    initial status; the click POSTs a fresh
//                    library entry with that status.
//
//   have entry       "<icon> <label> ▾" trigger reflecting the
//                    current status. Dropdown PATCHes to the new
//                    one. `dropped` means "không theo dõi nữa" —
//                    no separate remove action, no separate
//                    bookmark button.
//
// `material.status` (publication state, e.g. "Completed") is a
// different signal and renders as a passive chip elsewhere.

import { useEffect, useRef, useState } from 'react'
import { useMutation, useQueryClient } from '@tanstack/react-query'
import {
  BookOpen, BookmarkPlus, Check, CheckCircle2, ChevronDown,
  Loader2, Plus, XCircle,
} from 'lucide-react'

import { api } from '@shared/api/api'
import type { LibraryStatus } from '@shared/api/api'
import { qk } from '@shared/api/keys'
import { cn } from '@shared/lib/cn'


interface Option {
  code:  LibraryStatus
  label: string
  icon:  React.ReactNode
}

const OPTIONS: Option[] = [
  { code: 'reading', label: 'Đang đọc', icon: <BookOpen     size={13} /> },
  { code: 'plan',    label: 'Để dành',  icon: <BookmarkPlus size={13} /> },
  { code: 'done',    label: 'Đã đọc',   icon: <CheckCircle2 size={13} /> },
  { code: 'dropped', label: 'Đã bỏ',    icon: <XCircle      size={13} /> },
]


interface Props {
  workId:   number
  /** Null → no library entry yet for this Work. Trigger renders
   *  "Theo dõi"; selecting a status creates the entry. */
  entryId:  number | null
  status:   LibraryStatus | null
  /** Material currently shown — used to create the library entry
   *  (server resolves work_id from material_id and dedupes per
   *  user+Work). Required when `entryId` is null. */
  material: { id: number; title: string; cover_url: string | null } | null
}


export function StatusPicker({ workId, entryId, status, material }: Props) {
  const qc = useQueryClient()
  const [open, setOpen] = useState(false)
  const wrap = useRef<HTMLDivElement>(null)

  useEffect(() => {
    if (!open) return
    const onDoc = (e: MouseEvent) => {
      if (!wrap.current?.contains(e.target as Node)) setOpen(false)
    }
    const onEsc = (e: KeyboardEvent) => {
      if (e.key === 'Escape') setOpen(false)
    }
    document.addEventListener('mousedown', onDoc)
    document.addEventListener('keydown', onEsc)
    return () => {
      document.removeEventListener('mousedown', onDoc)
      document.removeEventListener('keydown', onEsc)
    }
  }, [open])

  const invalidate = () => {
    qc.invalidateQueries({ queryKey: qk.work.byId(workId) })
    qc.invalidateQueries({ queryKey: qk.library.all() })
  }

  const create = useMutation({
    mutationFn: (next: LibraryStatus) => {
      if (!material) throw new Error('material required to create entry')
      return api.createLibraryEntry({
        material_id: material.id,
        title:       material.title,
        cover_url:   material.cover_url,
        status:      next,
      })
    },
    onSuccess: () => { invalidate(); setOpen(false) },
  })

  const patch = useMutation({
    mutationFn: (next: LibraryStatus) =>
      api.patchLibraryEntry(entryId!, { status: next }),
    onSuccess: () => { invalidate(); setOpen(false) },
  })

  const pending = create.isPending || patch.isPending
  const hasEntry = entryId != null
  const current  = hasEntry
    ? OPTIONS.find((o) => o.code === status) ?? OPTIONS[0]!
    : null

  // Disable when this Work has no material the entry can attach
  // to (rare — only happens before the active source resolves).
  const disabled = !hasEntry && !material

  return (
    <div className="relative inline-block" ref={wrap}>
      <button
        type="button"
        onClick={() => !disabled && setOpen((v) => !v)}
        disabled={pending || disabled}
        className={cn(
          'inline-flex items-center gap-1.5 h-8 px-2.5 rounded-sm text-sm',
          'cursor-pointer transition-colors',
          hasEntry
            ? 'bg-surface-2 text-text hover:bg-hover'
            : 'bg-accent text-accent-fg hover:brightness-110',
          (pending || disabled) && 'opacity-60 cursor-wait',
        )}
        title={hasEntry ? 'Đổi trạng thái đọc' : 'Thêm vào Thư viện'}
      >
        {pending
          ? <Loader2 size={13} className="animate-spin" />
          : current?.icon ?? <Plus size={13} />}
        <span>{current?.label ?? 'Theo dõi'}</span>
        <ChevronDown size={12} className="opacity-70" />
      </button>

      {open && (
        <div
          role="listbox"
          className={cn(
            'absolute left-0 top-full mt-1 z-30 min-w-[180px]',
            'rounded-sm bg-surface border border-border shadow-md py-1',
          )}
        >
          {OPTIONS.map((opt) => {
            const active = hasEntry && opt.code === status
            return (
              <button
                key={opt.code}
                type="button"
                onClick={() => {
                  if (active) { setOpen(false); return }
                  if (hasEntry) patch.mutate(opt.code)
                  else          create.mutate(opt.code)
                }}
                className={cn(
                  'w-full text-left px-3 py-1.5 text-sm cursor-pointer',
                  'flex items-center gap-2 hover:bg-hover',
                  active && 'text-accent',
                )}
              >
                {opt.icon}
                <span className="flex-1">{opt.label}</span>
                {active && <Check size={12} />}
              </button>
            )
          })}
        </div>
      )}
    </div>
  )
}
