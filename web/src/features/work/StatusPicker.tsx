// StatusPicker — viewer's bookmark + reading status on a Work.
//
// Bookmark and status are two concerns; this control surfaces them
// in priority order:
//
//   no entry yet    "+ Theo dõi" single button. Click POSTs a fresh
//                   library entry with status='reading' — no menu,
//                   no choice; the most common path is one tap.
//
//   has entry       "<icon> <label> ▾" dropdown reflecting the
//                   current status. Menu offers the four status
//                   transitions plus an "Bỏ theo dõi" destructive
//                   item that DELETEs the entry. `dropped` here
//                   means "I've abandoned this story but want to
//                   remember I read it"; "Bỏ theo dõi" is the
//                   actual untrack action.
//
// Every viewer sees their OWN bookmark — the entry is keyed on
// (user, Work). Two readers can disagree on status, and that's fine.

import { useEffect, useRef, useState } from 'react'
import { useMutation, useQueryClient } from '@tanstack/react-query'
import {
  BookOpen, BookmarkPlus, Check, CheckCircle2, ChevronDown,
  Loader2, Star, Trash2, XCircle,
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
  /** Null → no library entry yet for this Work. Trigger collapses
   *  to a single "+ Theo dõi" action. */
  entryId:  number | null
  status:   LibraryStatus | null
  /** Material currently shown — required to create the library entry
   *  when `entryId` is null (server resolves work_id from material_id
   *  and dedupes per user+Work). */
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
    mutationFn: () => {
      if (!material) throw new Error('material required to create entry')
      return api.createLibraryEntry({
        material_id: material.id,
        title:       material.title,
        cover_url:   material.cover_url,
        status:      'reading',
      })
    },
    onSuccess: () => invalidate(),
  })

  const patch = useMutation({
    mutationFn: (next: LibraryStatus) =>
      api.patchLibraryEntry(entryId!, { status: next }),
    onSuccess: () => { invalidate(); setOpen(false) },
  })

  const remove = useMutation({
    mutationFn: () => api.deleteLibraryEntry(entryId!),
    onSuccess: () => { invalidate(); setOpen(false) },
  })

  const pending = create.isPending || patch.isPending || remove.isPending
  const hasEntry = entryId != null
  const current  = hasEntry
    ? OPTIONS.find((o) => o.code === status) ?? OPTIONS[0]!
    : null

  // ── No entry: single-button bookmark action ─────────────────
  //
  // No dropdown — the common path is one tap to start tracking with
  // status='reading'. The user can change status later through the
  // picker the entry exposes.
  if (!hasEntry) {
    return (
      <button
        type="button"
        onClick={() => create.mutate()}
        disabled={pending || !material}
        className={cn(
          'inline-flex items-center gap-1.5 h-8 px-2.5 rounded-sm text-sm',
          'bg-accent text-accent-fg hover:brightness-110',
          'cursor-pointer transition-[filter]',
          (pending || !material) && 'opacity-60 cursor-wait',
        )}
        title="Thêm vào Thư viện"
      >
        {pending
          ? <Loader2 size={13} className="animate-spin" />
          : <BookmarkPlus size={13} />}
        <span>Theo dõi</span>
      </button>
    )
  }

  // ── Has entry: status dropdown + destructive untrack ────────
  return (
    <div className="relative inline-block" ref={wrap}>
      <button
        type="button"
        onClick={() => setOpen((v) => !v)}
        disabled={pending}
        className={cn(
          'inline-flex items-center gap-1.5 h-8 px-2.5 rounded-sm text-sm',
          'bg-surface-2 text-text hover:bg-hover',
          'cursor-pointer transition-colors',
          pending && 'opacity-60 cursor-wait',
        )}
        title="Đã theo dõi — đổi trạng thái hoặc bỏ theo dõi"
      >
        {pending
          ? <Loader2 size={13} className="animate-spin" />
          : <Star size={13} className="text-accent fill-accent" />}
        <span>{current!.label}</span>
        <ChevronDown size={12} className="opacity-70" />
      </button>

      {open && (
        <div
          role="menu"
          className={cn(
            'absolute left-0 top-full mt-1 z-30 min-w-[200px]',
            'rounded-sm bg-surface border border-border shadow-md py-1',
          )}
        >
          {OPTIONS.map((opt) => {
            const active = opt.code === status
            return (
              <button
                key={opt.code}
                type="button"
                onClick={() => {
                  if (active) { setOpen(false); return }
                  patch.mutate(opt.code)
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

          <div className="my-1 border-t border-border-soft" />

          <button
            type="button"
            onClick={() => remove.mutate()}
            disabled={remove.isPending}
            className={cn(
              'w-full text-left px-3 py-1.5 text-sm cursor-pointer',
              'flex items-center gap-2 hover:bg-rose-500/10',
              'text-rose-400 hover:text-rose-300',
            )}
          >
            <Trash2 size={12} />
            <span>Bỏ theo dõi</span>
          </button>
        </div>
      )}
    </div>
  )
}
