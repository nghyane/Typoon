// TargetLangPicker — inline dropdown for the viewer's reading-lang
// preference on a Work.
//
// Surface: sits in `WorkHero` as a small button rendering "Đọc bằng
// Tiếng Việt"; click pops a 5-option dropdown (vi / en / ja / ko /
// zh). Saving PATCHes `library_entries.target_lang` and invalidates
// the work cache so chapter list / manifest fetch re-derive against
// the new language.
//
// Only rendered when the viewer has an entry (i.e. has bookmarked
// the work). Before that the SPA can't persist a preference — the
// chapter list falls back to the source's declared default.

import { useEffect, useRef, useState } from 'react'
import { useMutation, useQueryClient } from '@tanstack/react-query'
import { Check, ChevronDown, Languages, Loader2 } from 'lucide-react'

import { api } from '@shared/api/api'
import { qk } from '@shared/api/keys'
import { languageName } from '@shared/lib/lang'
import { cn } from '@shared/lib/cn'


/** Languages the user can pick to read in. Kept short by design —
 *  longer tail belongs in a full settings page, not this inline
 *  control. */
const OPTIONS: ReadonlyArray<string> = ['vi', 'en', 'ja', 'ko', 'zh']


interface Props {
  entryId:    number
  workId:     number
  targetLang: string
}


export function TargetLangPicker({ entryId, workId, targetLang }: Props) {
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

  const m = useMutation({
    mutationFn: (next: string) =>
      api.patchLibraryEntry(entryId, { target_lang: next }),
    onSuccess: () => {
      qc.invalidateQueries({ queryKey: qk.work.byId(workId) })
      qc.invalidateQueries({ queryKey: qk.library.all() })
      setOpen(false)
    },
  })

  return (
    <div className="relative inline-block" ref={wrap}>
      <button
        type="button"
        onClick={() => setOpen((v) => !v)}
        disabled={m.isPending}
        className={cn(
          'inline-flex items-center gap-1.5 h-8 px-2.5 rounded-sm text-sm',
          'text-text bg-surface-2 hover:bg-hover',
          'cursor-pointer transition-colors',
          m.isPending && 'opacity-60 cursor-wait',
        )}
        title="Đổi ngôn ngữ đọc"
      >
        {m.isPending
          ? <Loader2 size={13} className="animate-spin" />
          : <Languages size={13} />}
        <span>Đọc bằng {languageName(targetLang)}</span>
        <ChevronDown size={12} className="opacity-70" />
      </button>

      {open && (
        <div
          role="listbox"
          className={cn(
            'absolute left-0 top-full mt-1 z-30 min-w-[160px]',
            'rounded-sm bg-surface border border-border shadow-md py-1',
          )}
        >
          {OPTIONS.map((code) => {
            const active = code === targetLang
            return (
              <button
                key={code}
                type="button"
                onClick={() => {
                  if (active) { setOpen(false); return }
                  m.mutate(code)
                }}
                className={cn(
                  'w-full text-left px-3 py-1.5 text-sm cursor-pointer',
                  'flex items-center gap-2 hover:bg-hover',
                  active && 'text-accent',
                )}
              >
                {active
                  ? <Check size={12} />
                  : <span className="w-3" />}
                <span>{languageName(code)}</span>
              </button>
            )
          })}
        </div>
      )}
    </div>
  )
}
