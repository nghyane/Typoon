// Lazy-search combobox for the user's projects. Keyboard-first:
//   - typing focuses the listbox
//   - ↑/↓ moves selection
//   - Enter picks the highlighted item
//   - Esc closes
//
// Phase 1 trick: when the query has no match we offer "+ Tạo project"
// inline — but creating a project needs source/target language, so we
// just call back to the parent with a sentinel and let it open a modal.

import { useEffect, useRef, useState } from 'react'
import { cn } from '@shared/lib/cn'
import { input, Spinner } from '@shared/ui/Field'
import { useMyProjects } from '@shell/hooks/useMyProjects'
import type { ApiMeProject } from '@core/typoon'

export interface ProjectPickerProps {
  value:    ApiMeProject | null
  onChange: (p: ApiMeProject) => void
  /** Called when the user wants to create a project named `query`. */
  onCreate: (suggestedTitle: string) => void
}

export function ProjectPicker({ value, onChange, onCreate }: ProjectPickerProps) {
  const [query, setQuery] = useState(value?.title ?? '')
  const [open,  setOpen]  = useState(false)
  const [hi,    setHi]    = useState(0)
  const ref = useRef<HTMLInputElement>(null)

  const { projects, isLoading } = useMyProjects(query)
  const showCreate = query.trim().length > 0
    && !projects.some(p => p.title.toLowerCase() === query.trim().toLowerCase())
  const items = projects.length + (showCreate ? 1 : 0)

  useEffect(() => { setHi(0) }, [query])

  function commit(idx: number) {
    if (showCreate && idx === projects.length) {
      onCreate(query.trim())
    } else {
      const p = projects[idx]
      if (!p) return
      onChange(p)
      setQuery(p.title)
    }
    setOpen(false)
    ref.current?.blur()
  }

  return (
    <div className="relative">
      <input
        ref={ref}
        className={input}
        type="text"
        autoComplete="off"
        spellCheck={false}
        placeholder="Tìm hoặc tạo project…"
        value={query}
        onChange={e => { setQuery(e.target.value); setOpen(true) }}
        onFocus={() => setOpen(true)}
        onBlur={() => setTimeout(() => setOpen(false), 120)}
        onKeyDown={e => {
          if (e.key === 'ArrowDown') { e.preventDefault(); setHi(h => Math.min(h + 1, items - 1)) }
          else if (e.key === 'ArrowUp')   { e.preventDefault(); setHi(h => Math.max(h - 1, 0)) }
          else if (e.key === 'Enter')     { e.preventDefault(); commit(hi) }
          else if (e.key === 'Escape')    { setOpen(false) }
        }}
      />
      {open && (
        <ul
          role="listbox"
          className="absolute left-0 right-0 top-full mt-1 z-10 max-h-48 overflow-y-auto bg-surface-2 rounded-sm shadow-lg"
        >
          {isLoading && (
            <li className="px-3 py-2 text-xs text-text-subtle flex items-center gap-2">
              <Spinner size={12} /> Đang tải…
            </li>
          )}
          {!isLoading && projects.length === 0 && !showCreate && (
            <li className="px-3 py-2 text-xs text-text-subtle">Không có project nào.</li>
          )}
          {projects.map((p, i) => (
            <li
              key={p.project_id}
              role="option"
              aria-selected={i === hi}
              onMouseDown={() => commit(i)}
              onMouseEnter={() => setHi(i)}
              className={cn(
                'px-3 py-1.5 text-xs cursor-pointer truncate',
                i === hi ? 'bg-row-active text-text' : 'text-text-muted',
              )}
            >
              {p.title}
            </li>
          ))}
          {showCreate && (
            <li
              role="option"
              aria-selected={hi === projects.length}
              onMouseDown={() => commit(projects.length)}
              onMouseEnter={() => setHi(projects.length)}
              className={cn(
                'px-3 py-1.5 text-xs cursor-pointer border-t border-border-soft',
                hi === projects.length ? 'bg-row-active text-text' : 'text-accent-text',
              )}
            >
              + Tạo project: <b className="text-text">{query.trim()}</b>
            </li>
          )}
        </ul>
      )}
    </div>
  )
}
