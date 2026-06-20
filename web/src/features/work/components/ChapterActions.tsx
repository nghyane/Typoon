// ChapterActions — text-button strip rendered at the right end of
// each chapter row. State-driven; mutations come from WorkActionsContext.
//
// Translation is reader-local now; chapter rows only expose offline
// caching. The row itself opens the reader.
//   idle (no version) →  nothing

import { useState } from 'react'
import { CheckCircle, Download, Loader2 } from 'lucide-react'

import { cn } from '@shared/lib/cn'
import { toast } from '@shared/ui/Toaster'

import { useWorkActions } from '../contexts/WorkActionsContext'
import type { ChapterState, SourceVersion } from '../data/types'


interface Props {
  chapterRef:    string
  chapterNumber: string
  version:       SourceVersion | null
  state:         ChapterState
}


export function ChapterActions({
  chapterRef, chapterNumber, version, state,
}: Props) {
  const actions = useWorkActions()
  const [busy, setBusy] = useState<'save' | null>(null)

  const stop = (e: React.MouseEvent) => { e.preventDefault(); e.stopPropagation() }

  async function withBusy<T>(kind: 'save', fn: () => Promise<T>) {
    setBusy(kind)
    try { await fn() }
    catch (e) { toast.error((e as Error).message) }
    finally   { setBusy(null) }
  }

  const handleSaveRaw = (e: React.MouseEvent) => {
    stop(e)
    if (!version) return
    void withBusy('save', async () => {
      await actions.saveRawOffline(chapterRef, version)
      toast.success(`Đã lưu Ch.${chapterNumber} offline`)
    })
  }

  // ── Local UI busy override running/saving ──────────────────
  if (busy === 'save')  return <Status label="Đang lưu" />

  switch (state.status) {
    case 'saved-raw':
      return <CheckCircle size={13} className="text-success shrink-0" />

    case 'idle': {
      if (version) {
        return <ActionBtn onClick={handleSaveRaw}><Download size={13} /></ActionBtn>
      }
      return null
    }
  }
}


// ── Sub-components ─────────────────────────────────────────────


function Status({ label }: { label: string }) {
  return (
    <span className="inline-flex items-center gap-1.5 text-xs text-text-subtle whitespace-nowrap">
      <Loader2 size={12} className="animate-spin" />
      {label}
    </span>
  )
}


function ActionBtn({
  onClick, disabled, className, children,
}: {
  onClick:   (e: React.MouseEvent) => void
  disabled?: boolean
  className?: string
  children:  React.ReactNode
}) {
  return (
    <button
      type="button"
      title="Lưu offline"
      onClick={onClick}
      disabled={disabled}
      className={cn(
        'inline-flex items-center justify-center size-7 rounded-full text-xs text-text-subtle hover:text-text hover:bg-surface-2 transition-colors cursor-pointer',
        'disabled:opacity-50 disabled:cursor-not-allowed whitespace-nowrap',
        className,
      )}
    >
      {children}
    </button>
  )
}
