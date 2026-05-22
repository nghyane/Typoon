// ChapterActions — text-button strip rendered at the right end of
// each chapter row. State-driven; mutations come from WorkActionsContext.
//
// Visual rules (from design discussion):
//   running / saving  →  "Đang dịch ⟳" / "Đang lưu ⟳"   status text
//   error             →  [Thử lại]                        ghost sm error
//   saved-translated  →  ✓ [Dịch lại]                     icon + ghost sm
//   saved-raw         →  ✓                                icon only
//   done-online       →  [Lưu] [Dịch lại]                 2× ghost sm
//   done-expired      →  [Tải lại] [Dịch lại]             refresh + retranslate
//   needs translate   →  [Dịch]                           accent ghost sm
//   raw-only          →  [Lưu]                            ghost sm
//   idle (no version) →  nothing

import { useState } from 'react'
import { CheckCircle, Loader2 } from 'lucide-react'

import { cn } from '@shared/lib/cn'
import { toast } from '@shared/ui/Toaster'

import { useWorkActions } from '../contexts/WorkActionsContext'
import type { ChapterState, SourceVersion } from '../data/types'


interface Props {
  chapterRef:    string
  chapterNumber: string
  version:       SourceVersion | null
  targetLang:    string
  state:         ChapterState
}


export function ChapterActions({
  chapterRef, chapterNumber, version, targetLang, state,
}: Props) {
  const actions = useWorkActions()
  const [busy, setBusy] = useState<'spawn' | 'save' | null>(null)

  const stop = (e: React.MouseEvent) => { e.preventDefault(); e.stopPropagation() }

  async function withBusy<T>(kind: 'spawn' | 'save', fn: () => Promise<T>) {
    setBusy(kind)
    try { await fn() }
    catch (e) { toast.error((e as Error).message) }
    finally   { setBusy(null) }
  }

  const handleTranslate = (e: React.MouseEvent) => {
    stop(e)
    if (!version) return
    void withBusy('spawn', () => actions.spawnTranslate(chapterRef, version))
  }

  const handleSaveRaw = (e: React.MouseEvent) => {
    stop(e)
    if (!version) return
    void withBusy('save', async () => {
      await actions.saveRawOffline(chapterRef, version)
      toast.success(`Đã lưu Ch.${chapterNumber} offline`)
    })
  }

  const handleDownload = (e: React.MouseEvent) => {
    stop(e)
    if (!state.job?.id || !state.job.archive_url) return
    void withBusy('save', async () => {
      await actions.downloadTranslated(chapterRef, state.job!.id, state.job!.archive_url!)
      toast.success(`Đã lưu Ch.${chapterNumber} offline`)
    })
  }

  // ── Local UI busy override running/saving ──────────────────
  if (busy === 'save')  return <Status label="Đang lưu" />
  if (busy === 'spawn') return <Status label="Đang dịch" />

  switch (state.status) {
    case 'running':
      return <Status label="Đang dịch" />

    case 'error':
      return (
        <ActionBtn onClick={handleTranslate} className="text-error-text">
          Thử lại
        </ActionBtn>
      )

    case 'saved-translated':
      return (
        <span className="inline-flex items-center gap-2 shrink-0">
          <CheckCircle size={13} className="text-success shrink-0" />
          {version && (
            <ActionBtn onClick={handleTranslate}>Dịch lại</ActionBtn>
          )}
        </span>
      )

    case 'saved-raw':
      return <CheckCircle size={13} className="text-success shrink-0" />

    case 'done-online':
      return (
        <span className="inline-flex items-center gap-2 shrink-0">
          <ActionBtn onClick={handleDownload}>Lưu</ActionBtn>
          {version && (
            <ActionBtn onClick={handleTranslate}>Dịch lại</ActionBtn>
          )}
        </span>
      )

    case 'done-expired':
      // For now treat like done-online — reader refreshes URL on open.
      // Saving offline still requires a fresh URL; defer that to
      // a follow-up that wires `useRefreshArchiveUrl`.
      return (
        <span className="inline-flex items-center gap-2 shrink-0">
          {version && (
            <ActionBtn onClick={handleTranslate}>Dịch lại</ActionBtn>
          )}
        </span>
      )

    case 'idle': {
      const langOfRow = version?.lang ?? targetLang
      const needs = version && langOfRow !== targetLang
      if (needs) {
        return (
          <ActionBtn onClick={handleTranslate} className="text-accent-text">
            Dịch
          </ActionBtn>
        )
      }
      if (version) {
        return <ActionBtn onClick={handleSaveRaw}>Lưu</ActionBtn>
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
      onClick={onClick}
      disabled={disabled}
      className={cn(
        'text-xs text-text-subtle hover:text-text transition-colors cursor-pointer',
        'disabled:opacity-50 disabled:cursor-not-allowed whitespace-nowrap',
        className,
      )}
    >
      {children}
    </button>
  )
}
