import { type ReactNode } from 'react'
import { create } from 'zustand'
import { Modal } from './Modal'
import { Button } from './Button'

// =============================================================================
// Confirm dialog — imperative `confirm({...}): Promise<boolean>` API.
//
// Replaces `window.confirm()` / `window.alert()` / `window.prompt()`, which
// are silently blocked inside sandboxed iframes (Discord Activity). The trash
// button used to look dead ("ấn vào không có gì") because the native dialog
// never appeared.
//
// Mount <ConfirmHost /> once at the app root (AppLayout). Anywhere else:
//
//   if (await confirm({ title: 'Xoá Ch.12?', tone: 'danger' })) ...
//
// Same store pattern as <Toaster /> — single global host, callers stay
// stateless. ESC / backdrop / Cancel resolve `false`; Confirm resolves `true`.
// =============================================================================

type Tone = 'default' | 'danger'

interface ConfirmOptions {
  title:        string
  description?: ReactNode
  confirmText?: string
  cancelText?:  string
  tone?:        Tone
}

interface Pending extends ConfirmOptions {
  id:      number
  resolve: (ok: boolean) => void
}

interface State {
  current: Pending | null
  open:    (opts: ConfirmOptions, resolve: (ok: boolean) => void) => void
  close:   (ok: boolean) => void
}

let nextId = 1

const useConfirmStore = create<State>((set, get) => ({
  current: null,
  open: (opts, resolve) => {
    // If a confirm is already open, cancel it before showing the new one.
    // Two confirms stacked is never a deliberate UX — it means a stale
    // promise was leaked. Resolve the old one as `false` to free callers.
    const prev = get().current
    if (prev) prev.resolve(false)
    set({ current: { id: nextId++, ...opts, resolve } })
  },
  close: (ok) => {
    const cur = get().current
    if (!cur) return
    cur.resolve(ok)
    set({ current: null })
  },
}))

export function confirm(opts: ConfirmOptions): Promise<boolean> {
  return new Promise((resolve) => {
    useConfirmStore.getState().open(opts, resolve)
  })
}

export function ConfirmHost() {
  const current = useConfirmStore((s) => s.current)
  const close   = useConfirmStore((s) => s.close)

  return (
    <Modal
      open={!!current}
      onClose={() => close(false)}
      title={current?.title ?? ''}
      size="sm"
      layer="top"
      footer={
        <>
          <Button variant="ghost" onClick={() => close(false)}>
            {current?.cancelText ?? 'Huỷ'}
          </Button>
          <Button
            variant={current?.tone === 'danger' ? 'danger' : 'primary'}
            onClick={() => close(true)}
            autoFocus
          >
            {current?.confirmText ?? 'Xác nhận'}
          </Button>
        </>
      }
    >
      {current?.description != null && (
        <div className="px-5 py-4 text-sm text-text-muted leading-relaxed">
          {current.description}
        </div>
      )}
    </Modal>
  )
}
