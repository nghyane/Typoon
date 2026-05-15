// Toaster — global toast surface. Single store + a single host
// (mounted once in the app root). Toasts auto-dismiss after a
// configurable timeout; toasts with `action` stay visible longer
// so the user has time to react.
//
// Three tones (success / error / info) cover every reader signal
// at this stage. `action` is optional and renders as a secondary
// pill inside the toast — used by the reader's spawn pipeline to
// surface a "Đọc ngay" affordance once a translation finishes.

import { create } from 'zustand'
import { CheckCircle2, AlertCircle, Info, X } from 'lucide-react'
import { cn } from '@shared/lib/cn'

type Tone = 'success' | 'error' | 'info'

export interface ToastAction {
  label: string
  onClick: () => void
}

interface ToastEntry {
  id:     number
  tone:   Tone
  text:   string
  action: ToastAction | null
  /** Milliseconds the toast lives before auto-dismiss. */
  ttl:    number
}

interface ToastStore {
  items:   ToastEntry[]
  push:    (
    tone:   Tone,
    text:   string,
    opts?:  { action?: ToastAction; ttl?: number },
  ) => number
  dismiss: (id: number) => void
}

let nextId = 1

const DEFAULT_TTL  = 3500
const ACTION_TTL   = 8000   // longer so the user has time to click

const useToastStore = create<ToastStore>((set) => ({
  items: [],
  push: (tone, text, opts) => {
    const id = nextId++
    const ttl = opts?.ttl ?? (opts?.action ? ACTION_TTL : DEFAULT_TTL)
    set((s) => ({
      items: [...s.items, {
        id, tone, text, ttl,
        action: opts?.action ?? null,
      }],
    }))
    if (ttl > 0) {
      setTimeout(
        () => set((s) => ({ items: s.items.filter((t) => t.id !== id) })),
        ttl,
      )
    }
    return id
  },
  dismiss: (id) => set((s) => ({ items: s.items.filter((t) => t.id !== id) })),
}))

export const toast = {
  success: (text: string, opts?: { action?: ToastAction; ttl?: number }) =>
    useToastStore.getState().push('success', text, opts),
  error:   (text: string, opts?: { action?: ToastAction; ttl?: number }) =>
    useToastStore.getState().push('error', text, opts),
  info:    (text: string, opts?: { action?: ToastAction; ttl?: number }) =>
    useToastStore.getState().push('info', text, opts),
  dismiss: (id: number) => useToastStore.getState().dismiss(id),
}

const ICON: Record<Tone, typeof CheckCircle2> = {
  success: CheckCircle2,
  error:   AlertCircle,
  info:    Info,
}

const ICON_COLOR: Record<Tone, string> = {
  success: 'text-success-text',
  error:   'text-error-text',
  info:    'text-info-text',
}

export function Toaster() {
  const items = useToastStore((s) => s.items)
  const dismiss = useToastStore((s) => s.dismiss)

  return (
    <div
      className={cn(
        'fixed z-[60] flex flex-col gap-2 pointer-events-none',
        'right-[max(1rem,var(--sair))]',
        'bottom-[max(1rem,calc(var(--saib)+3.5rem+0.5rem))] sm:bottom-[max(1rem,var(--saib))]',
      )}
    >
      {items.map((t) => {
        const Icon = ICON[t.tone]
        return (
          <div
            key={t.id}
            className={cn(
              'pointer-events-auto flex items-center gap-2.5 min-w-72 max-w-md pl-3 pr-2 py-2.5',
              'rounded-md bg-surface text-text shadow-[0_8px_24px_rgb(0,0,0,0.4)] text-sm',
              'animate-[fadeIn_120ms_ease-out]',
            )}
          >
            <Icon size={15} className={cn('shrink-0', ICON_COLOR[t.tone])} />
            <span className="flex-1 truncate">{t.text}</span>
            {t.action && (
              <button
                onClick={() => {
                  t.action!.onClick()
                  dismiss(t.id)
                }}
                className={cn(
                  'shrink-0 h-7 px-2.5 rounded-md',
                  'bg-accent text-accent-fg text-xs font-medium',
                  'hover:bg-accent-strong transition-colors duration-150',
                  'cursor-pointer',
                )}
              >
                {t.action.label}
              </button>
            )}
            <button
              onClick={() => dismiss(t.id)}
              className="size-6 rounded-xs flex items-center justify-center text-text-subtle hover:text-text hover:bg-hover cursor-pointer"
              aria-label="Đóng"
            >
              <X size={12} />
            </button>
          </div>
        )
      })}
    </div>
  )
}
