import { create } from 'zustand'
import { CheckCircle2, AlertCircle, Info, X } from 'lucide-react'
import { cn } from '@shared/lib/cn'

type Tone = 'success' | 'error' | 'info'

interface ToastEntry {
  id:   number
  tone: Tone
  text: string
}

interface ToastStore {
  items:   ToastEntry[]
  push:    (tone: Tone, text: string) => void
  dismiss: (id: number) => void
}

let nextId = 1

const useToastStore = create<ToastStore>((set) => ({
  items: [],
  push: (tone, text) => {
    const id = nextId++
    set((s) => ({ items: [...s.items, { id, tone, text }] }))
    setTimeout(() => set((s) => ({ items: s.items.filter((t) => t.id !== id) })), 3500)
  },
  dismiss: (id) => set((s) => ({ items: s.items.filter((t) => t.id !== id) })),
}))

export const toast = {
  success: (text: string) => useToastStore.getState().push('success', text),
  error:   (text: string) => useToastStore.getState().push('error', text),
  info:    (text: string) => useToastStore.getState().push('info', text),
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
    <div className="fixed bottom-4 right-4 z-[60] flex flex-col gap-2 pointer-events-none">
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
