import { create } from 'zustand'
import { useEffect } from 'react'
import { CheckCircle2, AlertCircle, Info, X } from 'lucide-react'
import { cn } from '../lib/cn'

type Tone = 'success' | 'error' | 'info'

interface ToastEntry {
  id:   number
  tone: Tone
  text: string
}

interface ToastStore {
  items: ToastEntry[]
  push:  (tone: Tone, text: string) => void
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

const STYLE: Record<Tone, string> = {
  success: 'bg-white text-zinc-900 border-emerald-200',
  error:   'bg-white text-zinc-900 border-red-200',
  info:    'bg-white text-zinc-900 border-zinc-200',
}

const ICON_COLOR: Record<Tone, string> = {
  success: 'text-emerald-500',
  error:   'text-red-500',
  info:    'text-zinc-400',
}

export function Toaster() {
  const items = useToastStore((s) => s.items)
  const dismiss = useToastStore((s) => s.dismiss)
  // ensure root subscribes
  useEffect(() => {}, [items])

  return (
    <div className="fixed bottom-4 right-4 z-[60] flex flex-col gap-2 pointer-events-none">
      {items.map((t) => {
        const Icon = ICON[t.tone]
        return (
          <div
            key={t.id}
            className={cn(
              'pointer-events-auto flex items-center gap-2.5 min-w-72 max-w-md pl-3 pr-2 py-2.5',
              'rounded-lg border shadow-[0_8px_24px_rgb(0,0,0,0.08)] text-sm',
              'animate-[fadeIn_120ms_ease-out]',
              STYLE[t.tone],
            )}
          >
            <Icon size={15} className={cn('shrink-0', ICON_COLOR[t.tone])} />
            <span className="flex-1 truncate">{t.text}</span>
            <button
              onClick={() => dismiss(t.id)}
              className="size-6 rounded-md flex items-center justify-center text-zinc-400 hover:text-zinc-700 hover:bg-zinc-100 cursor-pointer"
            >
              <X size={12} />
            </button>
          </div>
        )
      })}
    </div>
  )
}
