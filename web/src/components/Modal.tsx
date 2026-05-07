import { useEffect, type ReactNode } from 'react'
import { X } from 'lucide-react'
import { cn } from '../lib/cn'

interface Props {
  open:     boolean
  onClose:  () => void
  title:    string
  size?:    'sm' | 'md' | 'lg'
  children: ReactNode
  footer?:  ReactNode
}

const SIZE = {
  sm: 'max-w-md',
  md: 'max-w-2xl',
  lg: 'max-w-4xl',
}

export function Modal({ open, onClose, title, size = 'md', children, footer }: Props) {
  useEffect(() => {
    if (!open) return
    const onKey = (e: KeyboardEvent) => e.key === 'Escape' && onClose()
    document.addEventListener('keydown', onKey)
    return () => document.removeEventListener('keydown', onKey)
  }, [open, onClose])

  if (!open) return null

  return (
    <div
      className="fixed inset-0 z-50 flex items-center justify-center p-4 bg-zinc-900/30 backdrop-blur-[2px]"
      onMouseDown={(e) => e.target === e.currentTarget && onClose()}
    >
      <div
        className={cn(
          'w-full bg-white rounded-2xl shadow-[0_24px_64px_rgb(0,0,0,0.18)] border border-zinc-200',
          'flex flex-col max-h-[88vh] overflow-hidden',
          SIZE[size],
        )}
      >
        <header className="flex items-center justify-between px-5 h-[52px] border-b border-zinc-100 shrink-0">
          <h2 className="text-base font-semibold text-zinc-900 tracking-tight truncate">{title}</h2>
          <button
            onClick={onClose}
            className="size-8 rounded-md flex items-center justify-center text-zinc-400 hover:text-zinc-700 hover:bg-zinc-100 transition-colors cursor-pointer"
          >
            <X size={15} />
          </button>
        </header>

        <div className="flex-1 overflow-auto">
          {children}
        </div>

        {footer && (
          <footer className="flex items-center justify-end gap-2 px-5 py-3 border-t border-zinc-100 bg-zinc-50/40 shrink-0">
            {footer}
          </footer>
        )}
      </div>
    </div>
  )
}
