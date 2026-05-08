import { BookOpen, Settings, Sparkles } from 'lucide-react'
import { cn } from '@shared/lib/cn'
import type { Tab } from './filter'

const TABS: { key: Tab; label: string; icon: typeof BookOpen }[] = [
  { key: 'chapters', label: 'Chương',     icon: BookOpen },
  { key: 'glossary', label: 'Thuật ngữ',  icon: Sparkles },
  { key: 'settings', label: 'Cài đặt',    icon: Settings },
]

interface Props {
  value:    Tab
  onChange: (t: Tab) => void
}

export function TabBar({ value, onChange }: Props) {
  return (
    <div className="flex items-center px-6 border-b border-border-soft">
      {TABS.map(({ key, label, icon: Icon }) => (
        <button
          key={key}
          onClick={() => onChange(key)}
          className={cn(
            'inline-flex items-center gap-2 h-10 px-3 text-[13px] font-medium cursor-pointer transition-colors',
            'border-b-2 -mb-px',
            value === key
              ? 'text-text border-accent'
              : 'text-text-muted border-transparent hover:text-text',
          )}
        >
          <Icon size={14} />
          {label}
        </button>
      ))}
    </div>
  )
}
