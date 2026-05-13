import { Wand2 } from 'lucide-react'
import { cn } from '@shared/lib/cn'

// =============================================================================
// ManualCreateRow — CTA that escapes the result list into manual mode.
//
// Two presentations, same component:
//   • hits === 0  primary path forward; framed as 'Tạo "X" thủ công?'
//                  styled with accent tint.
//   • hits > 0    secondary affordance; framed as 'Không thấy "X"?'
//                  rendered in the neutral surface palette.
//
// `seed` (the trimmed query) is forwarded so the manual form can
// pre-fill the title — saves the user re-typing.
// =============================================================================

export function ManualCreateRow({
  query, hits, onManualCreate,
}: {
  query:          string
  hits:           number
  onManualCreate: (seed: string) => void
}) {
  const seed = query.trim()
  return (
    <button
      type="button"
      onClick={() => onManualCreate(seed)}
      className={cn(
        'w-full flex items-center gap-2.5 px-3 py-2.5 rounded-md',
        'text-left transition-colors cursor-pointer',
        hits === 0
          ? 'bg-accent/10 border border-accent/20 hover:bg-accent/15'
          : 'bg-surface-2 hover:bg-hover',
      )}
    >
      <span className={cn(
        'inline-flex items-center justify-center size-8 rounded-sm shrink-0',
        hits === 0 ? 'bg-accent text-accent-fg' : 'bg-surface text-text-muted',
      )}>
        <Wand2 size={14} />
      </span>
      <div className="flex-1 min-w-0">
        <p className="text-sm text-text">
          {hits === 0
            ? `Không tìm thấy. Tạo "${seed}" thủ công?`
            : `Không thấy "${seed}"? Tạo thủ công`
          }
        </p>
        <p className="text-xs text-text-subtle mt-0.5">
          Manga không thuộc nguồn nào · tải chương từ file zip/cbz
        </p>
      </div>
    </button>
  )
}
