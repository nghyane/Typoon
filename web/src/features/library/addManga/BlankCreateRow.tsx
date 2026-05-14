import { Wand2, Loader2 } from 'lucide-react'

import { cn } from '@shared/lib/cn'

import type { ImportToLibrary } from './useImportToLibrary'

// Fallback CTA when no source matches the query: one click creates
// an upload-origin material seeded with the query as title. Cover +
// chapters happen on the Work hub. Accent palette when results are
// empty (primary path); neutral when results exist (nudge).

export function BlankCreateRow({
  query, hits, importer,
}: {
  query:    string
  hits:     number
  importer: ImportToLibrary
}) {
  const seed = query.trim()
  return (
    <button
      type="button"
      onClick={() => importer.importBlank(seed)}
      disabled={importer.isPending}
      className={cn(
        'w-full flex items-center gap-2.5 px-3 py-2.5 rounded-md',
        'text-left transition-colors cursor-pointer',
        'disabled:cursor-wait disabled:opacity-60',
        hits === 0
          ? 'bg-accent/10 border border-accent/20 hover:bg-accent/15'
          : 'bg-surface-2 hover:bg-hover',
      )}
    >
      <span className={cn(
        'inline-flex items-center justify-center size-8 rounded-sm shrink-0',
        hits === 0 ? 'bg-accent text-accent-fg' : 'bg-surface text-text-muted',
      )}>
        {importer.isPending
          ? <Loader2 size={14} className="animate-spin" />
          : <Wand2 size={14} />
        }
      </span>
      <div className="flex-1 min-w-0">
        <p className="text-sm text-text">
          {hits === 0
            ? (seed ? `Không tìm thấy. Tạo "${seed}" trống?` : 'Tạo manga trống')
            : (seed ? `Không thấy "${seed}"? Tạo trống`     : 'Tạo manga trống')
          }
        </p>
        <p className="text-xs text-text-subtle mt-0.5">
          Vào trang truyện để tải chương từ file zip/cbz hoặc ảnh.
        </p>
      </div>
    </button>
  )
}
