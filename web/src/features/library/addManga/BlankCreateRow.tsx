// BlankCreateRow — fallback CTA at the bottom of the result list.
//
// One click creates an empty Work seeded with the query as title and
// pins it to the library. Cover + sources can be added on the Work
// hub. Accent palette when results are empty (primary path); neutral
// otherwise (nudge).

import { Wand2 } from 'lucide-react'

import { cn } from '@shared/lib/cn'
import { Spinner } from '@shared/ui/primitives'

import type { ImportToLibrary } from './useImportToLibrary'

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
        'w-full flex items-center gap-3 px-3 py-2.5 rounded-md',
        'text-left transition-colors cursor-pointer',
        'disabled:cursor-wait disabled:opacity-60',
        hits === 0
          ? 'bg-accent-bg hover:brightness-110'
          : 'bg-surface-2 hover:bg-hover',
      )}
    >
      <span className={cn(
        'inline-flex items-center justify-center size-8 rounded-sm shrink-0',
        hits === 0 ? 'bg-accent text-accent-fg' : 'bg-surface text-text-muted',
      )}>
        {importer.isPending
          ? <Spinner size={14} />
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
        <p className="text-xs text-text-subtle mt-1">
          Vào trang truyện để liên kết nguồn đọc sau.
        </p>
      </div>
    </button>
  )
}
