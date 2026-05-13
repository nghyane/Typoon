// SourceChipRail — sibling-material picker for the Work page.
//
// One chip per material attached to the Work. Clicking a chip swaps
// the active source (URL state `?src=`), which in turn drives:
//   • which manifest the chapter spine is fetched from,
//   • which raw URLs the chapter rows expose.
//
// Materials whose source plugin isn't installed render in a muted
// state with a tooltip pointing to Settings → Sources. We don't hide
// them — visibility is the only honest way for users to learn that
// the Work has more raws available.

import { useMemo } from 'react'

import type { ApiMaterial } from '@shared/api/api'
import { useSources } from '@features/browse/sources'
import { languageName } from '@shared/lib/lang'
import { cn } from '@shared/lib/cn'


interface Props {
  materials:        ApiMaterial[]
  activeMaterialId: number | null
  onSelect:         (materialId: number) => void
}


export function SourceChipRail({
  materials, activeMaterialId, onSelect,
}: Props) {
  const installed = useSources((s) => s.sources)

  const rows = useMemo(() => materials.map((m) => {
    const src = m.source ? (installed[m.source] ?? null) : null
    const isInstalled = src !== null
    const isExtUpload = m.source === null
    const lang0 = m.languages?.[0] ?? ''
    return {
      id:        m.id,
      label:     src?.manifest.name ?? (isExtUpload ? 'Tải lên' : m.title),
      lang:      lang0 ? languageName(lang0) : '',
      installed: isInstalled || isExtUpload,
      tooltip:   isInstalled || isExtUpload
        ? null
        : `Plugin "${m.source}" chưa cài. Vào Cài đặt → Nguồn để cài.`,
    }
  }), [materials, installed])

  if (rows.length <= 1) return null  // single source — no chip rail

  return (
    <div className="flex items-center gap-1 flex-wrap">
      <span className="text-xs text-text-subtle mr-1">Nguồn:</span>
      {rows.map((r) => {
        const active = r.id === activeMaterialId
        return (
          <button
            key={r.id}
            type="button"
            disabled={!r.installed}
            title={r.tooltip ?? undefined}
            onClick={() => r.installed && onSelect(r.id)}
            className={cn(
              'h-7 px-2.5 rounded-sm text-xs whitespace-nowrap transition-colors',
              'inline-flex items-center gap-1.5',
              active
                ? 'bg-accent/15 text-accent border border-accent/30'
                : r.installed
                ? 'text-text-muted bg-surface-2 hover:bg-hover hover:text-text border border-transparent cursor-pointer'
                : 'text-text-subtle bg-surface-2/40 border border-transparent cursor-not-allowed',
            )}
          >
            <span>{r.label}</span>
            {r.lang && (
              <span className="text-[10px] text-text-subtle">{r.lang}</span>
            )}
          </button>
        )
      })}
    </div>
  )
}
