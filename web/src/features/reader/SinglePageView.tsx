// One-page view. Single <PageImage>; the user navigates with prev /
// next. Surrounding pages stay cached in the parent source so
// flipping back is instant.

import { Button } from '@shared/ui/Button'
import { PageImage } from './PageImage'
import type { ReaderPage } from './types'

interface Props {
  pages:    ReaderPage[]
  urls?:    ReadonlyMap<number, string>
  page:     number
  onChange: (p: number) => void
}

export function SinglePageView({ pages, urls, page, onChange }: Props) {
  const total = pages.length
  if (total === 0) return null
  const safe = Math.min(Math.max(0, page), total - 1)
  const p    = pages[safe]!

  return (
    <div className="max-w-3xl mx-auto py-4">
      <PageImage page={p} src={urls?.get(p.index) ?? p.url} inWindow />
      <div className="flex items-center justify-between mt-4 px-2">
        <Button onClick={() => onChange(safe - 1)} disabled={safe <= 0}>
          ← Trang trước
        </Button>
        <span className="text-sm text-text-muted tabular">
          <span className="text-text font-medium">{safe + 1}</span>
          <span className="opacity-50 mx-1">/</span>
          {total}
        </span>
        <Button onClick={() => onChange(safe + 1)} disabled={safe >= total - 1}>
          Trang sau →
        </Button>
      </div>
    </div>
  )
}
