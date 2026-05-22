// WorkDescription — description + stats summary.
//
// Reads from WorkIdentityContext (description from primary detail,
// updated_at from work) and WorkChaptersContext (totalChapters).
// No props — pure context consumer.

import { useState } from 'react'

import { useWorkIdentity } from '../contexts/WorkIdentityContext'
import { useWorkChapters } from '../contexts/WorkChaptersContext'
import { timeAgo } from '@shared/lib/time'
import { cn } from '@shared/lib/cn'


export function WorkDescription() {
  const { work, primaryDetail } = useWorkIdentity()
  const { totalChapters } = useWorkChapters()

  const [open, setOpen] = useState(false)
  const stripped = (primaryDetail?.description ?? '').replace(/<[^>]+>/g, '').trim()
  const hasDesc  = stripped.length > 0
  const overflows = stripped.length > 240

  const statParts: string[] = []
  if (totalChapters > 0) statParts.push(`${totalChapters} chương`)
  if (work.updated_at)   statParts.push(timeAgo(work.updated_at))

  if (!hasDesc && !statParts.length) return null

  return (
    <section className="px-4 sm:px-6 py-2 space-y-1.5">
      {hasDesc && (
        <>
          <p
            className={cn(
              'text-sm text-text-muted leading-relaxed whitespace-pre-line',
              !open && 'line-clamp-3',
            )}
          >
            {stripped}
          </p>
          {overflows && (
            <button
              type="button"
              onClick={() => setOpen(o => !o)}
              className="text-xs text-text-subtle hover:text-text transition-colors cursor-pointer"
            >
              {open ? 'Thu gọn' : 'Xem thêm'}
            </button>
          )}
        </>
      )}

      {statParts.length > 0 && (
        <p className="text-xs text-text-subtle tabular-nums">
          {statParts.join(' · ')}
        </p>
      )}
    </section>
  )
}
