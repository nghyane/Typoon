// WorkCard — grid item shared by Library and Explore search results.

import { Link } from '@tanstack/react-router'
import { cn } from '@shared/lib/cn'
import { Cover } from '@shared/ui/Cover'
import { Tag } from '@shared/ui/primitives'

export interface WorkCardData {
  id:        string
  title:     string
  cover_url: string | null
  source?:   string | null
  /** Free-form status text, e.g. last-read chapter. */
  badge?:    string | null
  nsfw?:     boolean
}

interface Props {
  work:       WorkCardData
  className?: string
}

export function WorkCard({ work, className }: Props) {
  return (
    <Link
      to="/w/$workId"
      params={{ workId: work.id }}
      search={{ tab: undefined }}
      className={cn(
        'group flex flex-col gap-2 rounded-sm overflow-hidden',
        'focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-accent',
        className,
      )}
    >
      <div className="relative aspect-[3/4] rounded-sm overflow-hidden bg-surface-2">
        <Cover
          src={work.cover_url}
          title={work.title}
          className="absolute inset-0 transition-transform group-hover:scale-[1.02]"
        />
        {work.nsfw && (
          <Tag tone="error" size="sm" className="absolute top-1.5 right-1.5">18+</Tag>
        )}
        {work.badge && (
          <div className="absolute bottom-0 inset-x-0 px-2 py-1 text-xs text-white bg-bg/90">
            {work.badge}
          </div>
        )}
      </div>
      <div className="px-0.5 space-y-0.5">
        <div className="text-xs font-medium text-text line-clamp-2 leading-tight">
          {work.title}
        </div>
        {work.source && (
          <div className="text-xs uppercase tracking-wider text-text-subtle">
            {work.source}
          </div>
        )}
      </div>
    </Link>
  )
}
