import { Link } from '@tanstack/react-router'
import { ArrowRight, Plus } from 'lucide-react'
import { cn } from '@shared/lib/cn'
import { Tag } from '@shared/ui/primitives'
import { MULTI_LANG } from '@shared/lib/lang'
import { SourceIcon } from './SourceIcon'
import type { InstalledSource } from '../manifest/types'

// =============================================================================
// SourceCard — entry tile in /browse hub.
//
// Discord-server-row pattern: 48px monogram tile on the left, content
// stack on the right (name + meta + blurb), arrow affordance far
// right. Same shape mobile and desktop — only padding/text size shift.
//
// Why this and not the rich-collage variant from earlier iterations:
//   • Real brand assets fetched from favicons never scale cleanly to
//     tile size. We tried.
//   • A 4-cover collage shifts content as covers load and confuses
//     two sources sharing a hot title.
//   • Discord users already parse this row shape from servers,
//     channels, friend rows. Zero learning curve.
//   • Content panel (name / meta / blurb) gets the visual weight,
//     icon is a landmark — matches a hub where the user is choosing
//     between sources by what's in them, not by who they are.
// =============================================================================

const ORIGIN_LABEL: Record<InstalledSource['origin'], string> = {
  bundled: 'Chính thức',
  repo:    'Từ repo',
  file:    'Tự cài',
}

const SOURCE_BLURBS: Record<string, string> = {
  community: 'Truyện do thành viên Hội Mê Truyện chia sẻ',
  happymh:  'Truyện Trung — phổ biến & mới cập nhật hằng ngày',
  otruyen:  'Truyện tiếng Việt do cộng đồng dịch',
  mangadex: 'Cộng đồng quốc tế đa ngôn ngữ',
}

interface Props {
  source: InstalledSource
}

export function SourceCard({ source }: Props) {
  const { manifest, origin, author } = source
  const langs   = manifest.languages
  const isMulti = langs.length > 1 || langs.includes(MULTI_LANG)
  const langTag = isMulti ? 'MULTI' : (langs[0] ?? '').toUpperCase()
  const blurb   = SOURCE_BLURBS[manifest.id]

  return (
    <Link
      to="/browse/$source"
      params={{ source: manifest.id }}
      className={cn(
        'group flex items-center gap-3 rounded-md bg-surface',
        'hover:bg-surface-2 transition-colors cursor-pointer',
        'px-3 py-2 sm:px-3.5',
      )}
    >
      <SourceIcon
        manifest={manifest}
        className="size-10 shrink-0"
        fontSize="text-xs"
      />

      <div className="flex-1 min-w-0">
        <div className="flex items-center gap-2 mb-0.5">
          <span className="text-sm font-semibold text-text truncate group-hover:text-accent-text transition-colors">
            {manifest.name}
          </span>
          {langTag && (
            <Tag tone={isMulti ? 'info' : 'outline'} size="sm" uppercase>
              {langTag}
            </Tag>
          )}
          {manifest.nsfw && (
            <Tag tone="error" size="sm" uppercase>NSFW</Tag>
          )}
        </div>
        <p className="text-[11px] text-text-subtle truncate">
          {manifest.host}
          <span aria-hidden className="mx-1.5 opacity-50">·</span>
          {ORIGIN_LABEL[origin]}
          {blurb && (
            <>
              <span aria-hidden className="mx-1.5 opacity-50">·</span>
              <span className="hidden sm:inline">{blurb}</span>
            </>
          )}
          {author && (
            <>
              <span aria-hidden className="mx-1.5 opacity-50">·</span>
              {author}
            </>
          )}
        </p>
      </div>

      <ArrowRight
        size={14}
        className="text-text-subtle group-hover:text-accent-text group-hover:translate-x-0.5 transition-all shrink-0"
      />
    </Link>
  )
}

// =============================================================================
// InstallSourceCard — last entry, same row shape with a quiet "+" icon.
// =============================================================================

export function InstallSourceCard() {
  return (
    <Link
      to="/settings"
      search={{ section: 'sources' } as never}
      className={cn(
        'group flex items-center gap-3 rounded-md',
        'bg-transparent ring-1 ring-inset ring-border-soft',
        'hover:ring-text-subtle hover:bg-surface transition-colors cursor-pointer',
        'px-3 py-2 sm:px-3.5',
      )}
    >
      <span
        className={cn(
          'size-10 rounded-md grid place-items-center shrink-0',
          'bg-surface-2 text-text-subtle',
          'ring-1 ring-inset ring-border-soft',
          'group-hover:text-text transition-colors',
        )}
      >
        <Plus size={16} />
      </span>
      <div className="flex-1 min-w-0">
        <p className="text-sm font-semibold text-text">Cài thêm nguồn</p>
        <p className="text-[11px] text-text-subtle truncate">
          Từ kho cộng đồng hoặc dán URL repo
        </p>
      </div>
      <ArrowRight
        size={14}
        className="text-text-subtle group-hover:text-accent-text group-hover:translate-x-0.5 transition-all shrink-0"
      />
    </Link>
  )
}
