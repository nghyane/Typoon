import { Cover, coverUrl } from '@shared/ui/Cover'
import { Tag } from '@shared/ui/primitives'
import { FollowButton } from '@features/library/views/LibraryCard'
import type { ApiLibraryEntry, ApiMaterial, LibraryStatus } from '@shared/api/api'

// =============================================================================
// HubHero — title page header.
//
// Layout (Anilist/MyAnimeList density without the cover poster eating
// half the viewport): 96px cover left, title + meta badges + action
// cluster right. Description hangs below as a collapsible <details>
// so the chapter list keeps top-of-fold.
// =============================================================================

const STATUS_LABEL: Record<LibraryStatus, string> = {
  reading: 'Đang đọc',
  plan:    'Kế hoạch',
  on_hold: 'Tạm dừng',
  done:    'Đã xong',
  dropped: 'Đã bỏ',
}

interface Props {
  entry:    ApiLibraryEntry
  material: ApiMaterial
}

export function HubHero({ entry, material }: Props) {
  const summary = entry.translation_summary
  return (
    <header className="px-4 sm:px-6 pt-4 sm:pt-6 pb-4 flex items-start gap-3 sm:gap-4">
      <Cover
        src={coverUrl(material.cover_url, material.updated_at)}
        title={material.title}
        fontSize="text-2xl"
        className="w-24 aspect-[2/3] rounded-md shrink-0"
      />
      <div className="flex-1 min-w-0">
        <div className="flex flex-col sm:flex-row sm:items-start sm:justify-between gap-2 sm:gap-3">
          <div className="min-w-0">
            <h1 className="text-lg sm:text-2xl font-semibold tracking-tight text-text line-clamp-2">
              {material.title}
            </h1>
            <MetaRow entry={entry} material={material} />
          </div>

          <div className="flex items-center gap-2 shrink-0 self-start">
            <FollowButton
              entryId={entry.id}
              materialId={material.id}
              title={material.title}
              cover={material.cover_url}
              targetLang={entry.target_lang}
              status={entry.status}
            />
          </div>
        </div>

        {summary && (summary.running > 0 || summary.error > 0 || summary.pending > 0) && (
          <ActivityRow
            running={summary.running}
            error={summary.error}
            pending={summary.pending}
          />
        )}
      </div>
    </header>
  )
}


function MetaRow({
  entry, material,
}: {
  entry: ApiLibraryEntry; material: ApiMaterial
}) {
  return (
    <div className="flex items-center gap-2 mt-2 flex-wrap text-xs text-text-subtle">
      <span className="inline-flex items-center gap-1 h-[22px] px-2 rounded-xs bg-surface-2 text-[11px] font-semibold uppercase tracking-wider text-text-muted">
        {material.languages[0]?.toUpperCase() ?? '?'}
        <span className="text-text-subtle">→</span>
        {(entry.target_lang ?? '?').toUpperCase()}
      </span>
      <Tag tone="neutral" size="sm">
        {STATUS_LABEL[entry.status]}
      </Tag>
      {material.author && <span>{material.author}</span>}
      {material.status && <span>· {material.status}</span>}
      {material.nsfw && (
        <span className="text-[11px] uppercase font-semibold px-1.5 py-0.5 rounded-xs bg-error/15 text-error-text">
          NSFW
        </span>
      )}
    </div>
  )
}


function ActivityRow({
  running, error, pending,
}: {
  running: number; error: number; pending: number
}) {
  return (
    <div className="mt-2 flex items-center gap-1.5">
      {running > 0 && (
        <Tag tone="info" size="sm">
          {running} đang dịch
        </Tag>
      )}
      {error > 0 && (
        <Tag tone="error" size="sm">
          {error} lỗi
        </Tag>
      )}
      {pending > 0 && (
        <Tag tone="warning" size="sm">
          {pending} chờ
        </Tag>
      )}
    </div>
  )
}
