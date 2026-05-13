import { Cover } from '@shared/ui/Cover'
import { Tag, input, label } from '@shared/ui/primitives'
import { proxify } from '@features/browse/proxy'
import type { LibraryStatus } from '@shared/api/api'
import type { Picked } from './types'

// =============================================================================
// PickedDetail — confirm form shown after the user picks a manga from
// search results or URL paste.
//
// Top: read-only preview card (cover + title + source tag + author).
// Below: reading-preference form (reading language + status).
//
// "Auto-translate" was removed when we made translation strictly
// client-pixel-driven: the server can't fetch raw pages without a
// browser-side manifest runtime, so a background poll-and-translate
// job has nothing to consume. Spawn is per-chapter, per-click only.
//
// 'Đổi' link returns the user to the search results without losing
// query or sidebar selection.
// =============================================================================

interface Props {
  picked:        Picked
  targetLang:    string
  setTargetLang: (s: string) => void
  status:        LibraryStatus
  setStatus:     (s: LibraryStatus) => void
  onChangePick:  () => void
}

export function PickedDetail({
  picked, targetLang, setTargetLang,
  status, setStatus,
  onChangePick,
}: Props) {
  return (
    <div className="space-y-4">
      <div className="flex items-start gap-3 p-3 rounded-md bg-surface-2">
        <Cover
          src={picked.cover ? proxify(picked.cover) : null}
          title={picked.title}
          className="w-14 aspect-[2/3] rounded-xs shrink-0"
          fontSize="text-xs"
        />
        <div className="flex-1 min-w-0">
          <p className="text-sm font-medium text-text line-clamp-2">{picked.title}</p>
          <div className="mt-1.5 flex flex-wrap items-center gap-2">
            <Tag tone="info" size="sm">{picked.source.manifest.name}</Tag>
            {picked.nsfw && <Tag tone="error" size="sm" uppercase>NSFW</Tag>}
          </div>
          {picked.author && (
            <p className="text-xs text-text-subtle mt-1.5 truncate">
              {picked.author}{picked.status && ` · ${picked.status}`}
            </p>
          )}
        </div>
        <button
          type="button"
          onClick={onChangePick}
          className="text-xs text-text-subtle hover:text-text shrink-0 underline-offset-2 hover:underline cursor-pointer"
        >
          Đổi
        </button>
      </div>

      <div className="grid grid-cols-[7rem_1fr] gap-x-3 gap-y-3">
        <div>
          <label className={label}>Đọc bằng</label>
          <select
            value={targetLang}
            onChange={(e) => setTargetLang(e.target.value)}
            className={input}
          >
            <option value="vi">Tiếng Việt</option>
            <option value="en">English</option>
            <option value="ja">日本語</option>
            <option value="ko">한국어</option>
            <option value="zh">中文</option>
          </select>
        </div>
        <div>
          <label className={label}>Tình trạng</label>
          <select
            value={status}
            onChange={(e) => setStatus(e.target.value as LibraryStatus)}
            className={input}
          >
            <option value="reading">Đang đọc</option>
            <option value="plan">Để dành</option>
            <option value="done">Đã đọc xong</option>
          </select>
        </div>
      </div>
    </div>
  )
}
