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
// Below: reading-preference form (target lang / status / auto-translate).
//
// 'Đổi' link returns the user to the search results without losing
// query or sidebar selection.
// =============================================================================

interface Props {
  picked:        Picked
  targetLang:    string
  setTargetLang: (s: string) => void
  autoTr:        boolean
  setAutoTr:     (b: boolean) => void
  status:        LibraryStatus
  setStatus:     (s: LibraryStatus) => void
  onChangePick:  () => void
}

export function PickedDetail({
  picked, targetLang, setTargetLang,
  autoTr, setAutoTr, status, setStatus,
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
          <div className="mt-1.5 flex flex-wrap items-center gap-1.5">
            <Tag tone="info" size="sm">{picked.source.manifest.name}</Tag>
            {picked.languages[0] && (
              <Tag tone="neutral" size="sm" uppercase>
                {picked.languages.slice(0, 3).join('/')}
              </Tag>
            )}
            {picked.nsfw && <Tag tone="error" size="sm" uppercase>NSFW</Tag>}
          </div>
          {picked.author && (
            <p className="text-[11px] text-text-subtle mt-1.5 truncate">
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
            <option value="plan">Kế hoạch</option>
            <option value="on_hold">Tạm dừng</option>
            <option value="done">Đã xong</option>
          </select>
        </div>

        <div className="col-span-2 flex items-center gap-2.5">
          <input
            id="auto-translate"
            type="checkbox"
            checked={autoTr}
            onChange={(e) => setAutoTr(e.target.checked)}
            className="size-4 cursor-pointer accent-accent"
          />
          <label
            htmlFor="auto-translate"
            className="text-sm text-text-muted cursor-pointer select-none"
          >
            Tự động dịch chương mới sang {targetLang.toUpperCase()}
            <span className="text-[11px] text-text-subtle ml-1">
              · tốn quota dịch cho mỗi chương mới
            </span>
          </label>
        </div>
      </div>
    </div>
  )
}
