// CoverSheet — edit the work's cover image. Tap-the-object pattern:
// the user taps the hero cover and this sheet opens.
//
// Two actions at MVP:
//
//   • Dán URL — paste a direct image URL. Validated as http/https,
//     previewed before save so a 404 URL doesn't silently destroy
//     the current cover.
//
//   • Khôi phục theo nguồn — drop the user override and re-sync to
//     whatever the primary source serves. Visible only when an
//     override is currently set.
//
// Upload from device is intentionally deferred — it needs blob
// storage + MIME validation + size limits. The sheet shape leaves
// a slot for it later without restructuring this surface.
//
// On commit the caller decides whether `cover_overridden` flips
// (true for URL paste, false for restore). This component only
// emits the URL value.

import { useEffect, useRef, useState } from 'react'
import { Link as LinkIcon, RotateCcw } from 'lucide-react'

import { BottomSheet } from '@shared/ui/BottomSheet'
import { Button } from '@shared/ui/Button'
import { Cover } from '@shared/ui/Cover'
import { cn } from '@shared/lib/cn'
import { useSourceFetch } from '@features/browse/SourceFetchProvider'


interface Props {
  open:    boolean
  onClose: () => void
  /** Current cover (already-resolved URL or null). Drives the
   *  initial preview frame. */
  currentUrl: string | null
  /** Show the "Khôi phục theo nguồn" affordance only when an
   *  override is currently in effect. */
  canRestore: boolean
  /** Title — used for the fallback monogram in the preview. */
  title: string
  /** User chose a custom URL. Caller persists + flips
   *  `cover_overridden=true`. Empty/invalid URLs never reach here. */
  onCommitUrl: (url: string) => void
  /** User chose to drop the override. Caller clears
   *  `cover_overridden` and lets attach-time auto-sync take over. */
  onRestore: () => void
}


export function CoverSheet({
  open, onClose, currentUrl, canRestore, title,
  onCommitUrl, onRestore,
}: Props) {
  // Edit mode flips the sheet from the "actions" view to the URL
  // input. Keeping them in the same sheet avoids a nested sheet
  // stack while still letting actions stay a single tap deep.
  const [mode, setMode] = useState<'actions' | 'url'>('actions')

  useEffect(() => {
    if (open) setMode('actions')
  }, [open])

  return (
    <BottomSheet
      open={open}
      onClose={onClose}
      title={mode === 'url' ? 'Ảnh bìa · Dán URL' : 'Ảnh bìa'}
    >
      {mode === 'actions' ? (
        <ActionsView
          currentUrl={currentUrl}
          title={title}
          canRestore={canRestore}
          onPickUrl={() => setMode('url')}
          onRestore={() => { onRestore(); onClose() }}
        />
      ) : (
        <UrlView
          initialUrl={currentUrl}
          title={title}
          onBack={() => setMode('actions')}
          onCommit={(u) => { onCommitUrl(u); onClose() }}
        />
      )}
    </BottomSheet>
  )
}


// ── Actions view ───────────────────────────────────────────────


function ActionsView({
  currentUrl, title, canRestore, onPickUrl, onRestore,
}: {
  currentUrl: string | null
  title:      string
  canRestore: boolean
  onPickUrl:  () => void
  onRestore:  () => void
}) {
  return (
    <div className="px-4 py-3 space-y-4">
      <div className="flex justify-center">
        <div className="w-32 aspect-[2/3] rounded-md overflow-hidden">
          <Cover src={currentUrl} title={title} className="w-full h-full" />
        </div>
      </div>

      <ul className="space-y-1">
        <ActionRow
          icon={<LinkIcon size={16} />}
          label="Dán URL ảnh"
          onClick={onPickUrl}
        />
        {canRestore && (
          <ActionRow
            icon={<RotateCcw size={16} />}
            label="Khôi phục theo nguồn"
            onClick={onRestore}
          />
        )}
      </ul>
    </div>
  )
}


function ActionRow({
  icon, label, onClick,
}: {
  icon:    React.ReactNode
  label:   string
  onClick: () => void
}) {
  return (
    <li>
      <button
        type="button"
        onClick={onClick}
        className={cn(
          'w-full flex items-center gap-3 h-11 px-3 rounded-sm text-left',
          'text-sm text-text hover:bg-hover',
          'transition-colors cursor-pointer',
        )}
      >
        <span className="size-5 inline-flex items-center justify-center text-text-subtle shrink-0">
          {icon}
        </span>
        <span className="flex-1">{label}</span>
      </button>
    </li>
  )
}


// ── URL view ───────────────────────────────────────────────────


/** URL paste view — input + live preview. The preview probes the
 *  URL through the same CDN proxy `<Cover>` uses, so what the user
 *  sees here is exactly what the rest of the app will render. */
function UrlView({
  initialUrl, title, onBack, onCommit,
}: {
  initialUrl: string | null
  title:      string
  onBack:     () => void
  onCommit:   (url: string) => void
}) {
  const [value,    setValue]    = useState(initialUrl ?? '')
  const [previewOk, setPreviewOk] = useState<boolean | null>(null)
  const ref = useRef<HTMLInputElement>(null)

  useEffect(() => {
    const t = window.setTimeout(() => ref.current?.focus(), 100)
    return () => window.clearTimeout(t)
  }, [])

  const trimmed = value.trim()
  const valid   = isHttpUrl(trimmed)
  const dirty   = trimmed !== (initialUrl ?? '')
  const canSave = valid && dirty && previewOk !== false

  function commit() {
    if (!canSave) return
    onCommit(trimmed)
  }

  return (
    <>
      <div className="px-4 py-3 space-y-3">
        <input
          ref={ref}
          type="url"
          inputMode="url"
          value={value}
          onChange={(e) => { setValue(e.target.value); setPreviewOk(null) }}
          onKeyDown={(e) => { if (e.key === 'Enter') commit() }}
          placeholder="https://…"
          className={cn(
            'w-full h-10 px-3 rounded-sm text-base',
            'bg-surface-2 text-text placeholder:text-text-subtle',
            'focus:outline-hidden focus:ring-1 focus:ring-accent/40',
          )}
        />

        <div className="flex justify-center">
          <div className="w-32 aspect-[2/3] rounded-md overflow-hidden bg-surface-2">
            {valid ? (
              <PreviewImg
                key={trimmed}
                url={trimmed}
                title={title}
                onLoad={() => setPreviewOk(true)}
                onError={() => setPreviewOk(false)}
              />
            ) : (
              <Cover src={null} title={title} className="w-full h-full" />
            )}
          </div>
        </div>

        <p className="text-xs text-text-subtle text-center">
          {!trimmed
            ? 'Dán link ảnh trực tiếp (jpg/png/webp).'
            : !valid
              ? 'URL không hợp lệ. Dùng đường dẫn http/https.'
              : previewOk === false
                ? 'Không tải được ảnh. Kiểm tra lại link.'
                : previewOk === true
                  ? 'Ảnh sẵn sàng.'
                  : 'Đang kiểm tra…'}
        </p>
      </div>

      <div className="px-4 pb-3 flex items-center justify-end gap-2 border-t border-border-soft pt-3">
        <Button variant="ghost" size="md" onClick={onBack}>
          Quay lại
        </Button>
        <Button variant="primary" size="md" onClick={commit} disabled={!canSave}>
          Lưu
        </Button>
      </div>
    </>
  )
}


function PreviewImg({
  url, title, onLoad, onError,
}: {
  url:     string
  title:   string
  onLoad:  () => void
  onError: () => void
}) {
  const { toBrowserUrl: proxify } = useSourceFetch()
  // Route through proxify so cross-origin headers / referer policy
  // matches what the live Cover will fetch.
  return (
    <img
      src={proxify(url)}
      alt={title}
      className="w-full h-full object-cover"
      onLoad={onLoad}
      onError={onError}
    />
  )
}


function isHttpUrl(s: string): boolean {
  if (!s) return false
  try {
    const u = new URL(s)
    return u.protocol === 'http:' || u.protocol === 'https:'
  } catch {
    return false
  }
}
