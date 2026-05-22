// WorkHero — cover (left) + identity + actions (right).
//
// Pure context consumer. Identity from `useWorkIdentity()`, read target
// from `useWorkChapters()`, mutations from `useWorkActions()`. Only the
// `onUpload` callback comes from the parent (it owns modal state).

import { useEffect, useRef, useState } from 'react'
import { Pencil, ChevronDown, BookmarkPlus, Upload } from 'lucide-react'

import { Cover } from '@shared/ui/Cover'
import { Button } from '@shared/ui/Button'
import { Tag } from '@shared/ui/primitives'
import { BottomSheet } from '@shared/ui/BottomSheet'
import { cn } from '@shared/lib/cn'
import { useIsDesktop } from '@shared/lib/useMediaQuery'

import { useWorkIdentity } from '../contexts/WorkIdentityContext'
import { useWorkChapters } from '../contexts/WorkChaptersContext'
import { useWorkActions } from '../contexts/WorkActionsContext'
import { CoverSheet } from '../CoverSheet'
import type { Work } from '../data/types'


interface Props {
  /** Open the upload-chapter dialog. Owned by the page composition. */
  onUpload: () => void
  /** Navigate to the read target. Owned by the route since it knows
   *  about TanStack Router. */
  onRead?: () => void
}
export function WorkHero({ onUpload, onRead }: Props) {
  const { work, primaryDetail } = useWorkIdentity()
  const { readTarget } = useWorkChapters()
  const { rename, setCover, resetCover } = useWorkActions()
  const navigate = useNavigateRead()

  const handleRead = onRead ?? navigate

  const cover  = work.cover_url ?? primaryDetail?.cover ?? null
  const author = primaryDetail?.author ?? null
  const [coverOpen, setCoverOpen] = useState(false)

  const readLabel = readTarget
    ? readTarget.isResume
      ? `Đọc tiếp ch.${readTarget.ref}`
      : 'Bắt đầu đọc'
    : null

  return (
    <section className="px-4 sm:px-6 pt-4 pb-3">
      <div className="flex items-start gap-4">
        {/* Cover */}
        <button
          type="button"
          onClick={() => setCoverOpen(true)}
          aria-label="Sửa ảnh bìa"
          className={cn(
            'group/cover relative w-[88px] sm:w-28 shrink-0',
            'aspect-[2/3] rounded-md overflow-hidden shadow-md cursor-pointer',
            'focus-visible:outline-2 focus-visible:outline-accent focus-visible:outline-offset-2',
          )}
        >
          <Cover src={cover} title={work.title} className="w-full h-full" />
          <span className={cn(
            'absolute inset-0 bg-bg/0 group-hover/cover:bg-bg/35',
            'transition-colors',
          )} />
        </button>

        {/* Identity + actions */}
        <div className="flex-1 min-w-0 space-y-2">
          <TitleBlock title={work.title} onRename={rename} />

          {author && (
            <p className="text-sm text-text-muted truncate -mt-1">{author}</p>
          )}

          <MetaRow
            nsfw={work.nsfw}
            status={primaryDetail?.status ?? null}
          />

          {/* Row 1: core actions */}
          <div className="flex items-stretch gap-2">
            {readLabel && (
              <Button variant="primary" size="md" onClick={handleRead}>
                {readLabel}
              </Button>
            )}
            <BookmarkButton work={work} />
          </div>

          {/* Row 2: utility actions */}
          <div className="flex items-center gap-2">
            <Button variant="ghost" size="sm" onClick={onUpload}>
              <Upload size={14} />
              Thêm chương
            </Button>
          </div>
        </div>
      </div>
      <CoverSheet
        open={coverOpen}
        onClose={() => setCoverOpen(false)}
        currentUrl={cover}
        canRestore={!!work.cover_overridden}
        title={work.title}
        onCommitUrl={setCover}
        onRestore={resetCover}
      />
    </section>
  )
}

// Hook to navigate to the work's read target without prop-drilling
// the navigate handler through the route.
import { useNavigate } from '@tanstack/react-router'

function useNavigateRead() {
  const { work } = useWorkIdentity()
  const { readTarget } = useWorkChapters()
  const nav = useNavigate()
  return () => {
    if (!readTarget) return
    nav({
      to:     '/r/$workId/$numberNorm',
      params: { workId: work.id, numberNorm: readTarget.ref },
    })
  }
}


// ── Title block ────────────────────────────────────────────────


function TitleBlock({ title, onRename }: { title: string; onRename: (t: string) => void }) {
  const isDesktop = useIsDesktop()
  const [editing, setEditing] = useState(false)

  const commit = (t: string) => { onRename(t); setEditing(false) }
  const cancel = () => setEditing(false)

  if (isDesktop && editing) {
    return <TitleInline initial={title} onCommit={commit} onCancel={cancel} />
  }

  if (isDesktop) {
    return (
      <div className="group/title flex items-start gap-1.5 min-w-0">
        <h1 className="text-lg font-semibold text-text leading-snug line-clamp-3 flex-1">
          {title}
        </h1>
        <button
          type="button"
          onClick={() => setEditing(true)}
          aria-label="Sửa tên"
          className={cn(
            'mt-0.5 shrink-0 size-7 rounded-sm flex items-center justify-center',
            'text-text-subtle hover:text-text hover:bg-hover',
            'opacity-0 group-hover/title:opacity-100 focus-visible:opacity-100',
            'transition-opacity cursor-pointer',
          )}
        >
          <Pencil size={13} />
        </button>
      </div>
    )
  }

  // Mobile: tap title → bottom sheet
  return (
    <>
      <button
        type="button"
        onClick={() => setEditing(true)}
        className="text-left min-w-0 w-full cursor-pointer"
      >
        <h1 className="text-lg font-semibold text-text leading-snug line-clamp-3">
          {title}
        </h1>
      </button>
      <RenameSheet
        open={editing}
        initial={title}
        onCommit={commit}
        onCancel={cancel}
      />
    </>
  )
}


function TitleInline({ initial, onCommit, onCancel }: {
  initial:  string
  onCommit: (t: string) => void
  onCancel: () => void
}) {
  const [v, setV] = useState(initial)
  const ref = useRef<HTMLInputElement>(null)

  const commit = () => {
    const t = v.trim()
    if (!t || t === initial) onCancel()
    else onCommit(t)
  }

  return (
    <input
      ref={ref}
      autoFocus
      type="text"
      value={v}
      onChange={e => setV(e.target.value)}
      onBlur={commit}
      onKeyDown={e => {
        if (e.key === 'Enter')  { e.preventDefault(); commit() }
        if (e.key === 'Escape') { e.preventDefault(); onCancel() }
      }}
      maxLength={300}
      className={cn(
        'w-full text-lg font-semibold text-text leading-snug',
        'bg-transparent border-b border-border-soft focus:border-accent focus:outline-none',
        'pb-0.5',
      )}
    />
  )
}


function RenameSheet({ open, initial, onCommit, onCancel }: {
  open:     boolean
  initial:  string
  onCommit: (t: string) => void
  onCancel: () => void
}) {
  const [v, setV] = useState(initial)
  const ref = useRef<HTMLInputElement>(null)

  const commit = () => {
    const t = v.trim()
    if (!t || t === initial) onCancel()
    else onCommit(t)
  }

  return (
    <BottomSheet
      open={open}
      onClose={onCancel}
      title="Sửa tên truyện"
      footer={
        <div className="flex justify-end gap-2">
          <Button variant="ghost" size="md" onClick={onCancel}>Huỷ</Button>
          <Button variant="primary" size="md" onClick={commit}
            disabled={!v.trim() || v.trim() === initial}>Lưu</Button>
        </div>
      }
    >
      <div className="px-4 py-3">
        <input
          ref={ref}
          autoFocus
          type="text"
          value={v}
          onChange={e => setV(e.target.value)}
          onKeyDown={e => { if (e.key === 'Enter') commit() }}
          maxLength={300}
          placeholder="Tên truyện"
          className="w-full h-10 px-3 rounded-sm text-base bg-surface-2 text-text placeholder:text-text-subtle focus:outline-hidden focus:ring-1 focus:ring-accent/40"
        />
        <p className="mt-2 text-xs text-text-subtle">
          Tên hiển thị trên thư viện. Nguồn gốc không bị đổi.
        </p>
      </div>
    </BottomSheet>
  )
}


// ── Meta + stats ──────────────────────────────────────────────


function MetaRow({ nsfw, status }: { nsfw: boolean; status: string | null }) {
  if (!status && !nsfw) return null
  const label = normalizeStatus(status)
  return (
    <div className="flex items-center gap-2 flex-wrap">
      {label && <Tag tone="neutral" size="sm">{label}</Tag>}
      {nsfw   && <Tag tone="error"   size="sm">18+</Tag>}
    </div>
  )
}

/** Normalize raw source status strings → Vietnamese display labels.
 *  Sources emit English or romanized strings; we map the common
 *  values and fall back to the raw string for unknown values. */
function normalizeStatus(s: string | null): string | null {
  if (!s) return null
  const key = s.toLowerCase().trim()
  const MAP: Record<string, string> = {
    ongoing:           'Đang tiến hành',
    'on going':        'Đang tiến hành',
    'on-going':        'Đang tiến hành',
    releasing:         'Đang tiến hành',
    publishing:        'Đang tiến hành',
    'đang cập nhật':   'Đang tiến hành',
    'đang tiến hành':  'Đang tiến hành',
    completed:         'Hoàn thành',
    complete:          'Hoàn thành',
    finished:          'Hoàn thành',
    'hoàn thành':      'Hoàn thành',
    'hoàn tất':        'Hoàn thành',
    hiatus:            'Tạm ngưng',
    'on hiatus':       'Tạm ngưng',
    'tạm ngưng':       'Tạm ngưng',
    cancelled:         'Đã huỷ',
    canceled:          'Đã huỷ',
    dropped:           'Đã huỷ',
  }
  return MAP[key] ?? s
}


// ── Secondary actions ─────────────────────────────────────────


const STATUS_OPTIONS = [
  { code: 'reading' as const, label: 'Đang đọc' },
  { code: 'plan'    as const, label: 'Để dành'  },
  { code: 'done'    as const, label: 'Đã đọc'   },
]

const STATUS_LABELS: Record<string, string> = {
  reading: 'Đang đọc',
  plan:    'Để dành',
  done:    'Đã đọc',
}


function BookmarkButton({ work }: { work: Work }) {
  const [open, setOpen] = useState(false)
  const ref  = useRef<HTMLDivElement>(null)
  const { addLibrary, removeLibrary, setStatus } = useWorkActions()

  const pinned = work.in_library
  const label  = pinned
    ? (STATUS_LABELS[work.library_status ?? ''] ?? 'Đang đọc')
    : 'Thư viện'

  useEffect(() => {
    if (!open) return
    const onDoc = (e: MouseEvent) => {
      if (ref.current && !ref.current.contains(e.target as Node)) setOpen(false)
    }
    const onEsc = (e: KeyboardEvent) => { if (e.key === 'Escape') setOpen(false) }
    document.addEventListener('mousedown', onDoc)
    document.addEventListener('keydown', onEsc)
    return () => {
      document.removeEventListener('mousedown', onDoc)
      document.removeEventListener('keydown', onEsc)
    }
  }, [open])

  if (!pinned) {
    return (
      <Button
        variant="secondary"
        size="md"
        onClick={addLibrary}
      >
        <BookmarkPlus size={14} />
        Thư viện
      </Button>
    )
  }

  return (
    <div ref={ref} className="relative">
      <Button
        variant="secondary"
        size="md"
        onClick={() => setOpen(v => !v)}
        aria-haspopup="menu"
        aria-expanded={open}
        className="bg-accent/15 text-accent-text hover:bg-accent/25 border-0"
      >
        {label}
        <ChevronDown size={12} className={cn('transition-transform', open && 'rotate-180')} />
      </Button>

      {open && (
        <div
          role="menu"
          className={cn(
            'absolute left-0 top-full mt-1 z-30 min-w-[160px]',
            'bg-surface rounded-md border border-border-soft',
            'shadow-[0_8px_24px_rgb(0,0,0,0.35)] py-1',
          )}
        >
          {STATUS_OPTIONS.map(opt => (
            <button
              key={opt.code}
              type="button"
              role="menuitemradio"
              aria-checked={work.library_status === opt.code}
              onClick={() => { setStatus(opt.code); setOpen(false) }}
              className={cn(
                'w-full flex items-center justify-between px-3 py-1.5 text-sm text-left',
                'transition-colors cursor-pointer hover:bg-hover',
                work.library_status === opt.code
                  ? 'text-text font-medium'
                  : 'text-text-muted hover:text-text',
              )}
            >
              {opt.label}
              {work.library_status === opt.code && (
                <span className="text-accent text-xs">✓</span>
              )}
            </button>
          ))}
          <div className="my-1 border-t border-border-soft" />
          <button
            type="button"
            role="menuitem"
            onClick={() => { removeLibrary(); setOpen(false) }}
            className="w-full px-3 py-1.5 text-sm text-left text-error-text hover:bg-hover transition-colors cursor-pointer"
          >
            Xoá khỏi thư viện
          </button>
        </div>
      )}
    </div>
  )
}
