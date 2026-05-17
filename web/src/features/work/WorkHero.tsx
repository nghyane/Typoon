// WorkHero — top of the Work page.
//
// Pro-design layout (flat dark, no blur, hierarchy-led):
//
//   ┌─────────────────────────────────────────────────────────────┐
//   │  ┌──────┐                                                   │
//   │  │      │  Title (h1, can wrap to 2 lines)                 │
//   │  │COVER │  title_native  (small, italic, muted)             │
//   │  │      │                                                   │
//   │  │      │  [Completed] [NSFW] · Author · JA → VI            │
//   │  │      │                                                   │
//   │  │      │  ┌Đọc tiếp ch.X┐ [Đang đọc ▾] [♡] [↗]            │
//   │  └──────┘                                                   │
//   │                                                             │
//   │  Mới nhất ch.64.6 · 92 chương · Cập nhật 2 giờ trước        │
//   │                                                             │
//   │  After the discussions with the temple master…              │
//   │  [▾ Mở rộng]                                                │
//   └─────────────────────────────────────────────────────────────┘
//
// No blur background — flat solid matches every other surface in
// the app. Status appears twice on purpose: a muted chip for the
// MANGA's publication state (from upstream), and a clickable picker
// for the USER's reading state (saved on the library entry).
//
// One Work may aggregate N sibling materials across sources. The
// hero picks a SINGLE primary material (`pickPrimaryMaterial`) for
// the metadata strip / status picker / bookmark / description; the
// title and cover come from their own resolvers, biased toward the
// viewer's reading language. The per-source picker rail is gone:
// chapter rows expose source info inline via `ChapterRow`.

import { useRef, useState } from 'react'
import { useMutation, useQueryClient } from '@tanstack/react-query'
import {
  BookOpen, Camera, ChevronDown, Globe, Share2, Upload,
} from 'lucide-react'

import { cn } from '@shared/lib/cn'
import { Cover } from '@shared/ui/Cover'
import { Button } from '@shared/ui/Button'
import { toast } from '@shared/ui/Toaster'
import { api } from '@shared/api/api'
import { qk } from '@shared/api/keys'
import { useSession } from '@features/auth/session'
import type {
  ApiMaterial, ApiRecentRead, ApiWorkViewerEntry,
} from '@shared/api/api'

import { TargetLangPicker } from './TargetLangPicker'
import { StatusPicker } from './StatusPicker'
import {
  collectAltTitles, pickPrimaryMaterial, resolveWorkCover,
  resolveWorkTitle,
} from './title'


interface Props {
  workId:           number
  materials:        ApiMaterial[]
  resumeFrom:       ApiRecentRead | null
  viewerEntry:      ApiWorkViewerEntry | null
  /** Resolved reading-lang from `useWorkData` (entry override → user
   *  default → fallback). Drives title / cover / primary-material
   *  resolution AND the TargetLangPicker label so both stay in sync
   *  whether the viewer has bookmarked the Work yet or not. */
  targetLang:       string
  latestChapterNum: string | null
  totalChapters?:   number
  onShare:          () => void
  onResume:         () => void
  onUpload:         () => void
}


export function WorkHero({
  workId, materials, resumeFrom, viewerEntry, targetLang,
  latestChapterNum, totalChapters,
  onShare, onResume, onUpload,
}: Props) {
  const { title, titleNative } = resolveWorkTitle(materials, targetLang)
  const cover                  = resolveWorkCover(materials, targetLang)
  const primary                = pickPrimaryMaterial(materials, targetLang)
  const altTitles              = collectAltTitles(materials, title)
  const isUserCreated          = materials.every((m) => m.origin !== 'source')

  return (
    <div className="px-4 sm:px-6 pt-6 pb-4">
      {/* Top row — cover + title block.
          `items-start` keeps the cover at its intrinsic
          aspect-ratio height; without it the flex row stretches
          every child to the tallest sibling, and an expanded
          alt-titles disclosure pulls the cover taller than its
          2:3 ratio (poster becomes warped). */}
      <div className="flex items-start gap-4 sm:gap-6">
        <CoverSlot
          material={primary}
          coverUrl={cover.coverUrl}
          title={title}
        />

        <div className="flex-1 min-w-0 flex flex-col gap-2 sm:gap-2.5">
          <TitleBlock
            title={title}
            native={titleNative}
            alts={altTitles}
          />

          <MetaStrip material={primary} />

          <ActionBar
            resumeFrom={resumeFrom}
            viewerEntry={viewerEntry}
            workId={workId}
            material={primary}
            targetLang={targetLang}
            isUserCreated={isUserCreated}
            onResume={onResume}
            onShare={onShare}
            onUpload={onUpload}
          />
        </div>
      </div>

      {/* Stats strip — small, subtle metadata that wasn't worth a
          chip but the user might still want at a glance. */}
      <StatsStrip
        latestChapterNum={latestChapterNum}
        totalChapters={totalChapters}
        updatedAt={primary?.updated_at ?? null}
      />

      {/* Description — full text, collapsed by default. */}
      {primary?.description && (
        <Description text={stripHtml(primary.description)} />
      )}
    </div>
  )
}


// ── Sub-blocks ─────────────────────────────────────────────────


// Match the server-side cap so we reject early instead of streaming a
// huge file up just to get a 413 back.
const COVER_MAX_BYTES        = 2 * 1024 * 1024
const COVER_ACCEPTED_MIMES   = ['image/jpeg', 'image/png', 'image/webp']


/** Cover slot with an inline "change cover" affordance for local
 *  (ext / upload) materials the viewer owns. Source-backed materials
 *  render the cover read-only — their state mirrors the manifest
 *  snapshot so a user-uploaded cover would just get overwritten on
 *  the next enrich. The button overlays the cover bottom-right on
 *  hover (desktop) and stays visible (mobile) so it's discoverable
 *  without obstructing the image. */
function CoverSlot({
  material, coverUrl, title,
}: {
  material: ApiMaterial | null
  coverUrl: string | null
  title:    string
}) {
  const qc = useQueryClient()
  const { user } = useSession()
  const fileInputRef = useRef<HTMLInputElement>(null)

  const canEdit =
       !!material
    && material.origin !== 'source'
    && !!user
    && material.imported_by === user.id

  const upload = useMutation({
    mutationFn: (f: File) => api.uploadCover(material!.id, f),
    onSuccess: () => {
      // Refresh both the active work payload (cover surfaces here)
      // and the library entry payload (cover surfaces on the card).
      // Cheap at beta scale — one entry per open tab.
      void qc.invalidateQueries({ queryKey: qk.work.all() })
      void qc.invalidateQueries({ queryKey: qk.library.all() })
      toast.success('Đã cập nhật ảnh bìa.')
    },
    onError: (e: Error) => toast.error(e.message),
  })

  function handlePick(file: File | null) {
    if (!file) return
    if (!COVER_ACCEPTED_MIMES.includes(file.type)) {
      toast.error('Chỉ chấp nhận ảnh JPG, PNG hoặc WebP.')
      return
    }
    if (file.size > COVER_MAX_BYTES) {
      toast.error(`Ảnh quá lớn (tối đa ${COVER_MAX_BYTES / 1024 / 1024} MB).`)
      return
    }
    upload.mutate(file)
  }

  return (
    <div className={cn(
      'relative w-24 sm:w-40 shrink-0 aspect-[2/3] rounded-md overflow-hidden shadow-md',
      'group',
    )}>
      <Cover
        src={coverUrl}
        title={title}
        version={material?.updated_at}
        className="w-full h-full"
      />
      {canEdit && (
        <>
          <input
            ref={fileInputRef}
            type="file"
            accept={COVER_ACCEPTED_MIMES.join(',')}
            className="sr-only"
            onChange={(e) => {
              handlePick(e.target.files?.[0] ?? null)
              // Reset value so picking the same file twice re-fires
              // onChange — common pattern after a failed upload.
              e.target.value = ''
            }}
          />
          <button
            type="button"
            onClick={() => fileInputRef.current?.click()}
            disabled={upload.isPending}
            title={upload.isPending ? 'Đang tải lên…' : 'Đổi ảnh bìa'}
            className={cn(
              'absolute bottom-1 right-1 inline-flex items-center justify-center',
              'h-7 w-7 rounded-full',
              'bg-bg/80 text-text hover:bg-bg backdrop-blur-sm',
              'border border-border-soft/60 shadow-sm',
              'transition-opacity cursor-pointer',
              // Always visible on touch; fade in on hover for desktop
              // so the image isn't permanently covered.
              'opacity-90 sm:opacity-0 sm:group-hover:opacity-100',
              'focus-visible:opacity-100 focus-visible:outline-none',
              'focus-visible:ring-1 focus-visible:ring-accent/50',
              upload.isPending && 'opacity-100 cursor-wait',
            )}
          >
            <Camera size={14} />
          </button>
        </>
      )}
    </div>
  )
}


function TitleBlock({
  title, native, alts,
}: {
  title:  string
  native: string | null
  alts:   string[]
}) {
  // Native title is included in `alts` already (collectAltTitles
  // walks every material's title_native). Render it inline in the
  // disclosure list — no separate italic subtitle. The `native`
  // prop is kept for callers that still want the raw value, but
  // not used here.
  void native

  return (
    <div className="min-w-0">
      <h1 className="text-lg sm:text-2xl font-semibold text-text leading-tight line-clamp-2">
        {title}
      </h1>
      {alts.length > 0 && <AltTitleStrip alts={alts} />}
    </div>
  )
}


/** Disclosure row: the trigger line ALWAYS shows the joined alts
 *  truncated to one line — that's the "button" that doesn't grow
 *  the cover row's height. Expanded just appends a separate list
 *  below; the trigger stays one line either way. */
function AltTitleStrip({ alts }: { alts: string[] }) {
  const [open, setOpen] = useState(false)
  return (
    <div className="mt-1.5 text-text-subtle">
      <button
        type="button"
        onClick={() => setOpen((v) => !v)}
        aria-expanded={open}
        className={cn(
          'flex items-center gap-2 w-full min-w-0 text-left',
          'text-xs sm:text-sm hover:text-text cursor-pointer',
          'transition-colors',
        )}
      >
        <Globe size={14} className="shrink-0" />
        <span className="flex-1 min-w-0 truncate">
          {alts.join(' / ')}
        </span>
        <ChevronDown
          size={14}
          className={cn(
            'shrink-0 transition-transform',
            open && 'rotate-180',
          )}
        />
      </button>
      {open && (
        <ul className="mt-2 ml-6 space-y-1.5 text-xs sm:text-sm text-text-muted">
          {alts.map((t) => (
            <li key={t} className="break-words">{t}</li>
          ))}
        </ul>
      )}
    </div>
  )
}


function MetaStrip({ material }: { material: ApiMaterial | null }) {
  const pubStatus = material?.status
  const author    = material?.author
  const nsfw      = material?.nsfw

  return (
    <div className="flex items-center gap-2 text-xs text-text-muted flex-wrap">
      {pubStatus && (
        <span className={cn(
          'inline-flex items-center h-5 px-1.5 rounded-sm',
          'bg-surface-2 text-text-muted capitalize',
        )}>
          {pubStatus}
        </span>
      )}
      {nsfw && (
        <span className={cn(
          'inline-flex items-center h-5 px-1.5 rounded-sm',
          'bg-rose-500/10 text-rose-400 font-medium',
        )}>
          18+
        </span>
      )}
      {author && (
        <span className="truncate max-w-[14rem]">{author}</span>
      )}
    </div>
  )
}


function ActionBar({
  resumeFrom, viewerEntry, workId, material, targetLang,
  isUserCreated,
  onResume, onShare, onUpload,
}: {
  resumeFrom:    ApiRecentRead | null
  viewerEntry:   ApiWorkViewerEntry | null
  workId:        number
  material:      ApiMaterial | null
  targetLang:    string
  isUserCreated: boolean
  onResume:      () => void
  onShare:       () => void
  onUpload:      () => void
}) {
  return (
    <div className="flex items-center gap-2 mt-1 flex-wrap">
      {resumeFrom && (
        <Button
          variant="primary"
          size="md"
          onClick={onResume}
          className="inline-flex items-center gap-1.5"
        >
          <BookOpen size={14} />
          <span>Đọc tiếp ch. {resumeFrom.chapter_number || '?'}</span>
        </Button>
      )}

      {/* Status picker covers both bookmark-create (no entry yet)
          and status-change (entry exists). Removing the entry is the
          explicit "xoá khỏi thư viện" action inside the picker — no
          separate button needed. */}
      <StatusPicker
        workId={workId}
        entryId={viewerEntry?.entry_id ?? null}
        status={viewerEntry?.status ?? null}
        materialId={material?.id ?? null}
        isUserCreated={isUserCreated}
      />

      {viewerEntry && (
        <TargetLangPicker
          entryId={viewerEntry.entry_id}
          workId={workId}
          targetLang={targetLang}
        />
      )}

      {/* Upload one's own chapter into this Work. Lazy material
          create happens server-side on first upload-init; this
          button always works regardless of whether the viewer has
          previously uploaded here. */}
      <Button
        variant="ghost"
        size="md"
        onClick={onUpload}
        className="inline-flex items-center gap-1.5"
        title="Tải lên chương"
      >
        <Upload size={14} />
        <span className="hidden sm:inline">Tải chương</span>
      </Button>

      <Button
        variant="ghost"
        size="md"
        onClick={onShare}
        className="inline-flex items-center gap-1.5"
        title="Chia sẻ"
      >
        <Share2 size={14} />
        <span className="hidden sm:inline">Chia sẻ</span>
      </Button>
    </div>
  )
}


function StatsStrip({
  latestChapterNum, totalChapters, updatedAt,
}: {
  latestChapterNum: string | null
  totalChapters?:   number
  updatedAt:        string | null
}) {
  const parts: string[] = []
  if (latestChapterNum) parts.push(`Mới nhất ch.${latestChapterNum}`)
  if (totalChapters)    parts.push(`${totalChapters} chương`)
  if (updatedAt)        parts.push(`Cập nhật ${relTime(updatedAt)}`)
  if (parts.length === 0) return null
  return (
    <div className="mt-2 text-xs text-text-subtle tabular flex flex-wrap items-center gap-x-1.5">
      {parts.map((p, i) => (
        <span key={i} className="inline-flex items-center gap-1.5">
          {i > 0 && <span className="text-border-soft">·</span>}
          <span>{p}</span>
        </span>
      ))}
    </div>
  )
}


function Description({ text }: { text: string }) {
  const [open, setOpen] = useState(false)
  // Only show the toggle when the content actually overflows.
  // Cheap heuristic: any string longer than ~280 chars (roughly 3
  // lines at the rendered width on desktop) is likely to clip.
  const overflows = text.length > 280
  return (
    <div className="mt-4">
      <p
        className={cn(
          'text-sm text-text-muted leading-relaxed whitespace-pre-line',
          !open && 'line-clamp-3 sm:line-clamp-4',
        )}
      >
        {text}
      </p>
      {overflows && (
        <button
          type="button"
          onClick={() => setOpen((o) => !o)}
          className={cn(
            'mt-1 inline-flex items-center gap-1 text-xs',
            'text-text-subtle hover:text-text cursor-pointer transition-colors',
          )}
        >
          {open ? 'Thu gọn' : 'Mở rộng'}
        </button>
      )}
    </div>
  )
}


// ── Helpers ────────────────────────────────────────────────────


function stripHtml(s: string): string {
  // Manifests sometimes hand back HTML in the description. Render
  // plain text — full markup is overkill for the hero blurb.
  return s.replace(/<[^>]+>/g, '').trim()
}


/** Short relative time for the stats strip — "2 giờ", "5 ngày". */
function relTime(iso: string): string {
  const t = new Date(iso.includes('T') ? iso : iso.replace(' ', 'T') + 'Z').getTime()
  const diff = Date.now() - t
  if (!Number.isFinite(diff)) return ''
  const min = Math.round(diff / 60_000)
  if (min < 60)        return `${min} phút trước`
  const hr = Math.round(min / 60)
  if (hr  < 24)        return `${hr} giờ trước`
  const day = Math.round(hr / 24)
  if (day < 30)        return `${day} ngày trước`
  const mo = Math.round(day / 30)
  if (mo  < 12)        return `${mo} tháng trước`
  return `${Math.round(mo / 12)} năm trước`
}


// ── Continue-reading bar (mobile, below hero) ──────────────────


export function ContinueReadingBar({
  resumeFrom, onResume,
}: {
  resumeFrom: ApiRecentRead | null
  onResume:   () => void
}) {
  if (!resumeFrom) return null
  return (
    <button
      type="button"
      onClick={onResume}
      className={cn(
        'w-full flex items-center gap-3 px-3 py-2 rounded-md',
        'bg-accent/10 hover:bg-accent/15 border border-accent/20',
        'text-left transition-colors cursor-pointer',
      )}
    >
      <BookOpen size={16} className="text-accent shrink-0" />
      <div className="flex-1 min-w-0">
        <div className="text-sm text-text">
          Tiếp tục ch. {resumeFrom.chapter_number || '?'}
        </div>
        {resumeFrom.chapter_label && (
          <div className="text-xs text-text-subtle truncate">
            {resumeFrom.chapter_label}
          </div>
        )}
      </div>
    </button>
  )
}
