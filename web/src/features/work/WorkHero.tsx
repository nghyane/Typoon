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
//   │  Nguồn:  [● MangaDex] [Bato.to] [+ 2]                       │
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

import { useState } from 'react'
import { BookOpen, Share2 } from 'lucide-react'

import { cn } from '@shared/lib/cn'
import { Cover } from '@shared/ui/Cover'
import { Button } from '@shared/ui/Button'
import type {
  ApiMaterial, ApiRecentRead, ApiWork, ApiWorkViewerEntry,
} from '@shared/api/api'

import { SourceChipRail } from './SourceChipRail'
import { TargetLangPicker } from './TargetLangPicker'
import { StatusPicker } from './StatusPicker'
import { ReferrersStrip } from './ReferrersStrip'
import { resolveWorkTitle } from './title'


interface Props {
  workId:           number
  /** Work payload — carries `cross_refs` for the Referrers strip. */
  work:             ApiWork | null
  activeMaterial:   ApiMaterial | null
  materials:        ApiMaterial[]
  resumeFrom:       ApiRecentRead | null
  viewerEntry:      ApiWorkViewerEntry | null
  latestChapterNum: string | null
  totalChapters?:   number
  onSelectSource:   (materialId: number) => void
  onShare:          () => void
  onResume:         () => void
}


export function WorkHero({
  workId, work, activeMaterial, materials, resumeFrom, viewerEntry,
  latestChapterNum, totalChapters,
  onSelectSource, onShare, onResume,
}: Props) {
  const m = activeMaterial
  // Canonical title comes from the WORK's siblings (deterministic
  // across viewers), not from `activeMaterial` (per-viewer choice).
  // Cover stays on the active material so swapping sources still
  // shows the user the cover they expect.
  const { title, titleNative } = resolveWorkTitle(materials)

  return (
    <div className="px-4 sm:px-6 pt-6 pb-4">
      {/* Top row — cover + title block. */}
      <div className="flex gap-4 sm:gap-6">
        <div className="w-24 sm:w-40 shrink-0 aspect-[2/3] rounded-md overflow-hidden shadow-md">
          <Cover
            src={m?.cover_url ?? null}
            title={title}
            version={m?.updated_at}
            className="w-full h-full"
          />
        </div>

        <div className="flex-1 min-w-0 flex flex-col gap-2 sm:gap-2.5">
          <TitleBlock title={title} native={titleNative} />

          <MetaStrip material={m} />

          <ActionBar
            resumeFrom={resumeFrom}
            viewerEntry={viewerEntry}
            workId={workId}
            material={m}
            onResume={onResume}
            onShare={onShare}
          />
        </div>
      </div>

      {/* Source picker — outside the title column so it can span the
          full width on mobile (cover + label width is too narrow). */}
      <div className="mt-4 flex items-center gap-2 flex-wrap">
        <span className="text-xs text-text-subtle shrink-0">Nguồn:</span>
        <SourceChipRail
          materials={materials}
          activeMaterialId={activeMaterial?.id ?? null}
          onSelect={onSelectSource}
        />
      </div>

      {/* Referrers — links out to the external identity services
          (Anilist, MAL, MangaDex, …) the Work's cross_refs resolve to.
          Auto-hides when cross_refs is empty; the auto-enrich hook
          will populate this strip silently on next mount. */}
      <div className="mt-2">
        <ReferrersStrip crossRefs={work?.cross_refs ?? null} />
      </div>

      {/* Stats strip — small, subtle metadata that wasn't worth a
          chip but the user might still want at a glance. */}
      <StatsStrip
        latestChapterNum={latestChapterNum}
        totalChapters={totalChapters}
        updatedAt={m?.updated_at ?? null}
      />

      {/* Description — full text, collapsed by default. */}
      {m?.description && (
        <Description text={stripHtml(m.description)} />
      )}
    </div>
  )
}


// ── Sub-blocks ─────────────────────────────────────────────────


function TitleBlock({
  title, native,
}: {
  title:  string
  native: string | null
}) {
  // Only render the native title when it's actually different from
  // the romanized one (some sources duplicate the same string into
  // both fields).
  const showNative = native && native.trim() && native.trim() !== title.trim()

  return (
    <div className="min-w-0">
      <h1 className="text-lg sm:text-2xl font-semibold text-text leading-tight line-clamp-2">
        {title}
      </h1>
      {showNative && (
        <p className="mt-1 text-xs sm:text-sm text-text-subtle italic truncate">
          {native}
        </p>
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
  resumeFrom, viewerEntry, workId, material, onResume, onShare,
}: {
  resumeFrom:  ApiRecentRead | null
  viewerEntry: ApiWorkViewerEntry | null
  workId:      number
  material:    ApiMaterial | null
  onResume:    () => void
  onShare:     () => void
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
          and status-change (entry exists). `dropped` = bỏ theo dõi
          — no separate remove button needed. */}
      <StatusPicker
        workId={workId}
        entryId={viewerEntry?.entry_id ?? null}
        status={viewerEntry?.status ?? null}
        material={material ? {
          id:        material.id,
          title:     material.title,
          cover_url: material.cover_url,
        } : null}
      />

      {viewerEntry && (
        <TargetLangPicker
          entryId={viewerEntry.entry_id}
          workId={workId}
          targetLang={viewerEntry.target_lang}
        />
      )}

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
