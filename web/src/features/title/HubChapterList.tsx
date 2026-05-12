import { useState } from 'react'
import { BookOpen, Sparkles, Loader2, AlertCircle, CheckCircle2 } from 'lucide-react'
import { cn } from '@shared/lib/cn'
import type { ApiChapter, ApiChapterTranslation } from '@shared/api/api'

// =============================================================================
// HubChapterList — chapter rows for the title page.
//
// Each row carries:
//   • number + label
//   • inline translation badges (per-lang done/running/error)
//   • action: 'Đọc' when the viewer's target_lang has a done
//     translation, otherwise 'Dịch' (spawn flow lands in slice 16).
//
// Sort: chapters descending by position so latest is on top — matches
// every manga reader's expectation (MangaDex/MangaPlus default).
// =============================================================================

interface Props {
  chapters:   ApiChapter[]
  targetLang: string | null
}

export function HubChapterList({ chapters, targetLang }: Props) {
  const [filter, setFilter] = useState<'all' | 'translated' | 'raw'>('all')

  if (chapters.length === 0) {
    return (
      <div className="rounded-md bg-surface-2 border border-dashed border-border-soft px-4 py-8 text-center">
        <p className="text-sm text-text-muted">Chưa có chương nào trong cơ sở dữ liệu</p>
        <p className="text-[11px] text-text-subtle mt-1">
          Mở reader hoặc bấm Dịch ở danh sách nguồn để vào hàng đợi pipeline.
        </p>
      </div>
    )
  }

  const sorted = [...chapters].sort((a, b) => b.position - a.position)
  const filtered = sorted.filter((c) => {
    if (filter === 'all') return true
    const done = c.translations.some((t) => t.state === 'done')
    return filter === 'translated' ? done : !done
  })

  return (
    <section className="space-y-2">
      <div className="flex items-center justify-between gap-2 px-0.5">
        <h2 className="text-[12px] uppercase tracking-wider text-text-subtle">
          {chapters.length} chương
        </h2>
        <FilterRow value={filter} onChange={setFilter} />
      </div>

      <ul className="rounded-md bg-surface-2 divide-y divide-border-soft overflow-hidden">
        {filtered.map((c) => (
          <ChapterRow key={c.id} chapter={c} targetLang={targetLang} />
        ))}
      </ul>
    </section>
  )
}


function FilterRow({
  value, onChange,
}: {
  value: 'all' | 'translated' | 'raw'
  onChange: (v: 'all' | 'translated' | 'raw') => void
}) {
  const items: Array<{ id: typeof value; label: string }> = [
    { id: 'all',        label: 'Tất cả'   },
    { id: 'translated', label: 'Đã dịch' },
    { id: 'raw',        label: 'Raw'      },
  ]
  return (
    <div className="flex items-center gap-0.5">
      {items.map((it) => (
        <button
          key={it.id}
          type="button"
          onClick={() => onChange(it.id)}
          className={cn(
            'h-7 px-2.5 rounded-sm text-[12px] transition-colors cursor-pointer',
            value === it.id
              ? 'bg-surface-2 text-text font-medium'
              : 'text-text-muted hover:bg-hover hover:text-text',
          )}
        >
          {it.label}
        </button>
      ))}
    </div>
  )
}


function ChapterRow({
  chapter, targetLang,
}: {
  chapter: ApiChapter; targetLang: string | null
}) {
  const readable = pickReadable(chapter.translations, targetLang)
  return (
    <li className="flex items-center gap-3 px-3 py-2 hover:bg-hover transition-colors">
      <div className="flex-1 min-w-0">
        <div className="flex items-center gap-1.5">
          <span className="text-[12px] font-medium text-text-muted tabular shrink-0">
            Ch.{chapter.number}
          </span>
          <p className="text-sm text-text truncate">
            {chapter.label ?? '—'}
          </p>
          <TranslationBadges translations={chapter.translations} />
        </div>
        {readable?.creator_name && (
          <p className="text-[11px] text-text-subtle truncate mt-0.5">
            Đọc bản của @{readable.creator_name}
          </p>
        )}
      </div>

      <Action chapter={chapter} readable={readable} targetLang={targetLang} />
    </li>
  )
}


function Action({
  readable, targetLang,
}: {
  chapter:    ApiChapter
  readable:   ApiChapterTranslation | null
  targetLang: string | null
}) {
  if (readable) {
    return (
      <button
        type="button"
        // TODO(slice 14): wire reader route /title/$entryId/ch/$chapterId?tx=
        disabled
        className="inline-flex items-center gap-1 h-7 px-2.5 rounded-sm text-[11px] font-medium bg-success/15 text-success-text cursor-not-allowed opacity-80"
      >
        <BookOpen size={11} />
        Đọc {readable.target_lang.toUpperCase()}
      </button>
    )
  }
  return (
    <button
      type="button"
      // TODO(slice 15): spawn translate inline
      disabled
      className="inline-flex items-center gap-1 h-7 px-2.5 rounded-sm text-[11px] font-medium bg-accent text-accent-fg cursor-not-allowed opacity-80"
      title={targetLang ? `Dịch sang ${targetLang.toUpperCase()}` : 'Chọn target_lang ở entry trước'}
    >
      <Sparkles size={11} />
      Dịch
    </button>
  )
}


function TranslationBadges({
  translations,
}: {
  translations: ApiChapterTranslation[]
}) {
  const done = translations.filter((t) => t.state === 'done')
  const running = translations.filter((t) => t.state === 'running' || t.state === 'pending')
  const errored = translations.filter((t) => t.state === 'error')

  // Distinct done languages.
  const doneLangs = new Map<string, ApiChapterTranslation>()
  for (const t of done) if (!doneLangs.has(t.target_lang)) doneLangs.set(t.target_lang, t)

  if (doneLangs.size === 0 && running.length === 0 && errored.length === 0) {
    return null
  }
  return (
    <div className="inline-flex items-center gap-1 shrink-0">
      {[...doneLangs.values()].slice(0, 3).map((t) => (
        <span
          key={t.id}
          className="inline-flex items-center gap-0.5 h-4 px-1 rounded-xs bg-success/15 text-success-text text-[10px] font-semibold uppercase"
          title={t.creator_name ? `Bản của ${t.creator_name}` : 'Bản dịch sẵn'}
        >
          <CheckCircle2 size={8} />
          {t.target_lang}
        </span>
      ))}
      {running.length > 0 && (
        <span
          className="inline-flex items-center gap-0.5 h-4 px-1 rounded-xs bg-info/15 text-info-text text-[10px] font-semibold"
          title={`${running.length} đang dịch`}
        >
          <Loader2 size={8} className="animate-spin" />
          {running.length}
        </span>
      )}
      {errored.length > 0 && (
        <span
          className="inline-flex items-center gap-0.5 h-4 px-1 rounded-xs bg-error/15 text-error-text text-[10px] font-semibold"
          title={`${errored.length} lỗi`}
        >
          <AlertCircle size={8} />
          {errored.length}
        </span>
      )}
    </div>
  )
}


function pickReadable(
  translations: ApiChapterTranslation[],
  targetLang:   string | null,
): ApiChapterTranslation | null {
  const done = translations.filter((t) => t.state === 'done')
  if (done.length === 0) return null
  if (targetLang) {
    const match = done.find((t) => t.target_lang === targetLang)
    if (match) return match
  }
  return done[0] ?? null
}
