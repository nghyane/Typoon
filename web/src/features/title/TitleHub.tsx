import { useEffect, useMemo, useState } from 'react'
import { Link } from '@tanstack/react-router'
import {
  ArrowLeft, AlertTriangle, BookOpen, Sparkles, Loader2,
  CheckCircle2, AlertCircle, Clock,
} from 'lucide-react'
import { Cover, coverUrl } from '@shared/ui/Cover'
import { EmptyState } from '@shared/ui/EmptyState'
import { Spinner, Tag } from '@shared/ui/primitives'
import { Button } from '@shared/ui/Button'
import { DataTable, Th } from '@shared/ui/DataTable'
import { DataToolbar, SearchInput } from '@shared/ui/DataToolbar'
import { cn } from '@shared/lib/cn'
import { timeAgo } from '@shared/lib/time'
import { useHeaderStore } from '../../store/header'
import { useSources } from '@features/browse/sources'
import { FollowButton } from '@features/library/views/LibraryCard'
import type {
  ApiChapterTranslation, ApiLibraryEntry, ApiMaterial, LibraryStatus,
} from '@shared/api/api'
import { useHubData, type HubChapterRow } from './useHubData'

// =============================================================================
// TitleHub — `/title/$entryId` detail page, pro-design.
//
// Reference: pre-refactor ProjectDetail (commit f8977dd). Same density
// ladder, same DataTable shell, same DataToolbar filter pattern. The
// material-architecture rebuild keeps the visual contract.
//
// Layout:
//   ① Hero          80px cover, title, lang pair + status + age,
//                   description inline (not collapsible), action
//                   cluster right.
//   ② Toolbar       filter segmented + search input.
//   ③ DataTable     Chương / Trạng thái / Cập nhật / Thao tác cols.
//
// Action buttons are disabled placeholders for slice 14 (reader) and
// slice 15 (inline spawn). Everything else is wired.
// =============================================================================

interface Props { entryId: number }

export function TitleHub({ entryId }: Props) {
  // Source registry hydration on direct refresh — same pattern as
  // /library and /settings.
  const ensureBundled = useSources((s) => s.ensureBundled)
  useEffect(() => { ensureBundled() }, [ensureBundled])

  const { entry, material, rows, loading, chaptersLoading, error } = useHubData(entryId)

  const setHeader   = useHeaderStore((s) => s.set)
  const clearHeader = useHeaderStore((s) => s.clear)
  useEffect(() => {
    if (material) {
      setHeader(material.material.title, [{ label: 'Thư viện', to: '/library' }])
    } else {
      setHeader('', [{ label: 'Thư viện', to: '/library' }])
    }
    return () => clearHeader()
  }, [material, setHeader, clearHeader])

  if (loading) {
    return (
      <div className="flex items-center justify-center py-24">
        <Spinner size={20} />
      </div>
    )
  }
  if (error || !entry) {
    return (
      <div className="px-4 sm:px-6 pt-12">
        <EmptyState
          icon={AlertTriangle}
          title="Không tải được"
          hint={(error as Error)?.message ?? 'Entry không tồn tại hoặc đã bị xoá.'}
        />
      </div>
    )
  }
  if (!material) {
    return (
      <div className="px-4 sm:px-6 pt-12">
        <EmptyState
          icon={AlertTriangle}
          title="Entry chưa có material chính"
          hint="Mở Cài đặt để link một material làm chính cho entry này."
        />
      </div>
    )
  }

  return (
    <div className="pb-16">
      <MobileBack />
      <Hero entry={entry} material={material.material} />
      <ChapterPanel
        entry={entry}
        rows={rows}
        loading={chaptersLoading}
      />
    </div>
  )
}


// ── Mobile back link ────────────────────────────────────────────────

function MobileBack() {
  return (
    <div className="sm:hidden px-4 pt-4">
      <Link
        to="/library"
        className="inline-flex items-center gap-1.5 text-sm text-text-subtle hover:text-text"
      >
        <ArrowLeft size={14} />
        Thư viện
      </Link>
    </div>
  )
}


// ── Hero ────────────────────────────────────────────────────────────

const STATUS_LABEL: Record<LibraryStatus, string> = {
  reading: 'Đang đọc',
  plan:    'Kế hoạch',
  on_hold: 'Tạm dừng',
  done:    'Đã xong',
  dropped: 'Đã bỏ',
}

function Hero({
  entry, material,
}: {
  entry: ApiLibraryEntry; material: ApiMaterial
}) {
  return (
    <div className="px-4 sm:px-6 pt-4 sm:pt-6 pb-4 sm:pb-5 flex items-start gap-3 sm:gap-4">
      <Cover
        src={coverUrl(material.cover_url, material.updated_at)}
        title={material.title}
        fontSize="text-xl"
        className="w-20 aspect-[2/3] rounded-md shrink-0"
      />
      <div className="flex-1 min-w-0">
        <div className="flex flex-col sm:flex-row sm:items-start sm:justify-between gap-2 sm:gap-3">
          <div className="min-w-0">
            <h1 className="text-lg sm:text-2xl font-semibold tracking-tight text-text line-clamp-2">
              {material.title}
            </h1>

            <div className="flex items-center gap-2 mt-2 flex-wrap text-xs text-text-subtle">
              <span className="inline-flex items-center gap-1 h-[22px] px-2 rounded-xs bg-surface-2 text-[11px] font-semibold uppercase tracking-wider text-text-muted">
                {material.languages[0]?.toUpperCase() ?? '?'}
                <span className="text-text-subtle">→</span>
                {(entry.target_lang ?? '?').toUpperCase()}
              </span>
              <Tag tone="neutral" size="sm">{STATUS_LABEL[entry.status]}</Tag>
              {material.author && <span>{material.author}</span>}
              {material.status && <span>· {material.status}</span>}
              {material.nsfw && (
                <span className="text-[11px] uppercase font-semibold px-1.5 py-0.5 rounded-xs bg-error/15 text-error-text">
                  NSFW
                </span>
              )}
              {material.updated_at && (
                <>
                  <span>·</span>
                  <span className="inline-flex items-center gap-1">
                    <Clock size={11} />
                    Cập nhật {timeAgo(material.updated_at)}
                  </span>
                </>
              )}
            </div>

            {material.description && (
              <p className="mt-3 text-sm text-text-muted leading-relaxed line-clamp-2 max-w-2xl">
                {material.description}
              </p>
            )}

            <ActivityRow summary={entry.translation_summary} />
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
      </div>
    </div>
  )
}


function ActivityRow({
  summary,
}: {
  summary: ApiLibraryEntry['translation_summary']
}) {
  if (!summary) return null
  if (summary.running === 0 && summary.error === 0 && summary.pending === 0) {
    return null
  }
  return (
    <div className="mt-2 flex items-center gap-1.5">
      {summary.running > 0 && (
        <Tag tone="info" size="sm">{summary.running} đang dịch</Tag>
      )}
      {summary.error > 0 && (
        <Tag tone="error" size="sm">{summary.error} lỗi</Tag>
      )}
      {summary.pending > 0 && (
        <Tag tone="warning" size="sm">{summary.pending} chờ</Tag>
      )}
    </div>
  )
}


// ── Chapter panel ───────────────────────────────────────────────────

type Filter = 'all' | 'translated' | 'untranslated' | 'in_progress'

const FILTERS: Array<{ key: Filter; label: string }> = [
  { key: 'all',           label: 'Tất cả'      },
  { key: 'translated',    label: 'Đã dịch'    },
  { key: 'in_progress',   label: 'Đang dịch'  },
  { key: 'untranslated',  label: 'Raw'         },
]

function ChapterPanel({
  entry, rows, loading,
}: {
  entry:   ApiLibraryEntry
  rows:    HubChapterRow[]
  loading: boolean
}) {
  const [filter, setFilter] = useState<Filter>('all')
  const [q,      setQ]      = useState('')

  const counts: Record<Filter, number> = useMemo(() => {
    let translated = 0, inProgress = 0, untranslated = 0
    for (const r of rows) {
      const done    = r.translations.some((t) => t.state === 'done')
      const running = r.translations.some((t) => t.state === 'running' || t.state === 'pending')
      if (done) translated++
      else if (running) inProgress++
      else untranslated++
    }
    return { all: rows.length, translated, in_progress: inProgress, untranslated }
  }, [rows])

  const filtered = useMemo(() => {
    const term = q.trim().toLowerCase()
    return rows.filter((r) => {
      const done    = r.translations.some((t) => t.state === 'done')
      const running = r.translations.some((t) => t.state === 'running' || t.state === 'pending')
      if (filter === 'translated'   && !done)    return false
      if (filter === 'in_progress'  && !running) return false
      if (filter === 'untranslated' && (done || running)) return false
      if (term && !`${r.number} ${r.label ?? ''}`.toLowerCase().includes(term)) {
        return false
      }
      return true
    })
  }, [rows, filter, q])

  return (
    <section className="px-4 sm:px-6">
      <DataToolbar>
        <div className="flex flex-wrap items-center gap-2 w-full">
          <Segmented value={filter} onChange={setFilter} counts={counts} />
          <SearchInput
            value={q}
            onChange={setQ}
            placeholder="Tìm chương…"
            className="flex-1 min-w-32"
          />
        </div>
      </DataToolbar>

      <DataTable className="overflow-x-auto">
        <thead>
          <tr className="bg-surface-2">
            <Th>Chương</Th>
            <Th className="w-72 hidden sm:table-cell">Trạng thái</Th>
            <Th className="w-24 hidden sm:table-cell">Cập nhật</Th>
            <Th className="w-32 text-right pr-3">Thao tác</Th>
          </tr>
        </thead>
        <tbody>
          {loading && rows.length === 0 && (
            Array.from({ length: 6 }).map((_, i) => (
              <tr key={i} className="border-b border-border-soft last:border-0">
                <td colSpan={4} className="px-4 py-3.5">
                  <div className="h-3 rounded bg-surface-2 animate-pulse" />
                </td>
              </tr>
            ))
          )}

          {!loading && filtered.length === 0 && (
            <tr>
              <td colSpan={4}>
                <EmptyState
                  icon={Sparkles}
                  title={rows.length === 0 ? 'Không có chương đọc được' : 'Không có chương phù hợp'}
                  hint={rows.length === 0
                    ? 'Nguồn không trả về chương nào ở ngôn ngữ này.'
                    : (q || filter !== 'all'
                        ? 'Thử từ khoá khác hoặc bỏ bộ lọc.'
                        : 'Không có dữ liệu để hiện.')}
                />
              </td>
            </tr>
          )}

          {!loading && filtered.map((r) => (
            <ChapterRow
              key={r.key}
              row={r}
              targetLang={entry.target_lang}
            />
          ))}
        </tbody>
      </DataTable>

      {!loading && filtered.length > 0 && (
        <p className="text-xs text-text-subtle mt-3 tabular">
          Hiển thị <span className="text-text-muted">{filtered.length}</span> trong{' '}
          <span className="text-text-muted">{rows.length}</span> chương
        </p>
      )}
    </section>
  )
}


function Segmented({
  value, onChange, counts,
}: {
  value:    Filter
  onChange: (v: Filter) => void
  counts:   Record<Filter, number>
}) {
  return (
    <div className="inline-flex items-center gap-0.5">
      {FILTERS.map(({ key, label }) => {
        const n = counts[key]
        const active = value === key
        return (
          <button
            key={key}
            onClick={() => onChange(key)}
            className={cn(
              'h-8 px-3 rounded-sm text-[13px] cursor-pointer transition-colors',
              'inline-flex items-center gap-2',
              active
                ? 'bg-surface-2 text-text font-medium'
                : 'text-text-muted hover:bg-hover hover:text-text',
            )}
          >
            {label}
            {n > 0 && (
              <span className={cn(
                'tabular text-[11px]',
                active ? 'text-text-subtle' : 'text-text-subtle/80',
              )}>
                {n}
              </span>
            )}
          </button>
        )
      })}
    </div>
  )
}


// ── Chapter row ─────────────────────────────────────────────────────

function ChapterRow({
  row, targetLang,
}: {
  row: HubChapterRow; targetLang: string | null
}) {
  const readable = pickReadable(row.translations, targetLang)
  const running  = row.translations.find((t) => t.state === 'running' || t.state === 'pending')
  const errored  = row.translations.find((t) => t.state === 'error')

  return (
    <tr className="group transition-colors border-b border-border-soft last:border-0 hover:bg-hover">
      <td className="px-3 py-3 min-w-0">
        <div className="flex items-baseline gap-2 min-w-0">
          <span className="font-semibold text-text tabular shrink-0">
            Ch.{row.number}
          </span>
          {row.label && (
            <span className="text-sm text-text-muted truncate">{row.label}</span>
          )}
        </div>
        {readable?.creator_name && (
          <div className="mt-1 text-xs text-text-subtle truncate">
            Bản của @{readable.creator_name}
          </div>
        )}
        {/* status inline on mobile */}
        <div className="sm:hidden mt-1">
          <StatusInline
            done={!!readable}
            running={!!running}
            errored={!!errored}
            translations={row.translations}
          />
        </div>
      </td>

      <td className="px-3 py-3 w-72 hidden sm:table-cell">
        <StatusInline
          done={!!readable}
          running={!!running}
          errored={!!errored}
          translations={row.translations}
        />
      </td>

      <td className="px-3 py-3 text-xs text-text-subtle whitespace-nowrap w-24 tabular hidden sm:table-cell">
        {/* HubChapterRow doesn't carry updated_at yet; placeholder dash
            keeps the column aligned. Slice 14 wires real timestamps. */}
        —
      </td>

      <td className="px-3 py-3 w-32">
        <div className="flex items-center gap-1 justify-end">
          <Action readable={readable} targetLang={targetLang} />
        </div>
      </td>
    </tr>
  )
}


function StatusInline({
  done, running, errored, translations,
}: {
  done:        boolean
  running:     boolean
  errored:     boolean
  translations: ApiChapterTranslation[]
}) {
  if (running) {
    return (
      <span className="inline-flex items-center gap-1.5 text-xs text-info-text">
        <Loader2 size={11} className="animate-spin" />
        Đang dịch
      </span>
    )
  }
  if (errored) {
    return (
      <span className="inline-flex items-center gap-1.5 text-xs text-error-text">
        <AlertCircle size={11} />
        Lỗi
      </span>
    )
  }
  if (done) {
    const langs = new Set<string>()
    for (const t of translations) {
      if (t.state === 'done') langs.add(t.target_lang)
    }
    return (
      <span className="inline-flex items-center gap-1.5 text-xs text-success-text">
        <CheckCircle2 size={11} />
        Đã dịch
        <span className="text-text-subtle uppercase ml-1">
          {[...langs].slice(0, 3).join(' · ')}
        </span>
      </span>
    )
  }
  return (
    <span className="inline-flex items-center gap-1.5 text-xs text-text-subtle">
      <span className="size-1.5 rounded-full bg-text-subtle" />
      Raw
    </span>
  )
}


function Action({
  readable, targetLang,
}: {
  readable:   ApiChapterTranslation | null
  targetLang: string | null
}) {
  if (readable) {
    // TODO(slice 14): wire reader route /title/$entryId/ch/$chapterId?tx=
    return (
      <Button size="sm" variant="ghost" disabled>
        <BookOpen size={12} />
        Đọc
      </Button>
    )
  }
  return (
    <Button
      size="sm"
      variant="primary"
      disabled
      title={targetLang ? `Dịch sang ${targetLang.toUpperCase()}` : 'Chọn target_lang trước'}
    >
      <Sparkles size={12} />
      Dịch
    </Button>
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
