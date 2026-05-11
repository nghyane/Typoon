import { createFileRoute, Link } from '@tanstack/react-router'
import { useEffect } from 'react'
import { useQuery } from '@tanstack/react-query'
import {
  BookOpen, Sparkles, CheckCircle2, AlertCircle, Loader2, Settings2,
} from 'lucide-react'
import { Cover } from '@shared/ui/Cover'
import { EmptyState } from '@shared/ui/EmptyState'
import { Spinner } from '@shared/ui/primitives'
import { cn } from '@shared/lib/cn'
import { timeAgo } from '@shared/lib/time'
import { useHeaderStore } from '../store/header'
import { api, type ApiMyTranslation, type DraftState } from '@shared/api/api'

// =============================================================================
// /translate — list of the user's translations, newest activity first.
//
// Cards group by (material, target_lang) so a user dịch 10 chapter cùng
// Naruto sang VN thấy 1 group "Naruto · VN · 10 chương". Click vào
// group → group detail (chapter list) or stay flat — we keep flat for
// now (one row per chapter) because that mirrors the LLM unit of work
// and aligns with the SSE / progress chip flow.
// =============================================================================

function MyTranslationsPage() {
  const setHeader   = useHeaderStore((s) => s.set)
  const clearHeader = useHeaderStore((s) => s.clear)
  useEffect(() => {
    setHeader('Bản dịch của tôi', [])
    return () => clearHeader()
  }, [setHeader, clearHeader])

  const { data: items = [], isPending } = useQuery({
    queryKey: ['translations', 'mine'],
    queryFn:  api.listMyTranslations,
    staleTime: 30_000,
    refetchOnWindowFocus: true,
  })

  if (isPending) {
    return (
      <div className="flex items-center justify-center py-24">
        <Spinner size={20} />
      </div>
    )
  }

  if (items.length === 0) {
    return (
      <div className="px-4 sm:px-6 pt-4 sm:pt-6">
        <EmptyState
          icon={Sparkles}
          title="Chưa có bản dịch nào"
          hint="Mở một truyện ở Duyệt nguồn và bấm Dịch trên chương bạn muốn."
          action={
            <Link
              to="/browse"
              className="inline-flex items-center gap-1.5 h-8 px-3 rounded-sm bg-accent text-accent-fg text-[13px] font-medium hover:bg-accent-hover cursor-pointer"
            >
              Duyệt nguồn
            </Link>
          }
        />
      </div>
    )
  }

  // Group by material — operator/translator works on a series at a time.
  const groups = groupByMaterial(items)

  return (
    <div className="px-4 sm:px-6 pt-4 sm:pt-6 pb-16">
      <div className="space-y-6">
        {groups.map((g) => (
          <MaterialGroup key={g.material_id} group={g} />
        ))}
      </div>
    </div>
  )
}

// ── Grouping ──────────────────────────────────────────────────────────

interface Group {
  material_id:    number
  material_title: string
  material_cover: string | null
  source:         string | null
  upstream_ref:   string | null
  items:          ApiMyTranslation[]
}

function groupByMaterial(items: ApiMyTranslation[]): Group[] {
  const by = new Map<number, Group>()
  for (const t of items) {
    let g = by.get(t.material_id)
    if (!g) {
      g = {
        material_id:    t.material_id,
        material_title: t.material_title,
        material_cover: t.material_cover,
        source:         t.material_source,
        upstream_ref:   t.material_upstream_ref,
        items:          [],
      }
      by.set(t.material_id, g)
    }
    g.items.push(t)
  }
  // Sort groups by most-recent chapter activity inside.
  return [...by.values()].sort((a, b) => {
    const am = a.items[0]?.updated_at ?? ''
    const bm = b.items[0]?.updated_at ?? ''
    return bm.localeCompare(am)
  })
}

// ── Material group card ──────────────────────────────────────────────

function MaterialGroup({ group }: { group: Group }) {
  // Sort chapters by position desc (latest first), matching the
  // chronological order users think in.
  const sorted = [...group.items].sort(
    (a, b) => b.chapter_position - a.chapter_position,
  )

  // Stats line: how many done / running / errored across this group.
  const stats = sorted.reduce(
    (acc, t) => {
      acc[t.state] = (acc[t.state] ?? 0) + 1
      return acc
    },
    {} as Record<DraftState, number>,
  )

  return (
    <section className="rounded-md bg-surface overflow-hidden">
      {/* Group header */}
      <header className="flex items-center gap-3 p-3 border-b border-border-soft">
        <Cover
          src={group.material_cover}
          title={group.material_title}
          className="size-12 rounded-sm shrink-0"
        />
        <div className="flex-1 min-w-0">
          <p className="text-sm font-semibold text-text truncate">
            {group.material_title}
          </p>
          <p className="text-[11px] text-text-subtle mt-0.5 flex items-center gap-2">
            <span>{sorted.length} chương</span>
            {stats.done    > 0 && <span className="text-success-text">✓ {stats.done}</span>}
            {stats.running > 0 && <span className="text-accent-text">⟳ {stats.running}</span>}
            {stats.error   > 0 && <span className="text-error-text">✗ {stats.error}</span>}
            {stats.pending > 0 && <span className="text-text-muted">… {stats.pending}</span>}
          </p>
        </div>
        {group.source && group.upstream_ref && (
          <Link
            to="/browse/$source/manga/$mangaId"
            params={{
              source:  group.source,
              mangaId: encodeURIComponent(group.upstream_ref),
            }}
            className="text-[11px] text-text-subtle hover:text-text shrink-0"
            title="Xem trang truyện"
          >
            <BookOpen size={14} />
          </Link>
        )}
      </header>

      {/* Chapter list */}
      <ul className="divide-y divide-border-soft">
        {sorted.map((t) => (
          <TranslationRow key={t.translation_id} t={t} />
        ))}
      </ul>
    </section>
  )
}

// ── Per-row ───────────────────────────────────────────────────────────

function TranslationRow({ t }: { t: ApiMyTranslation }) {
  const StateIcon = STATE_ICON[t.state]
  return (
    <li>
      <Link
        to="/translate/$translationId"
        params={{ translationId: String(t.translation_id) }}
        className="flex items-center gap-3 px-3 py-2.5 hover:bg-hover transition-colors cursor-pointer"
      >
        <span className={cn(
          'size-7 rounded-sm flex items-center justify-center shrink-0',
          STATE_BG[t.state],
        )}>
          <StateIcon size={13} className={STATE_FG[t.state]} />
        </span>

        <div className="flex-1 min-w-0">
          <p className="text-sm text-text truncate">
            {t.chapter_label ?? `Chương ${t.chapter_number}`}
          </p>
          <p className="text-[11px] text-text-subtle mt-0.5 flex items-center gap-2">
            <span className="uppercase font-semibold">{t.target_lang}</span>
            {t.updated_at && <span>· {timeAgo(t.updated_at)}</span>}
          </p>
        </div>

        <Settings2 size={13} className="text-text-subtle shrink-0" />
      </Link>
    </li>
  )
}

const STATE_ICON: Record<DraftState, typeof CheckCircle2> = {
  done:    CheckCircle2,
  running: Loader2,
  pending: Loader2,
  error:   AlertCircle,
}

const STATE_FG: Record<DraftState, string> = {
  done:    'text-success-text',
  running: 'text-accent-text',
  pending: 'text-text-muted',
  error:   'text-error-text',
}

const STATE_BG: Record<DraftState, string> = {
  done:    'bg-success/15',
  running: 'bg-accent/15',
  pending: 'bg-surface-2',
  error:   'bg-error/15',
}

export const Route = createFileRoute('/translate/')({
  component: MyTranslationsPage,
})
