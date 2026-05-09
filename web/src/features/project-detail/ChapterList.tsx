import { Plus, Sparkles } from 'lucide-react'
import type { ApiChapter } from '@shared/api/api'
import { Button } from '@shared/ui/Button'
import { cn } from '@shared/lib/cn'
import { DataToolbar, SearchInput } from '@shared/ui/DataToolbar'
import { DataTable, Th } from '@shared/ui/DataTable'
import { EmptyState } from '@shared/ui/EmptyState'
import { useDelayedFlag } from '@shared/lib/useDelayedFlag'
import { ChapterRow } from './ChapterRow'
import type { ChapterStats } from './chapter'
import { FILTERS, type Filter } from './filter'
import { useChapterMutations } from './mutations'

interface Props {
  projectId:  number
  isOwner:    boolean
  chapters:   ApiChapter[]
  loading:    boolean
  stats:      ChapterStats
  filter:     Filter
  setFilter:  (f: Filter) => void
  q:          string
  setQ:       (s: string) => void
  sel:        Set<number>
  toggleOne:  (id: number) => void
  toggleAll:  () => void
  allChecked: boolean
  onPull:     () => void
}

export function ChapterList({
  projectId, isOwner, chapters, loading, stats,
  filter, setFilter, q, setQ,
  sel, toggleOne, toggleAll, allChecked, onPull,
}: Props) {
  const mutations = useChapterMutations(projectId)
  // Only show skeleton if loading takes longer than 250ms — avoids a
  // 1-frame flash when the cache hits or the API responds quickly.
  const showSkeleton = useDelayedFlag(loading, 250)
  const showEmpty    = !loading && chapters.length === 0

  return (
    <>
      <DataToolbar>
        <div className="flex flex-wrap items-center gap-2 w-full">
          <div className="overflow-x-auto">
            <Segmented value={filter} onChange={setFilter} stats={stats} />
          </div>
          <SearchInput value={q} onChange={setQ} placeholder="Tìm chương…" className="flex-1 min-w-32" />
        </div>
      </DataToolbar>

      <DataTable className="overflow-x-auto">
        <thead>
          <tr className="bg-surface-2">
            <Th className="w-10">
              <button
                onClick={toggleAll}
                disabled={chapters.length === 0}
                className={cn(
                  'size-4 rounded-xs border flex items-center justify-center cursor-pointer transition-colors disabled:opacity-30 disabled:cursor-not-allowed',
                  allChecked
                    ? 'bg-accent border-accent text-accent-fg'
                    : 'border-text-subtle hover:border-text-muted',
                )}
                aria-label="Chọn tất cả"
              >
                {allChecked && <CheckIcon />}
              </button>
            </Th>
            <Th>Chương</Th>
            <Th className="w-64 hidden sm:table-cell">Trạng thái</Th>
            <Th className="w-24 hidden sm:table-cell">Cập nhật</Th>
            <Th className="w-20 text-right pr-3">Thao tác</Th>
          </tr>
        </thead>
        <tbody>
          {showSkeleton && Array.from({ length: 6 }).map((_, i) => (
            <tr key={i} className="border-b border-border-soft last:border-0">
              <td colSpan={5} className="px-4 py-3.5">
                <div className="h-3 rounded bg-surface-2 animate-pulse" />
              </td>
            </tr>
          ))}

          {showEmpty && (
            <tr>
              <td colSpan={5}>
                <EmptyState
                  icon={Sparkles}
                  title={stats.total === 0 ? 'Chưa có chương nào' : 'Không có chương phù hợp'}
                  hint={
                    stats.total === 0
                      ? 'Tải chương đầu tiên để bắt đầu dịch.'
                      : (q || filter !== 'all' ? 'Thử từ khoá khác hoặc bỏ bộ lọc.' : 'Thêm chương để bắt đầu.')
                  }
                  action={stats.total === 0 && isOwner && (
                    <Button variant="primary" onClick={onPull}>
                      <Plus size={12} />
                      Tải chương
                    </Button>
                  )}
                />
              </td>
            </tr>
          )}

          {!loading && chapters.map((ch) => (
            <ChapterRow
              key={ch.chapter_id}
              ch={ch}
              isOwner={isOwner}
              projectId={projectId}
              checked={sel.has(ch.chapter_id)}
              onToggle={() => toggleOne(ch.chapter_id)}
              mutations={mutations}
            />
          ))}
        </tbody>
      </DataTable>

      {!loading && chapters.length > 0 && (
        <p className="text-xs text-text-subtle mt-3 tabular">
          Hiển thị <span className="text-text-muted">{chapters.length}</span> trong <span className="text-text-muted">{stats.total}</span> chương
        </p>
      )}
    </>
  )
}

// ── internals ──────────────────────────────────────────────────────────────

function CheckIcon() {
  return (
    <svg width="9" height="9" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="3">
      <path d="M5 12l5 5 9-11" />
    </svg>
  )
}

function Segmented({
  value, onChange, stats,
}: {
  value:    Filter
  onChange: (f: Filter) => void
  stats:    ChapterStats
}) {
  const counts: Record<Filter, number> = {
    all:     stats.total,
    running: stats.running,
    idle:    stats.total - stats.done - stats.running - stats.error,
    done:    stats.done,
    error:   stats.error,
  }
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
