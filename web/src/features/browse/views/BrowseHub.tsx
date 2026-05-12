import { useEffect } from 'react'
import { Link } from '@tanstack/react-router'
import { Compass, Search } from 'lucide-react'
import { useHeaderStore } from '../../../store/header'
import { cn } from '@shared/lib/cn'
import { EmptyState } from '@shared/ui/EmptyState'
import { Button } from '@shared/ui/Button'
import { input as inputCls } from '@shared/ui/primitives'
import { useSources, useEnabledSources } from '../sources'
import { useLibrary } from '@features/library/store'
import { useShallow } from 'zustand/react/shallow'
import { LibraryRailCard } from '@features/library/views/LibraryCard'
import { SourceCard, InstallSourceCard } from './SourceCard'
import { Shelf } from './Shelf'

// =============================================================================
// BrowseHub — /browse landing.
//
// Pro design (slice 8): reader-first surface. Sections in order of how
// users actually open this page:
//
//   1. ⏯ Tiếp tục đọc      Cross-source rail from local reading history.
//                          Hides when user has nothing in progress.
//   2. 📚 Nguồn            Manifest sources cards. Demoted from top
//                          (Tachiyomi pattern) but still primary
//                          discovery for users who think in sources.
//   3. + Cài nguồn          Trailing affordance for repo / file installs.
//
// Global search placeholder remains — fanout search lands in slice 9.
// =============================================================================

export function BrowseHub() {
  const setHeader   = useHeaderStore((s) => s.set)
  const clearHeader = useHeaderStore((s) => s.clear)
  useEffect(() => {
    setHeader('Duyệt nguồn', [])
    return () => clearHeader()
  }, [setHeader, clearHeader])

  const ensureBundled = useSources((s) => s.ensureBundled)
  useEffect(() => { ensureBundled() }, [ensureBundled])

  const sources = useEnabledSources()

  // Local "Tiếp tục đọc" — across every source the user has touched.
  // Sorted by lastReadAt; limited so the rail stays one screen tall.
  const continueItems = useLibrary(
    useShallow((s) =>
      Object.values(s.items)
        .filter((e) => e.lastReadAt !== null)
        .sort((a, b) => (b.lastReadAt ?? 0) - (a.lastReadAt ?? 0))
        .slice(0, 12),
    ),
  )

  return (
    <div className="px-4 sm:px-6 pt-4 sm:pt-6 pb-12">
      {/* Search row — disabled stub until fanout search ships. */}
      <label
        className={cn(
          inputCls,
          'flex items-center gap-2 cursor-text h-9 mb-6 max-w-xl',
        )}
      >
        <Search size={14} className="text-text-subtle shrink-0" />
        <input
          type="text"
          placeholder="Tìm trên tất cả nguồn… (Sắp ra mắt)"
          disabled
          className="flex-1 bg-transparent outline-none text-sm placeholder:text-text-subtle text-text min-w-0 disabled:cursor-not-allowed"
        />
        <kbd className="hidden sm:inline text-[10px] font-mono bg-bg/40 rounded-xs px-1.5 py-0.5 text-text-subtle">
          ⌘K
        </kbd>
      </label>

      {/* Continue reading rail */}
      {continueItems.length > 0 && (
        <Shelf label="Tiếp tục đọc">
          {continueItems.map((entry) => (
            <LibraryRailCard
              key={`${entry.source}::${entry.mangaUrl}`}
              entry={entry}
            />
          ))}
        </Shelf>
      )}

      {/* Source picker */}
      <section className="mt-2">
        <div className="flex items-baseline gap-2 mb-3">
          <h2 className="text-sm font-semibold text-text">Nguồn</h2>
          <span className="text-[11px] text-text-subtle">
            {sources.length} nguồn đã cài
          </span>
        </div>

        {sources.length === 0 ? (
          <EmptyState
            icon={Compass}
            title="Chưa có nguồn nào"
            hint="Cài nguồn để bắt đầu duyệt truyện."
            action={
              <Link to="/settings">
                <Button variant="primary">Cài nguồn đầu tiên</Button>
              </Link>
            }
          />
        ) : (
          // Auto-fill so card count fills the row. Each card is a
          // Discord-channel-row shape — same density as /settings.
          <div className="grid grid-cols-[repeat(auto-fill,minmax(320px,1fr))] gap-2">
            {sources.map((s) => (
              <SourceCard key={s.manifest.id} source={s} />
            ))}
            <InstallSourceCard />
          </div>
        )}
      </section>
    </div>
  )
}
