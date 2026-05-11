import { useEffect } from 'react'
import { Link } from '@tanstack/react-router'
import { Compass, Search } from 'lucide-react'
import { useHeaderStore } from '../../../store/header'
import { cn } from '@shared/lib/cn'
import { EmptyState } from '@shared/ui/EmptyState'
import { Button } from '@shared/ui/Button'
import { input as inputCls } from '@shared/ui/primitives'
import { useSources, useEnabledSources } from '../sources'
import { SourceCard, InstallSourceCard } from './SourceCard'

// =============================================================================
// BrowseHub — /browse landing.
//
// Layout alignment with /projects (the sibling list page):
//   • Same full-bleed `px-6` (no centered max-width container)
//   • Same auto-fill grid pattern — cards size themselves into N
//     columns based on viewport. With 2-5 sources users still see a
//     full row; expanded viewports fill 4 columns; mobile collapses
//     to single column without media-query branching.
//   • Title lives in app header (useHeaderStore) — no duplicate hero.
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

  return (
    <div className="px-4 sm:px-6 pt-4 sm:pt-6 pb-12">
      {/* Subtitle / context strip */}
      <p className="text-sm text-text-subtle mb-5">
        {sources.length > 0
          ? `${sources.length} nguồn đã cài · Quản lý trong Cài đặt`
          : 'Khám phá truyện từ các nguồn bên ngoài'}
      </p>

      {/* Global search placeholder — disabled until phase 2 */}
      <label
        className={cn(
          inputCls,
          'flex items-center gap-2 cursor-text h-9 mb-6 max-w-md',
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
        // Auto-fill so card count fills the row: 1 col mobile, 2 col
        // sm+, 3-4 col on wide viewports. Mirrors `/projects` grid
        // pattern so /browse doesn't read as a different surface.
        <div className="grid grid-cols-[repeat(auto-fill,minmax(320px,1fr))] gap-2">
          {sources.map((s) => (
            <SourceCard key={s.manifest.id} source={s} />
          ))}
          <InstallSourceCard />
        </div>
      )}
    </div>
  )
}
